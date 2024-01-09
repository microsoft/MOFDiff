"""
CG Diffusion for MOF generation.
"""
from typing import Any

import numpy as np
import hydra
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter
from tqdm import tqdm

from mofdiff.common.data_utils import (
    EPSILON,
    cart_to_frac_coords,
    frac_to_cart_coords,
    min_distance_sqr_pbc,
    lengths_angles_to_volume,
    mard,
)


def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(fc_num_layers - 1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)


def subtract_cog(x, num_atoms):
    batch = torch.arange(num_atoms.size(0), device=num_atoms.device).repeat_interleave(
        num_atoms, dim=0
    )
    cog = scatter(x, batch, dim=0, reduce="mean").repeat_interleave(num_atoms, dim=0)
    return x - cog


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class VP(nn.Module):
    """
    variance preserving diffusion.
    """

    def __init__(self, num_steps=1000, s=0.0001, power=2, clipmax=0.999):
        super().__init__()
        t = torch.arange(0, num_steps + 1, dtype=torch.float)
        # cosine schedule introduced in https://arxiv.org/abs/2102.09672
        f_t = torch.cos((np.pi / 2) * ((t / num_steps) + s) / (1 + s)) ** power
        alpha_bars = f_t / f_t[0]
        betas = torch.cat(
            [torch.zeros([1]), 1 - (alpha_bars[1:] / alpha_bars[:-1])], dim=0
        )
        betas = betas.clamp_max(clipmax)
        sigmas = torch.sqrt(betas[1:] * ((1 - alpha_bars[:-1]) / (1 - alpha_bars[1:])))
        sigmas = torch.cat([torch.zeros([1]), sigmas], dim=0)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("betas", betas)
        self.register_buffer("sigmas", sigmas)

    def forward(self, h0, t):
        alpha_bar = self.alpha_bars[t]
        eps = torch.randn_like(h0)
        ht = (
            torch.sqrt(alpha_bar).view(-1, 1) * h0
            + torch.sqrt(1 - alpha_bar).view(-1, 1) * eps
        )
        return ht, eps

    def reverse(self, ht, eps_h, t):
        alpha = 1 - self.betas[t]
        alpha = alpha.clamp_min(1 - self.betas[-2])
        alpha_bar = self.alpha_bars[t]
        sigma = self.sigmas[t].view(-1, 1)

        z = torch.where(
            (t > 1)[:, None].expand_as(ht),
            torch.randn_like(ht),
            torch.zeros_like(ht),
        )

        return (1.0 / torch.sqrt(alpha + EPSILON)).view(-1, 1) * (
            ht - ((1 - alpha) / torch.sqrt(1 - alpha_bar + EPSILON)).view(-1, 1) * eps_h
        ) + sigma * z


class VE_pbc(nn.Module):
    """
    variance exploding diffusion under periodic boundary condition.
    """

    def __init__(self, num_steps, sigma_min, sigma_max):
        super().__init__()
        self.T = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.register_buffer(
            "sigmas",
            torch.exp(torch.linspace(np.log(sigma_min), np.log(sigma_max), self.T + 1)),
        )

    def forward(self, x0, t, lengths, angles, num_atoms, **kwargs):
        """
        x0 should be wrapped cart coords.
        """
        used_sigmas = self.sigmas[t].view(-1, 1)
        eps_x = torch.randn_like(x0) * used_sigmas
        frac_p_noisy = cart_to_frac_coords(x0 + eps_x, lengths, angles, num_atoms)
        cart_p_noisy = frac_to_cart_coords(frac_p_noisy, lengths, angles, num_atoms)
        _, wrapped_eps_x = min_distance_sqr_pbc(
            cart_p_noisy,
            x0,
            lengths,
            angles,
            num_atoms,
            x0.device,
            return_vector=True,
        )
        return frac_p_noisy, wrapped_eps_x, used_sigmas

    def reverse(self, xt, epx_x, t, lengths, angles, num_atoms):
        """
        xt should be wrapped cart coords.
        """
        sigmas = self.sigmas[t].view(-1, 1)
        adjacent_sigmas = torch.where(
            (t == 0).view(-1, 1),
            torch.zeros_like(sigmas),
            self.sigmas[t - 1].view(-1, 1),
        )
        cart_p_mean = xt - epx_x * (sigmas**2 - adjacent_sigmas**2)
        # the sign of eps_p here is related to the verification above.
        cart_p_rand = torch.sqrt(
            (adjacent_sigmas**2 * (sigmas**2 - adjacent_sigmas**2))
            / (sigmas**2)
        ) * torch.randn_like(xt)
        cart_p_next = cart_p_mean + cart_p_rand  # before wrapping
        frac_p_next = cart_to_frac_coords(cart_p_next, lengths, angles, num_atoms)
        return frac_p_next


class MOFDiff(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.global_training_step = nn.Parameter(
            torch.tensor(0, dtype=torch.float), requires_grad=False
        )

        bb_emb_dim = kwargs.get("bb_emb_dim")
        self.bb_emb_dim = bb_emb_dim

        # encode
        self.encoder = hydra.utils.instantiate(
            self.hparams.encoder,
            num_targets=self.hparams.latent_dim,
            type_dim=bb_emb_dim,
        )

        self.lattice_natom_shortcut = build_mlp(
            self.hparams.latent_dim + self.hparams.max_bbs + 7,
            self.hparams.hidden_dim,
            self.hparams.fc_num_layers,
            self.hparams.latent_dim,
        )

        self.fc_mu = nn.Linear(self.hparams.latent_dim, self.hparams.latent_dim)
        self.fc_var = nn.Linear(self.hparams.latent_dim, self.hparams.latent_dim)

        # stats
        self.fc_lattice = build_mlp(
            self.hparams.latent_dim,
            self.hparams.hidden_dim,
            self.hparams.fc_num_layers,
            6,
        )

        self.fc_num_atoms = build_mlp(
            self.hparams.latent_dim,
            self.hparams.hidden_dim,
            self.hparams.fc_num_layers,
            self.hparams.max_bbs + 1,
        )

        self.fc_n_metal = build_mlp(
            self.hparams.latent_dim,
            self.hparams.hidden_dim,
            self.hparams.fc_num_layers,
            self.hparams.max_bbs + 1,
        )

        # for property prediction.
        if self.hparams.predict_property:
            self.fc_property = build_mlp(
                self.hparams.latent_dim,
                self.hparams.hidden_dim,
                self.hparams.fc_num_layers,
                self.hparams.property_dim,
            )

        # decode
        self.t_emb = GaussianFourierProjection(
            self.hparams.t_emb_dim // 2, self.hparams.fourier_scale
        )

        self.decoder = hydra.utils.instantiate(
            self.hparams.decoder,
            in_size_atom=bb_emb_dim + self.hparams.t_emb_dim,
            output_atom_types=bb_emb_dim,
        )

        self.beta = self.hparams.beta

        # diffusion
        self.T = self.hparams.diffusion_step

        self.pos_diffusion = VE_pbc(
            self.T,
            sigma_min=self.hparams.diffusion.pos.sigma_min,
            sigma_max=self.hparams.diffusion.pos.sigma_max,
        )

        self.type_diffusion = VP(
            self.T,
            power=self.hparams.diffusion.type.power,
            clipmax=self.hparams.diffusion.type.clipmax,
        )

        # normalization from kwargs and saved as part of network.
        norm_x = kwargs.get("norm_x", self.hparams.norm_x)
        norm_h = kwargs.get("norm_h", self.hparams.norm_h)

        if norm_x != self.hparams.norm_x or norm_h != self.hparams.norm_h:
            print(
                f"normalization factors from dataset: x: {norm_x:.2f} | h: {norm_h:.2f}"
            )

        self.register_buffer("norm_x", torch.tensor(norm_x, dtype=torch.float))
        self.register_buffer("norm_h", torch.tensor(norm_h, dtype=torch.float))

        self.bb_encoder = None
        self.lattice_scaler = None
        self.scaler = None

    def encode(self, batch):
        """
        encode crystal structures to latents.
        """
        hidden = self.encoder(batch, no_emb=True)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)

        # normalize the lattice angles by 90.
        shortcut = torch.cat(
            [
                batch.scaled_lattice[:, :3],
                batch.scaled_lattice[:, 3:6] / 90,
                # pylint: disable=E1102
                F.one_hot(batch.num_atoms, self.hparams.max_bbs + 1),
            ],
            dim=-1,
        )
        mu = self.lattice_natom_shortcut(torch.cat([mu, shortcut], dim=-1))

        # clamp log_var to avoid training instability.
        log_var = torch.clamp(log_var, min=-5, max=5)

        z = self.reparameterize(mu, log_var)

        return mu, log_var, z

    def decode_stats(
        self,
        z,
        gt_num_atoms=None,
        gt_lengths=None,
        gt_angles=None,
        teacher_forcing=False,
    ):
        """
        decode key stats from latent embeddings.
        batch is input during training for teach-forcing.
        """
        if gt_num_atoms is not None:
            lengths_and_angles, lengths, angles = self.predict_lattice(z, gt_num_atoms)
            num_atoms = self.predict_num_atoms(z)
            if self.hparams.teacher_forcing_lattice and teacher_forcing:
                lengths = gt_lengths
                angles = gt_angles
        else:
            num_atoms = self.predict_num_atoms(z).argmax(dim=-1)
            lengths_and_angles, lengths, angles = self.predict_lattice(z, num_atoms)
        return num_atoms, lengths_and_angles, lengths, angles

    def normalize(self, x, h, lengths):
        x = x / self.norm_x
        lengths = lengths / self.norm_x
        h = h / self.norm_h
        return x, h, lengths

    def unnormalize(self, x, h, lengths):
        x = x * self.norm_x
        lengths = lengths * self.norm_x
        h = h * self.norm_h
        return x, h, lengths

    def phi(self, z, x_t, h_t, t_int, num_atoms, lengths=None, angles=None, frac=False):
        t = self.type_diffusion.betas[t_int].view(-1, 1)
        t_emb = self.t_emb(t)
        h_time = torch.cat([h_t, t_emb], dim=1)
        frac_x_t = x_t if frac else cart_to_frac_coords(x_t, lengths, angles, num_atoms)
        pred_eps_x, pred_eps_h = self.decoder(
            z, frac_x_t, h_time, num_atoms, lengths, angles, no_emb=True
        )
        used_sigmas_x = self.pos_diffusion.sigmas[t_int].view(-1, 1)
        pred_eps_x = subtract_cog(pred_eps_x, num_atoms)
        return pred_eps_x / used_sigmas_x, pred_eps_h

    def compute_error(self, pred_eps, eps, batch, weights=None):
        """Computes error, i.e. the most likely prediction of x."""
        if weights is None:
            error = scatter(((eps - pred_eps) ** 2), batch.batch, dim=0, reduce="mean")
        else:
            error = scatter(
                weights * ((eps - pred_eps) ** 2), batch.batch, dim=0, reduce="mean"
            )
        if len(error.shape) > 1:
            error = error.sum(-1)
        return error

    def diffusion_loss(self, z, x, h, batch, lengths=None, angles=None, t_int=None):
        """
        input x has to be cart coords.
        """
        # Sample a timestep t.
        if t_int is None:
            t_int = torch.randint(
                1, self.T + 1, size=(batch.num_atoms.size(0), 1), device=x.device
            ).long()
        else:
            t_int = (
                torch.ones((batch.num_atoms.size(0), 1), device=x.device).long() * t_int
            )
        t_int = t_int.repeat_interleave(batch.num_atoms, dim=0)

        # Sample noise.
        frac_x_t, target_eps_x, used_sigmas_x = self.pos_diffusion(
            x, t_int, lengths, angles, batch.num_atoms
        )
        h_t, eps_h = self.type_diffusion(h, t_int)

        # Compute the prediction.
        pred_eps_x, pred_eps_h = self.phi(
            z, frac_x_t, h_t, t_int, batch.num_atoms, lengths, angles, frac=True
        )

        # Compute the error.
        error_x = self.compute_error(
            pred_eps_x,
            target_eps_x / used_sigmas_x**2,
            batch,
            0.5 * used_sigmas_x**2,
        )  # likelihood reweighting
        error_h = self.compute_error(pred_eps_h, eps_h, batch)

        loss = self.hparams.cost_coord * error_x + self.hparams.cost_type * error_h

        return {
            "t": t_int.squeeze(),
            "diffusion_loss": loss.mean(),
            "coord_loss": error_x.mean(),
            "type_loss": error_h.mean(),
            "pred_eps_x": pred_eps_x,
            "pred_eps_h": pred_eps_h,
            "eps_x": target_eps_x,
            "eps_h": eps_h,
        }

    def forward(self, batch, teacher_forcing, t_int=None):
        mu, log_var, z = self.encode(batch)

        (
            pred_num_atoms,
            pred_lengths_and_angles,
            pred_lengths,
            pred_angles,
        ) = self.decode_stats(
            z, batch.num_components, batch.lengths, batch.angles, teacher_forcing
        )
        pred_n_metal = self.fc_n_metal(z)

        n_metal_loss = F.cross_entropy(pred_n_metal, batch.n_metal)
        num_atom_loss = self.num_atom_loss(pred_num_atoms, batch)
        lattice_loss = self.lattice_loss(pred_lengths_and_angles, batch)
        kld_loss = self.kld_loss(mu, log_var)
        property_loss = (
            self.property_loss(z, batch) if self.hparams.predict_property else 0.0
        )

        x = frac_to_cart_coords(
            batch.frac_coords, batch.lengths, batch.angles, batch.num_atoms
        )
        h = batch.atom_types
        x, h, normed_lengths = self.normalize(x, h, batch.lengths)

        loss_dict = self.diffusion_loss(
            z, x, h, batch, normed_lengths, batch.angles, t_int=t_int
        )

        loss_dict.update(
            {
                "n_metal_loss": n_metal_loss,
                "num_atom_loss": num_atom_loss,
                "lattice_loss": lattice_loss,
                "kld_loss": kld_loss,
                "property_loss": property_loss,
                "pred_num_atoms": pred_num_atoms,
                "pred_n_metal": pred_n_metal,
                "pred_lengths_and_angles": pred_lengths_and_angles,
                "pred_lengths": pred_lengths,
                "pred_angles": pred_angles,
                "target_cart_coords": x,
                "target_atom_types": batch.atom_types,
                "z": z,
                "y": batch.y,
                "num_atoms": batch.num_atoms,
                "batch": batch,
            }
        )

        loss = loss_dict["diffusion_loss"]
        return loss_dict

    def compute_stats(self, batch, outputs, prefix):
        diffusion_loss = outputs["diffusion_loss"]
        n_metal_loss = outputs["n_metal_loss"]
        num_atom_loss = outputs["num_atom_loss"]
        lattice_loss = outputs["lattice_loss"]
        coord_loss = outputs["coord_loss"]
        type_loss = outputs["type_loss"]
        kld_loss = outputs["kld_loss"]
        property_loss = outputs["property_loss"]
        z = outputs["z"]

        batch = outputs["batch"]

        loss = (
            diffusion_loss
            + self.hparams.cost_natom * num_atom_loss
            + self.hparams.cost_natom * n_metal_loss
            + self.hparams.cost_lattice * lattice_loss
            + self.beta * kld_loss
            + self.hparams.cost_property * property_loss
        )

        log_dict = {
            f"{prefix}_loss": loss,
            f"{prefix}_diffusion_loss": diffusion_loss,
            f"{prefix}_n_metal_loss": n_metal_loss,
            f"{prefix}_natom_loss": num_atom_loss,
            f"{prefix}_lattice_loss": lattice_loss,
            f"{prefix}_coord_loss": coord_loss,
            f"{prefix}_type_loss": type_loss,
            f"{prefix}_property_loss": property_loss,
            f"{prefix}_kld_loss": kld_loss,
            f"{prefix}_kld_beta": self.beta,
            f"{prefix}_z_norm": z.norm(2, dim=-1).mean().detach().item(),
        }

        if prefix != "train":
            if self.scaler is not None:
                preds = self.predict_property(z)
                for i, prop in enumerate(self.prop_list):
                    mae = torch.mean(torch.abs(preds[:, i] - batch.y[:, i]))
                    log_dict.update({f"{prefix}_mae: {prop}": mae})

            # validation with diffusion loss
            loss = diffusion_loss

            # evaluate num_atom prediction.
            pred_num_atoms = outputs["pred_num_atoms"].argmax(dim=-1)
            num_atom_accuracy = (
                pred_num_atoms == batch.num_atoms
            ).sum() / batch.num_graphs

            pred_n_metals = outputs["pred_n_metal"].argmax(dim=-1)
            n_metal_accuracy = (pred_n_metals == batch.n_metal).sum() / batch.num_graphs

            # evalute lattice prediction.
            pred_lengths_and_angles = outputs["pred_lengths_and_angles"]
            assert self.lattice_scaler is not None
            scaled_preds = self.lattice_scaler.inverse_transform(
                pred_lengths_and_angles
            )
            pred_lengths = scaled_preds[:, :3]
            pred_angles = scaled_preds[:, 3:]

            if self.hparams.data.lattice_scale_method == "scale_length":
                pred_lengths = pred_lengths * batch.num_atoms.view(-1, 1).float() ** (
                    1 / 3
                )
            lengths_mard = mard(batch.lengths, pred_lengths)
            angles_mae = torch.mean(torch.abs(pred_angles - batch.angles))

            pred_volumes = lengths_angles_to_volume(pred_lengths, pred_angles)
            true_volumes = lengths_angles_to_volume(batch.lengths, batch.angles)
            volumes_mard = mard(true_volumes, pred_volumes)

            log_dict.update(
                {
                    f"{prefix}_loss": loss,
                    f"{prefix}_property_loss": property_loss,
                    f"{prefix}_natom_accuracy": num_atom_accuracy,
                    f"{prefix}_n_metal_accuracy": n_metal_accuracy,
                    f"{prefix}_lengths_mard": lengths_mard,
                    f"{prefix}_angles_mae": angles_mae,
                    f"{prefix}_volumes_mard": volumes_mard,
                }
            )

        return log_dict, loss

    @torch.no_grad()
    def sample(
        self,
        n_samples=None,
        z=None,
        save_freq=False,
        disable_bar=False,
    ):
        if z is None:
            z = torch.randn(n_samples, self.hparams.latent_dim, device=self.device)
        else:
            n_samples = z.shape[0]

        num_atoms, _, lengths, angles = self.decode_stats(z)
        lengths = lengths / self.norm_x

        x = (
            torch.randn([num_atoms.sum(), 3], device=self.device)
            * self.hparams.diffusion.pos.sigma_max
        )
        frac_x = cart_to_frac_coords(x, lengths, angles, num_atoms)
        x = frac_to_cart_coords(frac_x, lengths, angles, num_atoms)

        h = torch.randn([num_atoms.sum(), self.bb_emb_dim], device=self.device)

        if save_freq:
            all_x = [x.clone().cpu()]
            all_h = [h.clone().cpu()]

        for t in tqdm(reversed(range(1, self.T)), disable=disable_bar):
            t = torch.full((num_atoms.sum(),), fill_value=t, device=self.device)

            score_x, score_h = self.phi(
                z, frac_x, h, t, num_atoms, lengths, angles, frac=True
            )
            frac_x = self.pos_diffusion.reverse(
                x, score_x, t, lengths, angles, num_atoms
            )
            x = frac_to_cart_coords(frac_x, lengths, angles, num_atoms)
            h = self.type_diffusion.reverse(h, score_h, t)

            if save_freq and (t[0] % save_freq == 0):
                all_x.append(x.clone().cpu())
                all_h.append(h.clone().cpu())

        if save_freq:
            all_x.append(x.clone().cpu())
            all_h.append(h.clone().cpu())

        x, h, lengths = self.unnormalize(x, h, lengths)

        output = {
            "x": x,
            "h": h,
            "z": z,
            "num_atoms": num_atoms,
            "lengths": lengths,
            "angles": angles,
        }

        if save_freq:
            output.update(
                {
                    "all_x": torch.stack(all_x, dim=0),
                    "all_h": torch.stack(all_h, dim=0),
                }
            )
        return output

    def predict_num_atoms(self, z):
        return self.fc_num_atoms(z)

    def predict_property(self, z):
        if not self.hparams.predict_property:
            raise ValueError(
                "self.hparams.predict_property is False. Cannot predict property."
            )
        self.scaler.match_device(z)
        return self.scaler.inverse_transform(self.fc_property(z))

    def predict_lattice(self, z, num_atoms):
        self.lattice_scaler.match_device(z)
        pred_lengths_and_angles = self.fc_lattice(z)  # (N, 6)
        scaled_preds = self.lattice_scaler.inverse_transform(pred_lengths_and_angles)
        pred_lengths = scaled_preds[:, :3]
        pred_angles = scaled_preds[:, 3:]
        if self.hparams.data.lattice_scale_method == "scale_length":
            pred_lengths = pred_lengths * num_atoms.view(-1, 1).float() ** (1 / 3)
        return pred_lengths_and_angles, pred_lengths, pred_angles

    def num_atom_loss(self, pred_num_atoms, batch):
        return F.cross_entropy(pred_num_atoms, batch.num_atoms)

    def property_loss(self, z, batch):
        self.scaler.match_device(z)
        return F.mse_loss(self.fc_property(z), self.scaler.transform(batch["y"]))

    def lattice_loss(self, pred_lengths_and_angles, batch):
        self.lattice_scaler.match_device(pred_lengths_and_angles)
        if self.hparams.data.lattice_scale_method == "scale_length":
            target_lengths = batch.lengths / batch.num_atoms.view(-1, 1).float() ** (
                1 / 3
            )
        target_lengths_and_angles = torch.cat([target_lengths, batch.angles], dim=-1)
        target_lengths_and_angles = self.lattice_scaler.transform(
            target_lengths_and_angles
        )
        return F.mse_loss(pred_lengths_and_angles, target_lengths_and_angles)

    def kld_loss(self, mu, log_var):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )
        return kld_loss

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        teacher_forcing = self.current_epoch <= self.hparams.teacher_forcing_max_epoch
        outputs = self(batch, teacher_forcing)
        log_dict, loss = self.compute_stats(batch, outputs, prefix="train")
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.global_training_step += 1
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix="val")
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix="test")
        self.log_dict(
            log_dict,
        )
        return loss

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {
            "optimizer": opt,
            "lr_scheduler": scheduler,
            "monitor": "val_diffusion_loss",
        }