"""
contrastive learning for building block embedding.
"""
import hydra
import omegaconf

import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter

from mofdiff.common.sys_utils import PROJECT_ROOT
from mofdiff.model.utils import build_mlp
from mofdiff.model.gemnet_oc.layers.radial_basis import RadialBasis


class BBEncoder(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.type_mapper = kwargs.get("type_mapper", None)
        self.hparams.use_type_mapper = True

        self.num_types = self.type_mapper.num_types
        self.cp_type = self.type_mapper.cp_type

        self.global_training_step = nn.Parameter(
            torch.tensor(0, dtype=torch.float), requires_grad=False
        )

        # encode
        self.encoder = hydra.utils.instantiate(
            self.hparams.encoder, num_targets=self.hparams.latent_dim
        )
        self.shortcut = build_mlp(
            self.hparams.latent_dim
            + self.hparams.max_atoms
            + 1
            + self.hparams.max_cps
            + 1
            + 64,  # 64 is the number of radial basis
            self.hparams.hidden_dim,
            self.hparams.fc_num_layers,
            self.hparams.latent_dim,
        )

        # contrastive learning
        self.fc_project = build_mlp(
            self.hparams.latent_dim,
            self.hparams.hidden_dim,
            self.hparams.fc_num_layers,
            self.hparams.project_dim,
        )

        # stats
        self.fc_num_atoms = build_mlp(
            self.hparams.latent_dim,
            self.hparams.hidden_dim,
            self.hparams.fc_num_layers,
            self.hparams.max_atoms + 1,
        )

        self.fc_num_cps = build_mlp(
            self.hparams.latent_dim,
            self.hparams.hidden_dim,
            self.hparams.fc_num_layers,
            self.hparams.max_cps + 1,
        )

        self.fc_diameter = build_mlp(
            self.hparams.latent_dim,
            self.hparams.hidden_dim,
            self.hparams.fc_num_layers,
            1,
        )

        self.d_basis = RadialBasis(
            num_radial=64,
            cutoff=kwargs.get("max_diameter", 20.0),
            rbf={"name": "gaussian"},
            envelope={"name": "polynomial", "exponent": 5},
            scale_basis=False,
        )

    def encode(self, batch):
        latent = self.shortcut(
            torch.cat(
                [
                    self.encoder(batch),
                    # pylint: disable=E1102
                    F.one_hot(batch.num_atoms, self.hparams.max_atoms + 1),
                    # pylint: disable=E1102
                    F.one_hot(batch.num_cps, self.hparams.max_cps + 1),
                    self.d_basis(batch.diameter),
                ],
                dim=-1,
            )
        )
        return latent

    def forward(self, batch):
        self.type_mapper.match_device(batch.atom_types)
        batch.atom_types = self.type_mapper.transform(batch.atom_types)
        z = self.encode(batch)

        # decode high level stats and compute loss.
        pred_num_atoms = self.fc_num_atoms(z)
        pred_num_cps = self.fc_num_cps(z)
        pred_diameter = self.fc_diameter(z).flatten()

        num_atom_loss = self.num_atom_loss(pred_num_atoms, batch)
        num_cp_loss = self.num_cp_loss(pred_num_cps, batch)
        diameter_loss = (pred_diameter - batch.diameter).pow(2).mean()
        id_loss = self.identity_loss(z, batch)

        loss_dict = {
            "num_atom_loss": num_atom_loss,
            "num_cp_loss": num_cp_loss,
            "diameter_loss": diameter_loss,
            "id_loss": id_loss,
            "pred_num_atoms": pred_num_atoms,
            "pred_num_cps": pred_num_cps,
            "pred_diameter": pred_diameter,
            "target_atom_types": batch.atom_types,
            "z": z,
        }

        return loss_dict

    def identity_loss(self, z, batch):
        projected = F.normalize(self.fc_project(z), dim=-1)
        mask = batch.identity.view(1, -1) == batch.identity.view(-1, 1)
        similarity = torch.exp(projected @ projected.T / self.hparams.temperature)
        loss = (
            (similarity * (~mask).float()).sum(dim=1).log()
            - (similarity * mask.float()).sum(dim=1).log()
        ).mean()

        return loss

    def compute_stats(self, batch, outputs, prefix):
        num_atom_loss = outputs["num_atom_loss"]
        num_cp_loss = outputs["num_cp_loss"]
        diameter_loss = outputs["diameter_loss"]
        id_loss = outputs["id_loss"]
        z_norm = outputs["z"].norm(dim=-1).mean()

        loss = (
            self.hparams.cost_id * id_loss
            + self.hparams.cost_natom * num_atom_loss
            + self.hparams.cost_ncp * num_cp_loss
            + self.hparams.cost_d * diameter_loss
            + self.hparams.cost_z * z_norm
        )

        log_dict = {}
        for k, v in outputs.items():
            if "loss" in k:
                log_dict[f"{prefix}_{k}"] = v.detach().item()

        log_dict[f"{prefix}_loss"] = loss
        log_dict["z_norm"] = z_norm.detach().item()

        if prefix != "train":
            pred_num_atoms = outputs["pred_num_atoms"].argmax(dim=-1)
            num_atom_accuracy = (
                pred_num_atoms == batch.num_atoms
            ).sum() / batch.num_graphs
            pred_num_cps = outputs["pred_num_cps"].argmax(dim=-1)
            num_cps = scatter(batch.is_anchor.long(), batch.batch, dim=0, reduce="sum")
            num_cp_accuracy = (pred_num_cps == num_cps).sum() / batch.num_graphs
            diameter_mae = (outputs["pred_diameter"] - batch.diameter).abs().mean()

            log_dict.update(
                {
                    f"{prefix}_natom_accuracy": num_atom_accuracy,
                    f"{prefix}_ncp_accuracy": num_cp_accuracy,
                    f"{prefix}_d_mae": diameter_mae,
                }
            )

        return log_dict, loss

    def num_atom_loss(self, pred_num_atoms, batch):
        return F.cross_entropy(pred_num_atoms, batch.num_atoms)

    def num_cp_loss(self, pred_num_cps, batch):
        target = scatter(batch.is_anchor.long(), batch.batch, dim=0, reduce="sum")
        return F.cross_entropy(pred_num_cps, target)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        outputs = self(batch)
        log_dict, loss = self.compute_stats(batch, outputs, prefix="train")
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.global_training_step += 1
        return loss

    # building block embedding space learning does not involve validation or testing.
    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        outputs = self(batch)
        log_dict, loss = self.compute_stats(batch, outputs, prefix="val")
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        outputs = self(batch)
        log_dict, loss = self.compute_stats(batch, outputs, prefix="test")
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
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
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}