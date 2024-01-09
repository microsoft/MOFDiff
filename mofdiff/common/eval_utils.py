from pathlib import Path
import os
import torch
import numpy as np

import hydra
from omegaconf import OmegaConf
from hydra.experimental import compose
from hydra import initialize_config_dir

from mofdiff.common.sys_utils import PROJECT_ROOT

def load_bb_encoder(model_path):
    cfg = OmegaConf.load(os.path.join(model_path, "hparams.yaml"))    
    cfg.model.encoder.scale_file = str(
        Path(PROJECT_ROOT) / "mofdiff/model/gemnet_oc/gemnet-oc.pt"
    )
    typer_mapper_path = Path(model_path) / "type_mapper.pt"
    type_mapper = torch.load(typer_mapper_path)

    model = hydra.utils.instantiate(
        cfg.model,
        data=cfg.data,
        type_mapper=type_mapper,
        _recursive_=False,
    )

    ckpts = list(Path(model_path).glob("*epoch*.ckpt"))
    if (Path(model_path) / "state_dict.pt").exists():
        model.load_state_dict(torch.load(Path(model_path) / "state_dict.pt"))
    elif len(ckpts) > 0:
        ckpt_epochs = np.array(
            [int(ckpt.parts[-1].split("-")[0].split("=")[1]) for ckpt in ckpts]
        )
        ckpt_ix = ckpt_epochs.argsort()[-1]
        ckpt = str(ckpts[ckpt_ix])
        print(f"Load BB encoder ckpt: {ckpt}.")
        ckpt = torch.load(ckpt)
        ckpt["hyper_parameters"]["encoder"]["scale_file"] = cfg.model.encoder.scale_file
        model.load_state_dict(ckpt["state_dict"])
    else:
        raise ValueError("No checkpoint found.")

    if "decoder" in model.hparams:
        del model.decoder
    for params in model.parameters():
        params.requires_grad = False

    return model


def load_mofdiff_model(model_path, config_overrides=None, load_last=False):
    """Loads a model checkpoint and config from a path."""
    if not isinstance(model_path, Path):
        model_path = Path(model_path)

    with initialize_config_dir(str(model_path)):
        cfg = compose(
            config_name="hparams",
            overrides=[] if config_overrides is None else config_overrides,
        )

        bb_encoder = load_bb_encoder(cfg.data.bb_encoder_path)
        print("load model.")
        model = hydra.utils.instantiate(
            cfg.model,
            data=cfg.data,
            _recursive_=False,
            bb_emb_dim=bb_encoder.hparams.latent_dim,
        )

        if load_last and (model_path / "last.ckpt").exists():
            ckpt = model_path / "last.ckpt"
        elif (model_path / "state_dict.pt").exists():
            model.load_state_dict(torch.load(model_path / "state_dict.pt"))
        else:
            ckpts = list(model_path.glob("*epoch*.ckpt"))
            if len(ckpts) > 0:
                ckpt_epochs = np.array(
                    [int(ckpt.parts[-1].split("-")[0].split("=")[1]) for ckpt in ckpts]
                )
                ckpt_ix = ckpt_epochs.argsort()[-1]
                ckpt = str(ckpts[ckpt_ix])
            ckpt = torch.load(ckpt)
            ckpt["hyper_parameters"]["decoder"]["scale_file"] = str(
                Path(PROJECT_ROOT) / "cdvae/model/gemnet_oc/gemnet-oc.pt"
            )
            model.load_state_dict(ckpt["state_dict"])
            
        model.lattice_scaler = torch.load(model_path / "lattice_scaler.pt")
        if os.path.exists(model_path / "prop_scaler.pt"):
            model.scaler = torch.load(model_path / "prop_scaler.pt")
        else:
            model.scaler = None

    return model, cfg, bb_encoder