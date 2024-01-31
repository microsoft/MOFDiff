from pathlib import Path
from typing import List
import hydra
import torch
import omegaconf
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

import math

from mofdiff.common.sys_utils import log_hyperparameters, PROJECT_ROOT
from mofdiff.common.data_utils import load_datamodule
from mofdiff.data.datamodule import DataModule


def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if "lr_monitor" in cfg.logging:
        hydra.utils.log.info("Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in cfg.train:
        hydra.utils.log.info("Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in cfg.train:
        hydra.utils.log.info("Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(HydraConfig.get().run.dir),
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                save_last=cfg.train.model_checkpoints.save_last,
                verbose=cfg.train.model_checkpoints.verbose,
                every_n_epochs=cfg.train.model_checkpoints.every_n_epochs,
                filename="{epoch}-{val_loss:.2f}",
            )
        )

    return callbacks


def run(cfg: DictConfig) -> None:
    """
    Generic train loop

    :param cfg: run configuration, defined by Hydra in /conf
    """

    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)

    if cfg.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Debug mode <{cfg.train.pl_trainer.fast_dev_run=}>. "
            f"Forcing debugger friendly configuration!"
        )
        # Debuggers don't like GPUs nor multiprocessing
        cfg.train.pl_trainer.gpus = 0
        cfg.data.datamodule.num_workers.train = 0
        cfg.data.datamodule.num_workers.val = 0
        cfg.data.datamodule.num_workers.test = 0

        # Switch wandb mode to offline to prevent online logging
        if "wandb" in cfg.logging:
            cfg.logging.wandb.mode = "offline"

    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)
    hydra_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    cache_path = cfg.data.get("data_cache_path", None)
    load_cached = cfg.data.get("load_cached", False)
    save_cached = cfg.data.get("save_cached", False)
    datamodule: DataModule = load_datamodule(cfg, cache_path, load_cached, save_cached)

    if cfg.data.use_type_mapper:
        datamodule.get_type_mapper()

    if datamodule.train_dataset is None:
        datamodule.setup("fit")
    assert datamodule.train_dataset is not None
    batch_size = cfg.data.datamodule.batch_size.train
    accumulate_grad_batches = cfg.train.pl_trainer.accumulate_grad_batches
    training_instances = len(datamodule.train_dataset)
    steps_per_epoch = math.ceil(
        training_instances / (batch_size * accumulate_grad_batches)
    )

    # Instantiate model
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")


    if cfg.config_for == "mof":
        extra_kwargs = {
            "bb_emb_dim": datamodule.bb_emb_dim,
            "norm_x": datamodule.train_dataset.mean_lattice,
            "norm_h": datamodule.train_dataset.mean_bb_emb,
        }
    elif cfg.config_for == "bb":
        extra_kwargs = {
            "type_mapper": datamodule.type_mapper
        }
    else:
        raise NotImplementedError(
            f"<config_for> is either 'mof' or 'bb', not {cfg.config_for}"
        )
    
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
        steps_per_epoch=steps_per_epoch,
        **extra_kwargs
    )

    # Pass scaler from datamodule to model
    hydra.utils.log.info(
        f"Passing scaler from datamodule to model <{datamodule.scaler}>"
    )
    model.lattice_scaler = (
        datamodule.lattice_scaler.copy()
        if datamodule.lattice_scaler is not None
        else None
    )
    model.scaler = datamodule.scaler.copy() if datamodule.scaler is not None else None
    model.prop_list = datamodule.prop_list if datamodule.prop_list is not None else None
    model.type_mapper = (
        datamodule.type_mapper.copy() if datamodule.type_mapper is not None else None
    )

    if datamodule.lattice_scaler is not None:
        torch.save(datamodule.lattice_scaler, hydra_dir / "lattice_scaler.pt")
    if datamodule.scaler is not None:
        torch.save(datamodule.scaler, hydra_dir / "prop_scaler.pt")
    if datamodule.type_mapper is not None:
        torch.save(datamodule.type_mapper, hydra_dir / "type_mapper.pt")

    # Instantiate the callbacks
    callbacks: List[Callback] = build_callbacks(cfg=cfg)

    # Logger instantiation/configuration
    logger = None

    if "wandb" in cfg.logging:
        hydra.utils.log.info("Instantiating <WandbLogger>")
        wandb_config = cfg.logging.wandb
        logger = WandbLogger(
            **wandb_config,
            tags=cfg.core.tags,
        )
        hydra.utils.log.info("W&B is now watching <{cfg.logging.wandb_watch.log}>!")
        logger.watch(
            model,
            log=cfg.logging.wandb_watch.log,
            log_freq=cfg.logging.wandb_watch.log_freq,
        )
    else:
        logger = TensorBoardLogger(**cfg.logging.tensorboard)
        hydra.utils.log.info(
            "TensorBoard Logger logs into <{cfg.logging.tensorboard.save_dir}>!"
        )

    # Store the YaML config separately into the wandb dir
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (hydra_dir / "hparams.yaml").write_text(yaml_conf)

    if (hydra_dir / "last.ckpt").exists():
        ckpt = hydra_dir / "last.ckpt"
        hydra.utils.log.info(f"found checkpoint: {ckpt}")
    else:
        ckpt = None

    hydra.utils.log.info("Instantiating the Trainer")
    trainer = pl.Trainer(
        default_root_dir=hydra_dir,
        logger=logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        check_val_every_n_epoch=cfg.logging.val_check_interval,
        progress_bar_refresh_rate=cfg.logging.progress_bar_refresh_rate,
        resume_from_checkpoint=ckpt,
        **cfg.train.pl_trainer,
    )
    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    hydra.utils.log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Logger closing to release resources/avoid multi-run conflicts
    if logger is not None and isinstance(logger, WandbLogger):
        # only WandbLogger has "finish()"
        logger.experiment.finish()


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="mofdiff")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
