import random
from typing import Optional, Sequence
from pathlib import Path
import os

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader

from mofdiff.common.data_utils import get_scaler_from_data_list
from mofdiff.common.atomic_utils import remap_values
from mofdiff.common.eval_utils import load_bb_encoder


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


class TypeMapper(object):
    def __init__(self, in_map, out_map):
        super().__init__()
        self.in_map = in_map
        self.out_map = out_map
        self.num_types = len(out_map)
        self.cp_type = out_map[in_map == 2].item()

    def match_device(self, tensor):
        if self.in_map.device != tensor.device:
            self.in_map = self.in_map.to(tensor.device)
            self.out_map = self.out_map.to(tensor.device)

    def copy(self):
        return TypeMapper(self.in_map.clone().detach(), self.out_map.clone().detach())

    def transform(self, atom_types):
        return remap_values((self.in_map, self.out_map), atom_types)

    def inverse_transform(self, transformed_atom_types):
        return remap_values((self.out_map, self.in_map), transformed_atom_types)

    def __repr__(self):
        return f"{self.__class__.__name__}\n in_map: {self.in_map}\n out_map: {self.out_map}"


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        scaler_path=None,
        device="cuda",
        bb_encoder_path=None,
    ):
        super().__init__()
        if bb_encoder_path is not None:
            self.bb_encoder = load_bb_encoder(bb_encoder_path).to(device)
            self.bb_emb_dim = self.bb_encoder.hparams.encoder.num_targets
        else:
            self.bb_encoder = None
            self.bb_emb_dim = None

        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

        self.get_scaler(scaler_path)
        self.type_mapper = None

        self.prepare_data_per_node = True

    def get_type_mapper(self, type_mapper_path=None):
        if type_mapper_path is None:
            unique_atom_types = torch.unique(
                torch.cat(
                    [d.atom_types for d in self.train_dataloader()]
                    + [d.atom_types for d in self.val_dataloader()],
                    dim=0,
                )
            )
            atom_index = torch.arange(unique_atom_types.shape[0])
            self.type_mapper = TypeMapper(unique_atom_types, atom_index)
        else:
            self.type_mapper = None
            if os.path.exists(Path(type_mapper_path) / "type_mapper.pt"):
                self.type_mapper = torch.load(Path(type_mapper_path) / "type_mapper.pt")
        return self.type_mapper

    def get_scaler(self, scaler_path):
        # Load once to compute property scaler
        if scaler_path is None:
            self.train_dataset = hydra.utils.instantiate(
                self.datasets.train, bb_encoder=self.bb_encoder
            )

            if "scaled_lattice" in self.train_dataset.cached_data[0].keys():
                self.lattice_scaler = get_scaler_from_data_list(
                    self.train_dataset.cached_data, key="scaled_lattice"
                )
            else:
                self.lattice_scaler = None
            if self.train_dataset.prop:
                self.scaler = get_scaler_from_data_list(
                    self.train_dataset.cached_data, key="y"
                )
                self.prop_list = self.train_dataset.prop_list
            else:
                self.scaler = None
                self.prop_list = None

            self.train_dataset.lattice_scaler = self.lattice_scaler
            self.train_dataset.scaler = self.scaler
        else:
            self.lattice_scaler = None
            self.scaler = None
            if os.path.exists(Path(scaler_path) / "lattice_scaler.pt"):
                self.lattice_scaler = torch.load(
                    Path(scaler_path) / "lattice_scaler.pt"
                )
            if os.path.exists(Path(scaler_path) / "prop_scaler.pt"):
                self.scaler = torch.load(Path(scaler_path) / "prop_scaler.pt")

    def setup(self, stage: Optional[str] = None):
        """
        construct datasets and assign data scalers.
        """
        if stage is None or stage == "fit":
            if self.train_dataset is None:
                self.train_dataset = hydra.utils.instantiate(
                    self.datasets.train, bb_encoder=self.bb_encoder
                )
                self.train_dataset.lattice_scaler = self.lattice_scaler
                self.train_dataset.scaler = self.scaler

            if self.val_dataset is None:
                self.val_dataset = hydra.utils.instantiate(
                        self.datasets.val, bb_encoder=self.bb_encoder
                    )
                self.val_dataset.lattice_scaler = self.lattice_scaler
                self.val_dataset.scaler = self.scaler

        if self.bb_encoder is not None:
            del self.bb_encoder

    def train_dataloader(self, shuffle=True) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=shuffle,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
            worker_init_fn=worker_init_fn,
        )

    def test_dataloader(self) -> Sequence[DataLoader]:
        return []

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )
