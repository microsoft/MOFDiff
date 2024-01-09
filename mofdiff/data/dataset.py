import bisect
import pickle
import lmdb
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from mofdiff.common.atomic_utils import (
    frac2cart,
    compute_distance_matrix,
    get_bb_fingerprint,
)
from mofdiff.common.data_utils import (
    lattice_params_to_matrix_torch,
    frac_to_cart_coords,
    logmod,
)

from mofdiff.common.sys_utils import DATASET_DIR


class BBDataset(Dataset):
    """
    Dataset class for building block representation learning.
    Args:
        name (str): name of dataset.
        path (str): path to dataset.
        max_bbs (int): maximum number of building blocks in the MOF.
        max_atoms (int): maximum number of atoms in a building block.
        max_cps (int): maximum number of connection points in a building block.
    Dataset can be specified by one of the following args:
        n_points (int): number of data points to use.
        split_file (str): path to split file.
    """

    def __init__(
        self,
        name,
        path,
        max_bbs=20,
        max_atoms=200,
        max_cps=20,
        n_points=None,
        split_file=None,
        **kwargs,
    ):
        super().__init__()

        self.name = name
        self.path = Path(path)
        self.prop = None

        if not self.path.is_file():
            db_paths = sorted(self.path.glob("*.lmdb"))
            assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"
            self._keys, self.envs = [], []
            for db_path in db_paths:
                self.envs.append(self.connect_db(db_path))
                length = pickle.loads(
                    self.envs[-1].begin().get("length".encode("ascii"))
                )
                self._keys.append(list(range(length)))
            keylens = [len(k) for k in self._keys]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self.num_samples = sum(keylens)
        else:
            self.envs = self.connect_db(self.path)
            self._keys = [
                f"{j}".encode("ascii") for j in range(self.envs.stat()["entries"])
            ]
            self.num_samples = len(self._keys) - 1

        if n_points is not None:
            self.samples = np.random.randint(0, self.num_samples, n_points)
            self.num_samples = n_points
        else:
            self.samples = list(range(self.num_samples))

        if split_file is not None:
            self.samples = np.loadtxt(split_file, dtype=int)
            self.num_samples = len(self.samples)
            print(f"load split file {split_file} with {self.num_samples} samples.")

        self.max_bbs = max_bbs
        self.max_atoms = max_atoms
        self.max_cps = max_cps

        self.cached = False
        self.cache_to_memory()

    def cache_to_memory(self):
        def mof_criterion(mof):
            if mof.num_components > self.max_bbs:
                return False

            cell = lattice_params_to_matrix_torch(mof.lengths, mof.angles).squeeze()
            distances = compute_distance_matrix(
                cell, frac2cart(mof.cg_frac_coords, cell)
            ).fill_diagonal_(5.0)
            return (not (distances < 1.0).any()) and mof.num_components <= self.max_bbs

        def bb_criterion(bb):
            bb.num_cps = bb.is_anchor.long().sum()
            if (bb.num_atoms > self.max_atoms) or (bb.num_cps > self.max_cps):
                return None, False

            cart_coords = frac_to_cart_coords(
                bb.frac_coords, bb.lengths, bb.angles, bb.num_atoms
            )
            pdist = torch.cdist(cart_coords, cart_coords).fill_diagonal_(5.0)

            # detect BBs with problematic bond info.
            edge_index = bb.edge_index
            j, i = edge_index
            bond_dist = (cart_coords[i] - cart_coords[j]).pow(2).sum(dim=-1).sqrt()

            success = (
                pdist.min() > 0.25
                and bond_dist.max() < 5.0
                and (bb.num_atoms <= self.max_atoms)
                and (bb.num_cps <= self.max_cps)
            )
            return cart_coords, success

        self.cached_ids = []
        self.cached_data = []

        print(f"process dataset {self.name} to BBs.")
        for i in tqdm(self.samples):
            try:
                data = self[i]
                if mof_criterion(data):
                    bbs = []
                    for bb in data.bbs:
                        cart_coords, success = bb_criterion(bb)
                        if success:
                            bb.num_nodes = bb.num_atoms
                            if "fp" not in bb:
                                bb.fp = get_bb_fingerprint(bb).view(1, -1)
                            bb.diameter = torch.pdist(cart_coords).max()
                            bbs.append(bb)
                    self.cached_data = self.cached_data + bbs
                    self.cached_ids.append(data.m_id)
            except Exception as e:
                print(i, e)
                continue

        print(f"gathered {len(self.cached_data)} BBs from {self.num_samples} MOFs.")
        self.num_samples = len(self.cached_data)

        self.cached = True

        self.close_db()
        del self.envs

        all_fps = torch.stack([x.fp for x in self.cached_data])
        _, reverse_index = torch.unique(all_fps, dim=0, return_inverse=True)
        for i in range(self.num_samples):
            self.cached_data[i].identity = reverse_index[i]
            del self.cached_data[i].fp

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.cached:
            data_object = self.cached_data[idx]
        else:
            if not self.path.is_file():
                # Figure out which db this should be indexed from.
                db_idx = bisect.bisect(self._keylen_cumulative, idx)
                # Extract index of element within that db.
                el_idx = idx
                if db_idx != 0:
                    el_idx = idx - self._keylen_cumulative[db_idx - 1]
                assert el_idx >= 0

                # Return features.
                datapoint_pickled = (
                    self.envs[db_idx]
                    .begin()
                    .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
                )
                data_object = pickle.loads(datapoint_pickled)
                data_object.id = f"{db_idx}_{el_idx}"
            else:
                datapoint_pickled = self.envs.begin().get(self._keys[idx])
                data_object = pickle.loads(datapoint_pickled)

            del data_object.prop_dict

            if "num_components" in data_object:
                data_object.num_components = data_object.num_components.squeeze()

        return data_object

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self):
        if not self.path.is_file():
            for env in self.envs:
                env.close()
        else:
            self.envs.close()


class MOFDataset(Dataset):
    """
    Dataset class for coarse-grained MOFs.

    Args:
        name (str): name of dataset.
        path (str): path to dataset.
        bb_encoder (nn.Module): building block encoder.
        device (str): device to run on.
        bb_batch_size (int): batch size for encoding building blocks.
        bb_emb_clipping (float): clipping value for building block embeddings.
        max_bbs (int): maximum number of building blocks in the MOF.
        max_atoms (int): maximum number of atoms in a building block.
        max_cps (int): maximum number of connection points in a building block.
        keep_bbs (bool): whether to keep building blocks all-atom structures in the dataset.
        bb_cache_path (str): path to cache the building block library. Useful for generation.
        logmod (bool): whether to use logmod transformation for the property.
    Dataset can be specified by one of the following args:
        samples (list): list of indices to use.
        n_points (int): number of data points to use.
        split_file (str): path to split file.
    """

    def __init__(
        self,
        name,
        path,
        prop_list,
        bb_encoder=None,
        device="cuda",
        bb_batch_size=512,
        bb_emb_clipping=100,
        max_bbs=20,
        max_atoms=1000,
        max_cps=20,
        keep_bbs=False,
        bb_cache_path=DATASET_DIR,
        logmod=False,
        samples=None,
        n_points=None,
        split_file=None,
        **kwags,
    ):
        super().__init__()
        self.name = name
        self.path = Path(path)

        self.batch_size = bb_batch_size
        if bb_encoder is not None:
            self.bb_encoder = bb_encoder.to(device)
            self.bb_encoder.type_mapper.match_device(self.bb_encoder)
        else:
            self.bb_encoder = None
        self.device = device
        self.bb_emb_clipping = bb_emb_clipping

        if not self.path.is_file():
            db_paths = sorted(self.path.glob("*.lmdb"))
            assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"
            self._keys, self.envs = [], []
            for db_path in db_paths:
                self.envs.append(self.connect_db(db_path))
                length = pickle.loads(
                    self.envs[-1].begin().get("length".encode("ascii"))
                )
                self._keys.append(list(range(length)))
            keylens = [len(k) for k in self._keys]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self.num_samples = sum(keylens)
        else:
            self.envs = self.connect_db(self.path)
            self._keys = [
                f"{j}".encode("ascii") for j in range(self.envs.stat()["entries"])
            ]
            self.num_samples = len(self._keys) - 1

        self.samples = np.arange(self.num_samples)

        # priority: samples > n_points > split_file
        if samples is not None:
            self.samples = samples
            self.num_samples = len(self.samples)
        elif n_points is not None:
            # in-place shuffle index if use n_points.
            np.random.shuffle(self.samples)
            self.num_samples = n_points
            print(f"only use {self.num_samples} samples before filtering.")
        elif split_file is not None:
            self.samples = np.loadtxt(split_file, dtype=int)
            self.num_samples = len(self.samples)
            print(
                f"load split file {split_file} with {self.num_samples} samples before filtering."
            )

        self.max_bbs = max_bbs
        self.max_atoms = max_atoms
        self.max_cps = max_cps
        self.keep_bbs = keep_bbs
        self.logmod = logmod

        if bb_encoder is not None:
            self.bb_cache_path = (
                Path(bb_cache_path)
                / f"{self.name}_{self.max_bbs}_{self.max_atoms}"
            )

        # property.
        self.prop_list = prop_list
        self.n_props = len(self.prop_list)
        self.prop = self.n_props > 0

        # cache data to memory.
        self.cached = False
        self.cache_to_memory()

    def cache_to_memory(self):
        def mof_criterion(mof):
            if (
                mof.num_components > self.max_bbs
                or mof.y.isnan().sum() > 0
                or mof.y.isinf().sum() > 0
            ):
                return False
            cell = lattice_params_to_matrix_torch(mof.lengths, mof.angles).squeeze()
            distances = compute_distance_matrix(
                cell, frac2cart(mof.cg_frac_coords, cell)
            ).fill_diagonal_(5.0)
            return (
                (not (distances < 1.0).any())
                and mof.num_components <= self.max_bbs
                and mof.y.isnan().sum() == 0
                and mof.y.isinf().sum() == 0
            )

        def bb_criterion(bb):
            bb.num_cps = bb.is_anchor.long().sum()
            if (bb.num_atoms > self.max_atoms) or (bb.num_cps > self.max_cps):
                return None, False

            cart_coords = frac_to_cart_coords(
                bb.frac_coords, bb.lengths, bb.angles, bb.num_atoms
            )
            pdist = torch.cdist(cart_coords, cart_coords).fill_diagonal_(5.0)

            # detect BBs with problematic bond info.
            edge_index = bb.edge_index
            j, i = edge_index
            bond_dist = (cart_coords[i] - cart_coords[j]).pow(2).sum(dim=-1).sqrt()

            success = (
                pdist.min() > 0.25
                and bond_dist.max() < 5.0
                and (bb.num_atoms <= self.max_atoms)
                and (bb.num_cps <= self.max_cps)
            )

            return cart_coords, success

        self.cached_ids = []
        self.cached_data = []
        all_bbs = []
        all_N_bbs = []

        print(f"cache dataset {self.name} to memory.")
        for i in tqdm(self.samples, total=self.num_samples):
            try:
                data = self[i]
                data.cell = data.cell.unsqueeze(0)
                if mof_criterion(data):
                    bb_all_success = True
                    for bb in data.bbs:
                        cart_coords, success = bb_criterion(bb)
                        if success:
                            bb.num_nodes = bb.num_atoms
                            bb.diameter = torch.pdist(cart_coords).max()
                            if "fp" not in bb:
                                bb.fp = get_bb_fingerprint(bb).view(1, -1)
                        else:
                            bb_all_success = False
                            break
                    if bb_all_success:
                        self.cached_ids.append(data.m_id)
                        data = self.get_cg(data)
                        all_bbs.extend(data.bbs)
                        all_N_bbs.append(len(data.bbs))

                        if not self.keep_bbs:
                            del data.bbs
                            del data.pyg_mols

                        if self.logmod:
                            data.y = logmod(data.y)
                            data.y_logmod = True
                        else:
                            data.y_logmod = False

                        self.cached_data.append(data)
            # handle pickle files that were not properly saved.
            except TypeError as e:
                print(e)
                continue

            if len(self.cached_ids) >= self.num_samples:
                break

        print(
            f"removed: {self.num_samples - len(self.cached_data)}"
            " samples b/c [position overlap] | [too many atoms] | [too big BBs]."
        )
        self.num_samples = len(self.cached_data)
        self.close_db()
        del self.envs

        # gather normalization stats
        self.mean_lattice = float(
            torch.stack([d["lengths"] for d in self.cached_data]).mean()
        )
        
        if self.bb_encoder is not None:
            self.embed_bb(all_bbs, all_N_bbs)
            del all_bbs, all_N_bbs
            del self.bb_encoder
            self.mean_bb_emb = float(
                torch.cat([d["atom_types"] for d in self.cached_data])
                .norm(dim=-1)
                .mean(dim=0)
            )

        self.cached = True

    def embed_bb(self, all_bbs, all_N_bbs):
        """
        get all building block embeddings.
        """
        total_N_bbs = len(all_bbs)
        n_batches = np.ceil(total_N_bbs / self.batch_size).astype(int)
        all_bb_emb = []

        print(f"embedding all {total_N_bbs} building blocks ({n_batches} batches).")

        for i in tqdm(range(n_batches)):
            batch = Batch.from_data_list(
                all_bbs[i * self.batch_size : (i + 1) * self.batch_size]
            ).to(self.device)
            batch.atom_types = self.bb_encoder.type_mapper.transform(batch.atom_types)
            with torch.no_grad():
                bb_emb = self.bb_encoder.encode(batch)
            all_bb_emb.append(bb_emb.cpu())
        all_bb_emb = torch.cat(all_bb_emb, dim=0)
        all_bb_emb = torch.clamp(
            all_bb_emb, -self.bb_emb_clipping, self.bb_emb_clipping
        )

        # save building block lookup table
        if hasattr(self, "bb_cache_path"):
            torch.save([all_bbs, all_bb_emb], self.bb_cache_path)

        offsets = torch.cumsum(torch.tensor([0] + all_N_bbs), dim=0)
        for i in range(len(self.cached_data)):
            self.cached_data[i].atom_types = all_bb_emb[offsets[i] : offsets[i + 1]]

    def get_cg(self, data):
        cg_data = Data(
            frac_coords=data.cg_frac_coords,
            lengths=data.lengths,
            angles=data.angles,
            cell=data.cell.view(1, 3, 3),
            edge_index=data.cg_edge_index,
            to_jimages=data.cg_to_jimages,
            num_bonds=data.num_cg_bonds,
            num_atoms=data.num_components,
            num_nodes=data.num_components,
            scaled_lattice=data.scaled_lattice,
            num_components=data.num_components,
            is_linker=data.is_linker,
            n_metal=data.is_linker.shape[0] - data.is_linker.sum(),
            n_linker=data.is_linker.sum(),
            y=data.y,
            node_embedding=True,
            bbs=data.bbs,
            pyg_mols=data.pyg_mols,
            m_id=data.m_id,
            top=data.top,  # extracted by MOFid
        )

        if "cg_edge_types" in data:
            cg_data.edge_types = data.cg_edge_types

        return cg_data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.cached:
            data_object = self.cached_data[idx]
        else:
            if not self.path.is_file():
                # Figure out which db this should be indexed from.
                db_idx = bisect.bisect(self._keylen_cumulative, idx)
                # Extract index of element within that db.
                el_idx = idx
                if db_idx != 0:
                    el_idx = idx - self._keylen_cumulative[db_idx - 1]
                assert el_idx >= 0

                # Return features.
                datapoint_pickled = (
                    self.envs[db_idx]
                    .begin()
                    .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
                )
                data_object = pickle.loads(datapoint_pickled)
                data_object.id = f"{db_idx}_{el_idx}"
            else:
                datapoint_pickled = self.envs.begin().get(self._keys[idx])
                data_object = pickle.loads(datapoint_pickled)

            if self.prop_list == ["scaled_lattice"]:
                data_object["y"] = data_object.scaled_lattice
            else:
                collected_prop = []
                for i, prop in enumerate(self.prop_list):
                    collected_prop.append(data_object.prop_dict[prop])
                data_object["y"] = torch.FloatTensor(collected_prop).view(1, -1)
            del data_object.prop_dict

            if "num_components" in data_object:
                data_object.num_components = data_object.num_components.squeeze()

        return data_object

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self):
        if not self.path.is_file():
            for env in self.envs:
                env.close()
        else:
            self.envs.close()