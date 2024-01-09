import torch
import torch.nn as nn

from torch_geometric.data import Data, Batch

from mofdiff.common.data_utils import frac_to_cart_coords
from mofdiff.model.gemnet_oc.gemnet_oc import GemNetOC

NUM_EDGE_TYPE = 5
MAX_ATOMIC_NUM = 100


class GemNetOCEncoder(nn.Module):
    def __init__(
        self,
        readout,
        type_dim=256,
        num_targets=256,
        hidden_dim=256,
        max_neighbors=20,
        radius=6.0,
        scale_file=None,
        use_pbc=True,
        otf_graph=False,
        **kwargs
    ):
        super().__init__()
        self.cutoff = radius
        self.max_num_neighbors = max_neighbors
        self.use_pbc = use_pbc

        self.gemnet = GemNetOC(
            num_targets=num_targets,
            emb_size_atom=hidden_dim,
            emb_size_edge=hidden_dim,
            in_size_atom=type_dim,
            cutoff=self.cutoff,
            max_neighbors=self.max_num_neighbors,
            use_pbc=self.use_pbc,
            otf_graph=otf_graph,
            scale_file=scale_file,
            use_latent=False,
            regress_forces=False,
            **kwargs
        )

        self.readout = readout

    def forward(self, data, no_emb=False):
        data.pos = frac_to_cart_coords(
            data.frac_coords, data.lengths, data.angles, data.num_atoms
        )
        data.natoms = data.num_atoms
        out = self.readout(
            self.gemnet(data, no_emb=no_emb), data.batch, data.num_atoms.shape[0]
        )
        return out


class GemNetOCDecoder(nn.Module):
    def __init__(
        self,
        num_blocks=3,
        hidden_dim=128,
        latent_dim=256,
        max_neighbors=20,
        radius=6.0,
        scale_file=None,
        use_pbc=True,
        output_atom_types=MAX_ATOMIC_NUM,
        in_size_atom=256,
        **kwargs  # compat
    ):
        super().__init__()
        self.cutoff = radius
        self.max_num_neighbors = max_neighbors
        self.use_pbc = use_pbc

        self.gemnet = GemNetOC(
            num_targets=output_atom_types,
            num_blocks=num_blocks,
            latent_dim=latent_dim,
            emb_size_atom=hidden_dim,
            emb_size_edge=hidden_dim,
            in_size_atom=in_size_atom,
            regress_forces=True,
            cutoff=self.cutoff,
            max_neighbors=self.max_num_neighbors,
            use_pbc=self.use_pbc,
            otf_graph=True,
            scale_file=scale_file,
            **kwargs
        )

    def forward(
        self,
        z,
        frac_x,
        h,
        num_atoms,
        lengths=None,
        angles=None,
        batch=None,
        no_emb=False,
    ):
        cart_coords = frac_to_cart_coords(frac_x, lengths, angles, num_atoms)
        offsets = torch.cat(
            [
                torch.LongTensor([0]).to(num_atoms.device),
                torch.cumsum(num_atoms, dim=0),
            ],
            dim=0,
        )
        data_list = []

        for idx in range(len(num_atoms)):
            data = Data(
                z=z[idx][None, :],
                pos=cart_coords[offsets[idx] : offsets[idx + 1]],
                frac_coords=frac_x[offsets[idx] : offsets[idx + 1]],
                lengths=lengths[idx][None, :],
                angles=angles[idx][None, :],
                num_atoms=num_atoms[idx],
                atom_types=h[offsets[idx] : offsets[idx + 1]],
            )
            data_list.append(data)
        batch = Batch.from_data_list(data_list)

        pred_eps_x, pred_eps_h = self.gemnet(batch, no_emb=no_emb)

        return pred_eps_x, pred_eps_h
