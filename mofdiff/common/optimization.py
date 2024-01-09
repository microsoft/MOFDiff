"""
Assembly algorithm.
"""
import numpy as np
import torch
import random

from scipy.optimize import minimize, linear_sum_assignment
from torch_geometric.data import Data
from openbabel import openbabel as ob
from ase.data import covalent_radii

from mofdiff.common.atomic_utils import (
    compute_distance_matrix,
    frac2cart,
    cart2frac,
    remap_values,
    compute_image_flag,
)
from mofdiff.common.constants import METALS
from mofdiff.common.data_utils import lattice_params_to_matrix_torch
from mofdiff.common.so3 import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    random_rotations,
)
from mofdiff.common.atomic_utils import pygmol2pybel

covalent_radii_tensor = torch.tensor(covalent_radii)
metal_atomic_numbers = torch.tensor(METALS)
non_metal_atomic_numbers = torch.tensor([x for x in np.arange(100) if x not in METALS])


def same_component_mask(component):
    """mask=1 for all pairs that belong to the same component."""
    mask = (component.unsqueeze(0) == component.unsqueeze(1)).to(torch.bool)
    return mask


def gaussian_ball_overlap_loss(
    cart_coords,
    component,
    cell,
    sigma=10.0,
    max_neighbors=30,
    cp_atom_types=None,
    type_mask=True,
):
    """loss for gaussian ball overlap at connection points."""
    n = cart_coords.shape[0]
    dist_mat = compute_distance_matrix(cell, cart_coords).squeeze()

    comp_mask = same_component_mask(component)
    dist_mat[comp_mask] = 1000.0

    if type_mask:
        assert cp_atom_types is not None
        # metal, metal mask -- bond cannot be formed between two metal atoms
        # non-metal, non-metal mask -- bond cannot be formed between two non-metal atoms
        cp_atom_is_metal = (cp_atom_types.view(-1, 1) == metal_atomic_numbers).any(
            dim=-1
        )
        cp_atom_is_not_metal = (
            cp_atom_types.view(-1, 1) == non_metal_atomic_numbers
        ).any(dim=-1)
        metal_mask = torch.logical_and(
            cp_atom_is_metal.view(-1, 1), cp_atom_is_metal.view(1, -1)
        )
        non_metal_mask = torch.logical_and(
            cp_atom_is_not_metal.view(-1, 1), cp_atom_is_not_metal.view(1, -1)
        )
        dist_mat[metal_mask] = 1000.0
        dist_mat[non_metal_mask] = 1000.0

    sorted_dist_mat = dist_mat.sort(dim=-1)[0]
    kNN_same_mask = (sorted_dist_mat >= 1000.0)[:, : min(max_neighbors, n - 1)]

    kNN_dists = sorted_dist_mat[:, : min(max_neighbors, n - 1)][~kNN_same_mask]
    return -torch.exp(-kNN_dists / sigma).sum() / n


def get_cp_coords(vecs, cg_frac_coords, bb_local_vectors, cell, remove_cg_coords=True):
    vecs = vecs.view(-1, 3)
    r = axis_angle_to_matrix(vecs).to(cg_frac_coords.dtype)
    frac_coords = cg_frac_coords
    num_atoms = cg_frac_coords.shape[0]
    num_nodes = num_atoms

    for i, local_vectors in enumerate(bb_local_vectors):
        local_vectors_r = local_vectors @ r[i]
        cp_frac_coords = cg_frac_coords[i] + local_vectors_r
        frac_coords = torch.cat([frac_coords, cp_frac_coords % 1], dim=0)
        num_cps = cp_frac_coords.shape[0]
        num_nodes = num_nodes + num_cps

    cart_coords = frac2cart(frac_coords, cell)
    atom_node = torch.zeros(num_nodes, dtype=torch.long, device=r.device)
    atom_node[:num_atoms] = 1

    if remove_cg_coords:
        frac_coords = frac_coords[~atom_node.bool()]
        cart_coords = cart_coords[~atom_node.bool()]

    return cart_coords, frac_coords


def fun(x, arg_dict):
    """
    Params:
        optimizable vars:
          vecs: num_bb x 3
          frac_coords: num_bb x 3
          lengths: 1 x 3
          angles: 1 x 3
        bb info:
          bb_local_vectors: List[Tensor], length=num_bb
          bb_atom_local_vectors: List[Tensor], length=num_bb
          bb_atom_types: List[Tensor], length=num_bb
          connecting_atom_index: Tensor, length=num_atoms
        opt params:
          sigma: gaussian overlap loss ball radius
          max_neighbor: maximum number of nearby connection points to compute for overlapping volume
    """

    # gather variables from arg_dict.
    vecs = arg_dict["vecs"]
    cg_frac_coords = arg_dict["cg_frac_coords"]
    lengths = arg_dict["lengths"]
    angles = arg_dict["angles"]
    bb_local_vectors = arg_dict["bb_local_vectors"]
    cp_components = arg_dict["cp_components"]
    connecting_atom_types = arg_dict["connecting_atom_types"]
    sigma = arg_dict.get("sigma", 1.0)
    max_neighbors = arg_dict.get("max_neighbors", 30)

    N_bb = len(bb_local_vectors)
    vecs = x.view(N_bb, 3)

    cell = lattice_params_to_matrix_torch(
        lengths.view(1, -1), angles.view(1, -1)
    ).squeeze()
    cp_cart_coords, _ = get_cp_coords(vecs, cg_frac_coords, bb_local_vectors, cell)
    cp_loss = gaussian_ball_overlap_loss(
        cp_cart_coords, cp_components, cell, sigma, max_neighbors, connecting_atom_types
    ).view(-1)

    return cp_loss


def grad_fun(x, arg_dict):
    grad = torch.autograd.grad(sum(fun(x, arg_dict)), x)[0]
    return grad


def fun_apply(x, arg_dict):
    return fun(torch.tensor(x, dtype=torch.float64), arg_dict).detach().numpy()


def grad_apply(x, arg_dict):
    return (
        grad_fun(torch.tensor(x, requires_grad=True, dtype=torch.float64), arg_dict)
        .detach()
        .numpy()
        .flatten()
    )


def layout_optimization(
    x0, args, bounds=None, return_traj=False, maxiter=100, tol=None
):
    x_traj = []
    f_traj = []

    def callback(x):
        x = torch.tensor(x)
        x_traj.append(x)
        cp_loss = fun(x, args)
        f_traj.append(cp_loss)

    result = minimize(
        x0=x0,
        args=args,
        fun=fun_apply,
        jac=grad_apply,
        method="L-BFGS-B",
        bounds=bounds,
        callback=callback if return_traj else None,
        tol=tol,
        options={"maxiter": maxiter, "disp": False},
    )

    result = dict(result)
    if return_traj:
        result.update({"x_traj": x_traj, "f_traj": f_traj})

    return result


def prepare_optimization_variables(mof, device=None):
    # fix these keys...
    device = mof.num_atoms.device if device is None else device
    cg_frac_coords = mof.cg_frac_coords if "cg_frac_coords" in mof else mof.frac_coords
    key = "pyg_mols" if "pyg_mols" in mof else "bbs"

    bb_local_vectors = [bb.local_vectors.to(device).double() for bb in mof[key]]
    bb_atom_types = [bb.atom_types.to(device).double() for bb in mof[key]]

    cp_components = torch.cat(
        [
            torch.ones(bb_local_vectors[i].shape[0]) * i
            for i in range(len(bb_local_vectors))
        ],
        dim=0,
    ).long()

    atom_types = []
    atom_components = []
    for i in range(len(bb_atom_types)):
        atom_types.append(bb_atom_types[i])
        atom_components.append(i * torch.ones(bb_atom_types[i].shape[0]).long())
    atom_types = torch.cat(atom_types, dim=0).long()
    atom_components = torch.cat(atom_components, dim=0).long()

    connecting_atom_index = get_connecting_atom_index(mof[key])
    connecting_atom_types = atom_types[connecting_atom_index]

    return {
        "cg_frac_coords": cg_frac_coords.to(device).double(),
        "lengths": mof.lengths.to(device).double(),
        "angles": mof.angles.to(device).double(),
        "bb_local_vectors": bb_local_vectors,
        "cp_components": cp_components,
        "connecting_atom_types": connecting_atom_types,
    }


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def annealed_optimization(
    mof,
    seed,
    sigma_schedule=np.linspace(3, 0.6, 10),
    max_neighbors_schedule=np.arange(25, 5, -2),
    return_traj=False,
    maxiter=100,
    tol=None,
    print_freq=1,
    verbose=False,
):
    set_seed(seed)
    N_bb = totaln = (
        mof.num_components.sum() if "pyg_mols" in mof else mof.num_atoms.sum()
    )
    vec_dim = N_bb * 3
    # init BB rotations
    r = random_rotations(totaln, device=mof.num_atoms.device).transpose(-1, -2)
    vecs = matrix_to_axis_angle(r).double()

    # get initial x and args
    args = prepare_optimization_variables(mof)
    args.update({"vecs": vecs})
    x0 = vecs.flatten().numpy()
    bounds = [(-100.0, 100.0)] * (N_bb * 3)

    x_traj = []
    n_steps = len(sigma_schedule)
    total_iters = 0
    for i in range(n_steps):
        args.update(
            {
                "vecs": vecs,
                "sigma": sigma_schedule[i],
                "max_neighbors": max_neighbors_schedule[i],
            }
        )

        results = layout_optimization(
            x0, args, bounds=bounds, return_traj=return_traj, maxiter=maxiter, tol=tol
        )
        x0 = results["x"]
        args["vecs"] = torch.from_numpy(x0[:vec_dim]).view(N_bb, 3)

        v = results["fun"]
        n_iter = results["nit"]
        sigma = args["sigma"]
        maxn = args["max_neighbors"]
        total_iters += n_iter

        if verbose and (i % print_freq == 0 or i == n_steps - 1):
            print(
                f"[{i+1}/{n_steps}] total iter: {total_iters}, sigma: {sigma:.2f}, max_neighbors: {maxn:.2f},"
                f"v: {v:.4f}"
            )
        if return_traj:
            x_traj.extend(results["x_traj"])

    results["total_iters"] = total_iters
    if return_traj:
        results["x_traj"] = x_traj

    return results, v


def match_cps(
    vecs,
    cg_frac_coords,
    bb_local_vectors,
    lengths,
    angles,
    cp_atom_types=None,
    type_mask=True,
):
    cell = lattice_params_to_matrix_torch(
        lengths.view(1, -1), angles.view(1, -1)
    ).squeeze()
    cart_coords, _ = get_cp_coords(vecs, cg_frac_coords, bb_local_vectors, cell)
    dist_mat = compute_distance_matrix(cell, cart_coords).squeeze()

    cp_components = torch.cat(
        [
            torch.ones(bb_local_vectors[i].shape[0]) * i
            for i in range(len(bb_local_vectors))
        ],
        dim=0,
    ).long()
    mask = same_component_mask(cp_components)
    dist_mat[mask] = 1000.0

    if type_mask:
        assert cp_atom_types is not None
        # metal, metal mask -- bond cannot be formed between two metal atoms
        # non-metal, non-metal mask -- bond cannot be formed between two non-metal atoms
        cp_atom_is_metal = (cp_atom_types.view(-1, 1) == metal_atomic_numbers).any(
            dim=-1
        )
        cp_atom_is_not_metal = (
            cp_atom_types.view(-1, 1) == non_metal_atomic_numbers
        ).any(dim=-1)
        metal_mask = torch.logical_and(
            cp_atom_is_metal.view(-1, 1), cp_atom_is_metal.view(1, -1)
        )
        non_metal_mask = torch.logical_and(
            cp_atom_is_not_metal.view(-1, 1), cp_atom_is_not_metal.view(1, -1)
        )
        dist_mat[metal_mask] = 1000.0
        dist_mat[non_metal_mask] = 1000.0

    row, col = linear_sum_assignment(dist_mat.cpu().numpy())
    cost = dist_mat[row, col]
    return {"row": torch.from_numpy(row), "col": torch.from_numpy(col), "cost": cost}

def get_mol_bonds(sbb):
    bonds = []
    bond_types = []
    sbb_pybel = pygmol2pybel(sbb)
    mol = sbb_pybel.OBMol
    for bond in ob.OBMolBondIter(mol):
        if bond.IsAromatic():
            btype = 0
        elif bond.GetBondOrder() == 1:
            btype = 1
        elif bond.GetBondOrder() == 2:
            btype = 2
        elif bond.GetBondOrder() == 3:
            btype = 3
        else:
            raise NotImplementedError
        begin, end = bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1
        bonds.append([begin, end])
        bond_types.append(btype)
    bonds = torch.tensor(bonds).T
    bond_types = torch.tensor(bond_types).long()
    return bonds, bond_types


def get_bb_bond_and_types(bb):
    new_bb = bb.clone()
    edge_index = new_bb.edge_index
    atom_index = (~bb.is_anchor).nonzero().flatten()
    anchor_index = bb.is_anchor.nonzero().T
    anchor_s_mask = (edge_index[0].view(-1, 1) == anchor_index).any(dim=1)
    anchor_t_mask = (edge_index[1].view(-1, 1) == anchor_index).any(dim=1)
    anchor_e_mask = torch.logical_or(anchor_s_mask, anchor_t_mask)
    edge_index = edge_index[:, ~anchor_e_mask]
    remapping = atom_index, torch.arange(len(atom_index))
    edge_index = remap_values(remapping, edge_index)
    edge_index = torch.unique(edge_index, dim=1)
    new_bb.edge_index = edge_index
    new_bb.frac_coords = bb.frac_coords[atom_index]
    new_bb.atom_types = bb.atom_types[atom_index]
    new_bb.num_nodes = len(new_bb.atom_types)
    new_bb.is_anchor = torch.zeros(new_bb.num_nodes).bool()

    # only apply get_mol_bonds to organic building blocks.
    # for metal nodes, use original MOFid bonds and bond type single.
    has_metal = len(np.intersect1d(bb.atom_types.numpy(), METALS)) > 0
    if len(edge_index) == 0:
        edge_index = torch.FloatTensor([])
        bond_types = torch.FloatTensor([])
    elif has_metal:
        bond_types = torch.ones(len(edge_index[0])).long()
    else:
        edge_index, bond_types = get_mol_bonds(new_bb)

    if len(edge_index) == 0:
        edge_index = torch.FloatTensor([])
        bond_types = torch.FloatTensor([])
    else:
        remapping = torch.arange(len(atom_index)), atom_index
        edge_index = remap_values(remapping, edge_index)
        rev_edge_index = torch.stack([edge_index[1], edge_index[0]])
        edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
        bond_types = bond_types.repeat(2)
    return anchor_e_mask, edge_index, bond_types


def get_unique_and_index(x, dim=0):
    # https://github.com/pytorch/pytorch/issues/36748
    unique, inverse, counts = torch.unique(
        x, dim=dim, sorted=True, return_inverse=True, return_counts=True
    )
    decimals = torch.arange(inverse.numel(), device=inverse.device) / inverse.numel()
    inv_sorted = (inverse + decimals).argsort()
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    index = index.sort().values
    return unique, index


def get_connecting_atom_index(bbs):
    all_connecting_atoms = []
    offset = 0
    for bb in bbs:
        # relying on: cp_index is sorted, edges are double-directed
        cp_index = (bb.atom_types == 2).nonzero().flatten()
        connecting_atom_index = (
            bb.edge_index[1, (bb.edge_index[0].view(-1, 1) == cp_index).any(dim=-1)]
            + offset
        )
        offset += bb.num_atoms
        all_connecting_atoms.append(connecting_atom_index)
    all_connecting_atoms = torch.cat(all_connecting_atoms)
    return all_connecting_atoms


def assemble_mof(
    cg_mof,
    vecs=None,
    bb_local_vectors=None,
    bbs=None,
):
    if vecs is not None:
        final_R = axis_angle_to_matrix(vecs)
        cg_mof.R = final_R.float()

    cell = lattice_params_to_matrix_torch(cg_mof.lengths, cg_mof.angles).squeeze()
    device = cell.device
    cg_cart_coords = frac2cart(cg_mof.frac_coords, cell)
    bbs = bbs if bbs is not None else cg_mof.bbs
    bbs = [bb.to(device) for bb in bbs]

    cart_coords, atom_types, component, edge_index, to_jimages = [], [], [], [], []
    natom_offset = 0
    natom = cg_mof.num_components if "num_components" in cg_mof else cg_mof.num_atoms
    bond_types = []
    for i in range(natom):
        bb = bbs[i]
        bb_cart_coords = frac2cart(bb.frac_coords, bb.cell)
        bb_cart_coords = bb_cart_coords - bb_cart_coords[bb.is_anchor].mean(dim=0)
        bb_cart_coords = bb_cart_coords @ cg_mof.R[i]
        bb_cart_coords = cg_cart_coords[i] + bb_cart_coords

        cart_coords.append(bb_cart_coords)
        atom_types.append(bb.atom_types)
        component.append(torch.ones(bb_cart_coords.shape[0]) * i)

        anchor_e_mask, bb_edge_index, bb_bond_types = get_bb_bond_and_types(bb)
        anchor_edges = bb.edge_index[:, anchor_e_mask]
        bb_edge_index = torch.cat([anchor_edges, bb_edge_index], dim=1)
        bb_bond_types = torch.cat([torch.ones(anchor_e_mask.sum()), bb_bond_types])
        edge_index.append(bb_edge_index + natom_offset)
        bond_types.append(bb_bond_types)
        natom_offset += bb_cart_coords.shape[0]

    # remove CPs
    cart_coords = torch.cat(cart_coords, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    frac_coords = cart2frac(cart_coords, cell) % 1
    atom_node = atom_types != 2
    num_atoms = atom_node.sum()
    atom_index = atom_node.nonzero().flatten()
    connecting_atom_index = get_connecting_atom_index(bbs)
    cp_atom_types = atom_types[connecting_atom_index]

    # cp_frac_coords = frac_coords[~atom_node]
    frac_coords = frac_coords[atom_node]
    atom_types = atom_types[atom_node]

    # resolving edges after removing CPs
    edge_index = torch.cat(edge_index, dim=1)
    bond_types = torch.cat(bond_types, dim=0)
    is_anchors = torch.cat([bb.is_anchor for bb in bbs])
    anchor_index = is_anchors.nonzero().T
    anchor_s_mask = (edge_index[0].view(-1, 1) == anchor_index).any(dim=1)
    anchor_neighs = torch.unique(edge_index[:, anchor_s_mask], dim=1)[1]
    cp_match = match_cps(
        vecs,
        cg_mof.frac_coords,
        bb_local_vectors,
        cg_mof.lengths,
        cg_mof.angles,
        cp_atom_types,
    )
    row, col = cp_match["row"], cp_match["col"]
    inter_BB_edges = torch.cat(
        [
            torch.stack([anchor_neighs[row], anchor_neighs[col]]),
            torch.stack([anchor_neighs[col], anchor_neighs[row]]),
        ],
        dim=1,
    )
    anchor_t_mask = (edge_index[1].view(-1, 1) == anchor_index).any(dim=1)
    anchor_e_mask = torch.logical_or(anchor_s_mask, anchor_t_mask)
    edge_index = edge_index[:, ~anchor_e_mask]
    bond_types = bond_types[~anchor_e_mask]

    edge_index = torch.cat([edge_index, inter_BB_edges], dim=1)
    bond_types = torch.cat(
        [bond_types, torch.ones(inter_BB_edges.shape[1])], dim=0
    ).long()
    remapping = atom_index, torch.arange(num_atoms)
    edge_index = remap_values(remapping, edge_index)
    edge_index, unique_index = get_unique_and_index(edge_index, dim=1)
    bond_types = bond_types[unique_index]
    to_jimages = compute_image_flag(
        cell, frac_coords[edge_index[0]], frac_coords[edge_index[1]]
    )

    return Data(
        frac_coords=frac_coords,
        atom_types=atom_types,
        num_atoms=num_atoms,
        cell=cell,
        lengths=cg_mof.lengths,
        angles=cg_mof.angles,
        edge_index=edge_index,
        to_jimages=to_jimages,
        bond_types=bond_types,
    )


def feasibility_check(cg_mof):
    """
    check the matched connection criterion.
    """
    if cg_mof is not None and cg_mof.bbs[0] is not None:
        atom_types = torch.cat([x.atom_types for x in cg_mof.bbs])
        connecting_atom_index = get_connecting_atom_index(cg_mof.bbs)
        connecting_atom_types = atom_types[connecting_atom_index]
        n_metal = (
            (connecting_atom_types.view(-1, 1) == metal_atomic_numbers)
            .any(dim=-1)
            .sum()
        )
        n_nonmetal = (
            (connecting_atom_types.view(-1, 1) == non_metal_atomic_numbers)
            .any(dim=-1)
            .sum()
        )
        return n_metal == n_nonmetal
    else:
        return False
