from pathlib import Path
import argparse
import pickle
import pandas as pd
from p_tqdm import p_umap
from mofdiff.common.atomic_utils import pyg_graph_from_cif, assemble_local_struct
from mofid.id_constructor import extract_topology

import multiprocessing as mp

mp.set_start_method("spawn", force=True)


def preprocess_graph(df_path, mofid_path, save_path, num_workers, device="cpu"):
    df = pd.read_csv(str(df_path))
    df.rename(columns={"MOFname": "material_id"}, inplace=True)
    save_path.mkdir(exist_ok=True, parents=True)

    def assemble_mof(m_id, use_asr=True):
        try:
            # use the metaloxo algorithm for deconstruction.
            comp_path = mofid_path / m_id / "MetalOxo"
            g_nodes = pyg_graph_from_cif(comp_path / "nodes.cif")
            g_linkers = pyg_graph_from_cif(comp_path / "linkers.cif")
            if g_nodes.num_atoms == 0 or g_linkers.num_atoms == 0:
                return None
            g_node_bridges = pyg_graph_from_cif(comp_path / "node_bridges.cif")
            if use_asr:
                g_asr = pyg_graph_from_cif(comp_path / "mof_asr.cif")
            else:
                g_asr = None
            data = assemble_local_struct(
                g_nodes, g_linkers, g_node_bridges, g_asr, device=device
            )
        except FileNotFoundError:
            return None
        except UnboundLocalError:
            return None
        except IndexError:
            return None
        return data

    def save_mof_graph_pkl(m_id):
        props = df.loc[m_id]
        if (save_path / f"{m_id}.pkl").exists():
            return m_id, False

        failed = True
        data = assemble_mof(m_id)
        if data is not None:
            data.m_id = m_id
            data.prop_dict = dict(props)
            top = extract_topology(mofid_path / "MetalOxo" / "topology.cgd")
            data.top = top
            with open(str(save_path / f"{m_id}.pkl"), "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            failed = False

        return m_id, failed

    done = [idx.parts[-1][:-4] for idx in save_path.glob("*.pkl")]
    undone = list(set(df["material_id"]) - set(done))
    num_data_points = len(undone)
    print(f"{num_data_points}/{len(df)} data points to process.")
    df["material_idx"] = df["material_id"]
    df.set_index("material_id", inplace=True)
    df["material_id"] = df["material_idx"]

    results = p_umap(
        save_mof_graph_pkl,
        undone,
        num_cpus=num_workers,
    )

    failed_ids = [x[0] for x in results if x[1]]
    with open(save_path / "failed_id.txt", "a+") as f:
        f.write("\n".join(failed_ids))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--df_path",
        type=str,
        help="path to dataframe of material id/properties.",
    )
    parser.add_argument(
        "--mofid_path",
        type=str,
        help="path to extracted mofids.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="path to save graph pickle files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    preprocess_graph(
        Path(args.df_path),
        Path(args.mofid_path),
        Path(args.save_path),
        args.num_workers,
        args.device,
    )
