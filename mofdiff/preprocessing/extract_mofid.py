"""
Extract mofid for a dataset.
"""
from pathlib import Path
import argparse
import pandas as pd
from p_tqdm import p_umap

from mofdiff.common.mof_utils import save_mofid


def preprocess_mofid(df_path, cif_path, save_path, num_workers):
    df = pd.read_csv(str(df_path))
    df.rename(columns={"MOFname": "material_id"}, inplace=True)
    save_path.mkdir(exist_ok=True, parents=True)

    def process_one(row):
        m_id = row["material_id"]
        try:
            save_mofid(cif_path / f"{m_id}.cif", save_path, True)
            return row["material_id"], False
        except ValueError as e:
            print(e)
            return row["material_id"], True
        except IndexError as e:
            print(e)
            return row["material_id"], True

    results = p_umap(
        process_one, [df.iloc[idx] for idx in range(len(df))], num_cpus=num_workers
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
        "--cif_path",
        type=str,
        help="path to cif files.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="path to save mofid.",
    )
    parser.add_argument("--num_workers", type=int, default=96)
    args = parser.parse_args()

    preprocess_mofid(
        Path(args.df_path),
        Path(args.cif_path),
        Path(args.save_path),
        args.num_workers,
    )
