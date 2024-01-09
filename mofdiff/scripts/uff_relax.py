from pathlib import Path
from functools import partial
import argparse
import json

from pymatgen.io.cif import CifWriter

from mofdiff.common.relaxation import lammps_relax
from mofdiff.common.mof_utils import save_mofid, mof_properties
from mofid.id_constructor import extract_topology

from p_tqdm import p_umap


def main(input_dir, max_natoms=2000, get_mofid=True, ncpu=96):
    """
    max_natoms: maximum number of atoms in a MOF primitive cell to run zeo++/mofid.
    """
    all_files = list((Path(input_dir) / "cif").glob("*.cif"))

    save_dir = Path(input_dir) / "relaxed"
    save_dir.mkdir(exist_ok=True, parents=True)

    def relax_mof(ciffile):
        name = ciffile.parts[-1].split(".")[0]
        try:
            struct, relax_info = lammps_relax(str(ciffile), str(save_dir))
        except TimeoutError:
            return None

        if struct is not None:
            struct = struct.get_primitive_structure()
            CifWriter(struct).write_file(save_dir / f"{name}.cif")
            relax_info["natoms"] = struct.frac_coords.shape[0]
            relax_info["path"] = str(save_dir / f"{name}.cif")
            return relax_info
        else:
            return None

    results = p_umap(relax_mof, all_files, num_cpus=ncpu)
    relax_infos = [x for x in results if x is not None]
    with open(save_dir / "relax_info.json", "w") as f:
        json.dump(relax_infos, f)

    # ZEO++ properties and validity checks
    zeo_dir = Path(input_dir) / "zeo"
    zeo_dir.mkdir(exist_ok=True)
    relaxed_files = [
        info["path"] for info in relax_infos if info["natoms"] <= max_natoms
    ]

    zeo_props = p_umap(
        partial(mof_properties, zeo_store_path=zeo_dir), relaxed_files, num_cpus=64
    )
    zeo_props = [x for x in zeo_props if x is not None]
    with open(Path(input_dir) / "zeo_props_relax.json", "w") as f:
        json.dump(zeo_props, f)

    valid_mof_paths = []
    for prop in zeo_props:
        if prop["all_check"]:
            valid_mof_paths.append(prop["path"])
    with open(Path(input_dir) / "valid_mof_paths.json", "w") as f:
        json.dump(valid_mof_paths, f)

    # MOFid
    if get_mofid:
        mofid_dir = Path(input_dir) / "mofid"
        mofid_dir.mkdir(exist_ok=True)

        def process_one(cif_file):
            cif_file = Path(cif_file)
            uid = cif_file.parts[-1].split(".")[0]
            try:
                (mofid_dir / f"{uid}").mkdir(exist_ok=True)
                save_mofid(cif_file, mofid_dir, primitive=True)
                top = extract_topology(mofid_dir / uid / "MetalOxo" / "topology.cgd")
                return {"uid": uid, "top": top}
            except Exception as e:
                print(e)
                return None

        # do not run mofid on data points.
        valid_sample_path = [
            x for x in valid_mof_paths if "data" not in Path(x).parts[-1]
        ]
        mofid_success_uids = p_umap(process_one, valid_sample_path, num_cpus=ncpu)
        with open(Path(input_dir) / "mofid_success_uids.json", "w") as f:
            json.dump(mofid_success_uids, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    args = parser.parse_args()
    main(args.input)