from pathlib import Path
import os
import subprocess
import json
import argparse
from functools import partial
from p_tqdm import p_umap
from textwrap import dedent
from openbabel.pybel import readfile

from mofdiff.common.sys_utils import EGULP_PATH, EGULP_PARAMETER_PATH


# assigns charges to the atoms in the simulation file using the MEPO Qeq charge equilibration method
def calculate_mepo_qeq_charges(
    cif_file: Path, outdir: Path, egulp_parameter_set="MEPO"
):
    uid = cif_file.parts[-1].split(".")[0]
    assert outdir.exists()

    # use openbabel to process
    sorbent = next(readfile("cif", str(cif_file)))
    rundir = outdir / uid
    rundir.mkdir(exist_ok=True)
    sorbent_file = rundir / "charges.cif"
    sorbent.write("cif", str(sorbent_file), overwrite=True)

    # write out egulp config file
    config = dedent(
        """
            build_grid 0
            build_grid_from_scratch 1 none 0.25 0.25 0.25 1.0 2.0 0 0.3
            save_grid 0 grid.cube
            calculate_pot_diff 0
            calculate_pot 0 repeat.cube
            skip_everything 0
            point_charges_present 0
            include_pceq 0
            imethod 0
            """.format(
            **locals()
        )
    ).strip()

    with open(rundir / f"temp_config_{uid}.input", "w") as file:
        file.write(config)

    # run egulp
    subprocess.run(
        [
            EGULP_PATH,
            str(sorbent_file),
            os.path.join(EGULP_PARAMETER_PATH, egulp_parameter_set + ".param"),
            f"temp_config_{uid}.input",
        ],
        cwd=str(rundir),
    )

    # cleanup
    os.system(
        "mv {} {}".format(
            str(rundir / "charges.cif"), str(rundir.parent / f"{uid}.cif")
        )
    )
    os.system("rm -rf {}".format(str(rundir)))


def main(input_dir, ncpu=32):
    rundir = Path(input_dir) / "mepo_qeq_charges"
    rundir.mkdir(exist_ok=True)

    if (Path(input_dir) / "valid_mof_paths.json").exists():
        with open(Path(input_dir) / "valid_mof_paths.json", "r") as f:
            all_files = [Path(x) for x in json.load(f)]
    else:
        all_files = list((Path(input_dir) / "cif").glob("*.cif"))

    p_umap(partial(calculate_mepo_qeq_charges, outdir=rundir), all_files, num_cpus=ncpu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    args = parser.parse_args()
    main(args.input)