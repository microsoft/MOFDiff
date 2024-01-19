"""
UFF relaxation with lammps-interface.
"""
from textwrap import dedent
from pathlib import Path
import os
import re
import subprocess
import numpy as np
import mendeleev

from lammps import lammps
from pymatgen.io.lammps import outputs as lmp_outputs
from pymatgen.core import Structure

from mofdiff.common.sys_utils import timeout


def extract_info_from_log(log_file_path):
    with open(log_file_path, "r") as file:
        lines = file.readlines()

    total_steps_part_1 = None
    total_steps_part_2 = None
    PotEng_start = None
    PotEng_after_part_1 = None
    PotEng_after_part_2 = None
    Volume_start = None
    Volume_end = None
    total_wall_time = None

    for i, line in enumerate(lines):
        if "ERROR" in line:
            return None

        if "Loop time of" in line:
            steps = int(line.split(" for ")[1].split(" steps")[0])
            if total_steps_part_1 is None:
                total_steps_part_1 = steps
            elif total_steps_part_2 is None:
                total_steps_part_2 = steps

        if "Step" in line and "PotEng" in line and "Volume" in line:
            next_line = lines[i + 1] if i + 1 < len(lines) else None
            if next_line is not None:
                splits = next_line.split()
                if PotEng_start is None:
                    PotEng_start = float(splits[1])
                    Volume_start = float(splits[2])
                elif PotEng_after_part_1 is None:
                    PotEng_after_part_1 = float(splits[1])
                else:
                    PotEng_after_part_2 = float(splits[1])
                    Volume_end = float(splits[2])

        if "Total wall time:" in line:
            time_str = line.split(": ")[1].strip()
            hours, minutes, seconds = map(int, time_str.split(":"))
            total_wall_time = hours * 3600 + minutes * 60 + seconds

    return {
        "step_1": total_steps_part_1,
        "step_2": total_steps_part_2,
        "PE_start": PotEng_start,
        "PE_1": PotEng_after_part_1,
        "PE_2": PotEng_after_part_2,
        "v_start": Volume_start,
        "v_end": Volume_end,
        "walltime": total_wall_time,
    }


@timeout(7200)
def initiate_lammps_with_force_field(cif_file, force_field="UFF"):
    """
    This function initiates a lammps instance where energies and forces are calculated 
    using a force field implemented within lammps. The returned lammps instance can then be 
    issued additional commands. Parameterization of the sorbent structure for use with 
    classical force fields is done using lammps-interface.
    """
    subprocess.run(
        ["lammps-interface", cif_file, "-ff", force_field], text=True, input="y"
    )

    # collect values from resulting input file ("in.cif_file_name")
    fname = cif_file.split("/")[-1].split(".")[0]
    with open(("in." + fname), "r") as input_file:
        # remove group definitions to avoid exceeding LAMMPS limit
        lammps_interface_string = re.findall(
            "(.*)#### Atom", input_file.read(), re.DOTALL
        )[0]

        # remove log definition
        lammps_interface_string = re.sub("log.+\n", "", lammps_interface_string)

    # start a lammps instance
    logname = str(Path(cif_file).parent.parent / "relaxed" / f"{fname}_log.lammps")
    lmp = lammps(cmdargs=["-log", logname])
    print(lammps_interface_string)
    # set up system in lammps
    lmp.commands_string(lammps_interface_string)

    # return initialized lammps instance
    return lmp


@timeout(7200)
def relax_structure_and_box(
    initialized_lammps,
    relax_store_path,
    uid,
    min_energy_threshold=1e-8,
    min_force_threshold=1e-8,
    min_max_iters=100000,
    min_max_force_evals=100000,
):
    # set parameters for minimization algorithm and calculation of thermodynamic properties
    initialized_lammps.commands_string(
        dedent(
            """       
        neighbor 1.0 nsq
        neigh_modify once no every 1 delay 0 check yes
        
        thermo          100
        thermo_style    custom step pe vol
        thermo_modify    norm no
        
        timer timeout 1:00:00 every 100
    """
        ).strip()
    )

    initialized_lammps.commands_string(
        dedent(
            """
        min_style       cg
        min_modify      dmax 0.01
        minimize        {min_energy_threshold} {min_force_threshold} {min_max_iters} {min_max_force_evals}
        fix             minimization_pressure all box/relax tri 0.0
        minimize        {min_energy_threshold} {min_force_threshold} {min_max_iters} {min_max_force_evals}
        unfix           minimization_pressure
        minimize        {min_energy_threshold} {min_force_threshold} {min_max_iters} {min_max_force_evals}
        fix             minimization_pressure all box/relax tri 0.0
        minimize        {min_energy_threshold} {min_force_threshold} {min_max_iters} {min_max_force_evals}
        unfix           minimization_pressure
    """.format(
                **locals()
            )
        ).strip()
    )

    # dump relaxed structure
    initialized_lammps.commands_string(
        dedent(
            f"""
        dump            output all atom 1 relaxed_structure_{uid}.dump
        run             0
        undump          output
    """.strip()
        )
    )

def dump_to_structure(dump_file, data_file):
    dump = next(lmp_outputs.parse_lammps_dumps(dump_file))

    # read atomic masses from data file
    with open(data_file, "r") as file:
        masses = re.findall(r"(?<=Masses\n{2})(.*?)(?=\n{2})", file.read(), re.DOTALL)[
            0
        ]
        masses = masses.split("\n")

    # lookup the atomic symbol associated with each atomic mass
    element_symbols = np.array(
        [element.symbol for element in mendeleev.elements.get_all_elements()]
    )
    element_mass = np.array(
        [element.mass for element in mendeleev.elements.get_all_elements()]
    )

    identifier_to_symbol = dict()
    for atom in masses:
        lammps_symbol = atom.split()[0]
        lammps_mass = float(atom.split()[1])
        identifier_to_symbol[lammps_symbol] = element_symbols[
            (np.abs(element_mass - lammps_mass)).argmin()
        ]

    dump.data["element"] = dump.data["type"].apply(
        lambda x: identifier_to_symbol[str(x)]
    )

    # build a pymatgen structure
    structure = Structure(
        dump.box.to_lattice(),
        dump.data.element,
        dump.data.loc[:, ["xs", "ys", "zs"]].to_numpy(),
    )

    return structure


@timeout(7200)
def lammps_relax(
    cif_file, relax_store_path="./", cleanup=True, force_field="UFF"
):
    uid = Path(cif_file).parts[-1].split(".")[0]
    if isinstance(relax_store_path, str):
        relax_store_path = Path(relax_store_path)

    try:
        cwd = os.getcwd()
        os.chdir(str(relax_store_path))
        lmp = initiate_lammps_with_force_field(cif_file, force_field)
        relax_structure_and_box(lmp, str(relax_store_path), uid)
        struct = dump_to_structure(
            str(relax_store_path / f"relaxed_structure_{uid}.dump"),
            str(relax_store_path / f"data.{uid}"),
        )
        if cleanup:
            os.remove(f"relaxed_structure_{uid}.dump")
            os.remove(f"data.{uid}")
            os.remove(f"in.{uid}")
        os.chdir(cwd)
    except Exception as e:
        print(e)
        struct = None

    fname = cif_file.split("/")[-1].split(".")[0]
    logname = Path(cif_file).parent.parent / "relaxed" / f"{fname}_log.lammps"
    if logname.exists():
        relax_info = extract_info_from_log(str(logname))
        if relax_info is not None:
            relax_info.update({"uid": uid})
    else:
        relax_info = None

    return struct, relax_info
