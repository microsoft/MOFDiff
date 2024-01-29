"""
A function to replicate the vacuum swing adsorption calculation performed by Boyd et al. 
in "Data-driven design of metal-organic frameworks for wet flue gas CO2 capture"
https://archive.materialscloud.org/record/2018.0016/v3

Mixture adsorption was simulated with the conditions 298K and 0.15:0.85 CO2/N2 
with a total pressure of 1 bar. The data file reports working capacities, which is 
the difference of adsorption of CO2 between two thermodynamic state points.
two desorption values were simulated; 0.1 bar CO2 at 363K (vacuum swing adsorption)
and 0.7 bar CO2 at 413K (temperature swing adsorption).
"""
import random
import numpy as np
from mofdiff.common.sys_utils import timeout
from mofdiff.gcmc import gcmc_wrapper
import re


def extract_raspa_output(raspa_output, has_N2=False):
    final_loading_section = re.findall(
        r"Number of molecules:\n=+[^=]*(?=)", raspa_output
    )[0]
    enthalpy_of_adsorption_section = re.findall(
        r"Enthalpy of adsorption:\n={2,}\n(.+?)\n={2,}", raspa_output, re.DOTALL
    )[0]

    CO2_subsection = re.findall(
        r"Component \d \[CO2\].*?(?=Component|\Z)", final_loading_section, re.DOTALL
    )[0]
    adsorbed_CO2 = float(
        re.findall(
            r"(?<=Average loading absolute \[mol/kg framework\])\s*\d*\.\d*",
            CO2_subsection,
        )[0]
    )

    if has_N2:
        CO2_enthalpy_subsection = re.findall(
            r"\[CO2\].*?(?=component|\Z)", enthalpy_of_adsorption_section, re.DOTALL
        )[0]

        enthalpy_of_adsorption_CO2 = float(
            re.findall(r"(?<=\[K\])\s*-?\d*\.\d*", CO2_enthalpy_subsection)[0]
        ) * 0.239
        heat_of_adsorption_CO2 = -1 * enthalpy_of_adsorption_CO2

        N2_subsection = re.findall(
            r"Component \d \[N2\].*?(?=Component|\Z)", final_loading_section, re.DOTALL
        )[0]
        adsorbed_N2 = float(
            re.findall(
                r"(?<=Average loading absolute \[mol/kg framework\])\s*\d*\.\d*",
                N2_subsection,
                re.DOTALL,
            )[0]
        )
        CO2_N2_selectivity = adsorbed_CO2 / adsorbed_N2

        N2_enthalpy_subsection = re.findall(
            r"\[N2\].*?(?=component|\Z)", enthalpy_of_adsorption_section, re.DOTALL
        )[0]
        enthalpy_of_adsorption_N2 = float(
            re.findall(r"(?<=\[K\])\s*-?\d*\.\d*", N2_enthalpy_subsection)[0]
        ) * 0.239
        heat_of_adsorption_N2 = -1 * enthalpy_of_adsorption_N2

        return (
            adsorbed_CO2,
            adsorbed_N2,
            CO2_N2_selectivity,
            heat_of_adsorption_CO2,
            heat_of_adsorption_N2,
        )

    else:
        CO2_enthalpy_subsection = re.findall(
            r"Total enthalpy of adsorption.*?(?=Q=-H|\Z)",
            enthalpy_of_adsorption_section,
            re.DOTALL,
        )[0]
        enthalpy_of_adsorption_CO2 = float(
            re.findall(r"(?<=\[K\])\s*-?\d*\.\d*", CO2_enthalpy_subsection)[0]
        ) * 0.239 # (kcal per mol)
        heat_of_adsorption_CO2 = -1 * enthalpy_of_adsorption_CO2

        return adsorbed_CO2, heat_of_adsorption_CO2

@timeout(36000)
def working_capacity_vacuum_swing(cif_file, calc_charges=True,
                                  rundir='./temp', rewrite_raspa_input=False):
    random.seed(4)
    np.random.seed(4)
    # adsorption conditions
    adsorbed = gcmc_wrapper.gcmc_simulation(
        cif_file,
        sorbates=["CO2", "N2"],
        sorbates_mol_fraction=[0.15, 0.85],
        temperature=298,
        pressure=100000,  # 1 bar
        rundir=rundir,
    )

    if calc_charges:
        gcmc_wrapper.calculate_mepo_qeq_charges(adsorbed)
    gcmc_wrapper.run_gcmc_simulation(
        adsorbed,
        rewrite_raspa_input=rewrite_raspa_input,
    )

    (
        adsorbed_CO2,
        adsorbed_N2,
        CO2_N2_selectivity_298,
        heat_of_adsorption_CO2_298,
        heat_of_adsorption_N2_298,
    ) = extract_raspa_output(adsorbed.raspa_output, has_N2=True)

    # desorption conditions
    residual = gcmc_wrapper.gcmc_simulation(
        cif_file,
        sorbates=["CO2"],
        sorbates_mol_fraction=[1],
        temperature=363,  # 363,
        pressure=10000,  # 10000 # 0.1 bar
        rundir=rundir,
    )
    
    if calc_charges:
        gcmc_wrapper.calculate_mepo_qeq_charges(residual)
    gcmc_wrapper.run_gcmc_simulation(
        residual,
        rewrite_raspa_input=rewrite_raspa_input,
    )

    residual_CO2, heat_of_adsorption_CO2_363 = extract_raspa_output(
        residual.raspa_output, has_N2=False
    )

    output = {
        "file": str(cif_file),
        "working_capacity_vacuum_swing": adsorbed_CO2 - residual_CO2,
        "CO2_N2_selectivity": CO2_N2_selectivity_298,
        "CO2_uptake_P0.15bar_T298K": adsorbed_CO2,
        "CO2_uptake_P0.10bar_T363K": residual_CO2,
        "CO2_heat_of_adsorption_P0.15bar_T298K": heat_of_adsorption_CO2_298,
        "CO2_heat_of_adsorption_P0.10bar_T363K": heat_of_adsorption_CO2_363,
        "N2_uptake_P0.85bar_T298K": adsorbed_N2,
        "N2_heat_of_adsorption_P0.85bar_T298K": heat_of_adsorption_N2_298,
    }

    return output

def run_or_fail(cif_path):
    try:
        return working_capacity_vacuum_swing(cif_path)
    except Exception as e:
        print(e)
        return None
