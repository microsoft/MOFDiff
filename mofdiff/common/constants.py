"""
Elemental information.
"""

COVALENT_RADII = {
    # Covalent radii revisited -- DOI:10.1039/B801115J
    "H": 0.31,
    "He": 0.28,
    "Li": 1.28,
    "Be": 0.96,
    "B": 0.84,
    "C": 0.76,  # for sp3; sp2 = 0.73; sp = 0.69
    "C_1": 0.69,
    "C_2": 0.73,
    "C_R": 0.73,
    "C_3": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Ne": 0.58,
    "Na": 1.66,
    "Mg": 1.41,
    "Al": 1.21,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Ar": 1.06,
    "K": 2.03,
    "Ca": 1.76,
    "Sc": 1.7,
    "Ti": 1.6,
    "V": 1.53,
    "Cr": 1.39,
    "Mn": 1.61,  # low spin = 1.39
    "Fe": 1.52,  # low spin = 1.32
    "Co": 1.5,  # low spin = 1.26
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    "Ga": 1.22,
    "Ge": 1.2,
    "As": 1.19,
    "Se": 1.2,
    "Br": 1.2,
    "Kr": 1.16,
    "Rb": 2.2,
    "Sr": 1.95,
    "Y": 1.9,
    "Zr": 1.5,  # 1.75  !! temporarily added to correct the UIO cluster bonding problem
    "Nb": 1.64,
    "Mo": 1.54,
    "Tc": 1.47,
    "Ru": 1.46,
    "Rh": 1.42,
    "Pd": 1.39,
    "Ag": 1.45,
    "Cd": 1.44,
    "In": 1.42,
    "Sn": 1.39,
    "Sb": 1.39,
    "Te": 1.38,
    "I": 1.39,
    "Xe": 1.4,
    "Cs": 2.44,
    "Ba": 2.15,
    "La": 2.07,
    "Ce": 2.04,
    "Pr": 2.03,
    "Nd": 2.01,
    "Pm": 1.99,
    "Sm": 1.98,
    "Eu": 1.98,
    "Gd": 1.96,
    "Tb": 1.94,
    "Dy": 1.92,
    "Ho": 1.92,
    "Er": 1.89,
    "Tm": 1.9,
    "Yb": 1.87,
    "Lu": 1.87,
    "Hf": 1.75,
    "Ta": 1.7,
    "W": 1.62,
    "Re": 1.51,
    "Os": 1.44,
    "Ir": 1.41,
    "Pt": 1.36,
    "Au": 1.36,
    "Hg": 1.32,
    "Tl": 1.45,
    "Pb": 1.46,
    "Bi": 1.48,
    "Po": 1.4,
    "At": 1.5,
    "Rn": 1.5,
    "Fr": 2.6,
    "Ra": 2.21,
    "Ac": 2.15,
    "Th": 2.06,
    "Pa": 2,
    "U": 1.96,
    "Np": 1.9,
    "Pu": 1.87,
    "Am": 1.8,
    "Cm": 1.69,
}

METALS = [3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
          37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 55, 56, 57,
          58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
          75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92, 93, 94,
          95, 96, 97, 98, 99, 100, 101, 102, 103]

METAL_LIKE = [
    "Li", "Be", "B", "Na", "Mg",
    "Al", "Si", "K", "Ca", "Sc",
    "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga",
    "Ge", "As", "Rb", "Sr", "Y",
    "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm",
    "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os",
    "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Po", "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np",
    "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
]

organic = set(["H", "C", "N", "O", "S"])
non_metals = set(["H", "He", "C", "N", "O", "F", "Ne",
                  "P", "S", "Cl", "Ar", "Se", "Br", "Kr",
                  "I", "Xe", "Rn"])
noble_gases = set(["He", "Ne", "Ar", "Kr", "Xe", "Rn"])
metalloids = set(["B", "Si", "Ge", "As", "Sb", "Te", "At"])
lanthanides = set(["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu",
                   "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"])
actinides = set(["Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
                 "Cf", "Es", "Fm", "Md", "No", "Lr"])
transition_metals = set(["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
                         "Cu", "Zn", "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
                         "Rh", "Pd", "Ag", "Cd", "Hf", "Ta", "W", "Re",
                         "Os", "Ir", "Pt", "Ir", "Pt", "Au", "Hg", "Rf",
                         "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn"])
alkali = set(["Li", "Na", "K", "Rb", "Cs", "Fr"])
alkaline_earth = set(["Be", "Mg", "Ca", "Sr", "Ba", "Ra"])
main_group = set(["Al", "Ga", "Ge", "In", "Sn", "Sb", "Tl", "Pb", "Bi",
                  "Po", "At", "Cn", "Uut", "Fl", "Uup", "Lv", "Uus"])

metals = main_group | alkaline_earth | alkali | transition_metals | metalloids | lanthanides 

INDEX2BTYPE = ['A', 'S', 'D', 'T']
