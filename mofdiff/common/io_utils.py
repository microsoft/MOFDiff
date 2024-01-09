import numpy as np
from ase.data import atomic_numbers


def readcif(name):
    with open(name, "r") as fi:
        EIF = fi.readlines()
        cond2 = False
        atom_props_count = 0
        atomlines = []
        counter = 0
        cell_parameter_boundary = [0.0, 0.0]
        for line in EIF:
            line_stripped = line.strip()
            if (not line) or line_stripped.startswith("#"):
                continue
            line_splitted = line.split()

            if line_stripped.startswith("_cell_length_a"):
                temp = line_splitted[1].replace(")", "")
                temp = temp.replace("(", "")
                cell_a = float(temp)
                cell_parameter_boundary[0] = counter + 1
            elif line_stripped.startswith("_cell_length_b"):
                temp = line_splitted[1].replace(")", "")
                temp = temp.replace("(", "")
                cell_b = float(temp)
            elif line_stripped.startswith("_cell_length_c"):
                temp = line_splitted[1].replace(")", "")
                temp = temp.replace("(", "")
                cell_c = float(temp)
            elif line_stripped.startswith("_cell_angle_alpha"):
                temp = line_splitted[1].replace(")", "")
                temp = temp.replace("(", "")
                cell_alpha = float(temp)
            elif line_stripped.startswith("_cell_angle_beta"):
                temp = line_splitted[1].replace(")", "")
                temp = temp.replace("(", "")
                cell_beta = float(temp)
            elif line_stripped.startswith("_cell_angle_gamma"):
                temp = line_splitted[1].replace(")", "")
                temp = temp.replace("(", "")
                cell_gamma = float(temp)
                cell_parameter_boundary[1] = counter + 1
            if cond2 and line_stripped.startswith("loop_"):
                break
            else:
                if line_stripped.startswith("_atom"):
                    atom_props_count += 1
                    if line_stripped == "_atom_site_label":
                        type_index = atom_props_count - 1
                    elif line_stripped == "_atom_site_fract_x":
                        fracx_index = atom_props_count - 1
                    elif line_stripped == "_atom_site_fract_y":
                        fracy_index = atom_props_count - 1
                    elif line_stripped == "_atom_site_fract_z":
                        fracz_index = atom_props_count - 1
                    cond2 = True
                elif cond2:
                    if len(line_splitted) == atom_props_count:
                        atomlines.append(line)
            counter += 1
        positions = []
        atomtypes = []
        for _, at in enumerate(atomlines):
            ln = at.strip().split()
            positions.append(
                [
                    float(ln[fracx_index].replace("(", "").replace(")", "")),
                    float(ln[fracy_index].replace("(", "").replace(")", "")),
                    float(ln[fracz_index].replace("(", "").replace(")", "")),
                ]
            )
            ln[type_index] = ln[type_index].strip("_")
            at_type = "".join([i for i in ln[type_index] if not i.isdigit()])
            atomtypes.append(atomic_numbers[at_type])

        lattice_params = np.array(
            [cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma]
        )
        positions = np.array(positions)
        atomtypes = np.array(atomtypes)
        return lattice_params, atomtypes, positions


def transform_jimages(item):
    if (item == np.array([0, 0, 0])).all():
        jimage = "."
    else:
        label = (item[0] + 5) * 100 + (item[1] + 5) * 10 + (item[2] + 5)
        jimage = f"1_{label}"
    return jimage


def writecif(
    fname, cellprm, fcoords, atom_labels, edge_index, distances, to_jimages, bond_types
):
    fname = str(fname)

    with open(fname, "w") as f_cif:
        f_cif.write("data_I\n")
        f_cif.write("_chemical_name_common  '%s'\n" % (fname.strip(".cif")))
        f_cif.write("_cell_length_a %8.05f\n" % (cellprm[0]))
        f_cif.write("_cell_length_b %8.05f\n" % (cellprm[1]))
        f_cif.write("_cell_length_c %8.05f\n" % (cellprm[2]))
        f_cif.write("_cell_angle_alpha %4.05f\n" % (cellprm[3]))
        f_cif.write("_cell_angle_beta  %4.05f\n" % (cellprm[4]))
        f_cif.write("_cell_angle_gamma %4.05f\n" % (cellprm[5]))
        f_cif.write("_space_group_name_H-M_alt      'P 1'\n\n\n")
        f_cif.write("loop_\n_space_group_symop_operation_xyz\n  'x, y, z' \n\n")
        f_cif.write("loop_\n")
        f_cif.write("_atom_site_label\n")
        f_cif.write("_atom_site_fract_x\n")
        f_cif.write("_atom_site_fract_y\n")
        f_cif.write("_atom_site_fract_z\n")
        f_cif.write("_atom_site_type_symbol\n")

        numbered_labels = []
        for i, atom in enumerate(atom_labels):
            numbered_labels.append(atom + str(i))
            f_cif.write(
                "%-5s %8s %8s %8s %5s\n"
                % (
                    atom + str(i),
                    fcoords[i, 0],
                    fcoords[i, 1],
                    fcoords[i, 2],
                    "%s" % (atom),
                )
            )

        # Add loop for bond information
        f_cif.write("loop_\n")
        f_cif.write("_geom_bond_atom_site_label_1\n")
        f_cif.write("_geom_bond_atom_site_label_2\n")
        f_cif.write("_geom_bond_distance\n")
        f_cif.write("_geom_bond_site_symmetry_2\n")
        f_cif.write("_ccdc_geom_bond_type\n")
        for i in range(len(edge_index)):
            atom1 = numbered_labels[edge_index[i, 0]]
            atom2 = numbered_labels[edge_index[i, 1]]
            dist = distances[i]
            jimage = transform_jimages(to_jimages[i])
            bond_type = bond_types[i] if bond_types is not None else "S"
            f_cif.write(
                "%-5s %-5s %8.05f %7s %5s\n" % (atom1, atom2, dist, jimage, bond_type)
            )
