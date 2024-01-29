from pathlib import Path 
import json
import argparse
import torch

from mofdiff.common.atomic_utils import graph_from_cif
from mofdiff.common.data_utils import lattice_params_to_matrix
from mofdiff.common.atomic_utils import frac2cart, compute_distance_matrix
from mofdiff.gcmc.simulation import working_capacity_vacuum_swing
from p_tqdm import p_umap


def main(input_dir, ncpu=24, rewrite_raspa_input=False):
    rundir = Path(input_dir) / 'gcmc'
    rundir.mkdir(exist_ok=True)
    
    if Path(input_dir).parts[-1].startswith('mepo'):
        all_files = list((Path(input_dir)).glob('*.cif'))
        calc_charges = False
    else:
        with open(Path(input_dir) / 'valid_mof_paths.json', "r") as f:
            all_files = [Path(x) for x in json.load(f)]
        calc_charges = True
    
    # skip data entries.
    all_files = [x for x in all_files if 'data' not in x.parts[-1]]

    def compute_gcmc(ciffile, max_natom=10000):
        uid = ciffile.parts[-1].split('.')[0]
        try:
            struct = graph_from_cif(ciffile).structure.get_primitive_structure()
            if struct.frac_coords.shape[0] > max_natom:
                adsorption_info = None
                info = f'too large: {struct.frac_coords.shape[0]}'
            else:
                frac_coords = torch.tensor(struct.frac_coords).float()
                cell = torch.from_numpy(lattice_params_to_matrix(*struct.lattice.parameters)).float()
                cart_coords = frac2cart(frac_coords, cell)
                dist_mat = compute_distance_matrix(cell, cart_coords).fill_diagonal_(5.)
                if dist_mat.min() < 0.5:
                    adsorption_info = None
                    info = 'atomic overlap'
                else:
                    adsorption_info = working_capacity_vacuum_swing(
                        str(ciffile),
                        calc_charges=calc_charges,
                        rundir=rundir,
                        rewrite_raspa_input=rewrite_raspa_input,
                    )
                    info = 'success'
        except Exception as e:
            print(f'Error in {ciffile}: {e}')
            adsorption_info = None
            info = str(e)
                
        return dict(uid=uid, info=info, adsorption_info=adsorption_info)
    
    results = p_umap(compute_gcmc, all_files, num_cpus=ncpu)
    with open(rundir / 'screening_results.json', 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--rewrite_raspa_input', action='store_true')
    parser.set_defaults(rewrite_raspa_input=False)
    args = parser.parse_args()
    main(args.input, rewrite_raspa_input=args.rewrite_raspa_input)
