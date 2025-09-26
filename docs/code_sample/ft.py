# -*- coding: utf-8 -*-
# 
# Find Time
#
# python ft.py \
#     --base_path /path/to/TNG100-1-Dark \
#     --config_file /path/to/tng100-1-dark.json \
#     --output_file /path/to/output/tng100-1-dark_postprocessed.hdf5

import argparse
import json
import gc
from pathlib import Path
import numpy as np
from tqdm import tqdm

from pyhipp.io import h5

from pyhipp_sims.sims.sim_info import SimInfo
from pyhipp_sims.sims.trees import TreeLoaderTngDark
from dwarf_assembly_bias.samples.model_dumper import SampleTngDark, ExtraPropTngDark

def main(args):
    print(f"Loading simulation config from: {args.config_file}")
    with open(args.config_file, 'r') as f:
        config_data = json.load(f)
    
    sim_info = SimInfo(
        name=config_data['formal_name'],
        root_dir=Path(args.base_path),
        root_file=Path(config_data['root_file']),
        sim_def=config_data['simulation_definition'],
        mahgic_ext = {'hubble_constant': config_data['hubble_constant']}
    )
    
    tr_ld = TreeLoaderTngDark(sim_info)

    h = sim_info.cosmology.hubble
    m_sub_lb_sim_units = (10**10.0) * h / 1e10 
    
    print(f"Selecting all central halos at z=0 above a low-mass threshold...")
    sample_selector = SampleTngDark(
        tr_ld, z_dst=0.0, only_c=True, m_sub_lb=m_sub_lb_sim_units
    )
    sample_selector.run()

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing and saving halo properties to: {output_file}")
    with h5.File(output_file, 'w') as f_out:
        sample_selector.dump_to(f_out.create_group('sample_info'))
        
        prop_extractor = ExtraPropTngDark(
            tr_ld,
            in_group=f_out['sample_info'],
            out_group=f_out.create_group('properties')
        )
        
        print("  - Loading base tree properties...")
        prop_extractor.load_tree_props()
        print("  - Calculating formation times (zf)...")
        prop_extractor.find_ftimes()
        print("  - Calculating peak mass/velocity (Mpeak/Vpeak)...")
        prop_extractor.find_peak()
        print("  - Calculating infall history...")
        prop_extractor.find_infall()

    print("\nPre-processing complete!")
    print(f"Full processed catalog saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Pre-process TNG merger trees to create a full halo catalog.")
    parser.add_argument('--base_path', required=True, help="Base path to the TNG simulation directory.")
    parser.add_argument('--config_file', required=True, help="Path to the simulation JSON configuration file.")
    parser.add_argument('--output_file', required=True, help="Path for the output full HDF5 halo catalog.")
    
    args = parser.parse_args()
    main(args)