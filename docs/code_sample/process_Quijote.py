# cals_a50:
# Smith et al., (2019). ytree: A Python package for analyzing merger trees. Journal of Open Source Software, 4(44), 1881, https://doi.org/10.21105/joss.01881

# Usage:
# Place this script in the same directory as the `/hlists` and `/trees` folders.
# Run the script to generate the target halo sample at `/halo_sample/halo.txt`.

import ytree
import numpy as np
from pathlib import Path
import yt
from tqdm import tqdm

MIN_MASS = 10**13.5
MAX_MASS = 10**14

# --- File Paths ---
HLIST_PATH = Path('./hlists/hlist_1.00000.list')
OUTPUT_DIR = Path('./halo_sample/')
OUTPUT_FILE = OUTPUT_DIR / 'halo_Quijote.txt'

# --- Analysis and Filter Functions (for AnalysisPipeline) ---

def halo_filter(node):
    is_main_halo = (node['pid'] == -1)
    is_in_mass_range = (node['mass'] >= MIN_MASS) and (node['mass'] < MAX_MASS)
    return is_main_halo and is_in_mass_range

def calc_a50(node):
    try:
        pmass = node["prog", "mass"]
        pscale = node["prog", "scale_factor"]
        
        if len(pmass) < 2:
            node["a50"] = node["scale_factor"]
            return
    except (KeyError, IndexError):
        node["a50"] = -1.0
        return

    mh = 0.5 * node["mass"]
    m50 = pmass <= mh

    if not m50.any():
        node["a50"] = node["scale_factor"]
    else:
        i = np.where(m50)[0][0]
        if i == 0:
            node["a50"] = pscale[i]
        else:
            slope = (pscale[i - 1] - pscale[i]) / (pmass[i - 1] - pmass[i])
            node["a50"] = slope * (mh - pmass[i]) + pscale[i]

# --- Diagnostic Function ---

def run_diagnostics(arbor):
    main_halos_mask = arbor['pid'] == -1
    main_halo_masses = arbor['mass'][main_halos_mask]
    
    if len(main_halo_masses) == 0:
        print("CRITICAL ERROR: No main halos (pid == -1) found.")
        return False
        
    print(f"Found {len(main_halo_masses)} main halos in total.")
    print(f"  - Min Mass: {np.min(main_halo_masses):.2e} Msun/h | Max Mass: {np.max(main_halo_masses):.2e} Msun/h | Median Mass: {np.median(main_halo_masses):.2e} Msun/h")
    return True

# --- Main Execution ---

def main():
    if not HLIST_PATH.exists():
        print(f"Error: Cannot find hlist file: {HLIST_PATH}")
        return
    a = ytree.load(str(HLIST_PATH))
    
    if not run_diagnostics(a):
        return

    a.add_analysis_field("a50", "")

    ap = ytree.AnalysisPipeline()
    ap.add_operation(halo_filter)
    ap.add_operation(calc_a50)

    for tree in tqdm(a, desc="Processing Halos"):
        ap.process_target(tree)

    final_selection_mask = (a['pid'] == -1) & \
                            (a['mass'] >= MIN_MASS) & \
                            (a['mass'] < MAX_MASS) & \
                            (a['a50'] > 0)

    num_halos_selected = np.sum(final_selection_mask)

    if num_halos_selected == 0:
        print("\nError: No valid halos were found after processing.")
        print("This could be due to an incorrect mass range or issues with the merger trees.")
        return
        
    print(f"Successfully selected {num_halos_selected} valid halos for the final catalog.")
    

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    header = (
    "# Columns:\n"
    "# 1. mass [Msun]\n"
    "# 2. zf\n"
    "# 3. x [Mpc/h]\n"
    "# 4. y [Mpc/h]\n"
    "# 5. z [Mpc/h]\n"
    "# 6. vx [km/s]\n"
    "# 7. vy [km/s]\n"
    "# 8. vz [km/s]\n"
)
    
    # Create the output array by applying the final mask to each field array
    output_array = np.column_stack([
        a['mass'][final_selection_mask].value, 
        1 / a['a50'][final_selection_mask].value - 1,
        a['x'][final_selection_mask].value, 
        a['y'][final_selection_mask].value, 
        a['z'][final_selection_mask].value,
        a['vx'][final_selection_mask].value, 
        a['vy'][final_selection_mask].value, 
        a['vz'][final_selection_mask].value
    ])
    
    np.savetxt(
        OUTPUT_FILE, output_array, fmt='%.6e', delimiter=' ',
        header=header, comments=''
    )
    
    print("\nScript finished successfully!")


if __name__ == "__main__":
    main()