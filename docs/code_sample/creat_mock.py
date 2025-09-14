# create_mock_tng_tree.py (Corrected Version)

import h5py
import numpy as np
from pathlib import Path

FILENAME = "tree_extended.hdf5"
HUBBLE_CONST = 0.6774  # h value for TNG

SNAPSHOTS = [99, 91, 84, 78, 72, 67, 59, 50, 40, 33]

# --- Global lists to store data for all halos from all trees ---
tree_data = {
    "SubhaloID": [], "SnapNum": [], "DescendantID": [],
    "FirstProgenitorID": [], "NextProgenitorID": [], "SubfindID": [],
    "Group_M_Crit200": [], "GroupPos": [], "GroupVel": [], "GroupFirstSub": []
}
subhalo_id_counter = 0

def generate_branch(snap_idx, mass, pos, vel, is_main_branch=True):
    """
    Recursively generates a branch of a merger tree, returning the ID of its root node.
    """
    global subhalo_id_counter
    
    # --- 1. Create the current node and add ALL its fields with placeholders ---
    current_id = subhalo_id_counter
    subhalo_id_counter += 1
    
    # Get the index for the current node *before* appending
    current_idx = len(tree_data["SubhaloID"])
    
    # Append physical properties
    tree_data["SubhaloID"].append(current_id)
    tree_data["SnapNum"].append(SNAPSHOTS[snap_idx])
    tree_data["Group_M_Crit200"].append(mass)
    tree_data["GroupPos"].append(pos)
    tree_data["GroupVel"].append(vel)
    
    # Append Subfind/Group info
    subfind_id_at_snap = current_idx # Simple unique ID for this mock
    tree_data["SubfindID"].append(subfind_id_at_snap)
    is_central_at_z0 = (SNAPSHOTS[snap_idx] == 99 and is_main_branch)
    tree_data["GroupFirstSub"].append(subfind_id_at_snap if is_central_at_z0 else -1)

    # *** CORRECTION START ***
    # Append PLACEHOLDERS for pointer fields immediately to keep lists synchronized.
    tree_data["DescendantID"].append(-1)
    tree_data["FirstProgenitorID"].append(-1)
    tree_data["NextProgenitorID"].append(-1)
    # *** CORRECTION END ***

    # Base case: If we are at the earliest snapshot, this is a leaf node
    if snap_idx >= len(SNAPSHOTS) - 1:
        return current_id

    # --- 2. Recursively generate progenitors for the next snapshot ---
    main_prog_mass = mass * np.random.uniform(0.6, 0.9)
    main_prog_pos = pos + np.random.normal(0, 50, 3)
    main_prog_vel = vel + np.random.normal(0, 20, 3)
    
    progenitor_ids = []
    # Occasionally, create a merger event for the main branch
    if is_main_branch and np.random.rand() < 0.3 and snap_idx < len(SNAPSHOTS) - 2:
        minor_prog_mass = mass * np.random.uniform(0.1, 0.3)
        minor_prog_pos = pos + np.random.normal(0, 200, 3)
        minor_prog_vel = vel + np.random.normal(0, 50, 3)
        minor_prog_id = generate_branch(snap_idx + 1, minor_prog_mass, minor_prog_pos, minor_prog_vel, is_main_branch=False)
        progenitor_ids.append(minor_prog_id)

    # Always generate the main progenitor
    main_prog_id = generate_branch(snap_idx + 1, main_prog_mass, main_prog_pos, main_prog_vel, is_main_branch=is_main_branch)
    progenitor_ids.insert(0, main_prog_id)
    
    # --- 3. Link progenitors to the current node by UPDATING the placeholders ---
    
    # Update the current node's FirstProgenitorID pointer
    tree_data["FirstProgenitorID"][current_idx] = progenitor_ids[0]
    
    # Link the chain of progenitors using NextProgenitorID
    for i in range(len(progenitor_ids) - 1):
        prog_idx = tree_data["SubhaloID"].index(progenitor_ids[i])
        tree_data["NextProgenitorID"][prog_idx] = progenitor_ids[i+1]
        
    # Link all progenitors back to the current node as their descendant
    for prog_id in progenitor_ids:
        prog_idx = tree_data["SubhaloID"].index(prog_id)
        tree_data["DescendantID"][prog_idx] = current_id

    return current_id

def create_mock_sublink_file(num_trees=5, filename=FILENAME):
    """ Main function to generate and save the mock SubLink HDF5 file. """
    print(f"Generating {num_trees} mock merger trees...")
    
    for i in range(num_trees):
        z0_mass = 10**np.random.uniform(13.0, 14.5) / 1e10 # In 10^10 Msun/h
        z0_pos = np.random.uniform(0, 75000, 3) # TNG100 box size is ~75 Mpc/h
        z0_vel = np.random.normal(0, 200, 3)
        print(f"  - Tree {i+1}: z=0 mass = {z0_mass * 1e10 / HUBBLE_CONST:.2e} Msun")
        generate_branch(snap_idx=0, mass=z0_mass, pos=z0_pos, vel=z0_vel)

    print(f"\nTotal halos generated: {len(tree_data['SubhaloID'])}")

    output_path = Path(filename)
    if output_path.exists(): output_path.unlink()

    print(f"Saving to HDF5 file: {filename}")
    with h5py.File(filename, 'w') as hf:
        # Data types are chosen to match the TNG specification document
        hf.create_dataset('SubhaloID', data=np.array(tree_data['SubhaloID'], dtype=np.int64))
        hf.create_dataset('SnapNum', data=np.array(tree_data['SnapNum'], dtype=np.int32))
        hf.create_dataset('DescendantID', data=np.array(tree_data['DescendantID'], dtype=np.int64))
        hf.create_dataset('FirstProgenitorID', data=np.array(tree_data['FirstProgenitorID'], dtype=np.int64))
        hf.create_dataset('NextProgenitorID', data=np.array(tree_data['NextProgenitorID'], dtype=np.int64))
        hf.create_dataset('SubfindID', data=np.array(tree_data['SubfindID'], dtype=np.int32))
        hf.create_dataset('Group_M_Crit200', data=np.array(tree_data['Group_M_Crit200'], dtype=np.float32))
        hf.create_dataset('GroupPos', data=np.array(tree_data['GroupPos'], dtype=np.float32))
        hf.create_dataset('GroupVel', data=np.array(tree_data['GroupVel'], dtype=np.float32))
        hf.create_dataset('GroupFirstSub', data=np.array(tree_data['GroupFirstSub'], dtype=np.int32))
        
        header = hf.create_group('Header')
        header.attrs.create('Ntrees_Total', num_trees)
        header.attrs.create('Nhalos_Total', len(tree_data['SubhaloID']))

    print("Mock file created successfully.")

if __name__ == "__main__":
    create_mock_sublink_file()