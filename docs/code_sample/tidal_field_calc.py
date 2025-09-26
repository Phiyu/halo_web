import numpy as np
import h5py
import os
from scipy.fft import rfftn, irfftn
from pathlib import Path

# --- 1. SCRIPT PARAMETERS ---
# These are the parameters you might need to change for different simulations.

# Input parameters for TNG100-1-Dark
snapshot_dir = Path('/Volumes/T7/TNG100-1-Dark/output/snapdir_099/')
sim_info_file = Path('tng100-1-dark.json') # Assumes you have this file
snapshot_num = 99

# Output file
output_file = Path('/Volumes/T7/tidal_field.hdf5')

# T-Web Algorithm Parameters
grid_size = 512  # The number of grid cells along one dimension
smoothing_scale = 2.0  # Gaussian smoothing scale in Mpc/h, as used in theory.ipynb
lambda_th = 0.2  # Eigenvalue threshold for web classification, as used in theory.ipynb

# --- 2. UTILITY FUNCTIONS ---

def load_tng_snapshot_data(snap_dir: Path, snap_num: int) -> tuple[np.ndarray, float, float]:
    """
    Loads particle positions, box size, and hubble parameter from a TNG snapshot.
    Handles multi-file snapshots.
    """
    print(f"Loading snapshot {snap_num} from {snap_dir}...")
    
    # Get the header from the first file to read BoxSize and Cosmology
    header_file = h5py.File(snap_dir / f'snap_{snap_num:03d}.0.hdf5', 'r')
    header = header_file['Header'].attrs
    box_size_kpc_h = header['BoxSize']  # Box size in ckpc/h
    box_size_mpc_h = box_size_kpc_h / 1000.0
    hubble_param = header['HubbleParam']
    header_file.close()

    all_positions = []
    num_files = header['NumFilesPerSnapshot']

    for i in range(num_files):
        file_path = snap_dir / f'snap_{snap_num:03d}.{i}.hdf5'
        with h5py.File(file_path, 'r') as f:
            # Check if there are any DM particles in this file
            if 'PartType1' in f and 'Coordinates' in f['PartType1']:
                pos = f['PartType1']['Coordinates'][:] / 1000.0  # to Mpc/h
                all_positions.append(pos)
                print(f"  - Loaded {pos.shape[0]} particles from file {i}/{num_files-1}")

    positions = np.concatenate(all_positions, axis=0)
    print(f"Total DM particles loaded: {positions.shape[0]}")
    return positions, box_size_mpc_h, hubble_param

def cic_deposit(positions: np.ndarray, box_size: float, n_grid: int) -> np.ndarray:
    """
    Deposits particles onto a grid using the Cloud-in-Cell (CIC) method.
    """
    print("Depositing particles onto grid using CIC...")
    density_field = np.zeros((n_grid, n_grid, n_grid), dtype=np.float32)
    
    # Normalize positions to grid coordinates
    pos_grid = positions / box_size * n_grid
    
    # Find the integer and fractional parts of the coordinates
    i = np.floor(pos_grid).astype(int)
    f = pos_grid - i
    
    # Calculate weights for the 8 neighboring cells
    w000 = (1 - f[:, 0]) * (1 - f[:, 1]) * (1 - f[:, 2])
    w001 = (1 - f[:, 0]) * (1 - f[:, 1]) * (    f[:, 2])
    w010 = (1 - f[:, 0]) * (    f[:, 1]) * (1 - f[:, 2])
    w011 = (1 - f[:, 0]) * (    f[:, 1]) * (    f[:, 2])
    w100 = (    f[:, 0]) * (1 - f[:, 1]) * (1 - f[:, 2])
    w101 = (    f[:, 0]) * (1 - f[:, 1]) * (    f[:, 2])
    w110 = (    f[:, 0]) * (    f[:, 1]) * (1 - f[:, 2])
    w111 = (    f[:, 0]) * (    f[:, 1]) * (    f[:, 2])

    # Add weights to the corresponding cells, handling periodic boundaries
    def add_to_grid(ii, jj, kk, weights):
        # Use np.add.at for atomic adds, preventing race conditions
        np.add.at(density_field, (ii % n_grid, jj % n_grid, kk % n_grid), weights)

    add_to_grid(i[:, 0],   i[:, 1],   i[:, 2],   w000)
    add_to_grid(i[:, 0],   i[:, 1],   i[:, 2]+1, w001)
    add_to_grid(i[:, 0],   i[:, 1]+1, i[:, 2],   w010)
    add_to_grid(i[:, 0],   i[:, 1]+1, i[:, 2]+1, w011)
    add_to_grid(i[:, 0]+1, i[:, 1],   i[:, 2],   w100)
    add_to_grid(i[:, 0]+1, i[:, 1],   i[:, 2]+1, w101)
    add_to_grid(i[:, 0]+1, i[:, 1]+1, i[:, 2],   w110)
    add_to_grid(i[:, 0]+1, i[:, 1]+1, i[:, 2]+1, w111)
    
    return density_field

# --- 3. MAIN SCRIPT LOGIC ---

if __name__ == '__main__':
    # Step 1: Load particle data
    positions, box_size, h = load_tng_snapshot_data(snapshot_dir, snapshot_num)
    
    # Step 2: Create density field
    rho_field = cic_deposit(positions, box_size, grid_size)
    
    # Calculate density contrast field delta = (rho / <rho>) - 1
    mean_density = positions.shape[0] / grid_size**3
    delta_field = rho_field / mean_density - 1.0
    
    print("Calculating tidal field via FFT...")
    # Step 3: Go to Fourier space
    delta_k = rfftn(delta_field)
    
    # Generate k vectors
    k_values = 2 * np.pi * np.fft.fftfreq(grid_size, d=box_size / grid_size)
    kx = k_values[:, np.newaxis, np.newaxis]
    ky = k_values[np.newaxis, :, np.newaxis]
    kz_rfft = k_values[np.newaxis, np.newaxis, :grid_size//2 + 1] # For rfftn
    
    k_sq = kx**2 + ky**2 + kz_rfft**2
    k_sq[0, 0, 0] = 1.0  # Avoid division by zero
    
    # Apply Gaussian smoothing in k-space
    smoothing_factor_k = np.exp(-0.5 * k_sq * smoothing_scale**2)
    delta_sm_k = delta_k * smoothing_factor_k
    
    # Calculate potential and tidal tensor components in k-space
    # phi_k = -delta_k / k_sq
    # We directly compute the traceless tidal tensor T_ij = (k_i k_j / k^2 - delta_ij / 3) * delta_k
    T11_k = (kx * kx / k_sq - 1/3) * delta_sm_k
    T12_k = (kx * ky / k_sq) * delta_sm_k
    T13_k = (kx * kz_rfft / k_sq) * delta_sm_k
    T22_k = (ky * ky / k_sq - 1/3) * delta_sm_k
    T23_k = (ky * kz_rfft / k_sq) * delta_sm_k
    T33_k = (kz_rfft * kz_rfft / k_sq - 1/3) * delta_sm_k

    # Step 4: Go back to real space
    print("Inverse transforming back to real space...")
    delta_sm_x = irfftn(delta_sm_k, s=(grid_size, grid_size, grid_size))
    T11 = irfftn(T11_k, s=(grid_size, grid_size, grid_size))
    T12 = irfftn(T12_k, s=(grid_size, grid_size, grid_size))
    T13 = irfftn(T13_k, s=(grid_size, grid_size, grid_size))
    T22 = irfftn(T22_k, s=(grid_size, grid_size, grid_size))
    T23 = irfftn(T23_k, s=(grid_size, grid_size, grid_size))
    T33 = irfftn(T33_k, s=(grid_size, grid_size, grid_size))
    
    # Step 5: Calculate eigenvalues at each grid point
    print("Calculating eigenvalues...")
    tidal_tensor = np.array([
        [T11, T12, T13],
        [T12, T22, T23],
        [T13, T23, T33]
    ])
    # Reshape to (N, N, N, 3, 3) for eigenvalue calculation
    tidal_tensor = np.moveaxis(tidal_tensor, [0, 1], [-2, -1])
    
    # Use eigh for symmetric matrices; it's faster and more stable
    eigenvalues = np.linalg.eigh(tidal_tensor)[0]
    # np.linalg.eigh returns eigenvalues in ascending order, so we reverse them
    eigenvalues = eigenvalues[:, :, :, ::-1] # Now lambda_1 > lambda_2 > lambda_3

    # Step 6: Classify cosmic web
    print("Classifying cosmic web...")
    n_above_thresh = (eigenvalues > lambda_th).sum(axis=3)
    # 3 -> Knot (3), 2 -> Filament (2), 1 -> Sheet (1), 0 -> Void (0)
    web_type = n_above_thresh 

    # Step 7: Save results to HDF5 file
    print(f"Saving results to {output_file}...")
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('lams', data=eigenvalues.astype(np.float32))
        f.create_dataset('delta_sm_x', data=delta_sm_x.astype(np.float32))
        f.create_dataset('web_type', data=web_type.astype(np.int8))
        f.attrs['l_box'] = box_size
        f.attrs['n_grids'] = grid_size
        f.attrs['smoothing_scale_mpc_h'] = smoothing_scale
        f.attrs['lambda_th'] = lambda_th
        
    print("Done!")