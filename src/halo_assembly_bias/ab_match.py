from .samples import halo
import numpy as np
from pathlib import Path

def ab_matching(halo: halo.HaloSample, numbers: np.ndarray) -> list[halo.HaloSample]:
    '''
    Match halos to galaxies by their indices. Return a dictionary of matched
    halo attributes.

    @halo: halo sample.
    @numbers: number of galaxies.
    '''
    assert np.all(numbers >= 0) and np.all(numbers.astype(int) == numbers), \
        "All numbers must be non-negative integers."
    
    abundance = ab_cal(halo.n_objs, numbers)
    
    zf = halo['zf']
    idx_sorted = np.argsort(zf) 
    
    split_indices = np.cumsum(abundance)[:-1]
    idx_groups = np.split(idx_sorted, split_indices)
    groups = [halo.subset(idx) for idx in idx_groups]
    return groups


def ab_cal(n: int, frac: np.ndarray[np.float64]) -> np.ndarray[np.int_]:
    '''
    Calculate the number of halos in each bin.

    @n: total number of halos.
    @frac: fraction in each bin.
    '''
    abundance = frac / (np.sum(frac).astype(int)) * n
    abundance = np.around(abundance).astype(int)
    diff = n - np.sum(abundance)
    if diff != 0:
        idx = np.argsort(-abundance if diff > 0 else abundance)
        for i in range(abs(diff)):
            abundance[idx[i % len(abundance)]] += 1 if diff > 0 else -1

    assert np.sum(abundance) == n, \
        "The sum of abundance must be equal to the number of halos."
    
    return abundance