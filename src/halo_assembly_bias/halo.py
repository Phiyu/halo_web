import numpy as np
from typing import Self
import numpy as np
from pathlib import Path
from pyhipp.core import abc, DataTable
from pyhipp.io import h5

class HaloSample(abc.HasLog, abc.HasDictRepr):
    
    repr_attr_keys = ('n_objs',)
    
    '''
    A simple halo sample.

    @data: attributes of the halos, e.g. {'mass': ..., 'radius': ...}.
    @verbose: whether to print log messages.
    @copy: whether to copy the data.
    '''
    def __init__(self, data: dict[str, np.ndarray], verbose=True, copy=True):

        keys = tuple(data.keys())
        n_objs = len(data[keys[0]])
        for k, v in data.items():
            assert len(v) == n_objs, f"Size of {k} != {n_objs}"
        
        super().__init__(verbose=verbose)
        self.data = DataTable(data=data, copy=copy)
        self.log(f'HaloSample: {n_objs=}, {keys=}')
        
    def __getitem__(self, key: str | tuple[str, ...]):
        '''
        Get the attribute of the halo sample.
        
        @key: key of the attribute.
        '''
        return self.data[key]
    
    def add_proper(self, data: dict[str, np.ndarray], overwrite=False) -> None:
        '''
        Add new attributes to the halo sample.

        @data: dict, keys are attribute names, values are attribute values (length equals n_objs).
        @overwrite: whether to overwrite existing attributes.
        '''
        for key, value in data.items():
            if not overwrite:
                assert key not in self.data, f"Key {key} already exists."
            assert len(value) == self.n_objs, f"Size of {key} != {self.n_objs}"
            self.data[key] = value


    @property
    def n_objs(self):
        key = next(iter(self.keys()))    
        val = self[key]
        return len(val)
    
    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()
    
    def items(self):
        return self.data.items()
    
    def __contains__(self, key: str):
        return key in self.data
    
    def subset(self, args: np.ndarray | slice, copy=True) -> Self:
        '''
        Row-wise subsetting by `args`, applied to each value. Return a new 
        (copied) sample.
        
        @copy: if False, values are not guaranteed to be copied.
        '''
        return HaloSample({
            k: v[args] for k, v in self.data.items()
        }, copy=copy, verbose=self.verbose)
        
    def subset_by_val(self, key: str, lo=None, hi=None, eq=None) -> Self:
        sel = np.ones(self.n_objs, dtype=bool)
        val = self[key]
        if lo is not None:
            sel &= val >= lo
        if hi is not None:
            sel &= val < hi
        if eq is not None:
            sel &= val == eq
        return self.subset(sel, copy=False)
    
    def subset_by_p(self, key: str, p_lo=None, p_hi=None) -> Self:
        val = self[key]
        lo, hi = None, None
        if p_lo is not None:
            lo = np.quantile(val, p_lo)
        if p_hi is not None:
            hi = np.quantile(val, p_hi)
        return self.subset_by_val(key, lo=lo, hi=hi)    
    
    @classmethod
    def from_file(cls, path: Path | str, source = 'Quijote', **init_kw):
        '''
        Create the sample from a file. The file should be in HDF5 format,
        with datasets containing the halo attributes.
        
        @path: path to the file.
        @init_kw: additional keyword arguments passed to __init__().
        '''
        if source == 'TNG':
            dict_data = h5.File.load_from(path)
        if source == 'Quijote':
            keys = ['mass', 'zf', 'x', 'y', 'z', 'vx', 'vy', 'vz']
            data = np.loadtxt(path)
            dict_data = {k: data[:, i] for i, k in enumerate(keys)}
        return cls(data = dict_data, copy=False, **init_kw)
    

