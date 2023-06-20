import numpy as np
import numpy.linalg as la
import h5py

from common import *

import cppimport.import_hook
from cpp_ext.tt_module import Tensor, SparseTensor 

class PySparseTensor:
    def __init__(self, filename, lookup, preprocessing=None):
        print("Loading sparse tensor...")
        f = h5py.File(filename, 'r')

        self.max_idxs = f['MAX_MODE_SET'][:]
        self.min_idxs = f['MIN_MODE_SET'][:]
        self.N = len(self.max_idxs)
        self.shape = self.max_idxs[:]

        # The tensor must have at least one mode
        self.nnz = len(f['MODE_0']) 

        self.tensor_idxs = np.zeros((self.nnz, self.N), dtype=np.uint32) 

        for i in range(self.N): 
            self.tensor_idxs[:, i] = f[f'MODE_{i}'][:] - self.min_idxs[i]

        self.values = f['VALUES'][:]
        self.data_norm = la.norm(self.values)
        print("Loaded tensor values from disk...")

        if preprocessing is not None:
            if preprocessing == "log_count":
                self.values = np.log(self.values + 1.0)
            else:
                print(f"Unknown preprocessing option '{preprocessing}' specified!")

        # FOR DEBUGGING ONLY! ---------------------------
        self.shape = [3, 3, 3]
        self.N = 3
        self.nnz = np.prod(self.shape) 
        self.tensor_idxs = np.zeros((self.nnz, self.N), dtype=np.uint32)
        self.values = np.zeros(self.nnz, dtype=np.double)
        for i in range(self.nnz):
            self.tensor_idxs[i, :] = np.unravel_index(i, self.shape)
            self.values[i] = np.sum(np.unravel_index(i, self.shape))

        self.data_norm = la.norm(self.values)
        # FOR DEBUGGING ONLY! ---------------------------

        self.ten = SparseTensor(self.tensor_idxs, self.values, lookup) 
        print("Finished loading sparse tensor...")