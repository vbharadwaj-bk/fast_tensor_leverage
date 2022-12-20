import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import time
import json
import pickle
import h5py

from common import *

import cppimport.import_hook
from cpp_ext.efficient_krp_sampler import CP_ALS 
from cpp_ext.als_module import Tensor, LowRankTensor, SparseTensor, ALS
from cpp_ext.efficient_krp_sampler import CP_ALS

class PyLowRank:
    def __init__(self, dims, R, allow_rhs_mttkrp=False, J=None, init_method="gaussian", seed=None):
        if seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(seed)
        if init_method=="gaussian":
            self.dims = []
            for i in dims:
                if i < R:
                    self.dims.append(i)
                else:
                    self.dims.append(divide_and_roundup(i, R) * R)

            self.N = len(dims)
            self.R = R
            self.U = [rng.normal(size=(i, R)) for i in self.dims]
            if allow_rhs_mttkrp:
                self.ten = LowRankTensor(R, J, 10000, self.U)
            else:
                self.ten = LowRankTensor(R, self.U)
        else:
            assert(False)

    def compute_diff_resid(self, rhs):
        '''
        Computes the residual of the difference between two low-rank tensors.
        '''
        sigma_lhs = np.zeros(self.R, dtype=np.double) 
        self.ten.get_sigma(sigma_lhs, -1)
        normsq = rhs.ten.compute_residual_normsq(sigma_lhs, self.U)
        return np.sqrt(normsq)

    def compute_estimated_fit(self, rhs_ten):
        diff_norm = self.compute_diff_resid(rhs_ten)
        rhs_norm = np.sqrt(rhs_ten.ten.get_normsq())
        return 1.0 - diff_norm / rhs_norm

class PySparseTensor:
    def __init__(self, filename, preprocessing=None):
        print("Loading sparse tensor...")
        f = h5py.File(filename, 'r')

        self.max_idxs = f['MAX_MODE_SET'][:]
        self.min_idxs = f['MIN_MODE_SET'][:]
        self.N = len(self.max_idxs)
        self.dims = self.max_idxs[:]

        # The tensor must have at least one mode
        self.nnz = len(f['MODE_0']) 

        self.tensor_idxs = np.zeros((self.nnz, self.N), dtype=np.uint32) 

        for i in range(self.N): 
            self.tensor_idxs[:, i] = f[f'MODE_{i}'][:] - self.min_idxs[i]

        self.values = f['VALUES'][:]

        if preprocessing is not None:
            if preprocessing == "log_count":
                self.values = np.log(self.values + 1.0)
            else:
                print(f"Unknown preprocessing option '{preprocessing}' specified!")
                exit(1)

        self.ten = SparseTensor(self.tensor_idxs, self.values) 
        print("Finished loading sparse tensor...")

from numba import cfunc, types, carray

#@jit(void(uint64,uint64,uint32,uint32,uint32,uint32,uint32), nopython=True)
#def test_function(out_buffer_, samples_, j, row_pos, M, Ij, tensor_dim):
#    delta_X = 0.01

#    out_buffer = carray(out_buffer_, (M, Ij), dtype=np.double)
#    samples = carray(samples_, (M, tensor_dim), dtype=np.uint64)

#    for i in range(row_pos, row_pos + M):
#        samples[i, j] = 0
#        temp_sum = np.cumsum(samples[i, :])
#        for k in range(Ij):
#            out_buffer[i, k] = np.sin((temp_sum + k) * delta_X)

class FunctionTensor:
    def __init__(self, bounds, subdivisions, func=None, func_batch=None):
        self.bounds = bounds
        self.subdivisions = subdivisions
        self.ten = None 
