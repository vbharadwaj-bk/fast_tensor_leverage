import numpy as np
import numpy.linalg as la
import scipy
import matplotlib.pyplot as plt
import time
import json
import pickle
import h5py

from common import *

import cppimport.import_hook
from cpp_ext.als_module import Tensor, LowRankTensor, SparseTensor, ALS
from cpp_ext.als_module import DenseTensor_float, DenseTensor_double 

class PyLowRank:
    def __init__(self, dims, R, allow_rhs_mttkrp=False, init_method="gaussian", seed=None):
        if seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(seed)
        if init_method=="gaussian":
            self.dims = []
            for i in dims:
                self.dims.append(i)

            self.N = len(dims)
            self.R = R
            self.U = [rng.normal(size=(i, R)) for i in self.dims]

            if allow_rhs_mttkrp:
                self.ten = LowRankTensor(R, 10000, self.U)
            else:
                self.ten = LowRankTensor(R, self.U)
        else:
            assert(False)
    
    def set_sigma(self, new_sigma):
        self.ten.set_sigma(new_sigma)

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

    def compute_integral(self, dx):
        '''
        Computes the integral of the CP decomposition. 
        '''  
        sigma = np.zeros(self.R, dtype=np.double) 
        self.ten.get_sigma(sigma, -1)

        buffers = [sigma]
        for i in range(self.N):
            buffers.append(scipy.integrate.simpson(self.U[i], axis=0, dx=dx[i]))

        return np.sum(chain_had_prod(buffers))

#from numba import cfunc, types, carray, void, uint32, float64, uint64, jit 

#@jit(void(float64[:, :],uint64[:, :],uint32,uint32,uint32,uint32), nopython=True)
#def test_function(out_buffer, samples, j, row_pos, M, Ij):
#    delta_X = 0.01
#
#    for i in range(row_pos, row_pos + M):
#        samples[i, j] = 0
#        temp_sum = np.sum(samples[i, :])
#
#        for k in range(Ij):
            #out_buffer[i-row_pos, k] = np.sin((temp_sum + k) * delta_X)
            #out_buffer[i-row_pos, k] = (temp_sum + k) * delta_X
#            out_buffer[i-row_pos, k] = samples[i, 0] 


#def test_wrapper(out_buffer_, samples_, j, row_pos, M, Ij, tensor_dim):
#    out_ptr = ctypes.c_void_p(out_buffer_)
#    samples_ptr = ctypes.c_void_p(samples_)
#    out_buffer = carray(out_ptr, (M, Ij), dtype=np.double)
#    samples = carray(samples_ptr, (M, tensor_dim), dtype=np.uint64)
#    test_function(out_buffer, samples, j, row_pos, M, Ij)

class FunctionTensor:
    def __init__(self, dims, J, dx, func=None, func_batch=None):
        self.N = len(dims)
        self.J = J
        #self.bounds = bounds
        #self.subdivisions = subdivisions
        self.dims = np.array(dims, dtype=np.uint64)
        self.ten = PyFunctionTensor(self.dims, J, 10000, dx)
