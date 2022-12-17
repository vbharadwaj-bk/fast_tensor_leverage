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
    def __init__(self, dims, R, allow_rhs_mttkrp=False, J=None, init_method="gaussian", seed=42):
        if init_method=="gaussian":
            rng = np.random.default_rng(seed)
            self.dims = []
            for i in dims:
                if i < R:
                    self.dims.append(i)
                else:
                    self.dims.append(divide_and_roundup(i, R) * R)
            #self.dims = [divide_and_roundup(i, R) * R for i in dims]
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


def als(lhs, rhs, J, method, iter):
    data = []

    als = ALS(lhs.ten, rhs.ten)
    als.initialize_ds_als(J, method)

    residual = lhs.compute_diff_resid(rhs)
    rhs_norm = np.sqrt(rhs.ten.get_normsq())

    #try:
    if True:
        for i in range(iter):
            for j in range(lhs.N):
                # This is used just to check for NaN values.
                g = chain_had_prod([lhs.U[i].T @ lhs.U[i] for i in range(lhs.N) if i != j])
                detected_nan = np.any(np.isnan(g))

                if detected_nan:
                    print("Found a NaN value!")

                sigma_lhs = np.zeros(lhs.R, dtype=np.double) 
                lhs.ten.get_sigma(sigma_lhs, -1)

                als.execute_exact_als_update(j, True, True)
                residual = lhs.compute_diff_resid(rhs)
                #residual = 0.0

                als.execute_ds_als_update(j, True, True)
                residual_approx = lhs.compute_diff_resid(rhs)

                if residual > 0:
                    ratio = residual_approx / residual
                else:
                    ratio = 1.0

                #print(f"Condition #: {la.cond(g)}")
                fit = lhs.compute_estimated_fit(rhs)
                print(f"Ratio: {ratio}, Residual: {residual_approx / rhs_norm}, Fit: {fit}")

                if j == 1:
                    mat = lhs.U[j]
                    print(mat)

                    exit(1)

                data_entry = {}
                data_entry["fit"] = fit 
                data_entry["exact_solve_residual"] = residual
                data_entry["approx_solve_residual"] = residual_approx
                data_entry["exact_solve_residual_normalized"] = residual / rhs_norm
                data_entry["approx_solve_residual_normalized"] = residual_approx / rhs_norm
                data_entry["rhs_norm"] = rhs_norm
                data_entry["ratio"] = ratio
                data_entry["j"] = j 
                
                data.append(data_entry)
        return data 
    #except:
    #    print("Caught SVD unconverged exception, terminating and returning trace...")
    #    return data

def sparse_als(lhs, rhs, J, method, iter):
    data = []

    als = ALS(lhs.ten, rhs.ten)
    als.initialize_ds_als(J, method)

    estimated_fit = lhs.compute_estimated_fit(rhs)
    print(f"Estimated Fit: {estimated_fit}")

    for i in range(iter):
        for j in range(lhs.N):
            als.execute_ds_als_update(j, True, True) 
            estimated_fit = lhs.compute_estimated_fit(rhs)
            print(f"Estimated Fit: {estimated_fit}")

    return data

if __name__=='__main__':
    J = 65536

    trial_count = 3
    iterations = 20
    result = {}

    samplers = ["larsen_kolda"]
    #R_values = [4, 8, 16, 32, 64, 128]
    R_values = [32]

    for R in R_values: 
        result[R] = {}
        for sampler in samplers:
            result[R][sampler] = []
            for trial in range(trial_count):
                #rhs = PyLowRank([2 ** 4] * N, R, allow_rhs_mttkrp=True, J=J, seed=479873)
                #rhs.ten.renormalize_columns(-1)
                rhs = PySparseTensor("/home/vbharadw/tensors/uber.tns_converted.hdf5")
                lhs = PyLowRank(rhs.dims, R, seed=923845)
                lhs.ten.renormalize_columns(-1)
                result[R][sampler].append(als(lhs, rhs, J, sampler, iterations))
                exit(1)
                #result[sampler].append(sparse_als(lhs, rhs, J, sampler, iterations))

    #with open('outputs/synthetic_sparse_comparison.json', 'w') as outfile:
    #    json.dump(result, outfile, indent=4)