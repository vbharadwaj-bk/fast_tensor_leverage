import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import time
import json
import pickle
import h5py
import ctypes

from common import *

import cppimport.import_hook
from cpp_ext.efficient_krp_sampler import CP_ALS 
from cpp_ext.als_module import Tensor, LowRankTensor, SparseTensor, PyFunctionTensor, ALS
from cpp_ext.efficient_krp_sampler import CP_ALS
from tensors import *

def als(lhs, rhs, J, method, iter):
    data = []

    als = ALS(lhs.ten, rhs.ten)

    if method != "exact":
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

                if method != "exact":
                    als.execute_ds_als_update(j, True, True)
                    residual_approx = lhs.compute_diff_resid(rhs)
                else: 
                    residual_approx = residual

                if residual > 0:
                    ratio = residual_approx / residual
                else:
                    ratio = 1.0

                fit = lhs.compute_estimated_fit(rhs)
                print(f"Ratio: {ratio}, Residual: {residual_approx / rhs_norm}, Fit: {fit}")

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

def sparse_tensor_test():
    J = 65536

    trial_count = 5
    iterations = 40
    result = {}

    samplers = ["efficient"]
    #R_values = [4, 8, 16, 32, 64, 128]
    R_values = [25]

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

    with open('outputs/lk_uber_comparison.json', 'w') as outfile:
        json.dump(result, outfile, indent=4)

def low_rank_test():
    J = 30000 

    trial_count = 1
    iterations = 20
    result = {}

    samplers = ["efficient"]
    R_values = [32]

    I = 2 ** 9
    N = 4

    for R in R_values: 
        result[R] = {}
        for sampler in samplers:
            result[R][sampler] = []
            for trial in range(trial_count):
                rhs = PyLowRank([I] * N, R, allow_rhs_mttkrp=True, J=J, seed=479873)
                rhs.ten.renormalize_columns(-1)

                # Specify a seed here to make everything deterministic
                lhs = PyLowRank(rhs.dims, R)
                lhs.ten.renormalize_columns(-1)
                result[R][sampler].append(als(lhs, rhs, J, sampler, iterations))

    #with open('outputs/low_rank_comparison.json', 'w') as outfile:
    #    json.dump(result, outfile, indent=4)

def numerical_integration_test():
    I = 100
    J = 5000
    N = 2
    R = 25
    dims = [I] * N
    iterations = 10

    rhs = FunctionTensor(N, J)
    print("Initialized Function Tensor!")

    lhs = PyLowRank(dims, R, seed=923845)
    lhs.ten.renormalize_columns(-1)

    dx = [0.01] * N

    method = "efficient"
    als = ALS(lhs.ten, rhs.ten)
    als.initialize_ds_als(J, method)

    integral = lhs.compute_integral(dx)
    print(f"Integral: {integral}")

    for i in range(iterations):
        for j in range(lhs.N):
            als.execute_ds_als_update(j, True, True) 

            g = chain_had_prod([lhs.U[i].T @ lhs.U[i] for i in range(lhs.N) if i != j])
            detected_nan = np.any(np.isnan(g))

            if detected_nan:
                print("Found a NaN value!")

            integral = lhs.compute_integral(dx)
            print(f"Integral: {integral}")

if __name__=='__main__':
    numerical_integration_test()
    #low_rank_test()
