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

def als_exact_comparison(lhs, rhs, J, method, iter):
    data = []

    als = ALS(lhs.ten, rhs.ten)

    if method != "exact":
        als.initialize_ds_als(J, method)

    residual = lhs.compute_diff_resid(rhs)
    rhs_norm = np.sqrt(rhs.ten.get_normsq())

    for i in range(iter):
        for j in range(lhs.N):
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