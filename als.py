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
from cpp_ext.als_module import ALS 
from tensors import *

def als_exact_comparison(lhs, rhs, J, method, iter):
    data = []

    als = ALS(lhs.ten, rhs.ten)

    if method != "exact":
        als.initialize_ds_als(J, method)

    rhs_norm = np.sqrt(rhs.ten.get_normsq())

    for i in range(iter):
        for j in range(lhs.N):
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

            fit = lhs.compute_exact_fit(rhs)
            #fit=-1
            print(f"Ratio: {ratio}, Residual: {residual_approx / rhs_norm}, AResidual: {residual_approx}")
            print(f"Fit {fit}")

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

def als_prod(lhs, rhs, J, method, max_iter, stop_tolerance, epoch_length=5, verbose=False):
    def verb_print(arg):
        if verbose:
            print(arg)
    #print("Starting ALS Algorithm...")
    data = []

    als = ALS(lhs.ten, rhs.ten)

    iterations = []
    fits = []
    update_times = []
    fit_computation_times = []

    if method != "exact":
        als.initialize_ds_als(J, method)

    iterations.append(0)
    fits.append(lhs.compute_exact_fit(rhs))
    verb_print(f"Before ALS:\tFit: {fits[-1]}")

    for i in range(max_iter):
        verb_print(f"Starting Iteration {i+1}.")
        for j in range(lhs.N):
            start = time.time()
            if method == "exact":
                als.execute_exact_als_update(j, True, True)
            else:
                als.execute_ds_als_update(j, True, True)
            elapsed = time.time() - start
            update_times.append(elapsed)

        if (i + 1) % epoch_length == 0:
            start = time.time()
            iterations.append(i + 1)
            fits.append(lhs.compute_exact_fit(rhs))
            verb_print(f"Iteration: {i+1}\tFit: {fits[-1]}")
            elapsed = time.time() - start
            fit_computation_times.append(elapsed)

            # Use the same stopping condition as L&K 
            if len(fits) > 4 and np.max(fits[-3:]) < np.max(fits[:-3]) + stop_tolerance:
                break

    data = {"iterations": iterations, 
            "fits": fits, 
            "update_times": update_times, 
            "fit_computation_times": fit_computation_times}
    return data