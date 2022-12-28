import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import time
import json
import pickle
import h5py
import ctypes
from PIL import Image

from common import *
from tensors import *
from als import *

import cppimport.import_hook
from cpp_ext.efficient_krp_sampler import CP_ALS 
from cpp_ext.als_module import Tensor, LowRankTensor, SparseTensor, ALS 
from cpp_ext.efficient_krp_sampler import CP_ALS

def sparse_tensor_test():
    J = 2 ** 16 

    trial_count = 5
    iterations = 40 
    result = {}

    samplers = ["efficient"]
    #R_values = [4, 8, 16, 32, 64, 128]
    R_values = [25]

    #rhs = PySparseTensor("/pscratch/sd/v/vbharadw/tensors/uber.tns_converted.hdf5", lookup="sort")
    rhs = PySparseTensor("/pscratch/sd/v/vbharadw/tensors/amazon-reviews.tns_converted.hdf5", lookup="sort")

    for R in R_values: 
        result[R] = {}
        for sampler in samplers:
            result[R][sampler] = []
            for trial in range(trial_count):
                lhs = PyLowRank(rhs.dims, R)
                lhs.ten.renormalize_columns(-1)
                result[R][sampler].append(als_prod(lhs, rhs, J, sampler, iterations))

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
                rhs = PyLowRank([I] * N, R, allow_rhs_mttkrp=True, seed=479873)
                rhs.ten.renormalize_columns(-1)

                # Specify a seed here to make everything deterministic
                lhs = PyLowRank(rhs.dims, R)
                lhs.ten.renormalize_columns(-1)
                result[R][sampler].append(als_exact_comparison(lhs, rhs, J, sampler, iterations))

    #with open('outputs/low_rank_comparison.json', 'w') as outfile:
    #    json.dump(result, outfile, indent=4)

def numerical_integration_test():
    I = 10000
    J = 10000
    N = 10
    R = 25 
    dims = [I] * N
    iterations = 20

    dx = 9.0 / (I - 1)
    dx_array = [dx] * N
    #dx = np.array([1.0 / (I - 1) for _ in range(N)], dtype=np.double)

    rhs = FunctionTensor(dims, J, dx)
    print("Initialized Function Tensor!")

    lhs = PyLowRank(dims, R, seed=923845)
    lhs.ten.renormalize_columns(-1)

    method = "larsen_kolda"
    als = ALS(lhs.ten, rhs.ten)
    als.initialize_ds_als(J, method)

    # For some very small test functions, we will manually compute the
    # ground truth

    #ground_truth = np.zeros((I, I), dtype=np.double)
    #for i in range(I):
    #    for j in range(I):
    #        ground_truth[i, j] = (i + j) * 0.01 

    #integral = lhs.compute_integral(dx)
    #print(f"Integral: {integral}")

    for i in range(iterations):
        for j in range(lhs.N):
            als.execute_ds_als_update(j, True, True) 

            g = chain_had_prod([lhs.U[i].T @ lhs.U[i] for i in range(lhs.N) if i != j])
            detected_nan = np.any(np.isnan(g))

            if detected_nan:
                print("Found a NaN value!")

            integral = lhs.compute_integral(dx_array)
            print(f"Integral: {integral}")

            #sigma_lhs = np.zeros(R, dtype=np.double) 
            #lhs.ten.get_sigma(sigma_lhs, -1)
            #test = np.einsum('i,ji,ki->jk', sigma_lhs, lhs.U[0], lhs.U[1])

def image_test():
    J = 30000 

    trial_count = 1
    iterations = 3
    result = {}

    samplers = ["efficient"]
    R_values = [32]

    img = Image.open("data/mandrill.png")
    img.load()

    max_value = 255.0
    img_array = np.asarray(img, dtype="double") / max_value
    tensor_dims = np.array(img_array.shape)
    rhs = PyDenseTensor(img_array) 
    print("Constructed dense tensor...")

    method = "efficient"

    for R in R_values: 
        result[R] = {}
        for sampler in samplers:
            lhs = PyLowRank(tensor_dims, R, seed=923845)
            lhs.ten.renormalize_columns(-1)

            als = ALS(lhs.ten, rhs.ten)
            als.initialize_ds_als(J, method)

            for i in range(iterations):
                for j in range(lhs.N):
                    als.execute_ds_als_update(j, True, True) 
    
                sigma_lhs = np.zeros(R, dtype=np.double) 
                lhs.ten.get_sigma(sigma_lhs, -1)
                approx = np.einsum('r,ir,jr,kr->ijk', sigma_lhs, lhs.U[0], lhs.U[1], lhs.U[2])
                loss = la.norm(approx - img_array)
                print(loss)


if __name__=='__main__':
    #low_rank_test() 
    #numerical_integration_test() 
    #sparse_tensor_test()
    image_test()
