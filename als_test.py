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
#from image_classification import * 

import cppimport.import_hook
import cpp_ext.als_module as ALS_Module
from cpp_ext.als_module import Tensor, LowRankTensor, SparseTensor, ALS, Sampler

def sparse_tensor_test():
    J = 2 ** 16

    trial_count = 1
    max_iterations = 5
    stop_tolerance = 1e-5

    result = {}

    samplers = ["efficient"]
    #R_values = [4, 8, 16, 32, 64, 128]
    R_values = [25]

    #rhs = PySparseTensor("/pscratch/sd/v/vbharadw/tensors/uber.tns_converted.hdf5", lookup="sort")
    rhs = PySparseTensor("/pscratch/sd/v/vbharadw/tensors/amazon-reviews.tns_converted.hdf5", lookup="sort")
    #rhs = PySparseTensor("/pscratch/sd/v/vbharadw/tensors/reddit-2015.tns_converted.hdf5", lookup="sort", preprocessing="log_count")
    #rhs = PySparseTensor("/pscratch/sd/v/vbharadw/tensors/enron.tns_converted.hdf5", lookup="sort", preprocessing="log_count")

    #rhs = PySparseTensor("/pscratch/sd/v/vbharadw/tensors/nell-1.tns_converted.hdf5", lookup="sort")
    #rhs = PySparseTensor("/pscratch/sd/v/vbharadw/tensors/nips.tns_converted.hdf5", lookup="sort", preprocessing="log_count")
    #rhs = PySparseTensor("/pscratch/sd/v/vbharadw/tensors/flickr-4d.tns_converted.hdf5", lookup="sort")

    for R in R_values: 
        result[R] = {}
        for sampler in samplers:
            result[R][sampler] = []
            for trial in range(trial_count):
                lhs = PyLowRank(rhs.dims, R)
                lhs.ten.renormalize_columns(-1)

                #print("Starting exact initialization")
                #als = ALS(lhs.ten, rhs.ten)
                #for j in range(lhs.N):
                #    als.execute_exact_als_update(j, True, True)

                # Initialize with a round of exact ALS
                #fit = lhs.compute_estimated_fit(rhs)
                #print(f"Fit after initialization: {fit}")

                start = time.time()
                result[R][sampler].append(als_prod(lhs, rhs, J, sampler, max_iterations, stop_tolerance, verbose=True))
                elapsed = time.time() - start
                print(f"Elapsed: {elapsed}")

    #with open('outputs/lk_uber_comparison.json', 'w') as outfile:
    #    json.dump(result, outfile, indent=4)

def low_rank_test():
    J = 60000 

    trial_count = 1
    result = {}

    samplers = ["larsen_kolda"]
    R_values = [128]

    I = 2 ** 16
    N = 5

    max_iterations = 50
    stop_tolerance = 1e-4

    for R in R_values: 
        rhs = PyLowRank([I] * N, R, allow_rhs_mttkrp=True, seed=479873)
        rhs.ten.multiply_random_factor_entries(0.1, 5.0)
        rhs.ten.renormalize_columns(-1)
        result[R] = {}
        for sampler in samplers:
            result[R][sampler] = []
            for trial in range(trial_count):
                # Specify a seed here to make everything deterministic
                lhs = PyLowRank(rhs.dims, R)
                lhs.ten.renormalize_columns(-1)
                print("=" * 20)
                print("Starting trial...")
                result[R][sampler].append(als_exact_comparison(lhs, rhs, J, sampler, max_iterations))
                print("Finished trial...")
                print("=" * 20)

    with open('outputs/low_rank_comparison.json', 'w') as outfile:
        json.dump(result, outfile, indent=4)

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
    J = 40000

    trial_count = 1
    iterations = 70
    result = {}

    samplers = ["larsen_kolda"]

    img = Image.open("data/mandrill.png")
    img.load()

    max_value = 255.0
    img_array = np.asarray(img, dtype="double") / max_value
    tensor_dims = np.array(img_array.shape)
    rhs = PyDenseTensor(img_array)
    rhs_norm = la.norm(img_array)
    print("Constructed dense tensor...")

    method = "efficient"
    R = 120

    lhs = PyLowRank(tensor_dims, R, seed=923845)
    lhs.ten.renormalize_columns(-1)

    als = ALS(lhs.ten, rhs.ten)
    als.initialize_ds_als(J, method)

    def generate_approx(ten, ground_truth):
        sigma_lhs = np.zeros(R, dtype=np.double) 
        lhs.ten.get_sigma(sigma_lhs, -1)
        approx = np.einsum('r,ir,jr,kr->ijk', sigma_lhs, lhs.U[0][:tensor_dims[0]], lhs.U[1][:tensor_dims[1]], lhs.U[2])
        
        return approx

    start = time.time()
    for i in range(iterations):
        print(f"Starting iteration {i}")
        for j in range(lhs.N):
            als.execute_ds_als_update(j, True, True) 

        #approx = generate_approx(lhs, img_array)
        #diff_norm = la.norm(approx - img_array)
        #fit = 1 - diff_norm / rhs_norm
        #print(f"Fit: {fit}")

    end = time.time()
    print(f"Elapsed: {end-start}")

    approx = generate_approx(lhs, img_array) 
    approx_norm = np.maximum(np.minimum(approx, 1.0), 0.0)
    image = Image.fromarray(np.uint8(approx_norm * max_value))
    #image.save("data/lowrank_approximation.png")

def image_classification_test():
    J = 800 
    classifier = TensorClassifier("cifar10", J, "efficient", R=50, max_iter=40)
    classifier.train()
    print("Completed training...")

def dsyrk_multithreading_test():
    I, R = 4000000, 100
    U = np.random.rand(I, R)
    start = time.time()
    ALS_Module.test_dsyrk_multithreading(U)
    elapsed = time.time() - start
    print(f"Elapsed: {elapsed}s")

def krp(U):
    running_krp = U[0]
    cols = U[0].shape[1]
    for i in range(1, len(U)):
        height_init = running_krp.shape[0]
        running_krp = np.einsum('ir,jr->ijr', running_krp, U[i]).reshape(height_init * U[i].shape[0], cols)
    return running_krp

def kronecker_product_test():
    I = 8 
    N = 3
    J = 100
    A_cols = 4
    A = PyLowRank([I] * N, A_cols, init_method="gaussian") 
    b = PyLowRank([I] * N, 1, allow_rhs_mttkrp=True, init_method="gaussian") 
    sampler = Sampler(A.U, J, A_cols, "efficient") 

    # Buffers required for sampling
    A_downsampled = np.zeros((J, A_cols), dtype=np.double)
    b_downsampled = np.zeros((J, 1), dtype=np.double)
    samples = np.zeros((N, J), dtype=np.uint64)
    weights = np.zeros((J), dtype=np.double)

    sampler.KRPDrawSamples_materialize(N+1, samples, A_downsampled, weights) 
    b.ten.materialize_rhs(samples.T.copy(), N, b_downsampled) 

    A_ds_reweighted = np.einsum('i,ir->ir', weights, A_downsampled)
    soln_approx, residual_approx, _, _ = la.lstsq(A_ds_reweighted, b_downsampled, rcond=None)
    print(residual_approx)

    soln_exact, residual_exact, _, _ = la.lstsq(krp(A.U), krp(b.U), rcond=None)
    print(residual_exact)


if __name__=='__main__':
    #low_rank_test() 
    #numerical_integration_test() 
    #sparse_tensor_test()
    #image_test()
    #image_classification_test()
    #dsyrk_multithreading_test()

    kronecker_product_test()


