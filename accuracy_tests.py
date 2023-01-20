import numpy as np
import numpy.linalg as la

import json

from common import *
from tensors import *

import cppimport.import_hook
import cpp_ext.als_module as ALS_Module
from cpp_ext.als_module import Tensor, LowRankTensor, SparseTensor, ALS, Sampler

def krp(U):
    running_krp = U[0]
    cols = U[0].shape[1]
    for i in range(1, len(U)):
        height_init = running_krp.shape[0]
        running_krp = np.einsum('ir,jr->ijr', running_krp, U[i]).reshape(height_init * U[i].shape[0], cols)
    return running_krp

def kronecker_product_test(I, R, N, J, methods, trial_count):
    results = {}
    for method in methods:
        results[method] = []

    for trial in range(trial_count):
        A = PyLowRank([I] * N, R, init_method="gaussian")
        A.ten.renormalize_columns(-1)
        b = PyLowRank([I] * N, 1, allow_rhs_mttkrp=True, init_method="gaussian") 
        A.ten.multiply_random_factor_entries(0.01, 10.0)

        # Buffers required for sampling
        A_downsampled = np.zeros((J, R), dtype=np.double)
        b_downsampled = np.zeros((J, 1), dtype=np.double)
        samples = np.zeros((N, J), dtype=np.uint64)
        weights = np.zeros((J), dtype=np.double)

        mttkrp = chain_had_prod([A.U[i].T @ b.U[i] for i in range(N)])
        gram = chain_had_prod([U.T @ U for U in A.U])
        soln_exact = la.pinv(gram) @ mttkrp
        exact_residual_norm = compute_diff_norm(A.U, b.U, soln_exact, np.ones(1)) 

        for method in methods:
            sampler = Sampler(A.U, J, R, method) 
            sampler.KRPDrawSamples_materialize(N+1, samples, A_downsampled, weights) 
            b.ten.materialize_rhs(samples.T.copy(), N+1, b_downsampled) 

            A_ds_reweighted = np.einsum('i,ir->ir', np.sqrt(weights), A_downsampled)
            b_ds_reweighted = np.einsum('i,ir->ir', np.sqrt(weights), b_downsampled)
            soln_approx, _, _, _ = la.lstsq(A_ds_reweighted, b_ds_reweighted, rcond=None)
            approx_residual_norm = compute_diff_norm(A.U, b.U, soln_approx, np.ones(1)) 

            epsilon = approx_residual_norm / exact_residual_norm
            results[method].append(epsilon)

    return results 

if __name__=='__main__':
    I = 2 ** 16
    J = 5000
    methods = ["efficient", "larsen_kolda"]
    results = {"N_trace": {}, "R_trace": {}}
    trial_count=10

    R = 64
    for N in range(3, 10):
        result = kronecker_product_test(I, R, N, J, methods, trial_count=trial_count)
        print("Completed trial...")
        results["N_trace"][N] = result
        print(result)

    N = 6
    for R in [2 ** i for i in range(4, 8)]:
        result = kronecker_product_test(I, R, N, J, methods, trial_count=trial_count)
        print("Completed trial...")
        results["R_trace"][R] = result
        print(result)

    with open('outputs/accuracy_bench.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)