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

def generate_random_matrix(I, R, max_singular_value, min_singular_value):
    U = np.random.normal(size=(I, R))
    U, _ = la.qr(U)

    V = np.random.normal(size=(R, R))
    V, _ = la.qr(V)
    singular_value_means = np.linspace(max_singular_value, min_singular_value, R)
    sigma= np.random.normal(singular_value_means, 0.1)
    return U @ np.diag(np.abs(sigma)) @ V.T 

def kronecker_product_test(I, R, N, J, methods, trial_count):
    results = {}
    for method in methods:
        results[method] = []

    for trial in range(trial_count):
        A_with_resid = PyLowRank([I] * N, R+1, init_method="gaussian")

        #for i in range(N):
        #    A_with_resid.U[i][:] = generate_random_matrix(I, R+1, 5.0, 0.5)

        A_with_resid.ten.renormalize_columns(-1)
        A_with_resid.ten.multiply_random_factor_entries(0.01, 10.0)
        
        #A_with_resid.set_sigma(np.ones(R+1, dtype=np.double))

        A = PyLowRank([I] * N, R, init_method="gaussian")

        for i in range(N):
            A.U[i][:] = A_with_resid.U[i][:, :-1]        

        A_downsampled = np.zeros((J, R), dtype=np.double)
        samples = np.zeros((N, J), dtype=np.uint64)
        weights = np.zeros((J), dtype=np.double)

        # Generate a Gaussian random vector in numpy
        soln_exact = np.random.normal(size=(R)) + 1.0
        soln_exact /= la.norm(soln_exact) * 10
        A.set_sigma(soln_exact)

        gram = chain_had_prod([U.T @ U for U in A.U])
        eigvals, V = la.eigh(gram)

        A_with_resid.set_sigma(rhs_sigma_padded)

        for method in methods:
            sampler = Sampler(A.U, J, R, method) 
            sampler.KRPDrawSamples_materialize(N+1, samples, A_downsampled, weights) 
            A_ds_reweighted = np.einsum('i,ir->ir', np.sqrt(weights), A_downsampled)
            SU = A_ds_reweighted @ V @ np.diag(np.sqrt(1 / eigvals)) 
            epsilon = la.cond(SU)

            results[method].append(epsilon)

    for method in results:
        print(method, np.mean(results[method]), np.std(results[method]))

    return results 

if __name__=='__main__':
    I = 2 ** 16 
    J = 5000 
    methods = ["efficient", "larsen_kolda"]
    results = {"N_trace": {}, "R_trace": {}}
    trial_count=50

    R = 64
    for N in range(3, 10):
        result = kronecker_product_test(I, R, N, J, methods, trial_count=trial_count)
        print("Completed trial...")
        results["N_trace"][N] = result 

    N = 6
    for R in [2 ** i for i in range(4, 8)]:
        result = kronecker_product_test(I, R, N, J, methods, trial_count=trial_count)
        print("Completed trial...")
        results["R_trace"][R] = result

    with open('outputs/accuracy_bench3.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)