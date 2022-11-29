import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import time
import json

import cppimport.import_hook
from cpp_ext.efficient_krp_sampler import CP_ALS 

def symmetrize(buf):
    return np.triu(buf, 1) + np.triu(buf, 1).T + np.diag(np.diag(buf))

def chain_had_prod(matrices):
    res = np.ones(matrices[0].shape)
    for mat in matrices:
        res *= mat
    return res

def inner_prod(U, V, sigma_U, sigma_V):
    elwise_prod = chain_had_prod([U[i].T @ V[i] for i in range(len(U))])
    elwise_prod *= np.outer(sigma_U, sigma_V)
    return np.sum(elwise_prod)

def compute_diff_norm(U, V, sigma_U, sigma_V):
    return np.sqrt(inner_prod(U, U, sigma_U, sigma_U) + inner_prod(V, V, sigma_V, sigma_V) - 2 * inner_prod(U, V, sigma_U, sigma_V))

def uniform_sample(U, j, J, R):
    samples = np.zeros((len(U), J), dtype=np.uint64)
    sampled_rows = np.ones((J, R), dtype=np.double)
    # Simple version of the Larsen & Kolda sampler 
    gram_matrices = [U[i].T @ U[i] for i in range(len(U))]

    rng = np.random.default_rng()

    for i in range(len(U)):
        buf = U[i] @ gram_matrices[i]
        buf *= U[i]
        leverage_scores = np.sum(buf, axis=1)
        C = np.sum(leverage_scores) 
        leverage_scores /= C
        if i != j:
            samples[i] = rng.choice(U[i].shape[0], size=J) 
            sampled_rows *= U[i][samples[i]]

    return samples, sampled_rows, 'uniform'

def larsen_kolda_sample(U, j, J, R):
    samples = np.zeros((len(U), J), dtype=np.uint64)
    sampled_rows = np.ones((J, R), dtype=np.double)
    # Simple version of the Larsen & Kolda sampler 
    gram_matrices = [U[i].T @ U[i] for i in range(len(U))]

    rng = np.random.default_rng()

    for i in range(len(U)):
        buf = U[i] @ gram_matrices[i]
        buf *= U[i]
        leverage_scores = np.sum(buf, axis=1)
        C = np.sum(leverage_scores) 
        leverage_scores /= C
        if i != j:
            samples[i] = rng.choice(U[i].shape[0], size=J, p=leverage_scores) 
            sampled_rows *= U[i][samples[i]]

    return samples, sampled_rows, 'larsen_kolda'

def fast_leverage_sample(U, j, J, R):
    cp_als = CP_ALS(J, R, U)
    samples = np.zeros((len(U), J), dtype=np.uint64)
    sampled_rows = np.zeros((J, R), dtype=np.double)
    cp_als.KRPDrawSamples(j, samples, sampled_rows)

    return samples, sampled_rows, 'fast_tensor_leverage'

def execute_leave_one_test(U_lhs, U_rhs, I, R, J, data, sample_function, N):
    from cpp_ext.als_module import Tensor, LowRankTensor, ALS
    j = N-1

    lhs_ten = LowRankTensor(R, U_lhs)
    rhs_ten = LowRankTensor(R, J, 10000, U_rhs)

    rhs_ten.renormalize_columns(-1)
    sigma_rhs = np.zeros(R, dtype=np.double) 
    rhs_ten.get_sigma(sigma_rhs)

    als = ALS(lhs_ten, rhs_ten)
    als.initialize_ds_als(J)

    samples, sampled_rows, algorithm = sample_function(U_lhs, j, J, R)

    g = chain_had_prod([U_lhs[i].T @ U_lhs[i] for i in range(N) if i != j])
    g_pinv = la.pinv(g)

    leverage_scores = np.sum((sampled_rows @ g_pinv) * sampled_rows, axis=1)
    weights = 1.0 / (leverage_scores * J / R)

    weighted_lhs = np.einsum('i,ij->ij', weights, sampled_rows)

    # Compute the true solution 
    elwise_prod = chain_had_prod([U_lhs[i].T @ U_rhs[i] for i in range(N) if i != j])
    true_soln = U_rhs[j] @ elwise_prod.T @ (g_pinv @ np.diag(sigma_rhs))

    mttkrp_res = np.zeros(U_lhs[j].shape, dtype=np.double)
    rhs_ten.execute_downsampled_mttkrp_py(samples, weighted_lhs, j, mttkrp_res) 

    sigma_lhs = np.zeros(R, dtype=np.double) 

    approx_soln = mttkrp_res @ g_pinv
    U_lhs[j][:] = true_soln

    lhs_ten.get_sigma(sigma_lhs)
    true_residual = compute_diff_norm(U_lhs, U_rhs, sigma_lhs, sigma_rhs)

    if algorithm == 'fast_tensor_leverage':
        als.execute_ds_als_update(j, True, False) 
    else:
        U_lhs[j] = approx_soln 

    lhs_ten.get_sigma(sigma_lhs)
    rhs_ten.get_sigma(sigma_rhs)

    approx_residual = compute_diff_norm(U_lhs, U_rhs, sigma_lhs, sigma_rhs)
    ratio = (approx_residual - true_residual) / true_residual
    data.append({"N": len(U_lhs), "I": I, "R": R, "J": J, "true_residual": true_residual, "approx_residual": approx_residual, 'ratio': ratio, 'alg': algorithm})
    print(data[-1])
    exit(1)

if __name__=='__main__':
    data = []
    R = 32
    for i in range(4, 19):
        for N in [3]:
            U_lhs = [np.random.rand(2 ** i, R) for _ in range(N)]
            U_rhs = [np.random.rand(2 ** i, R) for _ in range(N)]
            #execute_leave_one_test(U_lhs, U_rhs, 2 ** i, R, 10000, data, uniform_sample, N)
            #execute_leave_one_test(U_lhs, U_rhs, 2 ** i, R, 10000, data, larsen_kolda_sample, N)
            execute_leave_one_test(U_lhs, U_rhs, 2 ** i, R, 10000, data, fast_leverage_sample, N)
            break

    with open(f"outputs/increasing_i_comparison.json", "w") as outfile:
        json.dump(data, outfile) 

    data = []
    i = 16
    for N in [4, 5, 6]:
        for R in [4, 8, 16, 32, 64, 128]: 
            U_lhs = [np.random.rand(2 ** i, R) for _ in range(N)]
            U_rhs = [np.random.rand(2 ** i, R) for _ in range(N)]
            execute_leave_one_test(U_lhs, U_rhs, 2 ** i, R, 10000, data, uniform_sample, N)
            execute_leave_one_test(U_lhs, U_rhs, 2 ** i, R, 10000, data, larsen_kolda_sample, N)
            execute_leave_one_test(U_lhs, U_rhs, 2 ** i, R, 10000, data, fast_leverage_sample, N)

    with open(f"outputs/n_r_comparison.json", "w") as outfile:
        json.dump(data, outfile) 

    data = []
    i = 16
    R = 32
    for N in [2, 3, 4, 5, 6, 7, 8]:
        U_lhs = [np.random.rand(2 ** i, R) for _ in range(N)]
        U_rhs = [np.random.rand(2 ** i, R) for _ in range(N)]
        #execute_leave_one_test(U_lhs, U_rhs, 2 ** i, R, 10000, data, uniform_sample, N)
        #execute_leave_one_test(U_lhs, U_rhs, 2 ** i, R, 10000, data, larsen_kolda_sample, N)
        #execute_leave_one_test(U_lhs, U_rhs, 2 ** i, R, 10000, data, fast_leverage_sample, N)

    with open(f"outputs/n_comparison.json", "w") as outfile:
        json.dump(data, outfile) 


