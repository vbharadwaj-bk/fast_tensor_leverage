import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import time
import json

import cppimport.import_hook
from cpp_ext.efficient_krp_sampler import CP_ALS 

def symmetrize(buf):
    return np.triu(buf, 1) + np.triu(buf, 1).T + np.diag(np.diag(buf))

def krp(mats):
    if len(mats) == 1:
        return mats[0]
    else:
        running_mat = np.einsum('ik,jk->ijk', mats[0], mats[1]).reshape((mats[0].shape[0] * mats[1].shape[0], mats[0].shape[1]))
        
        for i in range(2, len(mats)):
            running_mat = np.einsum('ik,jk->ijk', running_mat, mats[i]).reshape((running_mat.shape[0] * mats[i].shape[0], mats[0].shape[1]))

        return running_mat

def chain_had_prod(matrices):
    res = np.ones(matrices[0].shape)
    for mat in matrices:
        res *= mat
    return res

def inner_prod(U, V):
    elwise_prod = chain_had_prod([U[i].T @ V[i] for i in range(len(U))])
    return np.sum(elwise_prod)

def compute_diff_norm(U, V):
    return np.sqrt(inner_prod(U, U) + inner_prod(V, V) - 2 * inner_prod(U, V))

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
    rhs_ten = LowRankTensor(R, U_rhs)
    als = ALS(lhs_ten, rhs_ten)
    als.initialize_ds_als(J) 

    print("Constructed ALS Sampler...")

    samples, sampled_rows, algorithm = sample_function(U_lhs, j, J, R)

    g = chain_had_prod([U_lhs[i].T @ U_lhs[i] for i in range(N) if i != j])
    g_pinv = la.pinv(g)
    leverage_score_sum = np.sum(g_pinv * g)

    leverage_scores = np.sum((sampled_rows @ g_pinv) * sampled_rows, axis=1)
    weights = 1.0 / np.sqrt(leverage_scores * J / leverage_score_sum)
    weighted_lhs = np.einsum('i,ij->ij', weights ** 2, sampled_rows)

    # Compute the true solution 
    elwise_prod = chain_had_prod([U_lhs[i].T @ U_rhs[i] for i in range(N) if i != j])
    true_soln = U_rhs[j] @ elwise_prod.T @ g_pinv 

    low_rank_ten = LowRankTensor(R, J, 17, U_rhs)
    mttkrp_res = np.zeros(U_lhs[j].shape, dtype=np.double)
    low_rank_ten.execute_downsampled_mttkrp_py(samples, weighted_lhs, j, mttkrp_res) 
    approx_soln = mttkrp_res @ g_pinv

    U_lhs[j] = true_soln
    true_residual = compute_diff_norm(U_lhs, U_rhs)

    U_lhs[j] = approx_soln 
    approx_residual = compute_diff_norm(U_lhs, U_rhs)
    ratio = (approx_residual - true_residual) / true_residual
    data.append({"N": len(U_lhs), "I": I, "R": R, "J": J, "true_residual": true_residual, "approx_residual": approx_residual, 'ratio': ratio, 'alg': algorithm})
    print(data[-1])

if __name__=='__main__':
    data = []
    R = 32 
    for i in range(4, 19):
        for N in [3]:
            U_lhs = [np.random.rand(2 ** i, R) for _ in range(N)]
            U_rhs = [np.random.rand(2 ** i, R) for _ in range(N)]
            execute_leave_one_test(U_lhs, U_rhs, 2 ** i, 32, 10000, data, uniform_sample, N)
            execute_leave_one_test(U_lhs, U_rhs, 2 ** i, 32, 10000, data, larsen_kolda_sample, N)
            execute_leave_one_test(U_lhs, U_rhs, 2 ** i, 32, 10000, data, fast_leverage_sample, N)

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
        execute_leave_one_test(U_lhs, U_rhs, 2 ** i, R, 10000, data, uniform_sample, N)
        execute_leave_one_test(U_lhs, U_rhs, 2 ** i, R, 10000, data, larsen_kolda_sample, N)
        execute_leave_one_test(U_lhs, U_rhs, 2 ** i, R, 10000, data, fast_leverage_sample, N)

    with open(f"outputs/n_comparison.json", "w") as outfile:
        json.dump(data, outfile) 


