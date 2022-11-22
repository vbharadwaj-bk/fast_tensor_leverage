import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import time
import json

import cppimport.import_hook
from cpp_ext.efficient_krp_sampler import CP_ALS, CP_Decomposition 

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

def materialize_4prod(U):
    return np.einsum('ir,jr,kr,lr->ijkl', U[0], U[1], U[2], U[3])

def execute_leave_one_test(I, R, J, data):
    from cpp_ext.als_module import Tensor, LowRankTensor, ALS

    N = 4
    j = 3
    U_lhs = [np.random.rand(I, R) for _ in range(N)]
    U_rhs = [np.random.rand(I, R) for _ in range(N)]

    cp_als = CP_ALS(J, R, U_lhs)
    samples = np.zeros((N, J), dtype=np.uint64)
    sampled_rows = np.zeros((J, R), dtype=np.double)
    g_pinv = np.zeros((R, R))

    print("Starting evaluation...")
    cp_als.KRPDrawSamples(j, samples, sampled_rows)
    cp_als.get_G_pinv(g_pinv)
    g_pinv = symmetrize(g_pinv)
    g = la.pinv(g_pinv)
    leverage_score_sum = np.sum(g_pinv * g)

    leverage_scores = np.sum((sampled_rows @ g_pinv) * sampled_rows, axis=1)
    weights = 1.0 / np.sqrt(leverage_scores * J / leverage_score_sum)

    weighted_lhs = np.einsum('i,ij->ij', weights ** 2, sampled_rows)

    rhs_implicit = CP_Decomposition(R, U_rhs)
    partial_evaluation = np.zeros((J, R))
    rhs_implicit.materialize_partial_evaluation(samples, j, partial_evaluation)
    unweighted_rhs = partial_evaluation @ U_rhs[j].T
    #weighted_rhs = np.einsum('i,ij->ij', weights, unweighted_rhs)

    # Compute the true solution 
    elwise_prod = chain_had_prod([U_lhs[i].T @ U_rhs[i] for i in range(N) if i != j])
    check_g_pinv = la.pinv(chain_had_prod([U_lhs[i].T @ U_lhs[i] for i in range(N) if i != j]))

    print(la.norm(check_g_pinv - g_pinv))

    lst = [U_lhs[i].T @ U_lhs[i] for i in range(N) if i != j]
    prev = np.ones((5, 5))
    for el in reversed(lst): 
        print(el[:5, :5] * prev)
        prev = prev * el[:5, :5]

    true_soln = U_rhs[j] @ elwise_prod.T @ check_g_pinv

    #low_rank_ten = LowRankTensor(R, J, 17, U_rhs)
    #mttkrp_res = np.zeros(U_lhs[j].shape, dtype=np.double)
    #low_rank_ten.execute_downsampled_mttkrp_py(samples, weighted_lhs, j, mttkrp_res) 
    #print(mttkrp_res)
    #print(unweighted_rhs.T @ weighted_lhs)

    approx_soln = unweighted_rhs.T @ weighted_lhs @ g_pinv 
    #approx_soln = mttkrp_res @ g_pinv

    U_lhs[j] = true_soln
    true_residual = compute_diff_norm(U_lhs, U_rhs)

    U_lhs[j] = approx_soln 
    approx_residual = compute_diff_norm(U_lhs, U_rhs)
    ratio = (approx_residual - true_residual) / true_residual
    data.append({"I": I, "R": R, "J": J, "true_residual": true_residual, "approx_residual": approx_residual, 'ratio': ratio})
    print(data[-1])

if __name__=='__main__':
    data = []
    for i in range(4, 5):
        execute_leave_one_test(2 ** i, 32, 10000, data)

    #with open(f"outputs/leave_one_rank_tests.json", "w") as outfile:
    #    json.dump(data, outfile) 

    #low_rank = LowRankTensor(5)
    #als = ALS()
    #als.test(low_rank)

