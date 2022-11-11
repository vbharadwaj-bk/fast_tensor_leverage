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

def execute_leave_one_tests():
    I = 4
    N = 4
    J = 1000
    R = 2
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

    # True leverage score sum
    full_lhs = krp(U_lhs[:j] + U_lhs[j+1:])
    full_rhs = krp(U_rhs[:j] + U_rhs[j+1:]) @ U_rhs[j].T

    #krp_q = la.qr(krp_materialized)[0]
    #krp_norms = la.norm(krp_q, axis=1) ** 2

    leverage_scores = np.sum((sampled_rows @ g_pinv) * sampled_rows, axis=1)
    weights = 1.0 / np.sqrt(leverage_scores * J / leverage_score_sum)

    weighted_lhs = np.einsum('i,ij->ij', weights, sampled_rows)

    rhs_implicit = CP_Decomposition(R, U_rhs)
    partial_evaluation = np.zeros((J, R))
    rhs_implicit.materialize_partial_evaluation(samples, j, partial_evaluation)
    unweighted_rhs = partial_evaluation @ U_rhs[j].T
    weighted_rhs = np.einsum('i,ij->ij', weights, unweighted_rhs)

    # Compute the true solution 
    elwise_prod = chain_had_prod([U_lhs[i].T @ U_rhs[i] for i in range(N) if i != j])
    true_soln = U_rhs[j] @ elwise_prod.T @ g_pinv
    approx_soln = weighted_rhs.T @ weighted_lhs @ g_pinv

    numpy_soln, _, _, _ = la.lstsq(full_lhs, full_rhs, rcond=None)
    numpy_soln = numpy_soln.T

if __name__=='__main__':
    execute_leave_one_tests()




