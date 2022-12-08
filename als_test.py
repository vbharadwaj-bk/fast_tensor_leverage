import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import time
import json
import pickle

from common import *

import cppimport.import_hook
from cpp_ext.efficient_krp_sampler import CP_ALS 
from cpp_ext.als_module import Tensor, LowRankTensor, ALS
from cpp_ext.efficient_krp_sampler import CP_ALS

class PyLowRank:
    def __init__(self, dims, R, allow_rhs_mttkrp=False, J=None, init_method="gaussian"):
        if init_method=="gaussian":
            self.N = len(dims)
            self.R = R
            self.U = [np.random.normal(size=(divide_and_roundup(i, R), R)) for i in dims]
            if allow_rhs_mttkrp:
                self.ten = LowRankTensor(R, J, 10000, self.U)
            else:
                self.ten = LowRankTensor(R, self.U)
        else:
            assert(False)

    def compute_diff_resid(self, rhs_ten):
        '''
        Computes the residual of the difference between two low-rank tensors.
        '''
        sigma_lhs, sigma_rhs = np.zeros(self.R, dtype=np.double), np.zeros(self.R, dtype=np.double)
        U_lhs = self.U
        self.ten.get_sigma(sigma_lhs, -1)
        U_rhs = rhs_ten.U
        rhs_ten.ten.get_sigma(sigma_rhs, -1)
        residual = compute_diff_norm(U_lhs, U_rhs, sigma_lhs, sigma_rhs)

        return residual

def als(lhs, rhs, J):
    data = []

    als = ALS(lhs.ten, rhs.ten)
    als.initialize_ds_als(J)

    residual = lhs.compute_diff_resid(rhs)
    print(f"Residual: {residual}")
    for i in range(80):
        for j in range(lhs.N):
            sigma_lhs, sigma_rhs = np.zeros(lhs.R, dtype=np.double), np.zeros(lhs.R, dtype=np.double)
            lhs.ten.get_sigma(sigma_lhs, j)
            rhs.ten.get_sigma(sigma_rhs, -1)

            g = chain_had_prod([lhs.U[i].T @ lhs.U[i] for i in range(N) if i != j])
            g_pinv = la.pinv(g) 

            elwise_prod = chain_had_prod([lhs.U[i].T @ rhs.U[i] for i in range(N) if i != j])
            elwise_prod *= np.outer(np.ones(R), sigma_rhs)
            true_soln = rhs.U[j] @ elwise_prod.T @ g_pinv @ np.diag(sigma_lhs ** -1)

            lhs.U[j][:] = true_soln
            lhs.ten.renormalize_columns(j)

            residual = lhs.compute_diff_resid(rhs)
            als.execute_ds_als_update(j, True, True) 

            residual_approx = lhs.compute_diff_resid(rhs)
            if residual > 0:
                ratio = residual_approx / residual
            else:
                ratio = 1.0

            #print(f"Condition #: {la.cond(g)}")
            print(f"Ratio: {ratio}, Residual: {residual}")
            data_entry = {}
            data_entry["exact_solve_residual"] = residual
            data_entry["approx_solve_residual"] = residual_approx
            data_entry["ratio"] = ratio
            data_entry["lhs"] = lhs.U
            data_entry["rhs"] = rhs.U
            data_entry["sigma_lhs"] = sigma_lhs
            data_entry["sigma_rhs"] = sigma_rhs
            data_entry["true_soln"] = true_soln
            data_entry["j"] = j 
            data.append(data_entry)

    #with open('data/lstsq_problems.pickle', 'wb') as handle:
    #    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    #print("Dumped data pickle")

if __name__=='__main__':
    i = 15
    R = 32
    N = 4
    J = 10000
    lhs = PyLowRank([2 ** i] * N, R)
    lhs.ten.renormalize_columns(-1)
    rhs = PyLowRank([2 ** i] * N, R, allow_rhs_mttkrp=True, J=J)
    rhs.ten.renormalize_columns(-1)

    als(lhs, rhs, J)
