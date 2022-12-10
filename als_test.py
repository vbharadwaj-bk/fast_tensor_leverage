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
    def __init__(self, dims, R, allow_rhs_mttkrp=False, J=None, init_method="gaussian", seed=42):
        if init_method=="gaussian":
            rng = np.random.default_rng(seed)
            self.N = len(dims)
            self.R = R
            self.U = [rng.normal(size=(divide_and_roundup(i, R) * R, R)) for i in dims]
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
        sigma_lhs, sigma_rhs = np.zeros(self.R, dtype=np.double), np.zeros(rhs.R, dtype=np.double)
        U_lhs = self.U
        self.ten.get_sigma(sigma_lhs, -1)
        U_rhs = rhs_ten.U
        rhs_ten.ten.get_sigma(sigma_rhs, -1)
        residual = compute_diff_norm(U_lhs, U_rhs, sigma_lhs, sigma_rhs)

        return residual

    def compute_norm(self):
        sigma_lhs = np.zeros(self.R, dtype=np.double)
        self.ten.get_sigma(sigma_lhs, -1)
        return np.sqrt(inner_prod(self.U, self.U, sigma_lhs, sigma_lhs))

def als(lhs, rhs, J, method, iter):
    data = []

    als = ALS(lhs.ten, rhs.ten)
    als.initialize_ds_als(J, method)

    rhs_norm = rhs.compute_norm()

    residual = lhs.compute_diff_resid(rhs)
    print(f"Residual: {residual / rhs_norm}")
    try:
        for i in range(iter):
            for j in range(lhs.N):
                sigma_lhs, sigma_rhs = np.zeros(lhs.R, dtype=np.double), np.zeros(rhs.R, dtype=np.double)
                lhs.ten.get_sigma(sigma_lhs, j)
                rhs.ten.get_sigma(sigma_rhs, -1)

                g = chain_had_prod([lhs.U[i].T @ lhs.U[i] for i in range(N) if i != j])
                print(la.cond(g))
                g_pinv = la.pinv(g) 

                elwise_prod = chain_had_prod([lhs.U[i].T @ rhs.U[i] for i in range(N) if i != j])
                elwise_prod *= np.outer(np.ones(lhs.R), sigma_rhs)
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
                print(f"Ratio: {ratio}, Residual: {residual_approx / rhs_norm}")
                data_entry = {}
                data_entry["exact_solve_residual"] = residual
                data_entry["approx_solve_residual"] = residual_approx
                data_entry["exact_solve_residual_normalized"] = residual / rhs_norm
                data_entry["approx_solve_residual_normalized"] = residual_approx / rhs_norm
                data_entry["rhs_norm"] = rhs_norm
                data_entry["ratio"] = ratio
                data_entry["j"] = j 
                
                data.append(data_entry)
        return data 
    except:
        print("Caught SVD unconverged exception, terminating and returning trace...")
        return data

    #with open('data/lstsq_problems.pickle', 'wb') as handle:
    #    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    #print("Dumped data pickle")

if __name__=='__main__':
    i = 13
    R = 32
    N = 5
    J = 20000

    trial_count = 5
    iterations = 25
    result = {"I": 2 ** i, "R" : R, "N": N, "J": J}

    samplers = ["efficient"]

    for sampler in samplers:
        result[sampler] = []
        for trial in range(trial_count): 
            lhs = PyLowRank([2 ** i] * N, 2 * R, seed=923845)
            lhs.ten.renormalize_columns(-1)
            rhs = PyLowRank([2 ** i] * N, R, allow_rhs_mttkrp=True, J=J, seed=29348)
            rhs.ten.renormalize_columns(-1)
            result[sampler].append(als(lhs, rhs, J, sampler, iterations))

    #with open('outputs/synthetic_lowrank_comparison.json', 'w') as outfile:
    #    json.dump(result, outfile, indent=4)