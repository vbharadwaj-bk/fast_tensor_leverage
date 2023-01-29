import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
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

def run_distribution_comparison():
    N = 3
    I = 8
    R = 8
    J = 50000
    result = {"N": N, "I": I, "R": R, "J": J}
    A = PyLowRank([I] * N, R, init_method="gaussian")
    A.ten.renormalize_columns(-1)
    A.ten.multiply_random_factor_entries(0.01, 10.0)

    full_krp = krp(A.U)
    q, r = la.qr(full_krp)
    leverage_scores = la.norm(q, axis=1) ** 2
    result["true_distribution"] = list(leverage_scores)

    sampler = Sampler(A.U, J, R, "efficient")
    samples = np.zeros((N, J), dtype=np.uint64)
    weights = np.zeros((J), dtype=np.double)
    sampler.KRPDrawSamples(N+1, samples)
    offset_arr = np.array([I ** (N - i - 1) for i in range(N)])
    exact_distribution = samples.T @ offset_arr
    result["sts_sampler_draws"] = list(exact_distribution)

    with open('outputs/distribution_comparison.json', 'w') as outfile:
        json.dump(result, outfile, indent=4)

    plot_result()

def plot_result():
    with open("outputs/distribution_comparison.json", "r") as infile:
        result = json.load(infile)
        N = result["N"]
        I = result["I"]
        J = result["J"]
        krp_height = I ** N
        
        true_distribution = result['true_distribution']
        sts_sampler_draws = np.array(result['sts_sampler_draws']).astype(np.int32)
        bins = np.array(np.bincount(sts_sampler_draws, minlength=krp_height)) / len(sts_sampler_draws)
        
        scale=5
        fig, ax = plt.subplots(figsize=(2 * scale,1 * scale))
        ax.plot(true_distribution / np.sum(true_distribution), label="True Leverage Score Distribution")
        ax.plot(bins, label=f"Histogram of Draws from Our Sampler")
        ax.grid(True)
        ax.set_xlabel("Row Index from KRP")
        ax.set_ylabel("Density")
        ax.legend()
        fig.savefig("plotting/distribution_comparison_test_generated.png")
        #fig.show()
        #plt.show()

if __name__=='__main__':
    run_distribution_comparison()
