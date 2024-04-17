import numpy as np
import json
import os
import json
import statistics
import sys
from tt_als import *
from tensor_train import *
from tt_svd import *
from tensorly.tt_tensor import tt_to_tensor
from line_profiler import profile


def fit(ground_truth, approx):
    fitness = 1 - (np.linalg.norm(approx - ground_truth) / np.linalg.norm(ground_truth))
    return fitness


def generate_tt_full_tensors(true_rank, dims, std_noise):
    tt = TensorTrain(dims, true_rank, init_method="gaussian")
    tt_full = tt.materialize_dense()
    tt_full_dense = PyDenseTensor(tt_full)
    noise = np.random.normal(0, std_noise, tt_full.shape)
    full_data = tt_full + noise

    dense_noisy_data = PyDenseTensor(full_data)

    return dense_noisy_data, tt_full_dense

def compute_relative_error(N, dim, tests, r, J, n_trials, nsweeps, std_noise=None, true_rank=None):
    # for sz in dim:
    dims = [dim] * N
    ranks = [1] + [r] * (N - 1) + [1]
    tt_cores_svd = None
    tt_cores_rsvd = None

    for test in tests:
        total_fit = []
        total_time = []
        # outfile = f'outputs/dense_tt/synthetic/svd_{j}-{r}-{test}.json'
        outfile = f'outputs/dense_tt/synthetic/svd_{J}-{r}-{test}_{true_rank}_{dim}_{n_trials}_{nsweeps}.json'
        directory = os.path.dirname(outfile)
        if not os.path.exists(directory):
            os.makedirs(directory)

        for tr in range(n_trials):
            X, ground_truth = generate_tt_full_tensors(true_rank, dims, std_noise)
            if test == "tt-svd":
                start = time.time()
                cores_svd = tensor_train_svd(X.data, ranks, svd="truncated_svd", verbose=False)
                end = time.time()
                total_time.append(end-start)
                approx = tt_to_tensor(cores_svd)
                total_fit.append(fit(ground_truth.data, approx))
                tt_cores_svd = cores_svd

            elif test == "randomized_tt-svd":
                start = time.time()
                cores_rsvd = tensor_train_svd(X.data, ranks, svd="randomized_svd", verbose=False)
                end = time.time()
                total_time.append(end - start)
                approx = tt_to_tensor(cores_rsvd)
                total_fit.append(fit(ground_truth.data, approx))
                tt_cores_rsvd = cores_rsvd

            elif test == "tt-als":
                if tt_cores_svd is None:
                    raise ValueError("TT-SVD cores must be computed before TT-ALS")

                tt_approx = TensorTrain(dims, r, init_method=None, U=tt_cores_svd)
                tt_approx.place_into_canonical_form(0)
                start = time.time()
                tt_als = TensorTrainALS(ground_truth, tt_approx)
                tt_als.execute_exact_als_sweeps_slow(num_sweeps=nsweeps)
                end = time.time()
                total_time.append(end - start)
                fit_result = tt_als.compute_exact_fit()
                total_fit.append(fit_result)

            elif test == "randomized_tt-als":  # proposal
                if tt_cores_rsvd is None:
                    raise ValueError("TT-SVD cores must be computed before TT-ALS")

                tt_approx = TensorTrain(dims, r, init_method=None, U=tt_cores_rsvd)
                tt_approx.place_into_canonical_form(0)
                start = time.time()
                tt_approx.build_fast_sampler(0, J=J)
                tt_als = TensorTrainALS(ground_truth, tt_approx)
                tt_als.execute_randomized_als_sweeps(num_sweeps=nsweeps, J=J)
                end = time.time()
                total_time.append(end - start)

                fit_result = tt_als.compute_exact_fit()
                total_fit.append(fit_result)
            else:
                continue

        time_mean = np.mean(total_time)

        fit_mean = np.mean(total_fit)


        result = {
            "dim_size": dim,
            "sample_size": J,
            "test_name": test,
            "time": time_mean,
            "fit": fit_mean,
        }

        with open(outfile, 'w') as f:
            json.dump(result, f)


if __name__ == '__main__':
    n_trials = 5
    N = 10
    dim = [3,4,5]
    # true_rank = rank = range(5,15,5)
    true_rank = 10
    rank = 2
    std_noise = 1e-6
    tests = ["tt-svd", "randomized_tt-svd", "tt-als", "randomized_tt-als"]
    J = 2000
    nsweeps = 50
    for d in dim:
        print(f"{d} is running")
        compute_relative_error(N, d, tests, rank, J, n_trials, nsweeps, std_noise, true_rank)
