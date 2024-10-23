import numpy as np
import json
import os
import json
import statistics
import sys
from datasets_loader import *
from tt_svd_draft import *
from tt_als import *
from tensor_train import *
from tensorly.decomposition import tensor_train
from tensorly.tt_tensor import tt_to_tensor


# from synthetic_data import *

def fitness(ground_truth, approx):
    fitness = 1 - (np.linalg.norm(approx - ground_truth) / np.linalg.norm(ground_truth))
    return fitness

def compute_fit_real_data(dataset, tests, r, j):
    # global tensor_train_svd
    file_path = get_datasets(dataset)
    ground_truth = load_data(data, file_path)
    N = ground_truth.N
    dims = ground_truth.shape
    print(ground_truth.data.shape)
    ranks = [1] + [r] * (N - 1) + [1]
    tt_cores_svd = None
    tt_cores_rsvd = None

    for test in tests:
        total_fit = []
        total_time = []
        outfile = f'outputs/dense_tt/{dataset}/{test}_{r}_{j}.json'
        directory = os.path.dirname(outfile)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if test == "tt-svd":
            print(f"tt-svd is starting ...")
            start = time.time()
            cores_svd = tensor_train_svd(ground_truth.data, ranks, svd="truncated_svd")
            end = time.time()
            total_time.append(end - start)
            approx = tt_to_tensor(cores_svd)
            fit= fitness(ground_truth.data, approx)
            total_fit.append(fit)
            tt_cores_svd = cores_svd

        elif test == "randomized_tt-svd":
            print(f"rsvd is starting ...")
            start = time.time()
            cores_rsvd = tensor_train_svd(ground_truth.data, ranks, svd="randomized_svd")
            end = time.time()
            total_time.append(end - start)
            approx = tt_to_tensor(cores_rsvd)
            total_fit.append(fitness(ground_truth.data, approx))
            tt_cores_rsvd = cores_rsvd

        elif test == "tt-als":
            tt_approx = TensorTrain(dims, r, init_method=None, U=tt_cores_svd)
            tt_approx.place_into_canonical_form(0)
            start = time.time()
            tt_als = TensorTrainALS(ground_truth, tt_approx)
            tt_als.execute_exact_als_sweeps_slow(num_sweeps=5)
            end = time.time()
            total_time.append(end - start)
            fit = tt_als.compute_exact_fit()
            total_fit.append(fit)

        elif test == "randomized_tt-als":  # proposal
            tt_approx = TensorTrain(dims, r, init_method=None, U=tt_cores_rsvd)
            tt_approx.place_into_canonical_form(0)
            start = time.time()
            tt_approx.build_fast_sampler(0, J=j)
            tt_als = TensorTrainALS(ground_truth, tt_approx)
            tt_als.execute_randomized_als_sweeps(num_sweeps=nsweeps, J=j)
            end = time.time()
            total_time.append(end - start)
            fit = tt_als.compute_exact_fit()
            total_fit.append(fit)
        else:
            continue

        time_mean = np.mean(total_time)
        fit_mean = np.mean(total_fit)

        result = {
            "test_name": test,
            "time": time_mean.tolist(),
            "fit": fit_mean.tolist(),
        }

        with open(outfile, 'w') as f:
            json.dump(result, f)


if __name__ == '__main__':
    r = 5
    # dataset = ["pavia","bench-park", "cat"]
    J = 2000
    nsweeps = 5
    data = "pavia"
    tests = ["tt-svd","randomized_tt-svd","tt-als","randomized_tt-als"]
    compute_fit_real_data(data, tests, r, J)
