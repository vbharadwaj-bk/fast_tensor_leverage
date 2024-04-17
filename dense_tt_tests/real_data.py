import numpy as np
import json
import os
import json
import statistics
import sys
from tt_svd import *
from datasets_loader import *
from tt_als import *
from tensor_train import *


# from synthetic_data import *

def relative_error(ground_truth, approx):
    rel_error = np.linalg.norm(approx - ground_truth) / np.linalg.norm(ground_truth)
    return rel_error


def compute_relative_error_real_data(dataset, tests, r, j):
    file_path = get_datasets(dataset)
    ground_truth = load_data(data, file_path)
    dims = ground_truth.shape
    for test in tests:
        total_relative_error = []
        total_time = []
        outfile = f'outputs/dense_tt/{dataset}/{test}-{r}.json'
        directory = os.path.dirname(outfile)
        if not os.path.exists(directory):
            os.makedirs(directory)
        start = time.time()
        if test == "tt-svd":
            tensor_train_svd = TensorTrainSVD(ground_truth.data, r)
            cores_svd = tensor_train_svd.tt_svd()
            approx = tt_to_tensor(cores_svd)
            total_relative_error.append(relative_error(ground_truth.data, approx))

        elif test == "randomized_tt-svd":
            tensor_train_svd = TensorTrainSVD(ground_truth.data, r)
            cores = tensor_train_svd.tt_randomized_svd()
            approx = tt_to_tensor(cores)
            total_relative_error.append(relative_error(ground_truth.data, approx))

        elif test == "tt-als":
            tt_approx = TensorTrain(dims, r, init_method="tt-svd",
                                    input_tensor=ground_truth)  # gaussian/svd-based initialization
            tt_approx.place_into_canonical_form(0)
            tt_als = TensorTrainALS(ground_truth, tt_approx)
            tt_als.execute_exact_als_sweeps_slow(num_sweeps=10)
            relative_error_result = 1 - tt_als.compute_exact_fit()
            total_relative_error.append(relative_error_result)

        elif test == "randomized_tt-als":  # proposal
            tt_approx = TensorTrain(dims, r, init_method="randomized_tt-svd",
                                    input_tensor=ground_truth)  # gaussian/svd-based initialization
            tt_approx.place_into_canonical_form(0)
            tt_als = TensorTrainALS(ground_truth, tt_approx)
            tt_approx.build_fast_sampler(0, J=j)
            tt_als.execute_randomized_als_sweeps(num_sweeps=10, J=j)
            relative_error_result = 1 - tt_als.compute_exact_fit()
            total_relative_error.append(relative_error_result)
        else:
            continue

        end = time.time()
        total_time.append(end - start)

        relative_error_mean = np.mean(total_relative_error)
        time_mean = np.mean(total_time)

        result = {
            "test_name": test,
            "time": time_mean.tolist(),
            "relative_error": relative_error_mean.tolist(),
        }

        with open(outfile, 'w') as f:
            json.dump(result, f)


if __name__ == '__main__':
    rank = 10
    # dataset = ["pavia","bench-park", "cat"]
    j = 8000
    data = "pavia"
    tests = ["tt-svd", "randomized_tt-svd", "tt-als", "randomized_tt-als"]
    compute_relative_error_real_data(data, tests, rank, j)
