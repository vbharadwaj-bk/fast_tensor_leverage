import numpy as np
import json
import os
import json
import statistics
import sys
from tt_svd import *
from datasets_loader import *

current_script_path = os.path.dirname(os.path.abspath(__file__))
fast_tensor_leverage_path = os.path.dirname(current_script_path)
sys.path.append(fast_tensor_leverage_path)
from algorithms.tt_als import *
from tensors.tensor_train import *
from relative_error import *


def compute_fit_real_data(dataset, tests, r, num_sweeps, J):
    ground_truth = get_load_datasets(dataset)  # PyDenseTensor

    for test in tests:
        fit = []
        total_time = []
        outfile = f'outputs/dense_tt/PaviaU/{test}.json'
        directory = os.path.dirname(outfile)
        if not os.path.exists(directory):
            os.makedirs(directory)
        start = time.time()
        fit.append(fit_data(ground_truth, test, r, num_sweeps, J))
        end = time.time()
        total_time.append(end - start)

        mean_fit= np.mean(fit)
        mean_time = np.mean(total_time)

        result = {
            "test_name": test,
            "time": mean_time,
            "fit": mean_fit,
        }

        with open(outfile, 'w') as f:
            json.dump(result, f)


if __name__ == '__main__':
    r = 10
    num_sweeps = 10
    J = 100000
    data = "pavia"
    tests = ["tt-svd", "randomized_tt-svd", "tt-als", "randomized_tt-als"]
    compute_fit_real_data(data, tests, r, num_sweeps, J)
