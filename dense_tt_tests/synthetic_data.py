import numpy as np
import json
import os
import json
import statistics
import sys
from tt_svd import *

current_script_path = os.path.dirname(os.path.abspath(__file__))
fast_tensor_leverage_path = os.path.dirname(current_script_path)
sys.path.append(fast_tensor_leverage_path)
from tensors.dense_tensor import *
from algorithms.tt_als import *
from tensors.tensor_train import *
from relative_error import *

# tt_approx = TensorTrain([I] * N, [R] * (N - 1))
# tt_approx_GT = TensorTrain([I] * N, [R] * (N - 1))
# ground_truth = PyDenseTensor(tt_approx_GT.materialize_dense())


def generate_tt_full_tensors(r_true, N, dims, std_noise):
    R = [1] + [r_true] * (N - 1) + [1]
    tt = []
    dense_data = []
    dense_noisy_data = []
    tensor_size = [[I] * N for I in dims]
    for size in tensor_size:
        tt.append(TensorTrain(size, [r_true] * (N - 1)))
    for tt_object in tt:
        tt_full = tt_object.materialize_dense()
        noise = np.random.normal(0, std_noise, tt_full.shape)
        dense_data.append(PyDenseTensor(tt_full))
        dense_noisy_data.append(PyDenseTensor(tt_full + noise))

    return dense_noisy_data, dense_data


def compute_fit(ground_truths, tests, n_trials, r, num_sweeps, J):
    count = 0
    for ground_truth in ground_truths:
        count += 1
        for test in tests:
            fit = []
            total_time = []
            outfile = f'outputs/dense_tt/synthetic/{std_noise}_{test}_data{count}_rank{r_true}.json'
            directory = os.path.dirname(outfile)
            if not os.path.exists(directory):
                os.makedirs(directory)
            for i in range(n_trials):
                print(f"Starting trial {i}...")
                start = time.time()
                fit.append(fit_data(ground_truth, test, r, num_sweeps, J))
                end = time.time()
                total_time.append(end - start)

            mean_fit = np.mean(fit)
            mean_time = np.mean(total_time)

            result = {
                "data": ground_truth.data.shape,
                "test_name": test,
                "time": mean_time,
                "fit": mean_fit,
                "count": count,
            }

            with open(outfile, 'w') as f:
                json.dump(result, f)


if __name__ == '__main__':
    n_trials = 1
    N = 3
    dims = list(range(100, 201, 100))
    r = 5
    r_true = 5
    std_noise = 10
    ground_truths_noisy, ground_truths = generate_tt_full_tensors(r_true, N, dims, std_noise)
    tests = ["tt-svd", "randomized_tt-svd", "tt-als", "randomized_tt-als"]
    compute_fit(ground_truths_noisy, tests, n_trials, r, 1, 100000)
    # compute_fit(ground_truths, tests, n_trials, r, 1, 100000)
