import json
import os
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import logging, sys
import time

# current_script_path = os.path.dirname(os.path.abspath(__file__))
# fast_tensor_leverage_path = os.path.dirname(current_script_path)
# sys.path.append(fast_tensor_leverage_path)

from tensor_train import *
from dense_tensor import *
from tns_tt_test import *
from tensorly.decomposition import tensor_train
from tt_als import *
from coil100_data_loader import *
from tt_svd import *


def error(ground_truth, approx):
    err = np.linalg.norm(approx - ground_truth) / np.linalg.norm(ground_truth)
    return err


def feature_extraction_task(dataset, r, j, J):
    tests = ["tns-tt", "tt-svd", "randomized_tt-svd", "tt-als", "randomized_tt-als"]
    print("Loaded dataset...")
    ground_truth, labels = get_coil_tensor(dataset)
    N = ground_truth.N
    # dims = ground_truth.shape
    # Permutation is just for coil-100 dataset.
    unfolded_tensor = np.transpose(ground_truth.data, (1, 2, 3, 0))
    unfolded_tensor_dense = PyDenseTensor(unfolded_tensor)
    print(unfolded_tensor_dense.data.shape)
    dims = list(unfolded_tensor.shape)
    ranks = [1] + [r] * (N - 1) + [1]
    # x_train, x_test, y_train, y_test = train_test_split(ground_truth.data, labels, train_size=0.5, random_state=42)

    tt_cores_svd = None
    tt_cores_rsvd = None

    for test in tests:
        outfile = f'outputs/dense_tt/{dataset}/{test}-{r}.json'
        directory = os.path.dirname(outfile)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if test == "tns-tt":
            start = time.time()
            cores = tns_tt_test(unfolded_tensor_dense.data, r, j, nsweeps=100, init_method="gaussian", tt_decomp=None)
            end = time.time()
            total_time = end - start
            approx = tt_to_tensor(cores)
            error_result = error(unfolded_tensor, approx)

            cores[-1] = cores[-1].transpose((0, 1, 2))

        elif test == "tt-svd":
            start = time.time()
            cores = tl.decomposition.tensor_train(unfolded_tensor_dense.data, ranks, svd="truncated_svd", verbose=False)
            end = time.time()
            total_time = end - start
            approx = tt_to_tensor(cores)
            error_result = error(unfolded_tensor, approx)
            tt_cores_svd = cores

            cores[-1] = cores[-1].transpose((0, 1, 2))

        elif test == "randomized_tt-svd":
            start = time.time()
            cores = tl.decomposition.tensor_train(unfolded_tensor_dense.data, ranks, svd="randomized_svd",
                                                  verbose=False)
            end = time.time()
            total_time = end - start
            approx = tt_to_tensor(cores)
            error_result = error(unfolded_tensor, approx)
            tt_cores_rsvd = cores

            cores[-1] = cores[-1].transpose((0, 1, 2))

        elif test == "tt-als":
            print(f"Initialized tt-als ... ")
            if tt_cores_svd is None:
                raise ValueError("TT-SVD cores must be computed before TT-ALS")

            tt_approx = TensorTrain(dims, r, init_method=None, U=tt_cores_svd)
            tt_approx.place_into_canonical_form(0)
            start = time.time()
            tt_als = TensorTrainALS(unfolded_tensor_dense, tt_approx)
            tt_als.execute_exact_als_sweeps_slow(num_sweeps=5)
            end = time.time()
            total_time = (end - start)
            error_result = 1 - tt_als.compute_exact_fit()
            # total_error.append(error_result)
            u = tt_als.tt_approx.U[-1].transpose((0, 1, 2))
            u = u.transpose((0, 1, 2))
            print(u.shape)
            cores = [u]

        elif test == "randomized_tt-als":  # proposal
            if tt_cores_rsvd is None:
                raise ValueError("TT-SVD cores must be computed before TT-ALS")

            tt_approx = TensorTrain(dims, r, init_method=None, U=tt_cores_rsvd)
            tt_approx.place_into_canonical_form(0)
            start = time.time()
            tt_approx.build_fast_sampler(0, J=J)
            tt_als = TensorTrainALS(unfolded_tensor_dense, tt_approx)
            tt_als.execute_randomized_als_sweeps(num_sweeps=5, J=J)
            end = time.time()
            total_time = end - start
            error_result = 1 - tt_als.compute_exact_fit()
            # total_error.append(error_result)

            u = tt_als.tt_approx.U[-1].transpose((0, 1, 2))
            u = u.transpose((0, 1, 2))
            cores = [u]
        else:
            continue
        print(cores[-1].shape)
        last_core = cores[-1].reshape(cores[-1].shape[1], cores[-1].shape[0] * cores[-1].shape[2])
        print(last_core.shape)
        # print(labels.shape)
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(last_core, labels)
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        cv_scores = cross_val_score(knn, last_core, labels, cv=kf, scoring='accuracy')
        accuracy = 1 - np.mean(cv_scores)

        result = {
            "test_name": test,
            "time": total_time,
            "acc": accuracy,
            "error": error_result.item(),
        }
        with open(outfile, 'w') as f:
            json.dump(result, f)


if __name__ == '__main__':
    j = 10
    J = 10**3
    r = 10
    feature_extraction_task("coil-100", r, j, J)
