import json
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

import logging, sys
import time


# current_script_path = os.path.dirname(os.path.abspath(__file__))
# fast_tensor_leverage_path = os.path.dirname(current_script_path)
# sys.path.append(fast_tensor_leverage_path)

from tensor_train import *
from dense_tensor import *
from tt_als import *
from coil100_data_loader import *
from tt_svd import *

def error(ground_truth, approx):
    error = np.linalg.norm(approx - ground_truth) / np.linalg.norm(ground_truth)
    return error

def feature_extraction_task(dataset, r, j):
    tests = ["tt-svd", "randomized_tt-svd", "tt-als", "randomized_tt-als"]
    print("Loaded dataset...")
    ground_truth, labels = get_coil_tensor(dataset)
    # Permutation is just for coil-100 dataset.
    unfolded_tensor = np.transpose(ground_truth.data, (1, 2, 3, 0))
    # x_train, x_test, y_train, y_test = train_test_split(ground_truth.data, labels, train_size=0.5, random_state=42)

    output_dir = os.path.join(fast_tensor_leverage_path,'outputs', 'dense_tt', 'coil100')
    os.makedirs(output_dir, exist_ok=True)

    for test in tests:
        total_error = []
        outfile = os.path.join(output_dir, f'{test}.json')
        start = time.time()

        if test == "tt-svd":
            tensor_train_svd = TensorTrainSVD(unfolded_tensor, r)
            cores = tensor_train_svd.tt_svd()
            approx = tt_to_tensor(cores)
            print(approx.shape)
            print(unfolded_tensor.shape)

            error_result = error(unfolded_tensor, approx)

            cores[-1] = cores[-1].transpose((0, 1, 2))
            print(cores[-1].shape)

        elif test == "randomized_tt-svd":
            tensor_train_svd = TensorTrainSVD(unfolded_tensor, r)
            cores = tensor_train_svd.tt_randomized_svd()
            approx = tt_to_tensor(cores)
            error_result = error(unfolded_tensor, approx)

            cores[-1] = cores[-1].transpose((0, 1, 2))

        elif test == "tt-als":
            tt_approx = TensorTrain(ground_truth.shape, r, init_method="tt-svd", input_tensor=ground_truth)  # gaussian/svd-based initialization
            tt_approx.place_into_canonical_form(0)
            tt_als = TensorTrainALS(ground_truth, tt_approx)
            tt_als.execute_exact_als_sweeps_slow(num_sweeps=5)
            error_result = 1-tt_als.compute_exact_fit()
            # total_error.append(error_result)
            u = tt_als.tt_approx.U[0].transpose((0, 1, 2))
            u = u.transpose((0, 1, 2))
            cores = [u]

        elif test == "randomized_tt-als":  # proposal
            print("Starting rTT-ALS...")
            tt_approx = TensorTrain(ground_truth.shape, r, init_method="randomized_tt-svd",
                                input_tensor=ground_truth)
            tt_approx.place_into_canonical_form(0)
            tt_als = TensorTrainALS(ground_truth, tt_approx)

            tt_approx.build_fast_sampler(0, J=j)
            tt_als.execute_randomized_als_sweeps(num_sweeps=5, J=j)
            error_result = 1-tt_als.compute_exact_fit()
            # total_error.append(error_result)

            u = tt_als.tt_approx.U[0].transpose((0, 1, 2))
            u = u.transpose((0, 1, 2))
            cores = [u]

            # last_core = cores[-1].reshape(cores[-1].shape[1], cores[-1].shape[0] * cores[-1].shape[2])
        else:
            continue
        end = time.time()
        #
        # if test == "randomized_svd":
        #     last_core = cores
        # else:
        last_core = cores[-1].reshape(cores[-1].shape[1], cores[-1].shape[0] * cores[-1].shape[2])
        print(last_core.shape)
        # print(labels.shape)
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(last_core, labels)
        cv_scores = cross_val_score(knn, last_core, labels, cv=10, scoring='accuracy')
        accuracy = np.mean(cv_scores)

        total_time = end - start

        result = {
            "test_name": test,
            "time": total_time,
            "acc": accuracy,
            "error": error_result.item(),
        }
        with open(outfile, 'w') as f:
            json.dump(result, f)


if __name__ == '__main__':
    j = 1000
    feature_extraction_task("coil-100", 5, j)
