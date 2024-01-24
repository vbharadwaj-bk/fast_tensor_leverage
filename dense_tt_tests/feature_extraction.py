import json
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

import logging, sys
import time

current_script_path = os.path.dirname(os.path.abspath(__file__))
fast_tensor_leverage_path = os.path.dirname(current_script_path)
sys.path.append(fast_tensor_leverage_path)

from tensors.tensor_train import *
from tensors.dense_tensor import *
from algorithms.tt_als import *
from coil100_data_loader import *
from tt_svd import *


def feature_extraction_task(dataset, r, j, n_iter):
    tests = ["tt-svd", "randomized_tt-svd", "randomized_tt-als", "randomized_svd"]
    print("Loaded dataset...")
    ground_truth, labels = get_coil_tensor(dataset)
    print(type(ground_truth))
    # Permutation is just for coil-100 dataset.
    unfolded_tensor = np.transpose(ground_truth.data, (1, 2, 3, 0))
    # x_train, x_test, y_train, y_test = train_test_split(ground_truth.data, labels, train_size=0.5, random_state=42)

    output_dir = os.path.join(fast_tensor_leverage_path, 'outputs', 'dense_tt', 'coil100')
    os.makedirs(output_dir, exist_ok=True)

    for test in tests:
        outfile = os.path.join(output_dir, f'{test}.json')
        start = time.time()

        if test == "tt-svd":
            tensor_train_svd = TensorTrainSVD(unfolded_tensor, r)
            cores = tensor_train_svd.tt_svd()
            cores[-1] = cores[-1].transpose((2, 1, 0))

        elif test == "randomized_tt-svd":
            tensor_train_svd = TensorTrainSVD(unfolded_tensor, r)
            cores = tensor_train_svd.tt_randomized_svd()
            cores[-1] = cores[-1].transpose((2, 1, 0))

        elif test == "randomized_tt-als":  # proposal
            print("Starting rTT-ALS...")
            tt_approx = TensorTrain(ground_truth.shape, [r] * (ground_truth.N - 1))
            tt_approx.place_into_canonical_form(0)
            tt_als = TensorTrainALS(ground_truth, tt_approx)

            tt_approx.build_fast_sampler(0, J=j)
            tt_als.execute_randomized_als_sweeps(num_sweeps=n_iter, J=j)

            u = tt_als.tt_approx.U[0].transpose((2, 1, 0))
            u = u.transpose((2, 1, 0))
            cores = [u]

        elif test == "randomized_svd":
            ground_truth_reshaped = ground_truth.data.reshape((7200, 32 * 32 * 3))
            print("Starting rSVD...")
            cores, s, v = randomized_svd(ground_truth_reshaped, r)

        else:
            continue
        end = time.time()

        if test == "randomized_svd":
            last_core = cores
        else:
            last_core = cores[-1].reshape(cores[-1].shape[1], cores[-1].shape[0] * cores[-1].shape[2])

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(last_core, labels)
        cv_scores = cross_val_score(knn, last_core, labels, cv=10, scoring='accuracy')
        accuracy = np.mean(cv_scores)

        total_time = end - start

        result = {
            "n_iter": n_iter,
            "test_name": test,
            "time": total_time,
            "acc": accuracy,
        }
        with open(outfile, 'w') as f:
            json.dump(result, f)


if __name__ == '__main__':
    n_iter = 10
    j = 2000
    feature_extraction_task("coil-100", 5, j, n_iter)
