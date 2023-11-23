import numpy as np
import numpy.linalg as la
import json
import matplotlib.pyplot as plt

from sklearn.utils.extmath import randomized_svd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import tensorly as tl

import logging, sys
import time

import cppimport
import cppimport.import_hook

from tensors.tensor_train import *
from algorithms.tt_als import *
from coil100_data_loader import *
from tt_svd import *

N = 4
rho = [0.5,0.1]

def test_all_tt_dense_tensor(dataset, R, J):
    tests = ["tt-svd","rtt-als","rsvd"]
    print("Loaded dataset...")
    ground_truth, labels = get_coil_tensor(dataset)
    # x_train, x_test, y_train, y_test = train_test_split(ground_truth.data, labels, train_size=0.5, random_state=42)

    for test in tests:
        outfile = f'outputs/dense_tt/{test}_J_{j}.json'
        start = time.time()
        if test == "tt-svd":
            print("Starting SVD...")
            tt_svd = TensorTrainSVD(ground_truth)
            cores = tt_svd.tt_regular_svd(R)
        elif test == "rtt-als":
            tt_approx = TensorTrain(ground_truth.shape, [R] * (ground_truth.N - 1))
            tt_approx.place_into_canonical_form(0)

            tt_als = TensorTrainALS(ground_truth, tt_approx)

            # print(tt_als.compute_exact_fit())
            tt_approx.build_fast_sampler(0, J=j)
            tt_als.execute_randomized_als_sweeps(num_sweeps=1, J=j)
            # last_core = np.reshape(cores[-1], (7200, -1))


        elif test == "rsvd":
            ground_truth_reshaped = ground_truth.data.reshape((7200,32*32*3))
            print("Starting rSVD...")
            u,s,v = randomized_svd(ground_truth_reshaped, R)


            # last_core = np.reshape(np.transpose(cores[-1], (1, 0, 2)), (7200, -1))

        else:
            continue
        end = time.time()
        core_dims = cores[-1].shape
        new_dim1 = core_dims[1]
        new_dim2 = core_dims[0] * core_dims[2]
        last_core = np.reshape(np.transpose(cores[-1], (1, 0, 2)), (new_dim1, new_dim2))
        if test == "rsvd":
            last_core = u
        if last_core.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Shape mismatch: last_core has {last_core.shape[0]} samples, labels has {labels.shape[0]} samples")
        knn = KNeighborsClassifier(n_neighbors=1)
        cv_scores = cross_val_score(knn, last_core, labels, cv=10, scoring='accuracy')
        accuracy = np.mean(cv_scores)
        total_time = end - start
        result = {
            "J": J,
            "test_name": test,
            "time": total_time,
            "acc": accuracy,
            "loss": 1-accuracy
        }
        with open(outfile, 'w') as f:
            json.dump(result, f)


if __name__ == '__main__':
    j = 2000
    test_all_tt_dense_tensor("coil-100", R = 5, J = j)




# core_transpose = np.transpose(cores[-1], axes=[1, 0, 2]).reshape(7200,5)
