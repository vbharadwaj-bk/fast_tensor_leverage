import numpy as np
import numpy.linalg as la
import json
import matplotlib.pyplot as plt

from sklearn.utils.extmath import randomized_svd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


import logging, sys
import time

from tensors.tensor_train import *
from algorithms.tt_als import *
from coil100_data_loader import *
from tt_svd import *

N = 4
rho = [0.5,0.1]

def test_all_tt_dense_tensor(dataset, R, J):
    tests = ["tt-svd","rtt-svd","rtt-als","rsvd"]
    print("Loaded dataset...")
    ground_truth, labels = get_coil_tensor(dataset)
    print(labels.shape)
    # x_train, x_test, y_train, y_test = train_test_split(ground_truth.data, labels, train_size=0.5, random_state=42)

    for test in tests:
        outfile = f'outputs/dense_tt/{test}.json'
        start = time.time()
        u = None

        if test == "tt-svd":
            print("Starting TT-SVD...")
            tt_svd = TensorTrainSVD(ground_truth,R)
            cores = tt_svd.tt_regular_svd()
            u = cores[-1]

        elif test == "rtt-svd":
            print("Starting rTT-SVD...")
            cores = tt_svd.tt_randomized_svd()
            u = cores[-1]

        elif test == "rtt-als":
            print("Starting rTT-ALS...")
            tt_approx = TensorTrain(ground_truth.shape, [R] * (ground_truth.N - 1))
            tt_approx.place_into_canonical_form(0)

            tt_als = TensorTrainALS(ground_truth, tt_approx)

            # print(tt_als.compute_exact_fit())
            tt_approx.build_fast_sampler(0, J=j)
            tt_als.execute_randomized_als_sweeps(num_sweeps=1, J=j)
            u = tt_als.tt_approx.U[0]
            print(u.shape)
            u = u.transpose((2,1,0))
            print(u.shape)



        elif test == "rsvd":
            ground_truth_reshaped = ground_truth.data.reshape((7200,32*32*3))
            print("Starting rSVD...")
            u,s,v = randomized_svd(ground_truth_reshaped, R)
            print(u.shape)
            # last_core = np.reshape(np.transpose(cores[-1], (1, 0, 2)), (7200, -1))

        if u is not None and test != "rsvd":
            new_dim1, new_dim2 = u.shape[1], u.shape[0]*u.shape[2]
            last_core = u.reshape((new_dim1, new_dim2))
            print(last_core.shape)
        elif u is not None and test == "rsvd":
            last_core = u
        else:
            continue

        end = time.time()

        knn = KNeighborsClassifier(n_neighbors=1)
        cv_scores = cross_val_score(knn, last_core, labels, cv=10, scoring='accuracy')
        accuracy = np.mean(cv_scores)
        total_time = end - start

        result = {
            "J": J,
            "test_name": test,
            "time": total_time,
            "acc": accuracy,
        }
        with open(outfile, 'w') as f:
            json.dump(result, f)

if __name__ == '__main__':
    j = 2000
    test_all_tt_dense_tensor("coil-100", 5, j)
