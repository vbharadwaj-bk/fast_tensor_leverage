import numpy as np
import os
import json
import sys
from tt_svd import *
from relative_error import *
import scikit_tt
from scikit_tt.solvers import sle

current_script_path = os.path.dirname(os.path.abspath(__file__))
fast_tensor_leverage_path = os.path.dirname(current_script_path)
sys.path.append(fast_tensor_leverage_path)
from tensors.dense_tensor import *
from algorithms.tt_als import *
from tensors.tensor_train import *


def fit(ground_truth, approx):
    fit = 1 - np.linalg.norm(approx-ground_truth) / np.linalg.norm(ground_truth)
    return fit


def fit_data(ground_truth, test, r, num_sweeps, J):
    fit_result = None
    if test == "tt-svd":
        tensor_train_svd = TensorTrainSVD(ground_truth.data, r)
        cores = tensor_train_svd.tt_svd()
        approx = tt_to_tensor(cores)
        fit_result = fit(ground_truth.data, approx)

    elif test == "randomized_tt-svd":
        tensor_train_svd = TensorTrainSVD(ground_truth.data, r)
        cores = tensor_train_svd.tt_randomized_svd()
        approx = tt_to_tensor(cores)
        fit_result = fit(ground_truth.data, approx)

    elif test == "tt-als":
        tt_approx = TensorTrain(ground_truth.shape, [r] * (ground_truth.N - 1))
        tt_approx.place_into_canonical_form(0)
        tt_als = TensorTrainALS(ground_truth, tt_approx)

        print(tt_als.compute_exact_fit())
        tt_als.execute_exact_als_sweeps_slow(num_sweeps)
        fit_result = tt_als.compute_exact_fit()
        print(fit_result)

    elif test == "randomized_tt-als":  # proposal
        tt_approx = TensorTrain(ground_truth.shape, [r] * (ground_truth.N - 1))
        tt_approx.place_into_canonical_form(0)
        tt_als = TensorTrainALS(ground_truth, tt_approx)

        print(tt_als.compute_exact_fit())
        tt_approx.build_fast_sampler(0, J=J)
        tt_als.execute_randomized_als_sweeps(num_sweeps=num_sweeps, J=J)
        fit_result = tt_als.compute_exact_fit()
        print(fit_result)

    return fit_result


