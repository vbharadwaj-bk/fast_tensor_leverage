import numpy as np
import json

from tt_cross import *
from quantization import *

from function_tensor import *
from tensor_train import TensorTrain
from functions import *
import time

def deduplicate_and_add_evaluations(existing_evals, new_eval_list):
    if existing_evals is not None:
        new_eval_list.append(existing_evals)

    if len(new_eval_list) > 0:
        stacked = np.vstack(new_eval_list)
        return np.unique(stacked, axis=0)
    else:
        return existing_evals

def produce_trace_tt_cross(ground_truth, tt_approx, nswp, outfle):
    info, cache = {}, {}

    m         = None # Number of calls to target function
    e         = None   # Desired accuracy
    dr_min    = 0      # Cross parameter (minimum number of added rows)
    dr_max    = 0      # Cross parameter (maximum number of added rows)

    lstsq_problem_numbers = []
    fits = []
    unique_eval_counts = []
    lstsq_problem_number = 1

    evaluations = None 

    def wrapped_func(I):
        return ground_truth.evaluate(I)

    def step_callback(Y, i, R, direction, is_lstq_problem=True):
        nonlocal lstsq_problem_number
        nonlocal evaluations

        # Note - I switched the order of tensordot
        if direction == "left":
            #tt_approx.U[i] = np.tensordot(Y[i], R, 1)
            tt_approx.U[i] = Y[i].copy()
            tt_approx.update_internal_sampler(i, direction, False)
            if i < len(Y) - 1:
                tt_approx.U[i+1] = Y[i+1].copy()
                tt_approx.update_internal_sampler(i+1, direction, False)

        elif direction == "right":
            #tt_approx.U[i] = np.tensordot(R, Y[i], 1)
            tt_approx.U[i] = Y[i].copy()
            tt_approx.update_internal_sampler(i, direction, False)

            if i > 0: 
                tt_approx.U[i-1] = Y[i-1].copy()
                tt_approx.update_internal_sampler(i-1, direction, False)

        if is_lstq_problem:
            approx_fit = ground_truth.compute_approx_tt_fit(tt_approx)

            lstsq_problem_numbers.append(lstsq_problem_number)
            fits.append(approx_fit)

            evaluations = deduplicate_and_add_evaluations(evaluations, ground_truth.evals)
            if evaluations is None:
                unique_eval_counts.append(0)
            else:
                unique_eval_counts.append(evaluations.shape[0])
            ground_truth.evals = []
            lstsq_problem_number += 1

    Y = tt_approx.U
    tt_approx.build_fast_sampler(0, 100)
    Y = cross(wrapped_func, Y, m, e, nswp, dr_min=dr_min, dr_max=dr_max,
        info=info, cache=cache, step_cb=step_callback)

    result = {
        "lstsq_problem_numbers": lstsq_problem_numbers,
        "fits": fits,
        "unique_eval_counts": unique_eval_counts
    }

    with open(outfle, 'w') as f:
        json.dump(result, f)

if __name__=='__main__':
    lbound = 0.001
    ubound = 25

    test_name = "sin1"
    func = sin_test
    N = 1
    grid_bounds = np.array([[lbound, ubound] for _ in range(N)], dtype=np.double)
    subdivs = [2 ** 10] * N

    tt_rank   = 4      # TT-rank of the initial tensor
    nswp      = 3      # Sweep number

    quantization = Power2Quantization(subdivs, ordering="canonical")
    tt_approx = TensorTrain(quantization.qdim_sizes, [tt_rank] * (quantization.qdim - 1))

    ground_truth = FunctionTensor(grid_bounds, subdivs, func, quantization=quantization, track_evals=True)
    ground_truth.initialize_accuracy_estimation(method="randomized", 
                                                rsample_count=10000)

    outfile = f'outputs/tt_benchmarks/cross_{test_name}.json'
    produce_trace_tt_cross(ground_truth, tt_approx, nswp, outfile)