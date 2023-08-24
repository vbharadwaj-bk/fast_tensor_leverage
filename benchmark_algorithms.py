import numpy as np
import json

from tt_cross import *
from quantization import *

from function_tensor import *
from tensor_train import TensorTrain
from functions import *
import time

def produce_trace_tt_cross(ground_truth, tt_approx, sweep_count):
    info, cache = {}, {}

    m         = 8.E+3  # Number of calls to target function
    e         = None   # Desired accuracy
    dr_min    = 0      # Cross parameter (minimum number of added rows)
    dr_max    = 0      # Cross parameter (maximum number of added rows)

    lstsq_problem_numbers = []
    lstsq_fits    = []
    lstsq_sample_counts = []

    lstsq_problem_number = 1

    def wrapped_func(I):
        return ground_truth.evaluate_indices(I)

    def step_callback(Y, i, R, direction):
        nonlocal lstsq_problem_number

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

        approx_fit = ground_truth.compute_approx_tt_fit(tt_approx)
        lstsq_problem_number += 1

        lstsq_problem_numbers.append(lstsq_problem_number)
        lstsq_fits.append(approx_fit)
        print(approx_fit)

    Y = tt_approx.U
    tt_approx.build_fast_sampler(0, 100)
    Y = cross(wrapped_func, Y, m, e, nswp, dr_min=dr_min, dr_max=dr_max,
        info=info, cache=cache, step_cb=step_callback)

    return lstsq_problem_numbers, lstsq_fits, lstsq_sample_counts


if __name__=='__main__':
    lbound = 0.001
    ubound = 25
    func = sin_test
    N = 1
    grid_bounds = np.array([[lbound, ubound] for _ in range(N)], dtype=np.double)
    subdivs = [2 ** 10] * N

    tt_rank   = 4      # TT-rank of the initial tensor
    nswp      = 1      # Sweep number

    quantization = Power2Quantization(subdivs, ordering="canonical")
    tt_approx = TensorTrain(quantization.qdim_sizes, [tt_rank] * (quantization.qdim - 1))

    ground_truth = FunctionTensor(grid_bounds, subdivs, func, quantization=quantization, track_evals=False)
    ground_truth.initialize_accuracy_estimation(method="randomized", 
                                                rsample_count=10000)

    result = produce_trace_tt_cross(ground_truth, tt_approx, 1)
    print(result)
