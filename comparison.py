import numpy as np
from tensor_train import *
from dense_tensor import *
from tt_als import *
from functions import *
import teneva
from time import perf_counter as tpc
np.random.seed(42)

def test_tt_cross(d,cores,r,func_names,nswp,J):
    tt_cross_fit_results = {}
    rand_als_fit_results = {}
    dims = [d] * cores
    ranks = [r] * (cores- 1)
    tt_approx = TensorTrain(dims,ranks)
    tt_approx.place_into_canonical_form(0)
    initialize = tt_approx.U

    I_idx = np.vstack([np.random.choice(k, 100) for k in n]).T

    for name in func_names:
        for i in range(nswp):
            print(f"Starting sweep {i}...")
            func = getattr(Functions, name)
            tt_cross = teneva.cross(func, initialize, nswp)
            tt_cross = teneva.truncate(tt_cross)
            func_values = func(I_idx)
            tt_cross_values = teneva.get_many(tt_cross, I_idx)
            fit = 1 - teneva.act_two.accuracy(tt_cross_values, func_values)
            tt_cross_fit_results[f"tt_cross_fit_{name}"] = fit

        full_tt_cross = teneva.full(tt_cross)
        ground_truth = PyDenseTensor(full_tt_cross)
        tt_als = TensorTrainALS(ground_truth, tt_approx)
        tt_als.execute_exact_als_sweeps_slow(nswp)
        tt_approx.build_fast_sampler(0, J=J)
        tt_als.execute_randomized_als_sweeps(num_sweeps=nswp, J=J)
        rand_als_fit_results[f"rand_als_fit_{name}"] = tt_als.compute_exact_fit()

    return tt_cross_fit_results, rand_als_fit_results

tt_cross_fit_results, rand_als_fit_results = test_tt_cross(100,4,8,['schaffer','sine'],5,10000)


print(tt_cross_fit_results)
print(rand_als_fit_results)















