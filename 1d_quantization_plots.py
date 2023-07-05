import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt

import cppimport
import cppimport.import_hook

from tensor_train import *
from tt_als import *
from function_tensor import *

def create_plot(func, lbound, ubound, tt_approx, func_tensor, name):
    f_plot_x = np.linspace(lbound, ubound, 1000)
    f_plot_y = func(np.array([f_plot_x]).T).squeeze()

    num_points = func_tensor.dims[0]
    idxs = np.zeros((num_points, 1), dtype=np.uint64)
    idxs[:, 0] = np.arange(num_points)
    idxs_quant = func_tensor.quantization.quantize_indices(idxs)

    approx_x = func_tensor.indices_to_spatial_points(idxs).squeeze()
    approx_y = tt_approx.evaluate_at_idxs(idxs_quant).squeeze()

    fig, ax = plt.subplots()
    ax.plot(f_plot_x, f_plot_y, c='black')
    ax.plot(approx_x, approx_y, c='orange')

    # Save Figure
    fig.savefig(f'plotting/quantization_experiments/{name}')

def test_qtt_interpolation_points():
    def sin_test(idxs):
        return np.sin(idxs[:, 0]) / idxs[:, 0]

    lbound = 0.001
    ubound = 25
    func = sin_test

    J = 200
    tt_rank = 4
    n = 2 ** 10
    N = 1
    grid_bounds = np.array([[lbound, ubound] for _ in range(N)], dtype=np.double)
    subdivs = [n] * N

    quantization = Power2Quantization(subdivs, ordering="canonical")
    ground_truth = FunctionTensor(grid_bounds, subdivs, func, quantization=quantization)
    tt_approx = TensorTrain(quantization.qdim_sizes, [tt_rank] * (quantization.qdim - 1))

    tt_approx.place_into_canonical_form(0)
    tt_approx.build_fast_sampler(0, J=J)
    tt_als = TensorTrainALS(ground_truth, tt_approx)
    ground_truth.initialize_accuracy_estimation()

    print(tt_als.compute_approx_fit())
    tt_als.execute_randomized_als_sweeps(num_sweeps=5, J=J, epoch_interval=1, accuracy_method="approx")
    create_plot(func, lbound, ubound, tt_approx, ground_truth, name="plot1.png")

if __name__=='__main__':
    test_qtt_interpolation_points()
