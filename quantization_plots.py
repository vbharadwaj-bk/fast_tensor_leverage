import numpy as np
import numpy.linalg as la

from celluloid import Camera 
import matplotlib.pyplot as plt

import cppimport
import cppimport.import_hook

from tensor_train import *
from tt_als import *
from function_tensor import *
from functions import *

def create_plot(func, lbound, ubound, tt_approx, func_tensor, eval_points, name, animate=None):
    eval_points = np.arange(0, func_tensor.dims[0], 1, dtype=np.uint64)
    f_plot_x = eval_points.astype(np.double) * func_tensor.dx[0] + func_tensor.grid_bounds[0, 0] 

    eval_points_mat = eval_points.reshape(eval_points.shape[0], 1) 
    f_plot_y = func_tensor.evaluate_indices(eval_points_mat).squeeze()

    noise_magnitude = 0.1
    noise = func_tensor.generate_reproducible_noise(eval_points_mat) * noise_magnitude 

    print(noise[8])
    noise2 = func_tensor.generate_reproducible_noise(np.array([[8]])) * noise_magnitude 
    print(noise2[0])

    exit(1)

    f_plot_y += noise

    num_points = func_tensor.dims[0]
    idxs = np.zeros((num_points, 1), dtype=np.uint64)
    idxs[:, 0] = np.arange(num_points)
    idxs_quant = func_tensor.quantization.quantize_indices(idxs)

    approx_x = func_tensor.indices_to_spatial_points(idxs).squeeze()
    approx_y = tt_approx.evaluate_at_idxs(idxs_quant).squeeze()

    fig, ax, camera = None, None, None

    if animate is None:
        fig, ax = plt.subplots()
    else:
        ax, camera= animate

    ax.plot(f_plot_x, f_plot_y, c='black')
    ax.plot(approx_x, approx_y, c='orange')

    if eval_points is not None:
        ax.scatter(eval_points[0], eval_points[1], marker='x', c='red')

    # Set y limits of plot from -0.5 to 1
    ax.set_ylim([-0.5, 1])

    if animate is None:
        fig.savefig(f'plotting/quantization_experiments/{name}')
        plt.close(fig)
    else:
        camera.snap()

def test_qtt_interpolation_points():
    lbound = 0.001
    ubound = 25
    func = sin2_test
    num_sweeps = 1
    tt_rank = 4
    n = 2 ** 10
    N = 1
    grid_bounds = np.array([[lbound, ubound] for _ in range(N)], dtype=np.double)
    subdivs = [n] * N

    alg = 'iid_leverage'
    J = 30000
    J2 = None

    #alg = 'reverse_iterative_volume'
    #J = 200
    #J2 = 32

    alg = 'teneva_rect_maxvol'
    J = 200
    J2 = 16

    quantization = Power2Quantization(subdivs, ordering="canonical")
    ground_truth = FunctionTensor(grid_bounds, subdivs, func, quantization=quantization, track_evals=True)
    tt_approx = TensorTrain(quantization.qdim_sizes, [tt_rank] * (quantization.qdim - 1))

    tt_approx.place_into_canonical_form(0)
    tt_approx.build_fast_sampler(0, J=J)
    tt_als = TensorTrainALS(ground_truth, tt_approx)
    ground_truth.initialize_accuracy_estimation()

    def extract_evaluation_points(tt_approx, func_tensor):
        eval_x, eval_y = [], []
        for eval_set in func_tensor.evals:
            x = list(func_tensor.indices_to_spatial_points(eval_set).squeeze())
            y = list(tt_approx.evaluate_at_idxs(eval_set, quantization=quantization).squeeze())
            eval_x.extend(x)
            eval_y.extend(y)

        func_tensor.evals = []
        return eval_x, eval_y

    progress = 0
    fig, ax = plt.subplots()
    camera = Camera(fig)

    for i in range(num_sweeps):
        print(f"Starting sweep {i}...")
        for j in range(tt_approx.N - 1):
            tt_als.optimize_core_approx(j, J, alg, J2)
            tt_approx.orthogonalize_push_right(j)
            tt_approx.update_internal_sampler(j, "left", True)
            tt_approx.update_internal_sampler(j+1, "left", False)
            progress += 1
            eval_points = extract_evaluation_points(tt_approx, ground_truth)
            create_plot(func, lbound, ubound, tt_approx, ground_truth, eval_points,
                            name=f"step_{progress}.png", animate=(ax, camera)) 

        for j in range(tt_approx.N - 1, 0, -1):
            tt_als.optimize_core_approx(j, J, alg, J2)
            tt_approx.orthogonalize_push_left(j)
            tt_approx.update_internal_sampler(j, "right", True)
            tt_approx.update_internal_sampler(j-1, "left", False)
            progress += 1
            eval_points = extract_evaluation_points(tt_approx, ground_truth)
            create_plot(func, lbound, ubound, tt_approx, ground_truth, eval_points,
                            name=f"step_{progress}.png", animate=(ax, camera)) 

        tt_approx.update_internal_sampler(0, "left", False)
        print(f'Fit: {tt_als.compute_approx_fit()}') 

    create_plot(func, lbound, ubound, tt_approx, ground_truth, ([], []),
                    name=f"final.png") 
    animation = camera.animate()  
    animation.save('plotting/quantization_experiments/animated.gif', writer = 'pillow')

if __name__=='__main__':
    test_qtt_interpolation_points()
