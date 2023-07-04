import numpy as np
import numpy.linalg as la

import logging, sys

import cppimport
import cppimport.import_hook


from tensor_train import *
from tt_als import *
from sparse_tensor import *
from function_tensor import *
from tensor_io.torch_tensor_loader import get_torch_tensor
from tensor_io.amrex_tensor_loader import get_amrex_single_plot_tensor

def test_tt_sampling(I=20, R=4, N=3, J=10000, seed=20, test_direction="left"): 
    '''
    Test that our algorithm can match the true leverage
    score distribution from the left and right subchains
    '''
    import matplotlib.pyplot as plt

    dims = [I] * N 
    ranks = [R] * (N-1) 
    tt = TensorTrain(dims, ranks, seed)

    if test_direction == "left":
        tt.place_into_canonical_form(N-1)
        tt.build_fast_sampler(N-1, J)
        samples = tt.leverage_sample(j=N-1, J=J, direction="left")
        linear_idxs = np.array(tt.linearize_idxs_left(samples), dtype=np.int64)
        left_chain = tt.left_chain_matricize(N-1)
        normsq_rows_left = la.norm(left_chain, axis=1) ** 2
        normsq_rows_normalized_left = normsq_rows_left / np.sum(normsq_rows_left)
        true_dist = normsq_rows_normalized_left
    else:
        tt.place_into_canonical_form(0)
        tt.build_fast_sampler(0, J)
        samples = tt.leverage_sample(j=0, J=J, direction="right")
        linear_idxs = np.array(tt.linearize_idxs_right(samples), dtype=np.int64)
        right_chain = tt.right_chain_matricize(0)
        normsq_rows_right = la.norm(right_chain, axis=1) ** 2 
        normsq_rows_normalized_right = normsq_rows_right / np.sum(normsq_rows_right)
        true_dist = normsq_rows_normalized_right

    fig, ax = plt.subplots()
    ax.plot(true_dist, label="True leverage distribution")
    bins = np.array(np.bincount(linear_idxs, minlength=len(true_dist))) / J

    ax.plot(bins, label="Our sampler")
    ax.set_xlabel("Row Index")
    ax.set_ylabel("Probability Density")
    ax.grid(True)
    ax.legend()
    ax.set_title(f"{J} samples drawn by our method vs. true distribution")
    fig.savefig('plotting/distribution_comparison.png')

def test_tt_als(I=20, R=4, N=3, J=10000):
    '''
    Test that TT-ALS can recover a tensor with a known
    low TT-rank representation.
    '''
    I = 100
    R = 8
    N = 4

    tt_approx = TensorTrain([I] * N, [R] * (N - 1))
    tt_approx_GT = TensorTrain([I] * N, [R] * (N - 1))
    ground_truth = PyDenseTensor(tt_approx_GT.materialize_dense()) 

    tt_approx.place_into_canonical_form(0)
    tt_als = TensorTrainALS(ground_truth, tt_approx)

    print(tt_als.compute_exact_fit())
    tt_approx.build_fast_sampler(0, J=J)
    tt_als.execute_randomized_als_sweeps(num_sweeps=10, J=J)

def test_image_feature_extraction(dataset="mnist", R=10, J=10000):
    ground_truth = get_torch_tensor(dataset)
    print("Loaded dataset...")
    tt_approx = TensorTrain(ground_truth.shape, 
        [R] * (ground_truth.N - 1))

    tt_approx.place_into_canonical_form(0)
    tt_als = TensorTrainALS(ground_truth, tt_approx)

    print(tt_als.compute_exact_fit())
    tt_approx.build_fast_sampler(0, J=J)
    tt_als.execute_randomized_als_sweeps(num_sweeps=20, J=J)

def test_norm_computation():
    I = 100
    R = 2
    N = 4

    tt_approx = TensorTrain([I] * N, [R] * (N - 1))
    tt_approx.place_into_canonical_form(0)

    full_materialized = tt_approx.materialize_dense()
    norm_materialized = la.norm(full_materialized)
    norm_test = la.norm(tt_approx.U[0])

    print(f'Norm of materialized tensor: {norm_materialized}')
    print(f'Norm test: {norm_test}')

def test_dense_recovery():
    '''
    Test that TT-ALS can recover a tensor with a known
    low TT-rank representation.
    '''
    I = 3
    R = 2
    N = 3
    J = 1000

    tt_approx = TensorTrain([I] * N, [R] * (N - 1))

    data = np.zeros((I, I, I), dtype=np.double)
    for i in range(I):
        for j in range(I):
            for k in range(I):
                data[i, j, k] = i + j + k

    ground_truth = PyDenseTensor(data)

    tt_approx.place_into_canonical_form(0)
    tt_als = TensorTrainALS(ground_truth, tt_approx)

    print(tt_als.compute_exact_fit())
    tt_approx.build_fast_sampler(0, J=J)
    tt_als.execute_randomized_als_sweeps(num_sweeps=10, J=J)


def test_sparse_tensor_decomposition(tensor_name="uber", R=10, J=65000):
    param_map = {
        "uber": {
            "preprocessing": None,
            "initialization": None
        },
        "enron": {
            "preprocessing": "log_count",
            "initialization": None
        },
        "nell-2": {
            "preprocessing": "log_count",
            "initialization": None
        },
        "amazon-reviews": {
            "preprocessing": None, 
            "initialization": None
        }
    }

    preprocessing = param_map[tensor_name]["preprocessing"] 
    initialization = param_map[tensor_name]["initialization"]    
    ground_truth = PySparseTensor(f"/pscratch/sd/v/vbharadw/tensors/{tensor_name}.tns_converted.hdf5", lookup="sort", preprocessing=preprocessing)

    # ranks = [R] * (ground_truth.N - 1)
    ranks = [R, R+1, R]

    print("Loaded dataset...")
    tt_approx = TensorTrain(ground_truth.shape, ranks)

    tt_approx.place_into_canonical_form(0)
    tt_approx.build_fast_sampler(0, J=J)
    tt_als = TensorTrainALS(ground_truth, tt_approx)

    tt_als.execute_randomized_als_sweeps(num_sweeps=20, J=J, epoch_interval=5)

def print_tensor_param_counts(dims, rank_cp, rank_tt):
    dims = np.array(dims)
    cp_param_count = np.sum(dims) * rank_cp
    tt_param_count = np.sum(dims[1:-1]) * rank_tt * rank_tt + (dims[0] + dims[-1]) * rank_tt
    print(f"CP param count: {cp_param_count}")
    print(f"TT param count: {tt_param_count}")

def test_function_tensor_decomposition():
    def slater_function(idxs):
        norms = np.sqrt(np.sum(idxs ** 2, axis=1))
        return np.exp(-norms) / norms 
        #return np.sin((np.sum(idxs, axis=1)))

    J = 200
    tt_rank = 4
    L = 10.0
    N = 3
    subdivs_per_dim = 1000
    grid_bounds = np.array([[1.0, L] for _ in range(N)], dtype=np.double)
    subdivisions = [subdivs_per_dim] * N
    ground_truth = FunctionTensor(grid_bounds, subdivisions, slater_function)

    #idxs_test = np.array([[1, 1, 1]], dtype=np.uint64)
    #observation = ground_truth.compute_observation_matrix(idxs_test, 2)
    #print(observation)
    #exit(1)

    tt_approx = TensorTrain(subdivisions, [tt_rank] * (N - 1))
    tt_approx.place_into_canonical_form(0)
    tt_approx.build_fast_sampler(0, J=J)
    tt_als = TensorTrainALS(ground_truth, tt_approx)
    ground_truth.initialize_accuracy_estimation()

    print(tt_als.compute_approx_fit())
    tt_als.execute_randomized_als_sweeps(num_sweeps=5, J=J, epoch_interval=1, accuracy_method="approx")


def test_amrex_decomposition(
        filepath="/pscratch/sd/a/ajnonaka/rtil/data/plt0004600",
        J=65000,
        R=30):
    ground_truth = get_amrex_single_plot_tensor(filepath)
    print("Loaded dataset...")
    tt_approx = TensorTrain(ground_truth.shape, 
        [R] * (ground_truth.N - 1))

    tt_approx.place_into_canonical_form(0)
    tt_als = TensorTrainALS(ground_truth, tt_approx)

    print(tt_als.compute_exact_fit())
    tt_approx.build_fast_sampler(0, J=J)
    tt_als.execute_randomized_als_sweeps(num_sweeps=30, J=J)

    tt_full = tt_approx.materialize_dense() 
    sl = tt_full[128, :, :]

    # Plot sl and save to a png file
    fig, ax = plt.subplots()
    ax.imshow(sl)
    fig.savefig('outputs/charge_tt_comparison.png')

def test_quantization():
    quantization = Power2Quantization([2 ** 3] * 3, ordering="canonical")
    indices = np.array([[3, 4, 7]], dtype=np.uint64)
    q_indices = quantization.quantize_indices(indices)
    print(q_indices)
    recovered_indices = quantization.unquantize_indices(q_indices)
    print(recovered_indices)

def test_qtt_interpolation_points():
    def slater_function(idxs):
        norms = np.sqrt(np.sum(idxs ** 2, axis=1))
        return np.exp(-norms) / norms 
        #return np.sin((np.sum(idxs, axis=1)))

if __name__=='__main__':
    #test_sparse_tensor_decomposition() 
    #test_dense_recovery()
    #test_tt_als()

    #print_tensor_param_counts([60000, 28, 28], 
    #    rank_cp=25, rank_tt=21)
    #test_image_feature_extraction()

    #test_function_tensor_decomposition()

    #test_amrex_decomposition(
    #    filepath="/pscratch/sd/a/ajnonaka/rtil/data/plt0004600",
    #    J=10000)
    test_quantization()
