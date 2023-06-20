import numpy as np
import numpy.linalg as la

import logging, sys

import cppimport
import cppimport.import_hook


from tensor_train import *
from tt_als import *
from sparse_tensor import *
from tensor_io.torch_tensor_loader import get_torch_tensor

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
    tt_als.execute_randomized_als_sweeps(num_sweeps=5, J=J)

def test_image_feature_extraction(dataset="mnist", R=14, J=20000):
    ground_truth = get_torch_tensor(dataset)
    print("Loaded dataset...")
    tt_approx = TensorTrain(ground_truth.shape, 
        [R] * (ground_truth.N - 1))

    tt_approx.place_into_canonical_form(0)
    tt_als = TensorTrainALS(ground_truth, tt_approx)

    print(tt_als.compute_exact_fit())
    tt_approx.build_fast_sampler(0, J=J)
    tt_als.execute_randomized_als_sweeps(num_sweeps=10, J=J)

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


def test_sparse_tensor_decomposition(tensor_name="uber", R=2, J=10000):
    param_map = {
        "uber": {
            "preprocessing": None,
            "initialization": None
        }
    }

    preprocessing = param_map[tensor_name]["preprocessing"] 
    initialization = param_map[tensor_name]["initialization"]    
    ground_truth = PySparseTensor(f"/pscratch/sd/v/vbharadw/tensors/{tensor_name}.tns_converted.hdf5", lookup="sort", preprocessing=preprocessing)

    I = 3
    data = np.zeros((I, I, I), dtype=np.double)
    for i in range(I):
        for j in range(I):
            for k in range(I):
                data[i, j, k] = i + j + k

    #ground_truth = PyDenseTensor(data)

    print("Loaded dataset...")
    tt_approx = TensorTrain(ground_truth.shape, 
        [R] * (ground_truth.N - 1))

    tt_approx.place_into_canonical_form(0)
    tt_approx.build_fast_sampler(0, J=J)
    tt_als = TensorTrainALS(ground_truth, tt_approx)
    print(tt_als.compute_exact_fit())

    tt_als.execute_randomized_als_sweeps(num_sweeps=10, J=J, epoch_interval=1)

    #tt_values = tt_approx.evaluate_partial_fast(
    #        ground_truth.tensor_idxs,
    #        tt_approx.N, "left").squeeze()

    tt_values_A = tt_approx.evaluate_partial_fast(
            np.array([[0, 0, 0]], dtype=np.uint64),
            tt_approx.N - 1, "left").squeeze()
    
    tt_values_B = None


    #print(tt_values)
    #print(ground_truth.values)

    left_chain = tt_approx.left_chain_matricize(ground_truth.N)
    print(left_chain)

if __name__=='__main__':
    test_sparse_tensor_decomposition() 
    #test_dense_recovery()
    #test_tt_als()
