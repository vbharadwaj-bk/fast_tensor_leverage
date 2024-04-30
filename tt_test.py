import numpy as np
import numpy.linalg as la

import logging, sys, json, argparse, os, datetime

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


def test_sparse_tensor_decomposition(params):
    tensor_map = {
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

    tensor_name = params.input
    preprocessing = tensor_map[tensor_name]["preprocessing"] 
    initialization = tensor_map[tensor_name]["initialization"]    

    filename_prefix = '_'.join([params.input, str(params.trank), 
                                    str(params.iter), params.distribution, 
                                    params.algorithm, str(params.samples), 
                                    str(params.epoch_iter)])

    files = os.listdir(args.output_folder)
    filtered_files = [f for f in files if filename_prefix in f]

    trial_nums = []
    for f in filtered_files:
        with open(os.path.join(args.output_folder, f), 'r') as f_handle:
            exp = json.load(f_handle)
            trial_nums.append(exp["trial_num"])

    if len(remaining_trials) > 0:
            trial_num = remaining_trials[0] 
            output_filename = f'{filename_prefix}_{trial_num}.out'

    remaining_trials = [i for i in range(args.repetitions) if i not in trial_nums]

    if len(remaining_trials) == 0:
        print("No trials left to perform!")
        exit(0)

    # Proceed if there are trials remaining 
    trial_num = remaining_trials[0]
    output_filename = f'{filename_prefix}_{trial_num}.out'

    ground_truth = PySparseTensor(f"/pscratch/sd/v/vbharadw/tensors/{tensor_name}.tns_converted.hdf5", lookup="sort", preprocessing=preprocessing)

    ranks = [params.trank] * (ground_truth.N - 1)

    print("Loaded dataset...")
    tt_approx = TensorTrain(ground_truth.shape, ranks)

    tt_approx.place_into_canonical_form(0)
    tt_approx.build_fast_sampler(0, J=params.samples)
    tt_als = TensorTrainALS(ground_truth, tt_approx)
    optimizer_stats = None

    if params.algorithm == "exact":
        optimizer_stats = tt_als.execute_exact_als_sweeps_sparse(num_sweeps=params.iter, J=params.samples, epoch_interval=params.epoch_iter)
    elif params.algorithm == "random":
        optimizer_stats = tt_als.execute_randomized_als_sweeps(num_sweeps=params.iter, J=params.samples, epoch_interval=params.epoch_iter)

    now = datetime.now()
    output_dict = {
        'time': now.strftime('%m/%d/%Y, %H:%M:%S'), 
        'input': params.input,
        'target_rank': params.trank,
        'iterations': params.iter,
        'algorithm': params.algorithm,
        'sample_count': params.samples,
        'accuracy_epoch_length': params.epoch_iter,
        'trial_count': params.repetitions,
        'trial_num': trial_num,
        'initial_fit': initial_fit,
        'final_fit': final_fit,
        'thread_count': os.environ.get('OMP_NUM_THREADS'),
        'stats': optimizer_stats
    }

    print(json.dumps(output_dict, indent=4))
    print(f"Final Fit: {final_fit}")

    if output_filename is not None:
        with open(os.path.join(args.output_folder, output_filename), 'w') as f:
            f.write(json.dumps(output_dict, indent=4)) 


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', type=str, help='Tensor name to decompose', required=True)
    parser.add_argument("-t", "--trank", help="Rank of the target decomposition", required=True, type=int)
    parser.add_argument("-iter", help="Number of ALS iterations", required=True, type=int)
    parser.add_argument('-alg','--algorithm', type=str, help='Algorithm to perform decomposition')
    parser.add_argument("-s", "--samples", help="Number of samples taken from the KRP", required=False, type=int)
    parser.add_argument("-o", "--output_folder", help="Folder name to print statistics", required=False)
    parser.add_argument("-e", "--epoch_iter", help="Number of iterations per accuracy evaluation epoch", required=False, type=int, default=5)
    parser.add_argument("-r", "--repetitions", help="Number of repetitions for multiple trials", required=False, type=int, default=1)
    args = parser.parse_args()

    test_sparse_tensor_decomposition(args) 
    #test_dense_recovery()
    #test_tt_als()
