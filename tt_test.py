import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from tensor_train import *
from reference_implementation.segment_tree import * 

def test_tt_sampling():
    I = 20
    R = 4
    N = 3
    J = 300000

    dims = [I] * N 
    ranks = [R] * (N-1) 

    seed = 20
    tt = TensorTrain(dims, ranks, seed)

    test_direction = "left"

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

    J = 1
    upto = 3
    indices = np.zeros((J, N), dtype=np.uint64)
    indices[0] = [2, 3, 0]
    partial_evaluation = tt.evaluate_partial_fast(indices, upto=upto, direction="left")
    ground_truth = tt.evaluate_left(indices[0], upto=upto)

    print(partial_evaluation)
    print(ground_truth)


    #fig, ax = plt.subplots()
    #ax.plot(true_dist, label="True leverage distribution")
    #bins = np.array(np.bincount(linear_idxs, minlength=len(true_dist))) / J

    #ax.plot(bins, label="Our sampler")
    #ax.set_xlabel("Row Index")
    #ax.set_ylabel("Probability Density")
    #ax.grid(True)
    #ax.legend()
    #ax.set_title(f"{J} samples drawn by our method vs. true distribution")
    #fig.savefig('plotting/distribution_comparison.png')

if __name__=='__main__':
    #test_tt_functions_small()
    test_tt_sampling()
