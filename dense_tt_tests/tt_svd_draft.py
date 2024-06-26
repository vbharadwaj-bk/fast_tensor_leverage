import numpy as np
import numpy.linalg as la
import logging, sys

import tensorly as tl
from tensorly.tt_tensor import validate_tt_rank, TTTensor
from tensorly.tenalg.svd import svd_interface


def tensor_train_svd(input_tensor, rank, svd="truncated_svd"):
    tensor_size = input_tensor.shape
    n_dim = len(tensor_size)
    rank = validate_tt_rank(tl.shape(input_tensor), rank=rank)

    unfolding = input_tensor
    factors = [None] * n_dim

    # Getting the TT factors up to n_dim - 1
    for k in range(n_dim - 1):
        # Reshape the unfolding matrix of the remaining factors
        n_row = int(rank[k] * tensor_size[k])
        unfolding = tl.reshape(unfolding, (n_row, -1))

        # SVD of unfolding matrix
        (n_row, n_column) = unfolding.shape
        current_rank = min(n_row, n_column, rank[k + 1])
        if svd == "truncated_svd":
            U, S, V = svd_interface(unfolding, n_eigenvecs=current_rank, method="truncated_svd")
        elif svd == "randomized_svd":
            U, S, V = svd_interface(unfolding, n_eigenvecs=current_rank, method="randomized_svd")
        else:
            continue
        rank[k + 1] = current_rank

        # Get kth TT factor
        factors[k] = tl.reshape(U, (rank[k], tensor_size[k], rank[k + 1]))

        unfolding = tl.reshape(S, (-1, 1)) * V

    # Getting the last factor
    (prev_rank, last_dim) = unfolding.shape
    factors[-1] = tl.reshape(unfolding, (prev_rank, last_dim, 1))

    return TTTensor(factors)


