import numpy as np
import scipy
from scipy.linalg import diagsvd
import tensorly as tl
from tensorly.decomposition import tensor_train
from tensorly.tt_tensor import tt_to_tensor
from tensorly.tt_tensor import validate_tt_rank, TTTensor
from tensorly.tenalg import svd_interface
from sklearn.utils.extmath import randomized_svd


class TensorTrainSVD:
    def __init__(self, tensor, r):
        self.tensor = tensor
        self.r = r
        self.tensor_size = tensor.shape
        print("Initialized TT-SVD-Class!")

    def tt_svd(self):
        order = len(self.tensor_size)
        ranks = [1] + [self.r] * (order - 1) + [1]
        print("Initialized TT-SVD!")

        cores = tl.decomposition.tensor_train(self.tensor, ranks, svd='truncated_svd', verbose=False)
        return cores

    def tt_randomized_svd(self):
        order = len(self.tensor_size)
        ranks = [1] + [self.r] * (order - 1) + [1]

        print("Initialized Randomized TT-SVD!")

        ranks = validate_tt_rank(tl.shape(self.tensor), rank=ranks)
        factors = [None] * order

        # Getting the TT factors up to n_dim - 1
        for k in range(order - 1):
            # Reshape the unfolding matrix of the remaining factors
            n_row = int(ranks[k] * self.tensor_size[k])
            self.tensor = tl.reshape(self.tensor, (n_row, -1))

            # SVD of unfolding matrix
            (n_row, n_column) = self.tensor.shape
            current_rank = min(n_row, n_column, ranks[k + 1])
            U, S, V = svd_interface(self.tensor, n_eigenvecs=current_rank, method="randomized_svd")

            ranks[k + 1] = current_rank

            # Get kth TT factor
            factors[k] = tl.reshape(U, (ranks[k], self.tensor_size[k], ranks[k + 1]))

            # Get new unfolding matrix for the remaining factors
            self.tensor = tl.reshape(S, (-1, 1)) * V

        # Getting the last factor
        (prev_rank, last_dim) = self.tensor.shape
        factors[-1] = tl.reshape(self.tensor, (prev_rank, last_dim, 1))
        return factors

    