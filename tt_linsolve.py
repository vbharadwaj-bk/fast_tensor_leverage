import quimb
import numpy as np
from tensors.tensor_train import *

# This code depends on the Google tensor network package; given that this
# package is no longer under development, we should switch; fortunately,
# several packages offer tensor contraction. 

class MatrixProductOperator:
    '''
    The internal representation of an MPO is a tensor train with
    individual cores reshaped.
    '''
    def __init__(self, dims_row, dims_col, ranks, seed=None, init_method="gaussian"):
        self.N = len(dims_row)
        assert(self.N == len(dims_col))
        combined_core_dims = [dims_row[i] * dims_col[i] for i in range(self.N)]

        tt_internal = TensorTrain(combined_core_dims, ranks, seed, init_method)
        self.U = []

        for i, core in enumerate(tt_internal.U):
            rank_left, rank_right = core.shape[0], core.shape[2]
            self.U.append(core.reshape((rank_left, dims_row[i], dims_col[i], rank_right)).copy())

        self.nodes = [tn.Node(self.U[i]) for i in range(self.N)] 

    def materialize_matrix(self):


if __name__=='__main__':
    N = 10
    I = 2
    R = 20
    mpo = MatrixProductOperator([I] * N, [I] * N, [R] * (N - 1))

    print("Initialized Matrix Product Operator!")


