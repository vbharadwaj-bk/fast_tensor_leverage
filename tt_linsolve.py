import tensornetwork as tn
import numpy as np
from tensors.tensor_train import *

# This code depends on the Google tensor network package; given that this
# package is no longer under development, we should switch; fortunately,
# several packages offer tensor contraction. 

class MPS:
    def __init__(self, dims, ranks, seed=None, init_method="gaussian"):
        self.N = len(dims_row)
        self.tt = TensorTrain(dims, ranks, seed, init_method)
        self.U = self.tt.U
        self.nodes = [tn.Node(self.U[i], name=f"mps_core_{i}", axis_names=[f'b{i}', f'p{i}',f'b{i+1}']) for i in range(self.N)] 

class MPO:
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

        self.nodes = [tn.Node(self.U[i], name=f"mpo_core_{i}", axis_names=[f'b{i}', f'pr{i}', f'pc{i}',f'b{i+1}']) for i in range(self.N)] 
        
class MPO_MPS_System:
    '''
    This class creates and hooks up the "sandwich" MPS-MPO-MPS system.
    We can then copy out subsets of nodes to perform contractions. 
    '''
    def __init__(self, dims, ranks_mpo, ranks_mps):
        self.mpo = MPO(dims, dims, ranks_mpo)
        self.mps = MPS(dims, ranks_mps)




if __name__=='__main__':
    N = 10
    I = 2
    R_mpo = 5
    R_mps = 5

    system = MPO_MPS_System(dims, [R_mpo] * (N - 1), [R_mps] * (N - 1))
    print("Initialized sandwich system!")

