import tensornetwork as tn
import numpy as np
from tensors.tensor_train import *

# This code depends on the Google tensor network package; given that this
# package is no longer under development, we should switch; fortunately,
# several packages offer tensor contraction. 

# Also verified: each node is just a thin wrapper around data, with
# the same underlying pointer.

class MPS:
    def __init__(self, dims, ranks, seed=None, init_method="gaussian"):
        self.N = len(dims)
        self.tt = TensorTrain(dims, ranks, seed, init_method)
        self.U = self.tt.U
        self.nodes_right = [tn.Node(self.U[i], name=f"mps_core_r_{i}", axis_names=[f'b{i}',f'p{i}',f'b{i+1}']) for i in range(self.N)] 
        self.nodes_left = [tn.Node(self.U[i], name=f"mps_core_l_{i}", axis_names=[f'b{i}',f'p{i}',f'b{i+1}']) for i in range(self.N)] 

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

    def materialize_matrix(self):
        mpo_copy = tn.replicate_nodes(self.nodes)
        output_edge_order = [f'pr{i}' for i in range(self.N)] + [f'pc{i}' for i in range(self.N)] 
        result = tn.contractors.greedy(mpo_copy, output_edge_order=output_edge_order)
        print(result)

class MPO_MPS_System:
    '''
    This class creates and hooks up the "sandwich" MPS-MPO-MPS system.
    We can then copy out subsets of nodes to perform contractions. 
    '''
    def __init__(self, dims, ranks_mpo, ranks_mps):
        mpo = MPO(dims, dims, ranks_mpo)
        mps = MPS(dims, ranks_mps)

        # Connect all nodes of the sandwich

        self.mpo = mpo
        self.mps = mps

if __name__=='__main__':
    N = 10
    I = 2
    R_mpo = 4
    R_mps = 4

    system = MPO_MPS_System([I] * N, [R_mpo] * (N - 1), [R_mps] * (N - 1))
    print("Initialized sandwich system!")

    system.materialize_matrix()
