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
        self.nodes_l = [tn.Node(self.U[i], name=f"mps_core_l_{i}", axis_names=[f'b{i}',f'pr{i}',f'b{i+1}']) for i in range(self.N)] 
        self.nodes_r = [tn.Node(self.U[i], name=f"mps_core_l_{i}", axis_names=[f'b{i}',f'pc{i}',f'b{i+1}']) for i in range(self.N)] 

        # Connect bond dimensions 
        for i in range(1, N):
            tn.connect(self.nodes_l[i-1][f'b{i}'], self.nodes_l[i][f'b{i}'])

        self.vector_length = np.prod(dims)

    def materialize_vector(self):
        N = self.N
        mps_copy = tn.replicate_nodes(self.nodes_l)

        def gne(node, edge):
            return mps_copy[node].get_edge(edge)

        output_edge_order = [gne(0, 'b0')] + [gne(i, f'pr{i}') for i in range(N)] \
                        + [gne(N-1, f'b{N}')] 

        result = tn.contractors.greedy(mps_copy, output_edge_order=output_edge_order).tensor
        result = result.reshape(self.vector_length)
        return(result)


class MPO:
    '''
    The internal representation of an MPO is a tensor train with
    individual cores reshaped.
    '''
    def __init__(self, dims_row, dims_col, ranks, seed=None, init_method="gaussian"):
        self.N = len(dims_row)
        self.dims_row = dims_row
        self.dims_col = dims_col
        assert(self.N == len(dims_col))

        combined_core_dims = [dims_row[i] * dims_col[i] for i in range(self.N)]

        tt_internal = TensorTrain(combined_core_dims, ranks, seed, init_method)
        self.U = []

        for i, core in enumerate(tt_internal.U):
            rank_left, rank_right = core.shape[0], core.shape[2]
            self.U.append(core.reshape((rank_left, dims_row[i], dims_col[i], rank_right)).copy())

        self.nodes = []
        for i in range(self.N):
            labels = [f'b{i}', f'pr{i}', f'pc{i}',f'b{i+1}']
            self.nodes.append(tn.Node(self.U[i], name=f"mpo_core{i}", axis_names=labels))

        # Connect bond dimensions 
        for i in range(1, N):
            tn.connect(self.nodes[i-1][f'b{i}'], self.nodes[i][f'b{i}'], name=f'b{i}')

        self.total_rows = np.prod(dims_row)
        self.total_cols = np.prod(dims_col)

    def materialize_matrix(self):
        N = self.N
        mpo_copy = tn.replicate_nodes(self.nodes)

        def gne(node, edge):
            return mpo_copy[node].get_edge(edge)

        output_edge_order = [gne(0, 'b0')] + [gne(i, f'pr{i}') for i in range(N)] \
                        + [gne(i, f'pc{i}') for i in range(N)] \
                        + [gne(N-1, f'b{N}')] 

        result = tn.contractors.greedy(mpo_copy, output_edge_order=output_edge_order).tensor
        result = result.reshape(self.total_rows, self.total_cols)
        return(result)


class MPO_MPS_System:
    '''
    This class creates and hooks up the "sandwich" MPS-MPO-MPS system.
    We can then copy out subsets of nodes to perform contractions. 
    '''
    def __init__(self, dims, ranks_mpo, ranks_mps):
        N = len(dims)
        mpo = MPO(dims, dims, ranks_mpo)
        mps = MPS(dims, ranks_mps)

        # Connect the physical dimensions of the MPO to the copies of the
        # MPS nodes. 
        for i in range(0, N):
            tn.connect(mpo.nodes[i][f'pr{i}'], mps.nodes_l[i][f'pr{i}'])
            tn.connect(mpo.nodes[i][f'pc{i}'], mps.nodes_r[i][f'pc{i}'])

        self.mpo = mpo
        self.mps = mps
        self.N = N

if __name__=='__main__':
    N = 5
    I = 2
    R_mpo = 4
    R_mps = 4

    system = MPO_MPS_System([I] * N, [R_mpo] * (N - 1), [R_mps] * (N - 1))
    print("Initialized sandwich system!")

    system.mpo.materialize_matrix()
    print(system.mps.materialize_vector())
