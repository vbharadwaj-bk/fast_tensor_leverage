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
        N = len(dims)
        self.N = N
        self.tt = TensorTrain(dims, ranks, seed, init_method)
        self.U = self.tt.U

        self.nodes_l = []
        self.nodes_r = []

        for i in range(self.N):
            axis_names_left = [f'b{i}',f'pr{i}',f'b{i+1}'] 
            axis_names_right = [f'b{i}',f'pc{i}',f'b{i+1}'] 

            if i == 0:
                axis_names_left = axis_names_left[1:]
                axis_names_right = axis_names_right[1:]
            elif i == N - 1:
                axis_names_left = axis_names_left[:-1]
                axis_names_right = axis_names_right[:-1]

            node_l = tn.Node(self.U[i].squeeze(), 
                    name=f"mps_core_l_{i}", 
                    axis_names=axis_names_left) 
            node_r = tn.Node(self.U[i].squeeze(), 
                    name=f"mps_core_r_{i}", 
                    axis_names=axis_names_right) 

            self.nodes_l.append(node_l)
            self.nodes_r.append(node_r)

        # Connect bond dimensions 
        for i in range(1, N):
            tn.connect(self.nodes_l[i-1][f'b{i}'], self.nodes_l[i][f'b{i}'])
            tn.connect(self.nodes_r[i-1][f'b{i}'], self.nodes_r[i][f'b{i}'])

        self.vector_length = np.prod(dims)

    def materialize_vector(self):
        N = self.N
        mps_copy = tn.replicate_nodes(self.nodes_l)

        def gne(node, edge):
            return mps_copy[node].get_edge(edge)

        output_edge_order = [gne(i, f'pr{i}') for i in range(N)]

        result = tn.contractors.greedy(mps_copy, output_edge_order=output_edge_order).tensor
        result = result.reshape(self.vector_length)
        return(result)


class MPO:
    '''
    The internal representation of an MPO is a tensor train with
    individual cores reshaped.
    '''
    def __init__(self, dims_row, dims_col, ranks, seed=None, init_method="gaussian"):
        N = len(dims_row)
        self.N = N 
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
            axis_names = [f'b{i}', f'pr{i}', f'pc{i}',f'b{i+1}']

            if i == 0:
                axis_names = axis_names[1:]
            elif i == N-1:
                axis_names = axis_names[:-1]

            self.nodes.append(tn.Node(self.U[i].squeeze(), name=f"mpo_core{i}", axis_names=axis_names))

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

        output_edge_order = [gne(i, f'pr{i}') for i in range(N)] \
                        + [gne(i, f'pc{i}') for i in range(N)]

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

        self.contractions_up = []
        self.contractions_down = []

    def mpo_mps_multiply(self, reshape_into_vec=True):
        N = self.N
        replicated_system = tn.replicate_nodes(self.mpo.nodes + self.mps.nodes_r)

        # The first N nodes of the replicated system are the MPO cores
        def gne(node, edge):
            return replicated_system[node].get_edge(edge)

        output_edge_order = [gne(i, f'pr{i}') for i in range(N)]

        result = tn.contractors.greedy(replicated_system, output_edge_order=output_edge_order).tensor

        if reshape_into_vec:
            result = result.reshape(self.mpo.total_rows)

        return result

    def _contract_cache_sweep(self, i, direction):
        assert(direction == "up" or direction == "down")
        N = self.N
        mpo = self.mpo
        mps = self.mps

        nodes_to_replicate = [mps.nodes_l[i], mpo.nodes[i], mps.nodes_r[i]]

        previous, next = None, False 
        if direction == "up":
            if i < N - 1:
                previous = self.contractions_down[i+1]
                bond_label_in = f'b{i+1}'
                bond_label_out = f'b{i}'
            if i > 0:
                next = True
        elif direction == "down":
            if i > 0:
                previous = self.contractions_up[i-1]
                input_bond_label = f'b{i}'
                output_bond_label = f'b{i+1}'
            if i < N - 1:
                next = True

        if previous:
            nodes_to_replicate.append(previous)

        replicated_system = tn.replicate_nodes(nodes_to_replicate)

        def gne(node, edge):
            return replicated_system[node].get_edge(edge)

        if i < N - 1:
            tn.connect(replicated_system[0][bond_label_in], replicated_system[3]['bl']) 
            tn.connect(replicated_system[1][bond_label_in], replicated_system[3]['bm']) 
            tn.connect(replicated_system[2][bond_label_in], replicated_system[3]['br']) 

        if i > 0:
            output_edge_order = [gne(j, bond_label_out) for j in range(3)]
        else:
            output_edge_order = None

        result = tn.contractors.greedy(replicated_system, output_edge_order=output_edge_order)

        if i > 0:
            result.add_axis_names(['bl', 'bm', 'br'])

        if direction == "up":
            self.contractions_down[i] = None
            self.contractions_down[i] = result
        elif direction == "down":
            self.contractions_up[i] = None
            self.contractions_up[i] = result


    def execute_dmrg(self, rhs, num_sweeps, cold_start=True):
        '''
        Cold start places the MPS into canonical form with core 0
        non-orthogonal and computes a set of right_contractions. 
        '''
        N = self.N
        mps = self.mps
        mpo = self.mpo

        if cold_start:
            # Step 1: Place the MPS in canonical form w/ core 0 non-orthogonal
            mps.tt.place_into_canonical_form(0)

            self.contractions_down = [None] * self.N
            self.contractions_up = [None] * self.N

            for i in reversed(range(0, N)):
                self._contract_cache_sweep(i, "up")

            for i in reversed(range(0, N)):
                self._contract_cache_sweep(i, "down")

            print("Cold started DMRG!")

def verify_mpo_mps_contraction():
    N = 10
    I = 2
    R_mpo = 4
    R_mps = 4

    system = MPO_MPS_System([I] * N, [R_mpo] * (N - 1), [R_mps] * (N - 1))

    print("Initialized sandwich system!")

    mat = system.mpo.materialize_matrix()
    vec = system.mps.materialize_vector()

    print(system.mpo_mps_multiply())
    print(mat @ vec)


def test_dmrg():
    N = 3
    I = 2
    R_mpo = 4
    R_mps = 4

    system = MPO_MPS_System([I] * N, [R_mpo] * (N - 1), [R_mps] * (N - 1))
    system.execute_dmrg(None, 0, cold_start=True)

if __name__=='__main__':
    #verify_mpo_mps_contraction()

    test_dmrg()
