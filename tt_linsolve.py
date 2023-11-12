import tensornetwork as tn
import numpy as np
import numpy.linalg as la
from tensors.tensor_train import *

# This code depends on the Google tensor network package; given that this
# package is no longer under development, we should switch; fortunately,
# several packages offer tensor contraction. 

# The convention in this file is to assume that axis names between different
# nodes are only shared when there is a contraction involved. If three
# or more nodes share an edge, we raise an exception 
# (no hypergraph contractions here). Also, all axis names must be specified. 

def vec(X):
    prod_shape = np.prod(X.shape)
    return X.reshape(prod_shape)

def contract_nodes(nodes_in, 
                   contractor, 
                   output_order=None, 
                   out_row_modes=None, 
                   out_col_modes=None,
                   vectorize=False
                   ):

    if output_order == None \
        and (out_row_modes is None or out_col_modes is None):
        raise Exception("Must specify either output order or matricize row / col axes.")

    matricize = out_row_modes is not None
    assert(not (vectorize and matricize))

    if matricize:
        output_order = out_row_modes + out_col_modes
    elif output_order is None:
        output_order = []

    nodes = tn.replicate_nodes(nodes_in)
    edges = {} # Dictionary of axis name to nodes sharing the edge

    for node in nodes:
        for name in node.axis_names:
            if name not in edges:
                edges[name] = [] 

            edges[name].append(node)

    dangling_edges = {}
    for name in edges:
        if len(edges[name]) >= 3:
            node_names = ', '.join([node.name for node in edges[name]])
            raise Exception(f"Error, nodes {node_names} all share edge {name}")
        elif(len(edges[name]) == 1):
            dangling_edges[name] = edges[name][0].get_edge(name)
        else:
            tn.connect(edges[name][0][name], edges[name][1][name])

    output_edges = []
    for name in output_order:
        if name not in dangling_edges:
            exception_text = f"Error, edge {name} is not dangling.\n"

            if len(edges[name]) == 2:
                exception_text += f"Edge {name} spans nodes {edges[name][0]} and {edges[name][1]}."
            else:
                exception_text += f"Edge {name} not found."

            raise Exception(exception_text)

        else:
            output_edges.append(dangling_edges[name])

    for name in dangling_edges:
        if name not in output_order:
            raise Exception(f"Error, list of dangling edges is {dangling_edges}. Edge {name} is dangling and not in output order.")

    if len(output_edges) == 0:
        output_edges = None
    result = contractor(nodes, output_edge_order=output_edges)

    if vectorize:
        return vec(result.tensor)        
    elif matricize:
        shape = result.shape
        row_count = np.prod(shape[:len(out_row_modes)])
        col_count = np.prod(shape[len(out_row_modes):])

        return result.tensor.reshape(row_count, col_count)
    else:
        if len(output_order) > 0:
            result.add_axis_names(output_order)
        return result


class MPS:
    def __init__(self, dims, ranks, seed=None, init_method="gaussian"):
        N = len(dims)
        self.N = N
        self.tt = TensorTrain(dims, ranks, seed, init_method)
        self.U = self.tt.U

        self.nodes_l = []
        self.nodes_r = []

        for i in range(self.N):
            axis_names_left = [f'b_mpsl{i}',f'pr{i}',f'b_mpsl{i+1}'] 
            axis_names_right = [f'b_mpsl{i}',f'pc{i}',f'b_mpsl{i+1}'] 

            if i == 0:
                axis_names_left = axis_names_left[1:]
                axis_names_right = axis_names_right[1:]
            elif i == N - 1:
                axis_names_left = axis_names_left[:-1]
                axis_names_right = axis_names_right[:-1]

            node_l = tn.Node(self.U[i].squeeze(), 
                    name=f"mpsl_core_{i}", 
                    axis_names=axis_names_left) 
            node_r = tn.Node(self.U[i].squeeze(), 
                    name=f"mpsr_core_{i}", 
                    axis_names=axis_names_right) 

            self.nodes_l.append(node_l)
            self.nodes_r.append(node_r)

        self.vector_length = np.prod(dims)

    def materialize_vector(self):
        output_order = [f'pr{i}' for i in range(self.N)]

        return contract_nodes(self.nodes_l, 
                       contractor=tn.contractors.greedy, 
                       output_order=output_order,
                       vectorize=True)

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
            axis_names = [f'b_mpo{i}', f'pr{i}', f'pc{i}',f'b_mpo{i+1}']

            if i == 0:
                axis_names = axis_names[1:]
            elif i == N-1:
                axis_names = axis_names[:-1]

            self.nodes.append(tn.Node(self.U[i].squeeze(), name=f"mpo_core{i}", axis_names=axis_names))

        self.total_rows = np.prod(dims_row)
        self.total_cols = np.prod(dims_col)

    def materialize_matrix(self):
        out_row_modes =  [f'pr{i}' for i in range(self.N)]
        out_col_modes = [f'pc{i}' for i in range(self.N)]

        return contract_nodes(self.nodes, 
                       contractor=tn.contractors.greedy, 
                       out_row_modes=out_row_modes,
                       out_col_modes=out_col_modes)


class MPO_MPS_System:
    '''
    This class creates and hooks up the "sandwich" MPS-MPO-MPS system.
    We can then copy out subsets of nodes to perform contractions. 
    '''
    def __init__(self, dims, ranks_mpo, ranks_mps):
        N = len(dims)
        mpo = MPO(dims, dims, ranks_mpo)
        mps = MPS(dims, ranks_mps)

        self.mpo = mpo
        self.mps = mps
        self.N = N

        self.contractions_up = []
        self.contractions_down = []

    def mpo_mps_multiply(self, reshape_into_vec=True):
        N = self.N

        output_order = [f'pr{i}' for i in range(self.N)]

        return contract_nodes(self.mpo.nodes + self.mps.nodes_r, 
                       contractor=tn.contractors.greedy, 
                       output_order=output_order,
                       vectorize=reshape_into_vec)

    def _contract_cache_sweep(self, i, direction):
        assert(direction == "up" or direction == "down")
        N = self.N
        mpo = self.mpo
        mps = self.mps

        nodes = [mps.nodes_l[i], mpo.nodes[i], mps.nodes_r[i]]

        previous, next = None, False 
        if direction == "up":
            if i < N - 1:
                previous = self.contractions_down[i+1]
            if i > 0:
                next = True
        elif direction == "down":
            if i > 0:
                previous = self.contractions_up[i-1]
            if i < N - 1:
                next = True

        if previous:
            nodes.append(previous)

        if next:
            output_order = [f'b_mpsl{i+1}',
                                 f'b_mpo{i+1}',
                                 f'b_mpsr{i+1}']
        else:
            output_order = None

        result = contract_nodes(nodes, 
                    contractor=tn.contractors.greedy, 
                    output_order=output_order)

        if direction == "up":
            self.contractions_down[i] = result
        elif direction == "down":
            self.contractions_up[i] = result

    def form_lhs_system(self, i, contract_into_matrix=False):
        '''
        This method assumes that the correct up and down contractions have been 
        previously computed.
        '''
        N = self.N
        mpo = self.mpo
        mps = self.mps

        nodes = [mpo.nodes[i]]

        if i > 0:
            nodes.append(self.contractions_up[i-1])
        if i < N-1:
            nodes.append(self.contractions_down[i+1])
        
        output_edge_order = None
        if i > 0 and i < N - 1:
            output_order = [f'b_mpsl{i}', f'pr{i}', f'b_mpsl{i+1}', 
                                 f'b_mpsr{i}', f'pc{i}', f'b_mpsr{i+1}']
        elif i == 0:
            output_order = [f'pr{i}', f'b_mpsl{i+1}', 
                                 f'pc{i}', f'b_mpsr{i+1}']
        elif i == N - 1: 
            output_order = [f'pr{i}', f'b_mpsl{i+1}', 
                                 f'pc{i}', f'b_mpsr{i+1}']

        result = contract_nodes(nodes, 
                    contractor=tn.contractors.greedy, 
                    output_order=output_order)

        if contract_into_matrix:
            result = result.tensor
            shape = result.shape
            length = len(shape)
            result = result.reshape((np.prod(shape[0:length//2]), np.prod(shape[length//2:])))

        return result

    def contract_mps_with_rhs(self, rhs, i):
        mps = self.mps
        N = self.N
        rhs_node = tn.Node(rhs, axis_names=[f'pr{i}' for i in range(N)])
        nodes = [node for j, node in enumerate(mps.nodes_l) if j != i]
        nodes.append(rhs_node)

        output_order = None
        if i == 0:
            output_order = ['pr0', 'b_mpsl1'] 
        elif i == N - 1:
            output_order = [f'b_mpsl{i}', f'pr{i}'] 
        elif 0 < i < N - 1:
            output_order = [f'b_mpsl{i}', f'pr{i}', f'b_mpsl{i+1}'] 

        result = tn.contractors.greedy(nodes, output_order=output_order)
        return result.tensor

    def compute_error(self, rhs):
        Ax = self.mpo_mps_multiply()
        b = vec(rhs)

        return la.norm(b - Ax) 

    def execute_dmrg(self, rhs, num_sweeps, cold_start=True):
        '''
        Cold start places the MPS into canonical form with core 0
        non-orthogonal and computes a set of right_contractions. 
        '''
        N = self.N
        mps = self.mps
        mpo = self.mpo
        tt = mps.tt 

        if cold_start:
            # Step 1: Place the MPS in canonical form w/ core 0 non-orthogonal
            mps.tt.place_into_canonical_form(0)

            self.contractions_down = [None] * self.N
            self.contractions_up = [None] * self.N

            for i in reversed(range(0, N)): 
                self._contract_cache_sweep(i, "up")

        for iter in range(num_sweeps):
            for i in range(N-1):
                A = self.form_lhs_system(i, contract_into_matrix=True)
                b = vec(self.contract_mps_with_rhs(rhs, i))

                x = la.solve(A, b)

                print(f"Error before solve: {self.compute_error(rhs)}")
                tt.U[i][:] = x.reshape(tt.U[i].shape)
                print(f"Error after solve: {self.compute_error(rhs)}")
                tt.orthogonalize_push_right(i)
                self._contract_cache_sweep(i, "down")

            for i in reversed(range(1,N)):
                A = self.form_lhs_system(i, contract_into_matrix=True)
                b = vec(self.contract_mps_with_rhs(rhs, i))
                x = la.solve(A, b)

                tt.U[i][:] = x.reshape(tt.U[i].shape)

                tt.orthogonalize_push_left(i)
                self._contract_cache_sweep(i, "up")

def verify_mpo_mps_contraction():
    N = 3
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
    N = 6
    I = 2
    R_mpo = 4
    R_mps = 4

    system = MPO_MPS_System([I] * N, [R_mpo] * (N - 1), [R_mps] * (N - 1))
    rhs = system.mpo_mps_multiply().reshape([I] * N) * 1000
    system.mps.tt.reinitialize_gaussian()
    system.execute_dmrg(rhs, 5, cold_start=True)

if __name__=='__main__':
    #test_dmrg()
    verify_mpo_mps_contraction()
