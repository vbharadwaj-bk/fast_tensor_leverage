from algorithms.contractions import *
import numpy as np
import numpy.linalg as la
from tensors.tensor_train import *
from tensors.dense_tensor import *

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
            axis_names_right = [f'b_mpsr{i}',f'pc{i}',f'b_mpsr{i+1}'] 

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
    def __init__(self, dims_row, dims_col, ranks, seed=None, cores=None, init_method="gaussian"):
        N = len(dims_row)
        self.N = N 
        self.dims_row = dims_row
        self.dims_col = dims_col
        assert(self.N == len(dims_col))

        combined_core_dims = [dims_row[i] * dims_col[i] for i in range(self.N)]

        reduce_rank = cores is None 
        tt_internal = TensorTrain(combined_core_dims, 
                                  ranks, 
                                  seed, 
                                  init_method,
                                  reduce_rank=reduce_rank)
        self.U = []
        self.ranks = tt_internal.ranks

        for i, core in enumerate(tt_internal.U):
            rank_left, rank_right = core.shape[0], core.shape[2]
            self.U.append(core.reshape((rank_left, dims_row[i], dims_col[i], rank_right)).copy())

        if cores is not None:
            for i in range(self.N):
                self.U[i][:] = cores[i].reshape(self.U[i].shape)

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
    def __init__(self, mpo, mps):
        assert(mpo.N == mps.N)
        N = mps.N 

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
                next = i
        elif direction == "down":
            if i > 0:
                previous = self.contractions_up[i-1]
            if i < N - 1:
                next = i + 1 

        if previous:
            nodes.append(previous)

        if next:
            output_order = [f'b_mpsl{next}',
                            f'b_mpo{next}',
                            f'b_mpsr{next}']
        else:
            output_order = None

        result = contract_nodes(nodes, 
                    contractor=tn.contractors.greedy, 
                    output_order=output_order)

        if direction == "up":
            self.contractions_down[i] = result
        elif direction == "down":
            self.contractions_up[i] = result

    def form_lhs(self, i, contract_into_matrix=False):
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
        
        out_row_modes = [f'pr{i}']
        out_col_modes = [f'pc{i}']

        if i > 0:
            out_row_modes = [f'b_mpsl{i}'] + out_row_modes
            out_col_modes = [f'b_mpsr{i}'] + out_col_modes
        if i < N - 1:
            out_row_modes = out_row_modes + [f'b_mpsl{i+1}'] 
            out_col_modes = out_col_modes + [f'b_mpsr{i+1}'] 

        return contract_nodes(nodes, 
                    contractor=tn.contractors.greedy, 
                    out_row_modes=out_row_modes,
                    out_col_modes=out_col_modes)

    def form_lhs_debug(self, i, contract_into_matrix=True):
        N = self.N
        mpo = self.mpo
        mps = self.mps

        nodes = mpo.nodes + [node for j, node in enumerate(mps.nodes_l)
                             if j != i] \
                            + [node for j, node in enumerate(mps.nodes_r)
                                                if j != i]

        out_row_modes = [f'pr{i}']
        out_col_modes = [f'pc{i}']

        if i > 0:
            out_row_modes = [f'b_mpsl{i}'] + out_row_modes
            out_col_modes = [f'b_mpsr{i}'] + out_col_modes
        if i < N - 1:
            out_row_modes = out_row_modes + [f'b_mpsl{i+1}'] 
            out_col_modes = out_col_modes + [f'b_mpsr{i+1}'] 

        return contract_nodes(nodes, 
                       contractor=tn.contractors.greedy, 
                       out_row_modes=out_row_modes,
                       out_col_modes=out_col_modes)


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
 
        result = contract_nodes(nodes, 
                    contractor=tn.contractors.greedy, 
                    output_order=output_order)
         
        return result.tensor

    def compute_error(self, rhs):
        Ax = self.mpo_mps_multiply()
        b = vec(rhs)

        return la.norm(b - Ax) / la.norm(b) 

    def execute_dmrg_exact(self, rhs, num_sweeps, cold_start=True):
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

        print(f"Error before ALS: {self.compute_error(rhs)}")

        for iter in range(num_sweeps):
            for i in range(N-1):
                A = self.form_lhs(i, contract_into_matrix=True)
                b = vec(self.contract_mps_with_rhs(rhs, i))
                x = la.solve(A, b)

                tt.U[i][:] = x.reshape(tt.U[i].shape)
                tt.orthogonalize_push_right(i)

                self._contract_cache_sweep(i, "down")

            for i in reversed(range(1,N)):
                A = self.form_lhs(i, contract_into_matrix=True)
                b = vec(self.contract_mps_with_rhs(rhs, i))
                x = la.solve(A, b)

                tt.U[i][:] = x.reshape(tt.U[i].shape)
                tt.orthogonalize_push_left(i)
                self._contract_cache_sweep(i, "up")

            print(f"Error after sweep {iter}: {self.compute_error(rhs)}")

    def sampled_QTB(self, i, J, ground_truth):
        tt = self.mps.tt
        samples = np.zeros((J, self.N), dtype=np.uint64)

        left_rows, left_cols = None, None
        right_rows, right_cols = None, None
        if i > 0:
            left_samples = tt.leverage_sample(i, J, "left")
            samples[:, :i] = left_samples

            left_rows = tt.evaluate_partial_fast(samples, i, "left")
            left_cols = left_rows.shape[1]
        else:
            left_cols = 1
        if i < self.N - 1:
            right_samples = tt.leverage_sample(i, J, "right")
            samples[:, i+1:] = right_samples
            
            right_rows = tt.evaluate_partial_fast(samples, i, "right")
            right_cols = right_rows.shape[1]
        else:
            right_cols = 1

        design, samples_to_spmm = None, None

        if left_rows is None:
            design = right_rows
        elif right_rows is None:
            design = left_rows
        else:
            # Should probably write a custom kernel for this in C++ 
            design = np.einsum("ij,ik->ijk", left_rows, right_rows).reshape(J, -1)

        weights = la.norm(design, axis=1) ** 2 / design.shape[1] * J
        design = np.einsum("ij,i->ij", design, 1.0 / weights)
        samples_to_spmm = samples

        result = np.zeros((tt.dims[i], design.shape[1]), dtype=np.double)
        ground_truth.execute_sampled_spmm(
                samples_to_spmm,
                design,
                i,
                result)
        
        result = result.reshape(tt.dims[i], 
                                tt.ranks[i],
                                tt.ranks[i+1])
        
        result = vec(result.transpose((1, 0, 2)))  
        return result


    def execute_dmrg_randomized(self, rhs, num_sweeps, J, cold_start=True):
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

        tt.build_fast_sampler(0, J=J)

        print(f"Error before ALS: {self.compute_error(rhs)}")

        ground_truth = PyDenseTensor(rhs)
        J= 1000 

        for iter in range(num_sweeps):
            for i in range(N-1):
                A = self.form_lhs(i, contract_into_matrix=True) 
                b = self.sampled_QTB(i, J, ground_truth)
                x = la.solve(A, b)

                tt.U[i][:] = x.reshape(tt.U[i].shape)
                tt.orthogonalize_push_right(i)
                tt.update_internal_sampler(i, "right", True)
                self._contract_cache_sweep(i, "down")

            for i in reversed(range(1,N)):
                A = self.form_lhs(i, contract_into_matrix=True)
                b = self.sampled_QTB(i, J, ground_truth)
                x = la.solve(A, b)

                tt.U[i][:] = x.reshape(tt.U[i].shape)
                tt.orthogonalize_push_left(i)
                tt.update_internal_sampler(i, "left", True)
                self._contract_cache_sweep(i, "up")

            print(f"Error after sweep {iter}: {self.compute_error(rhs)}")

def verify_mpo_mps_contraction():
    N = 15
    I = 2
    R_mpo = 4
    R_mps = 4

    mpo = MPO([I] * N, [I] * N, [R_mpo] * (N - 1))
    mps = MPS([I] * N, [R_mps] * (N - 1))
    system = MPO_MPS_System(mpo, mps)

    print("Initialized sandwich system!")

    mat = system.mpo.materialize_matrix()
    vec = system.mps.materialize_vector()

    print(system.mpo_mps_multiply())
    print(mat @ vec)


def test_dmrg():
    N = 12
    I = 2
    R_mpo_ns = 4
    R_mpo = R_mpo_ns * R_mpo_ns
    R_mps = 10

    mpo_ns = MPO([I] * N, [I] * N, [R_mpo_ns] * (N - 1))

    # Create an symmetric MPO by multiplying the nonsymmetric MPO
    # with itself

    sym_cores = []
    r_nodes = tn.replicate_nodes(mpo_ns.nodes)
    for i in range(N):
        new_axis_names, output_order = None, None
        if i > 0 and i < N - 1:
            new_axis_names = [f'rb_mpo{i}', f'rpr{i}', f'pc{i}',f'rb_mpo{i+1}']
            output_order = [f'b_mpo{i}', f'rb_mpo{i}', 
                            f'rpr{i}', f'pr{i}', 
                            f'b_mpo{i+1}', f'rb_mpo{i+1}']
        elif i == 0:
            new_axis_names = [f'rpr{i}', f'pc{i}', f'rb_mpo{i+1}']
            output_order = [f'rpr{i}', f'pr{i}', 
                            f'b_mpo{i+1}', f'rb_mpo{i+1}'
                            ]
        elif i == N - 1:
            new_axis_names = [f'rb_mpo{i}', f'rpr{i}', f'pc{i}']
            output_order = [f'b_mpo{i}', f'rb_mpo{i}', 
                            f'rpr{i}', f'pr{i}']

        r_nodes[i].add_axis_names(new_axis_names)
        result = contract_nodes([mpo_ns.nodes[i], r_nodes[i]],  
                                contractor=tn.contractors.greedy, 
                                output_order=output_order)
        shape = result.shape
        output_shape = None

        if i > 0 and i < N - 1:
            output_shape =  [np.prod(shape[0:2]), shape[2], shape[3], np.prod(shape[4:6])]
        elif i == 0:
            output_shape =  [shape[0], shape[1], np.prod(shape[2:4])]
        elif i == N - 1:
            output_shape =  [np.prod(shape[0:2]), shape[2], shape[3]]

        sym_cores.append(result.tensor.reshape(output_shape))

    mpo = MPO([I] * N, [I] * N, [rank * rank for rank in mpo_ns.ranks[1:-1]], cores=sym_cores)
    mps = MPS([I] * N, [R_mps] * (N - 1)) 

    system = MPO_MPS_System(mpo, mps)
 
    rhs = system.mpo_mps_multiply().reshape([I] * N) * 1000
    system.mps.tt.reinitialize_gaussian()
    #system.execute_dmrg_exact(rhs, 200, cold_start=True)
    system.execute_dmrg_randomized(rhs, 
                                   200, 
                                   J=1000, 
                                   cold_start=True)

if __name__=='__main__':
    test_dmrg()
    #verify_mpo_mps_contraction()
