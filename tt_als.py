from tensor_train import *
from dense_tensor import *

import time

class TensorTrainALS:
    def __init__(self, ground_truth, tt_approx):
        self.ground_truth = ground_truth
        self.tt_approx = tt_approx

        print("Initialized TT-ALS!")

    def compute_exact_fit(self):
        '''
        This operation can be particularly expensive for dense tensors.
        '''
        if isinstance(self.ground_truth, PyDenseTensor):
            tt_materialization = self.tt_approx.materialize_dense()
            return 1.0 - la.norm(tt_materialization - self.ground_truth.data) / self.ground_truth.data_norm
        else:
            raise NotImplementedError

    def execute_exact_als_sweeps_slow(self, num_sweeps):
        '''
        Assumes that the TT is in orthogonal
        form with core 0 non-orthogonal. This is a slow
        implementation mean for debugging.

        This is the single-site version of TT-ALS. 
        '''
        if not isinstance(self.ground_truth, PyDenseTensor):
            raise NotImplementedError

        tt_approx = self.tt_approx
        N = tt_approx.N

        def optimize_core(i):
            left_chain = tt_approx.left_chain_matricize(i)
            right_chain = tt_approx.right_chain_matricize(i)

            if len(left_chain.shape) == 0:
                left_cols = 1
            else:
                left_cols = left_chain.shape[1]

            if len(right_chain.shape) == 0:
                right_cols = 1
            else:
                right_cols = right_chain.shape[1] 

            design = np.kron(left_chain, right_chain)
            target_modes = list(range(N))
            target_modes.remove(i)
            target_modes.append(i)

            data_t = np.transpose(self.ground_truth.data, target_modes)
            data_mat = data_t.reshape([-1, data_t.shape[-1]])
            mode_size = data_mat.shape[1]
            tt_approx.U[i] = (design.T @ data_mat).reshape(left_cols, right_cols, mode_size).transpose([0, 2, 1]).copy()

        for _ in range(num_sweeps):
            for i in range(N - 1):
                optimize_core(i)
                tt_approx.orthogonalize_push_right(i)
                print(tt_als.compute_exact_fit())

            for i in range(N - 1, 0, -1):
                optimize_core(i)
                tt_approx.orthogonalize_push_left(i)
                print(tt_als.compute_exact_fit())

    def execute_randomized_als_sweeps(self, num_sweeps, J, epoch_interval=5):
        print("Starting randomized ALS!")
        tt_approx = self.tt_approx
        N = tt_approx.N
        def optimize_core(j):
            samples = np.zeros((J, N), dtype=np.uint64)
            left_rows = None
            right_rows = None
            if j > 0:
                start = time.time()
                left_samples = tt_approx.leverage_sample(j, J, "left")
                end = time.time()
                print(f"Sampled {J} left indices in {end - start} seconds.")

                samples[:, :j] = left_samples

                start = time.time()
                left_rows = tt_approx.evaluate_partial_fast(samples, j, "left")
                end = time.time()
                print(f"Evaluated {J} left indices in {end - start} seconds.")
                left_cols = left_rows.shape[1]
            else:
                left_cols = 1
            if j < N - 1:
                start = time.time()
                right_samples = tt_approx.leverage_sample(j, J, "right")
                end = time.time()
                print(f"Sampled {J} right indices in {end - start} seconds.")
                samples[:, j+1:] = right_samples
                
                start = time.time()
                right_rows = tt_approx.evaluate_partial_fast(samples, j, "right")
                end = time.time()
                print(f"Evaluated {J} right indices in {end - start} seconds.")
                right_cols = right_rows.shape[1]
            else:
                right_cols = 1

            if left_rows is None:
                design = right_rows
            elif right_rows is None:
                design = left_rows
            else:
                # Should probably write a custom kernel for this in C++ 
                design = np.einsum("ij,ik->ijk", left_rows, right_rows).reshape(J, -1)

            weights = la.norm(design, axis=1) ** 2 / design.shape[1] * J

            # We do this in two steps so we can (potentially) compute the
            # gram matrix 
            design = np.einsum("ij,i->ij", design, np.sqrt(1.0 / weights))
            design_gram = design.T @ design
            design = np.einsum("ij,i->ij", design, np.sqrt(1.0 / weights))

            result = np.zeros((tt_approx.dims[j], design.shape[1]), dtype=np.double)
            self.ground_truth.ten.execute_downsampled_mttkrp(
                    samples,
                    design,
                    j,
                    result)

            result = result @ la.pinv(design_gram) 
            tt_approx.U[j] = result.reshape(tt_approx.dims[j], left_cols, right_cols).transpose([1, 0, 2]).copy()

        for i in range(num_sweeps):
            print(f"Starting sweep {i}...")
            for j in range(N - 1):
                optimize_core(j)
                tt_approx.orthogonalize_push_right(j)
                tt_approx.update_internal_sampler(j, "left", True)

            for j in range(N - 1, 0, -1):
                optimize_core(j)
                tt_approx.orthogonalize_push_left(j)
                tt_approx.update_internal_sampler(j, "right", True)

            if i % epoch_interval == 0:
                print(self.compute_exact_fit())

if __name__=='__main__': 
    I = 100
    R = 8
    N = 4

    tt_approx = TensorTrain([I] * N, [R] * (N - 1))
    tt_approx_GT = TensorTrain([I] * N, [R] * (N - 1))
    ground_truth = PyDenseTensor(tt_approx_GT.materialize_dense()) 

    tt_approx.place_into_canonical_form(0)
    tt_als = TensorTrainALS(ground_truth, tt_approx)

    print(tt_als.compute_exact_fit())

    J = 1000
    tt_approx.build_fast_sampler(0, J=J)
    tt_als.execute_randomized_als_sweeps(num_sweeps=5, J=J)
