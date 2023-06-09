from tensor_train import *
from dense_tensor import *

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

        N = self.tt_approx.N

        def optimize_core(i):
            left_chain = self.tt_approx.left_chain_matricize(i)
            right_chain = self.tt_approx.right_chain_matricize(i)

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
            mode_cols = data_mat.shape[1]
            self.tt_approx.U[i] = (design.T @ data_mat).reshape(mode_cols, left_cols, right_cols).transpose([1, 0, 2]).copy()

        for _ in range(num_sweeps):
            for i in range(N - 1):
                optimize_core(i)
                tt_approx.orthogonalize_push_right(i)
                print(tt_als.compute_exact_fit())

            for i in range(N - 1, 0, -1):
                optimize_core(i)
                tt_approx.orthogonalize_push_left(i)
                print(tt_als.compute_exact_fit())

if __name__=='__main__': 
    I = 20
    R = 4
    N = 3

    data = np.ones([I] * N) * 5
    ground_truth = PyDenseTensor(data)
    tt_approx = TensorTrain([I] * N, [R] * (N - 1))
    tt_approx.place_into_canonical_form(0)
    tt_als = TensorTrainALS(ground_truth, tt_approx)

    sweeps = 2
    tt_als.execute_exact_als_sweeps_slow(sweeps)