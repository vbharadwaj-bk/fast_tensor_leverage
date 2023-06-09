from tensor_train import *
from dense_tensor import *

class TensorTrainALS:
    def __init__(self, ground_truth, tt_approx):
        self.ground_truth = ground_truth
        self.tt_approx = tt_approx

    def compute_exact_fit(self):
        '''
        This operation can be particularly expensive for dense tensors.
        '''
        if isinstance(self.ground_truth, PyDenseTensor):
            tt_materialization = self.tt_approx.materialize_dense()
            return 1.0 - la.norm(tt_materialization - self.ground_truth.data) / self.ground_truth.data_norm
        else:
            raise NotImplementedError
        


if __name__=='__main__': 
    I = 2
    R = 4
    N = 3

    data = np.zeros([I] * N)
    print(data)

