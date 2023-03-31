import numpy as np
import numpy.linalg as la

class TensorTrain:
    def __init__(self, dims, ranks, seed=None, init_method="gaussian"):
        if seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(seed)

        if init_method=="gaussian":
            self.dims = dims
            self.N = len(dims)
            self.ranks = [1] + ranks + [1]

            self.U = [rng.normal(size=(self.ranks[i], self.dims[i], self.ranks[i+1])) for i in range(self.N)]
        else:
            assert(False)

    def evaluate_left(self, idxs, upto=-1):
        left = np.ones((1, 1), dtype=np.double)
 
        if upto == -1:
            upto = self.N 

        for i in range(upto):
            left = left @ self.U[i][:, idxs[i], :]

        return left

    def evaluate_right(self, idxs, upto=-1):
        right = np.ones((1, 1), dtype=np.double)

        for i in reversed(range(upto+1, self.N)):
            right = right @ self.U[i][:, idxs[i], :].T

        return right 

    def orthogonalize_push_right(self, idx):
        assert(idx < self.N - 1)
        dim = self.dims[idx]
        Q, R = la.qr(self.U[idx].view().reshape(self.ranks[idx] * dim, self.ranks[idx+1]))
        self.U[idx] = Q.reshape(self.ranks[idx], dim, self.ranks[idx+1])
        self.U[idx+1] = np.einsum('ij,jkl->ikl', R, self.U[idx + 1]) 

    def orthogonalize_push_left(self, idx):
        assert(idx > 0) 
        dim = self.dims[idx]
        Q, R = la.qr(self.U[idx].view().reshape(self.ranks[idx], dim * self.ranks[idx+1]).T)
        self.U[idx] = Q.T.reshape(self.ranks[idx], dim, self.ranks[idx+1])
        self.U[idx-1] = np.einsum('jkl,lm->jkm', self.U[idx - 1], R.T) 

def test_tt_functions_small():
    I = 2
    R = 2
    N = 3

    dims = [I] * N
    ranks = [R] * (N - 1)

    seed = 20
    tt = TensorTrain(dims, ranks, seed)

    # Test evaluation 
    print(tt.evaluate_left([1, 1, 1], upto=-1))
    print(tt.evaluate_right([1, 1, 1], upto=-1))

    # Test evaluations after a right-sweep orthogonalization 
    for i in range(N - 1):
        tt.orthogonalize_push_right(i)

    print(tt.evaluate_left([1, 1, 1], upto=-1))
    print(tt.evaluate_right([1, 1, 1], upto=-1))

    # Test evaluations after a left-sweep orthogonalization 
    for i in reversed(range(1, N)):
        tt.orthogonalize_push_left(i)

    print(tt.evaluate_left([1, 1, 1], upto=-1))
    print(tt.evaluate_right([1, 1, 1], upto=-1))


if __name__=='__main__':
    test_tt_functions_small()

