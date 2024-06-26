import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import tensorly as tl

import cppimport.import_hook
from cpp_ext.tt_module import TTSampler


def batch_dot_product(A, B):
    return np.einsum('ij,ij->j', A, B)


class TensorTrain:
    '''
    Core[i] has dimensions (ranks[i], dims[i], ranks[i+1]) 
    '''

    def __init__(self, dims, rank, seed=None, init_method=None, U=None, reduce_rank=True):
        if seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(seed)

        self.dims = np.array(dims, dtype=np.uint64)
        self.N = len(dims)
        self.rank = rank
        self.U = U
        self.ranks = np.array([1] + [self.rank] * (self.N - 1) + [1], np.uint64)
        self.rng = rng
        # self.std_noise = std_noise

        reduced_ranks = []

        for i in range(self.N - 1):
            left_dim = np.prod(self.dims[:i + 1])
            right_dim = np.prod(self.dims[i + 1:])
            oldrank = self.ranks[i + 1]

            if reduce_rank:
                self.ranks[i + 1] = min(left_dim, right_dim, self.ranks[i + 1])

            if self.ranks[i + 1] != oldrank:
                reduced_ranks.append(i + 1)

        if len(reduced_ranks) > 0:
            print(f"Reduced ranks at indices {reduced_ranks} to maximum allowable.")

        assert (len(self.ranks) == self.N + 1)

        if init_method == "gaussian" and self.U is None:
            self.U = [rng.normal(size=(self.ranks[i], self.dims[i], self.ranks[i + 1])) for i in range(self.N)]
            for i in range(self.N):
                self.U[i] /= la.norm(self.U[i])

        elif init_method is None and self.U is not None:
            for i in range(self.N):
                self.U[i] /= la.norm(self.U[i])
                print(U[i].shape)


        else:
            raise (NotImplementedError())

        self.matricizations = [None] * self.N
        self.internal_sampler = None

    def reinitialize_gaussian(self):
        for i in range(self.N):
            self.U[i][:] = self.rng.normal(size=(self.ranks[i], self.dims[i], self.ranks[i + 1]))
            self.U[i] /= la.norm(self.U[i])

    def clone(self):
        clone = TensorTrain(self.dims, list(self.ranks[1:-1]))

        for i in range(self.N):
            clone.U[i][:] = self.U[i]

        return clone

    def evaluate_left(self, idxs, upto=-1):
        left = np.ones((1, 1), dtype=np.double)

        if upto == -1:
            upto = self.N

        for i in range(upto):
            left = left @ self.U[i][:, idxs[i], :]

        return left

    def evaluate_right(self, idxs, upto=-1):
        right = np.ones((1, 1), dtype=np.double)

        for i in reversed(range(upto + 1, self.N)):
            right = right @ self.U[i][:, idxs[i], :].T

        return right

    def orthogonalize_push_right(self, idx):
        assert (idx < self.N - 1)
        dim = self.dims[idx]
        Q, R = la.qr(self.U[idx].view().reshape(self.ranks[idx] * dim, self.ranks[idx + 1]))
        self.U[idx][:] = Q.reshape(self.ranks[idx], dim, self.ranks[idx + 1])
        self.U[idx + 1][:] = np.einsum('ij,jkl->ikl', R, self.U[idx + 1])

    def orthogonalize_push_left(self, idx):
        assert (idx > 0)
        dim = self.dims[idx]
        Q, R = la.qr(self.U[idx].view().reshape(self.ranks[idx], dim * self.ranks[idx + 1]).T)
        self.U[idx][:] = Q.T.reshape(self.ranks[idx], dim, self.ranks[idx + 1])
        self.U[idx - 1][:] = np.einsum('jkl,lm->jkm', self.U[idx - 1], R.T)

    def place_into_canonical_form(self, idx_nonorthogonal):
        idx = idx_nonorthogonal
        for i in range(idx):
            self.orthogonalize_push_right(i)

        for i in reversed(range(idx + 1, self.N)):
            self.orthogonalize_push_left(i)

    def contract_cores(self, core1, core2):
        '''
        Returns a (3, 1) contraction of two provided TT-cores, reshaped 
        so the physical dimensions are combined and the output has
        dimension 3.
        '''
        ph_dim1, ph_dim2 = core1.shape[1], core2.shape[1]
        rank_left, rank_right = core1.shape[0], core2.shape[2]

        return np.einsum('abc,cde->abde', core1, core2).reshape(rank_left, ph_dim1 * ph_dim2, rank_right)

    def left_chain_matricize(self, j):
        if j == 0:
            contraction = np.ones((1, 1), dtype=np.double)
        else:
            contraction = self.U[0]

            for i in range(1, j):
                contraction = self.contract_cores(contraction, self.U[i])

        return np.squeeze(contraction)

    def right_chain_matricize(self, j):
        if j == self.N - 1:
            contraction = np.ones((1, 1), dtype=np.double)
        else:
            contraction = self.U[j + 1]

            for i in range(j + 2, self.N):
                contraction = self.contract_cores(contraction, self.U[i])

        return np.squeeze(contraction).T

    def materialize_dense(self):
        '''
        Materializes the dense tensor that this TT represents.
        Warning: This is an expensive call!
        '''
        tensor_vec = self.left_chain_matricize(self.N)
        return tensor_vec.reshape(self.dims)

    def update_internal_sampler(self, i, direction, rebuild_tree):
        '''
        Updates the matricization associated with a given core. The change
        is propagated to the internal sampler. 
        '''
        if direction == "left":
            self.matricizations[i] = self.U[i].view().transpose([1, 0, 2]).reshape(self.ranks[i] * self.dims[i],
                                                                                   self.ranks[i + 1]).copy()
            self.internal_sampler.update_matricization(self.matricizations[i], i, 1, rebuild_tree)
        elif direction == "right":
            self.matricizations[i] = self.U[i].view().transpose([1, 2, 0]).reshape(self.ranks[i + 1] * self.dims[i],
                                                                                   self.ranks[i]).copy()
            self.internal_sampler.update_matricization(self.matricizations[i], i, 0, rebuild_tree)
        else:
            raise (ValueError("Invalid direction!"))

    def build_fast_sampler(self, idx_nonorthogonal, J):
        '''
        Before calling this function, the TT
        must be in canonical form with the specified core
        non-orthogonal.
        '''
        self.internal_sampler = TTSampler(self.N,
                                          J,
                                          max(self.ranks),
                                          self.dims)

        for i in range(self.N):
            if i < idx_nonorthogonal:
                self.update_internal_sampler(i, "left", True)

            # The non-orthogonal core is always left-matricized,
            # but we don't build a tree for it
            if i == idx_nonorthogonal:
                self.update_internal_sampler(i, "left", False)

            if i > idx_nonorthogonal:
                self.update_internal_sampler(i, "right", True)

    def leverage_sample(self, j, J, direction):
        if direction == "left":
            direction_int = 1
            sample_idxs = np.zeros((j, J), dtype=np.uint64)
        elif direction == "right":
            sample_idxs = np.zeros((self.N - j - 1, J), dtype=np.uint64)
            direction_int = 0

        self.internal_sampler.sample(j, J, sample_idxs, direction_int)
        sample_idxs = sample_idxs.T
        return np.array(sample_idxs, dtype=np.uint64)

    def linearize_idxs_left(self, idxs):
        cols = idxs.shape[1]
        entry = 1
        vec = np.zeros(cols, dtype=np.uint64)
        for i in range(cols):
            vec[i] = entry
            entry *= self.dims[i]

        return idxs @ np.flip(vec)

    def linearize_idxs_right(self, idxs):
        cols = idxs.shape[1]
        entry = 1
        vec = np.zeros(cols, dtype=np.uint64)
        for i in reversed(range(self.N - cols, self.N)):
            vec[i - self.N + cols] = entry
            entry *= self.dims[i]

        return idxs @ vec

    def evaluate_partial_fast(self, idxs, upto, direction):
        J = idxs.shape[0]
        if direction == "left":
            result = np.zeros((J, self.ranks[upto]), dtype=np.double)
            direction_int = 1
        elif direction == "right":
            result = np.zeros((J, self.ranks[upto + 1]), dtype=np.double)
            direction_int = 0
        else:
            assert (False)

        self.internal_sampler.evaluate_indices_partial(idxs, upto, direction_int, result)
        return result

    def evaluate_at_idxs(self, idxs, quantization=None):
        if quantization is None:
            idxs_to_use = idxs
        else:
            idxs_to_use = quantization.quantize_indices(idxs)

        return self.evaluate_partial_fast(
            idxs_to_use,
            self.N, "left").squeeze()
