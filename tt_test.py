import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from reference_implementation.segment_tree import * 

import cppimport.import_hook
from cpp_ext.tt_module import TTSampler 

def batch_dot_product(A, B):
    return np.einsum('ij,ij->j', A, B)

class TensorTrain:
    '''
    Core[i] has dimensions (ranks[i], dims[i], ranks[i+1]) 
    '''
    def __init__(self, dims, ranks, seed=None, init_method="gaussian"): 
        if seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(seed)

        if init_method=="gaussian":
            self.dims = np.array(dims, dtype=np.uint64)
            self.N = len(dims)
            self.ranks = np.array([1] + ranks + [1], np.uint64)

            self.U = [rng.normal(size=(self.ranks[i], self.dims[i], self.ranks[i+1])) for i in range(self.N)]
        else:
            assert(False)

        self.matricizations = [None] * self.N
        self.internal_sampler = None

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

    def evaluate_partial_fast(self, idxs, upto, direction):
        J = idxs.shape[1]
        if direction == "left":
            result = np.zeros((J, self.ranks[upto]), dtype=np.double)
            direction_int = 1
        elif direction == "right":
            result = np.zeros((J, self.ranks[upto+1]), dtype=np.double)
            direction_int = 0
        else:
            assert(False)

        self.internal_sampler.evaluate_indices_partial(idxs, upto, direction_int, result)

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

    def place_into_canonical_form(self, idx_nonorthogonal):
        idx = idx_nonorthogonal
        for i in range(idx):
            self.orthogonalize_push_right(i)

        for i in reversed(range(idx+1, self.N)):
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
            contraction = np.ones(1, 1)
        else:
            contraction = self.U[0]

            for i in range(1, j):
                contraction = self.contract_cores(contraction, self.U[i])

        return np.squeeze(contraction)

    def right_chain_matricize(self, j):
        if j == self.N-1:
            contraction = np.ones(1, 1)
        else:
            contraction = self.U[j+1]

            for i in range(j+2, self.N):
                contraction = self.contract_cores(contraction, self.U[i])

        return np.squeeze(contraction).T

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
                self.matricizations[i] = self.U[i].view().transpose([1, 0, 2]).reshape(self.ranks[i] * self.dims[i], self.ranks[i+1]).copy()
                self.internal_sampler.update_matricization(self.matricizations[i], i, 1, True)

            # The non-orthogonal core is always left-matricized,
            # but we don't build a tree for it
            if i == idx_nonorthogonal:
                self.matricizations[i] = self.U[i].view().transpose([1, 0, 2]).reshape(self.ranks[i] * self.dims[i], self.ranks[i+1]).copy()
                self.internal_sampler.update_matricization(self.matricizations[i], i, 1, False)

            if i > idx_nonorthogonal:
                self.matricizations[i] = self.U[i].view().transpose([1, 2, 0]).reshape(self.ranks[i+1] * self.dims[i], self.ranks[i]).copy()
                self.internal_sampler.update_matricization(self.matricizations[i], i, 0, True)

    def leverage_sample(self, j, J, direction):
        '''
        TODO: Should allow drawing more than one sample! 
        '''
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

def test_tt_sampling():
    I = 20
    R = 4
    N = 3
    J = 300000

    dims = [I] * N 
    ranks = [R] * (N-1) 

    seed = 20
    tt = TensorTrain(dims, ranks, seed)

    test_direction = "right"

    if test_direction == "left":
        tt.place_into_canonical_form(N-1)
        tt.build_fast_sampler(N-1, J)
        samples = tt.leverage_sample(j=N-1, J=J, direction="left")
        linear_idxs = np.array(tt.linearize_idxs_left(samples), dtype=np.int64)
        left_chain = tt.left_chain_matricize(N-1)
        normsq_rows_left = la.norm(left_chain, axis=1) ** 2
        normsq_rows_normalized_left = normsq_rows_left / np.sum(normsq_rows_left)
        true_dist = normsq_rows_normalized_left
    else:
        tt.place_into_canonical_form(0)
        tt.build_fast_sampler(0, J)
        samples = tt.leverage_sample(j=0, J=J, direction="right")
        linear_idxs = np.array(tt.linearize_idxs_right(samples), dtype=np.int64)
        right_chain = tt.right_chain_matricize(0)
        normsq_rows_right = la.norm(right_chain, axis=1) ** 2 
        normsq_rows_normalized_right = normsq_rows_right / np.sum(normsq_rows_right)
        true_dist = normsq_rows_normalized_right

    fig, ax = plt.subplots()
    ax.plot(true_dist, label="True leverage distribution")
    bins = np.array(np.bincount(linear_idxs, minlength=len(true_dist))) / J

    ax.plot(bins, label="Our sampler")
    ax.set_xlabel("Row Index")
    ax.set_ylabel("Probability Density")
    ax.grid(True)
    ax.legend()
    ax.set_title(f"{J} samples drawn by our method vs. true distribution")

    fig.savefig('plotting/distribution_comparison.png')

if __name__=='__main__':
    #test_tt_functions_small()
    test_tt_sampling()
