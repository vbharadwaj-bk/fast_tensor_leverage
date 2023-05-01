import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from reference_implementation.segment_tree import * 

import cppimport.import_hook
from cpp_ext.tt_module import TTSampler 

def batch_dot_product(A, B):
    return np.einsum('ij,ij->j', A, B)

class LemmaSampler:
    def __init__(self, U, Y, F):
        self.I = U.shape[0]
        self.U = U
        self.F = F
        self.Y = Y
        self.G = {}
        self.Y_all_ones = np.sum(np.abs(Y - 1.0)) == 0

        tree = SegmentTree(self.I, self.F)
        self.tree = tree 
        for v in reversed(range(tree.node_count)):
            if tree.is_leaf(v):
                start, end = tree.S(v)
                self.G[v] = U[start:end].T @ U[start:end] 
            else:
                self.G[v] = self.G[tree.L(v)] + self.G[tree.R(v)]
                self.G[tree.R(v)] = None

    def m(self, h, v):
        return h @ (self.G[v] * self.Y) @ h.T

    def q(self, h, v):
        start, end = self.tree.S(v)
        W = self.U[start:end]
        if self.Y_all_ones:
            return (W @ h) ** 2
        else:
            return np.diag(W @ (self.Y * np.outer(h, h)) @ W.T)

    def RowSample(self, h):
        m = lambda v : self.m(h, v)
        q = lambda v : self.q(h, v)
        return self.tree.STSample(m, q)

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

        self.left_matricizations = [None] * self.N
        self.right_matricizations = [None] * self.N

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
        Warning: before calling this function, the TT
        must be in canonical form with the specified core
        non-orthogonal.
        '''
        self.internal_sampler = TTSampler(self.N, J, max(self.ranks))

        self.samplers = {}
        for i in range(self.N):
            if i < idx_nonorthogonal:
                cols = self.ranks[i+1]
                self.samplers[i] = LemmaSampler(
                    self.U[i].view().reshape(self.ranks[i] * self.dims[i], self.ranks[i+1]).copy(),
                    np.ones((cols, cols)),
                    cols 
                )
                self.left_matricizations[i] = self.U[i].view().reshape(self.ranks[i] * self.dims[i], self.ranks[i+1]).copy()
                self.internal_sampler.update_matricization(self.left_matricizations[i], i)

            # TODO: This is broken, need to fix it... 
            if i > idx_nonorthogonal:
                cols = self.ranks[i+1]
                self.samplers[i] = LemmaSampler(
                    self.U[i].view().reshape(self.ranks[i] * self.dims[i], self.ranks[i+1]).copy(),
                    np.ones((cols, cols)),
                    cols)

    def leverage_sample(self, j, J):
        '''
        TODO: Should allow drawing more than one sample! Also -
        this only draws a sample from the left contraction for now.
        '''
        rng = np.random.default_rng()
        sample_idxs = np.zeros((j, J), dtype=np.uint64)

        H_old = None

        for i in reversed(range(j)):
            H_new = np.zeros((J, self.ranks[i]))

            if i == j-1:
                I = self.U[i].shape[1]
                R = self.U[i].shape[2] 

                for k in range(J):
                    first_col_idx = rng.choice(R)
                    leverage_scores = la.norm(self.U[i][:, :, first_col_idx], axis=0) ** 2
                    leverage_scores /= np.sum(leverage_scores)
                    row_idx = rng.choice(range(I), p=leverage_scores)
                    sample_idxs[i, k] = row_idx
                    H_new[k, :] = self.U[i][:, row_idx, first_col_idx].T
            else:
                for k in range(J):
                    sample_idxs[i, k] = self.samplers[i].RowSample(H_old[k, :])

                for k in range(J):
                    idx = sample_idxs[i, k]
                    idx_mod = np.fmod(idx, np.array(self.dims[i], dtype=np.uint64))
                    sample_idxs[i, k] = idx_mod 
                    H_new[k, :] = H_old[k, :] @ self.U[i][:, idx_mod, :].T

            H_old = H_new

        sample_idxs = sample_idxs.T
        sample_rows = H_new

        return np.array(sample_idxs, dtype=np.uint64), np.array(sample_rows) 

    def linearize_idxs_left(self, idxs):
        cols = idxs.shape[1]
        entry = 1
        vec = np.zeros(cols, dtype=np.uint64) 
        for i in range(cols):
            vec[i] = entry
            entry *= self.dims[i]

        return idxs @ np.flip(vec)

def test_tt_sampling():
    I = 6
    R = 2
    N = 4
    J = 10000

    dims = [I] * N 
    ranks = [R] * (N-1) 

    seed = 20
    tt = TensorTrain(dims, ranks, seed)

    # Sweep so that the rightmost core is orthogonal
    #for i in range(N - 1):
    #    tt.orthogonalize_push_right(i) 

    tt.place_into_canonical_form(N-1)
    tt.build_fast_sampler(N-1, J)
    left_chain = tt.left_chain_matricize(N-1)

    U0 = tt.U[0]
    U1 = tt.U[1]

    normsq_rows = la.norm(left_chain, axis=1) ** 2

    #normsq_rows = left_chain ** 2
    normsq_rows_normalized = normsq_rows / np.sum(normsq_rows)

    #samples, rows = tt.leverage_sample_alternate(j=N-1, J=J)
    samples, rows = tt.leverage_sample(j=N-1, J=J)

    linear_idxs = np.array(tt.linearize_idxs_left(samples), dtype=np.int64)

    fig, ax = plt.subplots()
    ax.plot(normsq_rows_normalized, label="True leverage distribution")
    bins = np.array(np.bincount(linear_idxs, minlength=len(normsq_rows_normalized))) / J

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
