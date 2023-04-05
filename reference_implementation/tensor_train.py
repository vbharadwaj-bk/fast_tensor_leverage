import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from lemma_sampler import *

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

    def build_fast_sampler(self, idx_nonorthogonal):
        '''
        Warning: before calling this function, the TT
        must be in canonical form with the specified core
        non-orthogonal.
        '''
        self.samplers = {}
        for i in range(self.N):
            if i < idx_nonorthogonal:
                cols = self.ranks[i]
                self.samplers[i] = LemmaSampler(
                    self.U[i].view().reshape(self.ranks[i], self.dims[i] * self.ranks[i+1]).T.copy(),
                    np.ones((cols, cols)),
                    cols
                ) 
            if i > idx_nonorthogonal:
                cols = self.ranks[i+1]
                self.samplers[i] = LemmaSampler(
                    self.U[i].view().reshape(self.ranks[i] * self.dims[i], self.ranks[i+1]).copy(),
                    np.ones((cols, cols)),
                    cols)

        # Conduct a test on the second matrix in the list 
        i = 1
        reshaped_core = self.U[i].view().reshape(self.ranks[i], self.dims[i] * self.ranks[i+1]).T.copy()
        print(reshaped_core)
        print("----------------------")
        print(self.U[i][:, 1, :])
        print("----------------------")
        print(self.U[i-1])

        test_scores = np.einsum('ij,kj->ik', self.U[i-1].squeeze(), reshaped_core) ** 2
        print(np.sum(test_scores))
        exit(1)


    def leverage_sample(self, j, J):
        '''
        TODO: Should allow drawing more than one sample! Also -
        this only draws a sample from the left contraction for now.
        '''

        sample_idxs = []
        sample_rows = []
        for _ in range(J): 
            h_left = np.ones(1)
            idx_left = []
            
            for i in range(j):
                idx = self.samplers[i].RowSample(h_left)
                idx_mod = idx // self.ranks[i+1] # This is either div or mod, must figure out which 
                idx_left.append(idx_mod)
                h_left = h_left @ self.U[i][:, idx_mod, :]

            sample_idxs.append(idx_left)
            sample_rows.append(h_left)

        return np.array(sample_idxs, dtype=np.uint64), np.array(sample_rows) 

    def linearize_idxs_left(self, idxs):
        cols = idxs.shape[1]
        entry = 1
        vec = np.zeros(cols, dtype=np.uint64) 
        for i in range(cols):
            vec[i] = entry
            entry *= self.dims[i]

        return idxs @ np.flip(vec)


def test_tt_functions_small():
    I = 2
    R = 2
    N = 2

    dims = [I] * N
    ranks = [R] * (N - 1)

    seed = 20
    tt = TensorTrain(dims, ranks, seed)

    # Test evaluation at a particular index 
    print(tt.evaluate_left([1, 1, 1], upto=-1))
    print(tt.evaluate_right([1, 1, 1], upto=-1))

    # Test evaluations after a right-sweep orthogonalization 
    for i in range(N - 1):
        tt.orthogonalize_push_right(i)

    print(tt.evaluate_left([1, 1, 1], upto=-1))
    print(tt.evaluate_right([1, 1, 1], upto=-1))

    # Test orthogonality of matricization
    left_chain = tt.left_chain_matricize(2)
    print(left_chain.T @ left_chain)

    # Place into canonical form and build a sampler 
    tt.place_into_canonical_form(2)
    tt.build_fast_sampler(2)

    # Draw a tensor-train sample
    samples, rows = tt.leverage_sample(j=2, J=10)
    print(rows)
    linear_idxs = tt.linearize_idxs_left(samples)
    print(linear_idxs)
    print(left_chain)

    # Test evaluations after a left-sweep orthogonalization 
    #for i in reversed(range(1, N)):
    #    tt.orthogonalize_push_left(i)

    # Test orthogonality of matricization
    #right_chain = tt.right_chain_matricize(0)
    #print(right_chain.T @ right_chain)

    #print(tt.evaluate_left([1, 1, 1], upto=-1))
    #print(tt.evaluate_right([1, 1, 1], upto=-1))

def test_tt_sampling():
    I = 4
    R = 3
    N = 3

    dims = [I] * N 
    ranks = [R] * (N - 1)

    seed = 20
    tt = TensorTrain(dims, ranks, seed)

    # Sweep so that the rightmost core is orthogonal
    #for i in range(N - 1):
    #    tt.orthogonalize_push_right(i) 

    tt.place_into_canonical_form(N-1)
    tt.build_fast_sampler(N-1)
    left_chain = tt.left_chain_matricize(N-1)

    normsq_rows = la.norm(left_chain, axis=1) ** 2
    normsq_rows_normalized = normsq_rows / np.sum(normsq_rows)


    J = 100000
    samples, rows = tt.leverage_sample(j=N-1, J=J)
    linear_idxs = np.array(tt.linearize_idxs_left(samples), dtype=np.int64)

    fig, ax = plt.subplots()
    ax.plot(normsq_rows_normalized, label="True leverage distribution")
    bins = np.array(np.bincount(linear_idxs, minlength=len(normsq_rows_normalized))) / len(linear_idxs)
    ax.plot(bins, label="Our sampler")
    ax.set_xlabel("Row Index")
    ax.set_ylabel("Probability Density")
    ax.grid(True)
    ax.legend()
    ax.set_title(f"{J} samples drawn by our method vs. true distribution")

    fig.savefig('../plotting/distribution_comparison.png')


if __name__=='__main__':
    #test_tt_functions_small()
    test_tt_sampling()
