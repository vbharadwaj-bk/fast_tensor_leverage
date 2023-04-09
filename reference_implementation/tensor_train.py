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
        #reshaped_core = self.U[i].view().reshape(self.ranks[i], self.dims[i] * self.ranks[i+1]).T.copy()
        #print(reshaped_core)
        #print("----------------------")
        #print(self.U[i][:, 1, :])
        #print("----------------------")
        #print(self.U[i-1])

        #sampler = LemmaSampler(
        #    reshaped_core, 
        #    np.ones((cols, cols)),
        #    12
        #) 

        #total = 0.0
        #for k in range(4):
        #    test_scores = np.einsum('j,kj->k', self.U[i-1][:, k, :].squeeze(), reshaped_core) ** 2
        #    total += np.sum(test_scores)
        #    print(np.sum(test_scores) / 3)
        #    print(test_scores / np.sum(test_scores))

        #    h = self.U[i-1][:, k, :].view().squeeze()

        #    hist = np.zeros(12)
        #    J = 30000
        #    for _ in range(J):
        #        idx = sampler.RowSample(h)
        #        hist[idx] += 1/J

        #    print(hist)
        #    print('=' * 20)

        #print(total)

        #hist = np.zeros(4)
        #for _ in range(J):
        #    idx = self.samplers[0].RowSample(np.ones(1)) // self.ranks[1] 
        #    hist[idx] += 1/J

        #print(hist)

        #first_mat = self.U[0].squeeze()
        #print(first_mat.T @ first_mat)
        #exit(1)


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

    def leverage_sample_alternate(self, j, J):
        '''
        Okay, for now this function only works if the
        non-orthogonal core is the rightmost core in the
        chain, i.e. j = N-1. Also, this function only works
        right now on chains of order 2, so we can test if
        the algorithm works correctly.
        '''
        assert(j == self.N-1)
        I = self.U[0].shape[1]
        R = self.U[0].shape[2]
        sample_idxs = []
        sample_rows = [] 

        rng = np.random.default_rng()

        for _ in range(J): 
            idx_left = []

            col_idx1 = 0
            #col_idx1 = rng.choice(range(R))
            leverage_scores = la.norm(self.U[1][:, :, col_idx1], axis=0) ** 2
            leverage_scores /= np.sum(leverage_scores)
            row_idx1 = rng.choice(range(I), p=leverage_scores)
            column_normsq = self.U[1][:, row_idx1, col_idx1] ** 2

            # For debugging only!
            #column_normsq = np.ones(R)
            # End debugging session!

            column_normsq = column_normsq / np.sum(column_normsq) 
            col_idx0 = rng.choice(range(R), p=column_normsq)
            leverage_scores = self.U[0][:, :, col_idx0].squeeze() ** 2 

            #for idx in range(R):
                #scores = la.norm(self.U[0][:, :, idx], axis=0) ** 2
                #print(scores)

            #exit(1)
            leverage_scores /= np.sum(leverage_scores)

            true_scores = (self.U[0].squeeze() @ self.U[1][:, row_idx1, col_idx1]) ** 2
            true_scores /= np.sum(true_scores)

            row_idx0 = rng.choice(range(I), p=true_scores)

            #row_idx0 = rng.choice(range(I), p=leverage_scores)
            sample_idxs.append([row_idx0, row_idx1])


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
    I = 2
    R = 2
    N = 3 

    dims = [I] * N 
    ranks = [R, 1] 

    seed = 20
    tt = TensorTrain(dims, ranks, seed)

    # Sweep so that the rightmost core is orthogonal
    #for i in range(N - 1):
    #    tt.orthogonalize_push_right(i) 

    tt.place_into_canonical_form(N-1)
    tt.build_fast_sampler(N-1)
    left_chain = tt.left_chain_matricize(N-1)

    print(left_chain ** 2)

    U0 = tt.U[0]
    U1 = tt.U[1]

    print(la.norm(U1.reshape(2, 2), axis=0) ** 2)
    print(np.sum(la.norm(U1.reshape(2, 2), axis=0) ** 2))
    print("Done!")

    normsq_rows = left_chain ** 2
    normsq_rows_normalized = normsq_rows / np.sum(normsq_rows)

    print(f"True Leverage Scores: {normsq_rows}")

    J = 10000
    samples, rows = tt.leverage_sample_alternate(j=N-1, J=J)
    print("Drew samples!")

    linear_idxs = np.array(tt.linearize_idxs_left(samples), dtype=np.int64)

    fig, ax = plt.subplots()
    ax.plot(normsq_rows_normalized, label="True leverage distribution")
    bins = np.array(np.bincount(linear_idxs, minlength=len(normsq_rows_normalized))) / J

    print(normsq_rows_normalized)
    print(bins)


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
