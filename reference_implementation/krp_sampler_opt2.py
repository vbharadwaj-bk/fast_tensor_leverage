import numpy as np
import numpy.linalg as la

from reference_implementation.lemma_sampler import *

def batch_dot_product(A, B):
    return np.einsum('ij,ij->j', A, B)

def chain_had_prod(matrices):
    res = np.ones(matrices[0].shape)
    for mat in matrices:
        res *= mat
    return res

class EfficientKRPSampler:
    def __init__(self, U):
        '''
        U = [U_1, ..., U_N] is a list of matrices. All must have
        the same column dimension R.
        '''
        self.U = U
        self.N = len(U)
        self.R = U[0].shape[1]
        self.Z_samplers = {}
        for j in range(self.N):
            self.Z_samplers[j] = LemmaSampler(U[j], Y=np.ones((self.R, self.R)), F=self.R)

    def KRPDrawSamples(self, j, J):
        '''
        Returns J samples from the KRP excluding index j (to include all
        matrices, set j < 0 or j >= N). Samples drawn according to the
        exact leverage score distribution on the KRP.
        '''
        gram_matrices = [self.Z_samplers[k].G[0] for k in range(self.N) if k != j]
        G = chain_had_prod(gram_matrices)
        G_geq_k = la.pinv(G) 
        Lambda_VT = {}
        E_samplers = {}
        for k in reversed(range(self.N)):
            if k != j:
                W, V = la.eigh(G_geq_k)
                G_geq_k *= gram_matrices[k]
                Lambda_VT [k] = np.diag(np.sqrt(W)) @ V.T
                E_samplers[k] = LemmaSampler(Lambda_VT[k], Y=gram_matrices[k], F=1) 

        samples = []
        for _ in range(J):
            h = np.ones(self.R)
            sample = []
            for k in range(self.N):
                if k == j:
                    continue 

                u_k = E_samplers[k].RowSample(h)
                scaled_h = Lambda_VT [k][u_k] * h
                t_k = self.Z_samplers[k].RowSample(scaled_h)
                h *= self.U[k][t_k, :]
                sample.append(t_k)

            samples.append(sample)

        return np.array(samples, dtype=np.uint64)

