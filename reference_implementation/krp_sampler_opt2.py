import numpy as np
import numpy.linalg as la

from reference_implementation.partition_tree import *
from reference_implementation.lemma_sampler import *

def batch_dot_product(A, B):
    return np.einsum('ij,ij->j', A, B)

'''
This implementation uses the eigendecomposition to drive down 
the complexity even further. 
'''
class EfficientKRPSampler:
    def __init__(self, U, F):
        '''
        U = [U_1, ..., U_N] is a list of matrices. All must have
        the same column dimension R.

        F = [F_1, ..., F_N] is a list of integers >= 1 
        '''
        self.U = U
        self.N = len(U)
        self.R = U[0].shape[1]
        self.Z_samplers = {}
        for j in range(self.N):
            self.Z_samplers[j] = LemmaSampler(U[j], np.ones((self.R, self.R)), F[j])

    def computeM(self, j):
        gram_matrices = [self.Z_samplers[k].G[0] for k in range(self.N) if k != j]
        G = chain_had_prod(gram_matrices)
        M_buffer = la.pinv(G) 
        self.scaled_eigenvectors = {}
        self.E_samplers = {}

        for k in reversed(range(self.N)):
            if k != j:
                W, V = la.eigh(M_buffer)
                M_buffer *= gram_matrices[k]
                self.scaled_eigenvectors[k] = np.diag(np.sqrt(W)) @ V.T
                self.E_samplers[k] = LemmaSampler(self.scaled_eigenvectors[k], gram_matrices[k], F=1) 

    def KRPDrawSamples(self, j, J):
        '''
        Draws J samples from the KRP excluding J. Returns the scalar
        indices of each sampled row in the Khatri-Rao product. 
        '''
        self.computeM(j)
        gram_matrices = [self.Z_samplers[k].G[0] for k in range(self.N) if k != j]
        samples = []

        for _ in range(J):
            h = np.ones(self.R)
            vector_idxs = []
            for k in range(self.N):
                if k == j:
                    continue 

                Rc = self.E_samplers[k].RowSample(h)
                scaled_h = self.scaled_eigenvectors[k][Rc] * h
                ik = self.Z_samplers[k].RowSample(scaled_h)
                h *= self.U[k][ik, :]
                vector_idxs.append(ik)

            samples.append(vector_idxs)

        return np.array(samples, dtype=np.uint64)

