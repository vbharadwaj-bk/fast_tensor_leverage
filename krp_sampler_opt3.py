import numpy as np
import numpy.linalg as la

import cppimport.import_hook
from cpp_ext.partition_tree import PartitionTree as PartitionTreeOpt
from partition_tree import * 

def batch_dot_product(A, B):
    return np.einsum('ij,ij->j', A, B)

'''
This implementation uses the eigendecomposition to drive down 
the complexity even further. 
'''
class EfficientKRPSampler:
    def __init__(self, U, F):
        '''
        This is a close implementation of the pseudocode procedure
        "ConstructSampler". 

        U = [U_1, ..., U_N] is a list of matrices. All must have
        the same column dimension R.

        F = [F_1, ..., F_N] is a list of integers >= 1 
        '''
        self.U = U
        self.N = len(U)
        self.R = U[0].shape[1]
        self.trees = []
        self.G = [] 
        for j in range(self.N):
            tree = PartitionTree(U[j].shape[0], F[j])
            self.trees.append(tree)
            self.G.append({})

            for v in reversed(range(tree.node_count)):
                if tree.is_leaf(v):
                    start, end = tree.S(v)
                    self.G[j][v] = U[j][start:end].T @ U[j][start:end] 
                else:
                    self.G[j][v] = self.G[j][tree.L(v)] + self.G[j][tree.R(v)]
                    self.G[j][tree.R(v)] = None

    def m(self, h, k, v):
        return h @ (self.G[k][v]) @ h.T

    def q(self, h, k, v):
        start, end = self.trees[k].S(v)
        W = self.U[k][start:end]
        return (W @ h) ** 2

    def computeM(self, j):
        '''
        Compute M_k for the KRP of all matrices excluding
        U_j. Also compute the eigendecomposition of each M_k,
        which will be useful to us. 
        '''
        G = chain_had_prod([self.G[k][0] for k in range(self.N) if k != j])
        M_buffer = la.pinv(G) 

        self.M = {}
        self.eigvecs = {}
        self.eigvals = {}

        for k in reversed(range(self.N)):
            if k != j:
                self.M[k] = M_buffer.copy()
                W, V = la.eigh(M_buffer)
                self.eigvecs[k] = V
                self.eigvals[k] = W
                M_buffer *= self.G[k][0] 

    def KRPDrawSample(self, j):
        h = np.ones(self.R)
        vector_idxs = []
        scalar_idx = 0
        for k in range(self.N):
            if k == j:
                continue 

            Y = self.eigvecs[k]
            eig_weights = self.eigvals[k] * batch_dot_product(Y, np.outer(h, h) * self.G[k][0] @ Y)

            Rc = np.random.multinomial(1, eig_weights / np.sum(eig_weights))
            scaled_h = self.eigvecs[k][:, np.nonzero(Rc==1)[0][0]] * h

            m = lambda v : self.m(scaled_h, k, v)
            q = lambda v : self.q(scaled_h, k, v)

            ik = self.trees[k].PTSampleUpgraded(m, q)
            h *= self.U[k][ik, :]
            vector_idxs.append(ik)
            scalar_idx = (scalar_idx * self.U[k].shape[0]) + ik

        return h, scalar_idx, vector_idxs

    def KRPDrawSamples_scalar(self, j, J):
        '''
        Draws J samples from the KRP excluding J. Returns the scalar
        indices of each sampled row in the Khatri-Rao product. 
        '''
        self.computeM(j)
        samples = []
        for _ in range(J):
            samples.append(self.KRPDrawSample(j)[1])

        return samples

