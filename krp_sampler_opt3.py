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
    def __init__(self, U, F, J):
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
        self.opt_trees = []
        self.J = J
        for j in range(self.N):
            self.opt_trees.append(PartitionTreeOpt(U[j].shape[0], F[j], J, self.R))
            self.opt_trees[j].build_tree(U[j])

    def symmetrize(self, buf):
        return buf + buf.T - np.diag(np.diag(buf))

    def computeM(self, j):
        '''
        Compute M_k for the KRP of all matrices excluding 
        U_j. Also compute the eigendecomposition of each M_k,
        which will be useful to us. 
        '''

        lst = []
        for k in range(self.N):
            if k != j:
                buf = np.zeros((self.R, self.R))
                self.opt_trees[k].get_G0(buf)
                lst.append(buf)

        G = self.symmetrize(chain_had_prod(lst))
        M_buffer = la.pinv(G) 

        self.M = {}
        self.eigvecs = {}
        self.eigvals = {}
        self.scaled_eigvecs = {}
        self.eigen_trees = {}

        for k in reversed(range(self.N)):
            if k != j:
                self.M[k] = M_buffer.copy()
                W, V = la.eigh(M_buffer)
                self.eigvecs[k] = V
                self.eigvals[k] = W
                self.scaled_eigvecs[k] = (np.sqrt(W) * V).T.copy()
                self.eigen_trees[k] = PartitionTreeOpt(self.R, 1, self.J, self.R)
                self.eigen_trees[k].build_tree(self.scaled_eigvecs[k])

                buf = np.zeros((self.R, self.R))
                self.opt_trees[k].get_G0(buf)
                buf = self.symmetrize(buf)
                self.eigen_trees[k].multiply_against_numpy_buffer(buf)
                M_buffer *= buf 

    def Eigensample(self, k, h, scaled_h):
        scaled_h[:] = h
        ik_idxs = np.zeros(self.J, dtype=np.uint64)
        self.eigen_trees[k].PTSample(
            self.scaled_eigvecs[k],
            scaled_h,
            h,
            ik_idxs)

    def Treesample(self, k, h, scaled_h, samples):
        J = scaled_h.shape[0]
        ik_idxs = np.zeros(J, dtype=np.uint64)

        self.opt_trees[k].PTSample(
                self.U[k], 
                h,
                scaled_h,
                ik_idxs
                )

        samples *= self.U[k].shape[0]
        samples += ik_idxs

    def KRPDrawSamples_scalar(self, j, J):
        '''
        Draws J samples from the KRP excluding J. Returns the scalar
        indices of each sampled row in the Khatri-Rao product. 
        '''
        self.computeM(j)
        samples = np.zeros(J, dtype=np.uint64)
        h = np.ones((J, self.R), dtype=np.double)
        scaled_h = np.zeros((J, self.R), dtype=np.double)

        for k in range(self.N):
            if k == j:
                continue 

            self.Eigensample(k, h, scaled_h)
            self.Treesample(k, h, scaled_h, samples)

        return samples
