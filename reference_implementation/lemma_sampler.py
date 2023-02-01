import numpy as np
import numpy.linalg as la

from segment_tree import * 

def batch_dot_product(A, B):
    return np.einsum('ij,ij->j', A, B)

class LemmaSampler:
    def __init__(self, U, Y, F):
        self.I = U.shape[0]
        self.U = U 
        self.F = 1
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

