import numpy as np
from tensor_train import *
from dense_tensor import *
from tt_als import *
import teneva
from time import perf_counter as tpc
np.random.seed(42)

d = 100
N = 4
n = [d] * N
a = [-0.5] * N
b = [0.5] * N

class Functions:
    @staticmethod
    def schaffer(I):
        """Schaffer function."""
        X = teneva.ind_to_poi(I, a, b, n)
        Z = X[:, :-1]**2 + X[:, 1:]**2
        y = 0.5 + (np.sin(np.sqrt(Z))**2 - 0.5) / (1. + 0.001 * Z)**2
        return np.sum(y, axis=1)

    @staticmethod
    def sine(I):
        """#f(x1,...,xN) = sin(x1+...+xN)"""
        X = teneva.ind_to_poi(I, a, b, n)
        return np.sin(np.sum(X, axis=1))
    #
    # @staticmethod
    # def hilbert(I):
    #     """#Hilbert tensor"""
    #     X = teneva.ind_to_poi(I, a, b, n)
    #     return 1/np.sum(X, axis=1)


def sin_test(idxs):
    return np.sin(idxs[:, 0]) / idxs[:, 0]

def sin2_test(idxs):
    return np.sin(idxs[:, 0]) + 0.1 * np.sin(idxs[:, 0] * 5 + 3)
