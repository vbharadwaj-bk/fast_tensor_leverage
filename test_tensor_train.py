import numpy as np
import numpy.linalg as la

from tensors import TensorTrain

def test_tt_sampling():
    I = 2
    R = 2
    N = 3

    dims = [I] * N
    ranks = [R] * (N - 1)

    seed=20
    tt = TensorTrain(dims, ranks, seed)

if __name__=='__main__':
    test_tt_sampling()


