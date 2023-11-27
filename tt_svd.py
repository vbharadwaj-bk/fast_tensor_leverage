import numpy as np
import numpy.linalg as la
import json
import matplotlib.pyplot as plt

import logging, sys
import time

import cppimport
import cppimport.import_hook
from tensors.dense_tensor import *
from coil100_data_loader import *
from IPython import embed

class TensorTrainSVD:
    def __init__(self, ground_truth):
        self.ground_truth = ground_truth

        print("Initialized TT-SVD!")

    def tt_regular_svd(self, R):
        if isinstance(self.ground_truth, PyDenseTensor):
            order = self.ground_truth.N
            ranks = [R] * (order-1)
            ranks.insert(0, 1)
            cores = []
            unfolded_tensor = self.ground_truth.data
            unfolded_tensor = np.transpose(unfolded_tensor, (1, 2, 3, 0))
            dims = unfolded_tensor.shape
            print(unfolded_tensor.shape)
            for i in range(order-1):
                # reshape residual tensor
                m = int(ranks[i] * dims[i])
                n = np.prod(dims[i+1:])
                unfolded_tensor = np.reshape(unfolded_tensor, [m,n])

                # apply SVD in order to isolate modes
                m,n = unfolded_tensor.shape
                current_rank = min(m,n,ranks[i + 1])
                [u, s, v] = la.svd(unfolded_tensor, full_matrices=False)
                ranks[i + 1] = current_rank


                # define new TT core
                u = u[:,:ranks[i + 1]]
                s = s[:ranks[i + 1]]
                v = v[:ranks[i + 1], :]
                cores.append(np.reshape(u,[ranks[i],dims[i],ranks[i+1]]))
                print("Shape of u:", u.shape, "Desired shape:", (ranks[i], dims[i], ranks[i+1]))
                # embed()

                # set new residual tensor
                unfolded_tensor = np.diag(s) @ v

            # define last TT core
            old_rank, dim_final = unfolded_tensor.shape
            cores.append(np.reshape(unfolded_tensor, [old_rank, dim_final, 1]))

        else:
            raise NotImplementedError

        return cores

if __name__ == '__main__':
    dataset = "coil-100"
    ground_truth, labels = get_coil_tensor(dataset)
    tt_svd = TensorTrainSVD(ground_truth)
    R = 5
    cores = tt_svd.tt_regular_svd(R)
