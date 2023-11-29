import numpy as np
import numpy.linalg as la
import json
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd

import logging, sys
import time

import cppimport
import cppimport.import_hook
from tensors.dense_tensor import *
from coil100_data_loader import *
from IPython import embed

class TensorTrainSVD:
    def __init__(self, ground_truth,R):
        self.ground_truth = ground_truth
        self.R = R

        if isinstance(self.ground_truth,PyDenseTensor):
            print("Initialized TT-SVD-Class!")
        else:
            NotImplementedError

    def tt_regular_svd(self):
        cores = []
        unfolded_tensor = self.ground_truth.data
        order = self.ground_truth.N
        ranks = [self.R] * (order - 1)
        ranks.insert(0, 1)

        #Permutation is just for coil-100 dataset.
        unfolded_tensor = np.transpose(unfolded_tensor, (1, 2, 3, 0))
        # print(unfolded_tensor.shape)
        dims = unfolded_tensor.shape

        for i in range(order-1):
            # reshape residual tensor
            m = ranks[i] * dims[i]
            n = np.prod(dims[i+1:])
            unfolded_tensor = np.reshape(unfolded_tensor, [m,n])


            # apply SVD
            m,n = unfolded_tensor.shape
            svd_rank = min(m,n,ranks[i + 1])
            [u, s, v] = la.svd(unfolded_tensor, full_matrices=False)
            ranks[i + 1] = svd_rank


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


        return cores

    def tt_randomized_svd(self):

        print("Initialized rTT-SVD!")
        rtt_cores = []

        unfolded_tensor = self.ground_truth.data
        order = self.ground_truth.N
        ranks = [self.R] * (order - 1)
        ranks.insert(0, 1)
        unfolded_tensor = np.transpose(unfolded_tensor, (1, 2, 3, 0))
        print(unfolded_tensor.shape)
        dims = unfolded_tensor.shape

        for i in range(order - 1):
            # reshape residual tensor
            m = ranks[i] * dims[i]
            n = np.prod(dims[i + 1:])
            unfolded_tensor = np.reshape(unfolded_tensor, [m, n])
            print(unfolded_tensor.shape)
            [u,s,v] = randomized_svd(unfolded_tensor, ranks[i+1])
            # u = u[:,:j]
            # v = v[:,:j]
            # s = s[:j]
            print("Shape of u:", u.shape, "Desired shape:", (ranks[i], dims[i], ranks[i+1]))

            u = u.reshape([ranks[i],dims[i],ranks[i+1]])
            rtt_cores.append(u)

            unfolded_tensor = np.diag(s) @ v

        rank_final, dim_final = unfolded_tensor.shape
        rtt_cores.append(unfolded_tensor.reshape([rank_final,dim_final, 1]))
        return rtt_cores


if __name__ == '__main__':
    dataset = "coil-100"
    ground_truth, labels = get_coil_tensor(dataset)
    R = 5
    tt_svd = TensorTrainSVD(ground_truth,R)
    over_sample = 5
    cores = tt_svd.tt_regular_svd()
    cores_tt = tt_svd.tt_randomized_svd()
