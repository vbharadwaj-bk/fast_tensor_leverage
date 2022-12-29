import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import time
import os

from common import *
from tensors import *
from als import *

import cppimport.import_hook
from cpp_ext.als_module import Tensor, ALS

print("Loading torch and torchvision...")
import torch
import torchvision
print("Torch / Torchvision loaded!")

class TensorClassifier:
    def __init__(self, dataset_name, J, method, R, max_iter):
        self.dataset_name = dataset_name
        self.J = J
        self.R = R
        self.max_iter = max_iter
        self.method = method 

    def train(self):
        param_map = {
            "mnist": torchvision.datasets.MNIST
        } 

        root = os.getenv('TORCH_DATASET_FOLDER')  
        if root is None:
            print("Unset environment variable TORCH_DATASET_FOLDER.")
            exit(1) 
        else:
            dataset_class = param_map[self.dataset_name]
            dataset = dataset_class(f'{root}/', download=True, train=True, transform=torchvision.transforms.ToTensor())
            loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True, num_workers=16)
            images, labels = next(iter(loader))
            images_sq = torch.squeeze(images)

            images_np = images_sq.numpy()
            labels_np = labels.numpy()

            tensor_dims = images_np.shape
            N = len(tensor_dims)

            print(f"Loaded dataset {self.dataset_name}...") 
            rhs = PyDenseTensor(images_np)
            print("Initialized dense tensor...")

            rhs_norm = la.norm(images_np)

            lhs = PyLowRank(images_np.shape, self.R, seed=923845)
            lhs.ten.renormalize_columns(-1)
            als = ALS(lhs.ten, rhs.ten)
            als.initialize_ds_als(self.J, self.method)

            loss_frequency = 5

            def generate_approx():
                sigma_lhs = np.zeros(self.R, dtype=np.double) 
                lhs.ten.get_sigma(sigma_lhs, -1)

                U_trunc = [lhs.U[i][:tensor_dims[i]] for i in range(N)]
                if len(lhs.U) == 3:
                    approx = np.einsum('r,ir,jr,kr->ijk', sigma_lhs, U_trunc[0], U_trunc[1], U_trunc[2])
                elif len(lhs.U) == 4:
                    approx = np.einsum('r,ir,jr,kr,lr->ijkl', sigma_lhs, U_trunc[0], U_trunc[1], U_trunc[2], U_trunc[3])

                return approx

            print("Starting ALS...")
            for i in range(self.max_iter):
                print(f"Starting iteration {i}")
                for j in range(lhs.N):
                    als.execute_ds_als_update(j, True, True) 

                if i % loss_frequency == 0:
                    approx = generate_approx()
                    diff_norm = la.norm(approx - images_np)
                    fit = 1 - diff_norm / rhs_norm
                    print(f"Fit: {fit}")

