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
    def __init__(self, dataset_name, J, method):
        self.dataset_name = dataset_name
        self.J = J
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

            print(f"Loaded dataset {self.dataset_name}...") 
            rhs = PyDenseTensor(images_np)
            print("Initialized dense tensor...")


