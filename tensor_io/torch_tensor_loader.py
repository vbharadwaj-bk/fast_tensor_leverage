import numpy as np
from dense_tensor import *

import os

def get_torch_tensor(dataset_name):
    print("Loading torch and torchvision...")
    import torch
    import torchvision
    print("Torch / Torchvision loaded!")

    dataset_name = dataset_name

    param_map = {
        "mnist": torchvision.datasets.MNIST,
        "cifar10": torchvision.datasets.CIFAR10
    } 

    root = os.getenv('TORCH_DATASET_FOLDER')  
    if root is None:
        print("Unset environment variable TORCH_DATASET_FOLDER.")
        exit(1) 

    dataset_class = param_map[dataset_name]
    dataset = dataset_class(f'{root}/', download=True, train=True, transform=torchvision.transforms.ToTensor())
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True, num_workers=16)
    images, labels = next(iter(loader))
    images_sq = torch.squeeze(images)

    images_np = images_sq.numpy()
    labels_np = labels.numpy()

    tensor_dims = images_np.shape
    N = len(tensor_dims)

    print(f"Loaded dataset {dataset_name}...") 
    tensor = PyDenseTensor(images_np)
    print("Initialized dense tensor...")

    return tensor