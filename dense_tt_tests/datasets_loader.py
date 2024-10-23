import numpy as np
import os
import requests
import scipy.io
from scipy.io import loadmat
from zipfile import ZipFile
from PIL import Image
import glob
import sys
from moviepy.editor import VideoFileClip
import tifffile as tiff
import h5py
from dense_tensor import *
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_datasets(dataset):
    print("Loaded dataset...")
    filename = None  # Initialize filename variable

    home_directory = os.path.expanduser("~")
    #dataset_path = os.path.join(data_set path)
    if dataset == "pavia":
        file_url = "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat"
        filename = 'PaviaU.mat'
    elif dataset == "bench-park":
        file_url = 'https://www.pexels.com/video/853751/download/'
        filename = 'bench-park-video.mp4'
    elif dataset == "cat":
        file_url = "https://www.pexels.com/download/video/854982/?fps=25.0&h=720&w=1280"
        filename = 'cat-video.mp4'
    elif dataset == "dc-mall":
        file_url = "http://cobweb.ecn.purdue.edu/~biehl/Hyperspectral_Project.zip/dc.tif"
        filename = 'dc-mall'
    elif dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize the images
        ])
        file_url = datasets.MNIST(root='./dense_tt_sets', train=True, download=True, transform=transform)
        filename = 'mnist'
    print(f"Filename: {filename}")

    full_file_path = os.path.join(dataset_path, filename)
    os.makedirs(dataset_path, exist_ok=True)

    if not os.path.isfile(full_file_path):
        print(f"Downloading dataset to {full_file_path}")
        try:
            response = requests.get(file_url, stream=True)  # Use stream to handle large files
            response.raise_for_status()  # This will raise an HTTPError if the response was an HTTP error

            with open(full_file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Successfully downloaded {filename}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {filename}. Error: {e}")
    else:
        print(f"Dataset already exists at {full_file_path}")
    return full_file_path


def load_data(data, file_path):
    tensor = None
    if data == "pavia":
        hyperspectral_data = scipy.io.loadmat(file_path)
        hyperspectral_data = hyperspectral_data['paviaU']
        print(hyperspectral_data.shape)
        if hyperspectral_data is None:
            raise KeyError("Key 'paviaU' not found in the .mat file")

        hyperspectral_data = np.transpose(hyperspectral_data[:600, :320, :100],(2,1,0))
        # hyperspectral_data = hyperspectral_data[:600, :320, :100]
        hyperspectral_data = hyperspectral_data.reshape((100,320,30,20))
        hyperspectral_data = np.array(hyperspectral_data).astype(np.float32)

        print("Initialized dense tensor...")
        tensor = PyDenseTensor(hyperspectral_data)

    elif data == "bench-park":
        video_path = file_path
        bench_park_data = VideoFileClip(video_path)
        frames = [frame for frame in bench_park_data.iter_frames()]
        video_data = np.array(frames)
        video_data_grayscale = video_data.mean(axis=-1)
        target_shape = (24, 45, 32, 60, 28, 13)
        reshaped_data = video_data_grayscale.reshape(target_shape)
        print("Data reshaped successfully.")
        print("Initialized dense tensor...")
        tensor = PyDenseTensor(reshaped_data)

        try:
            if video_data_grayscale.size == np.product(target_shape):
                reshaped_data = video_data_grayscale.reshape(target_shape)
                print("Data reshaped successfully.")
                print("Initialized dense tensor...")
                tensor = PyDenseTensor(reshaped_data)
            else:
                print("Cannot reshape: total number of elements does not match the target shape.")
        except Exception as e:
            print(f"Error during reshaping: {e}")

    elif data == "cat":
        video_path = file_path
        bench_park_data = VideoFileClip(video_path)
        frames = [frame for frame in bench_park_data.iter_frames()]
        video_data = np.array(frames)
        video_data_grayscale = video_data.mean(axis=-1)
        print(video_data_grayscale.shape)
        video_data_grayscale = np.transpose(video_data_grayscale,(1,2,0))
        # target_shape = (16, 45, 32, 40, 13, 22)
        target_shape = (286,720,40,32)
        # target_shape = target_shape.reshape(32,,)
        try:
            if video_data_grayscale.size == np.product(target_shape):
                reshaped_data = video_data_grayscale.reshape(target_shape)
                print("Data reshaped successfully.")
                print("Initialized dense tensor...")
                tensor = PyDenseTensor(reshaped_data)
            else:
                print("Cannot reshape: total number of elements does not match the target shape.")
        except Exception as e:
            print(f"Error during reshaping: {e}")

    elif data == "dc-mall":
        file_name = 'dc-mall'
        hyper = tiff.imread('dc.tif')
        hyperspectral_data = np.array(hyper).astype(np.float32)
        print(f"Loaded hyperspectral data shape: {hyperspectral_data.shape}")

        hyperspectral_data = hyperspectral_data[:190,:,:306]
        print(hyperspectral_data.shape)
        hyperspectral_data = hyperspectral_data.reshape((1280,306,10,19))
        print("Initialized dense tensor...")
        tensor = PyDenseTensor(hyperspectral_data)

    elif data == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize the images
        ])
        data = datasets.MNIST(root='./dense_tt_sets', train=True, download=True, transform=transform)
        images_list = []
        mnist_loader = DataLoader(data, batch_size=2, shuffle=True)
        for images, labels in mnist_loader:
            images_np = images.numpy()
            images_np = images_np.astype(np.float64)
            images_list.append(images_np)

        images_np = np.array(images_list).astype(np.float64)
        mnist_data  = images_np.reshape((280,600,28,10))
        print("Initialized dense tensor...")

        tensor = PyDenseTensor(mnist_data)

    else:
        print(f"Dataset type {data} not supported for this operation")

    return tensor


# if __name__ == "__main__":
#     data= "mnist"
#     file_path = get_datasets(data)
#     data = load_data(data, file_path)
#     print(f"Dataset shape is {data.shape}")
