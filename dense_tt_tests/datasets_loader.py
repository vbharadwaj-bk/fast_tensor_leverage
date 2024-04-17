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
from PIL import Image

from dense_tensor import *


def get_datasets(dataset):
    print("Loaded dataset...")
    filename = None  # Initialize filename variable

    home_directory = os.path.expanduser("~")
    dataset_path = os.path.join(home_directory, "my_projects", "TT_Sampling", "fast_tensor_leverage",
                                "dense_tt_tests")
    if dataset == "pavia":
        file_url = "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat"
        filename = file_url.split('/')[-1]
    elif dataset == "bench-park":
        file_url = 'https://www.pexels.com/video/853751/download/'
        filename = 'bench-park-video.mp4'
    elif dataset == "cat":
        file_url = "https://www.pexels.com/download/video/854982/?fps=25.0&h=720&w=1280"
        filename = 'cat-video.mp4'
    # elif dataset == "coil-reshaped":
    #     file_url = "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip"
    #     filename = file_url.split('/')[-1]
    else:
        raise ValueError(f"not implemented yet")

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
        data = loadmat(file_path)
        hyperspectral_data = data.get('paviaU')
        if hyperspectral_data is None:
            raise KeyError("Key 'paviaU' not found in the .mat file")

        hyperspectral_data = np.array(hyperspectral_data).astype(np.float64)
        hyperspectral_data = hyperspectral_data[:600, :320, :100]
        hyperspectral_data = hyperspectral_data.reshape((100,100,80,24))

        print("Initialized dense tensor...")
        tensor = PyDenseTensor(hyperspectral_data)

    elif data == "bench-park":
        video_path = file_path
        bench_park_data = VideoFileClip(video_path)
        frames = [frame for frame in bench_park_data.iter_frames()]
        video_data = np.array(frames)
        video_data_grayscale = video_data.mean(axis=-1)
        target_shape = (24, 45, 32, 60, 28, 13)
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
        # target_shape = (16, 45, 32, 40, 13, 22)
        target_shape = (100,143,128,144)
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

    # elif data == "coil-reshape":
    #     red_truck_path = os.listdir(file_path)
    #     no_obj = 1
    #     pic_per_obj = 72
    #     strt = 4
    #     X = np.zeros((128, 128, 3, no_obj * pic_per_obj), dtype=np.uint8)
    #     cnt = 0
    #     for obj in range(1, no_obj + 1):
    #         for pic in range(1, pic_per_obj + 1):
    #             img_path = os.path.join(path, flist[strt + cnt])
    #             X[:, :, :, (obj - 1) * pic_per_obj + pic - 1] = np.array(Image.open(img_path))
    #             cnt += 1
    #
    #     if dataset == 'coil-reshape':
    #         X = X.reshape((8, 16, 8, 16, 3, 8, 9))
    #     else:
    #         X = X.astype(np.float64)

    else:
        print(f"Dataset type {data} not supported for this operation")

    return tensor


# if __name__ == "__main__":
#     data= "cat"
#     file_path = get_datasets(data)
#     cat_data = load_data(data, file_path)
#     print(f"Dataset shape is {cat_data.shape}")
