import numpy as np
import os
import requests
import scipy.io
from scipy.io import loadmat
from zipfile import ZipFile
from PIL import Image
import glob
import sys

current_script_path = os.path.dirname(os.path.abspath(__file__))
fast_tensor_leverage_path = os.path.dirname(current_script_path)
sys.path.append(fast_tensor_leverage_path)
from tensors.dense_tensor import *

def get_load_datasets(dataset):
    print("Loaded dataset...")
    if dataset == "pavia":
        file_url = "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat"
        filename = file_url.split('/')[-1]
        print(f"Filename: {filename}")

        home_directory = os.path.expanduser("~")
        dataset_path = os.path.join(home_directory, "my_projects", "TT_Sampling", "fast_tensor_leverage",
                                    "dense_tt_tests")
        full_file_path = os.path.join(dataset_path, filename)

        os.makedirs(dataset_path, exist_ok=True)

        if not os.path.isfile(full_file_path):
            print(f"Downloading dataset to {full_file_path}")
            response = requests.get(file_url)
            response.raise_for_status()
            with open(full_file_path, 'wb') as file:
                file.write(response.content)
        else:
            print(f"Dataset already exists at {full_file_path}")

        data = loadmat(full_file_path)
        hyperspectral_data = data.get('paviaU')
        if hyperspectral_data is None:
            raise KeyError("Key 'paviaU' not found in the .mat file")

        hyperspectral_data = np.array(hyperspectral_data).astype(np.float64)
        hyperspectral_data = hyperspectral_data[:600, :320, :100]
        hyperspectral_data = hyperspectral_data.reshape((24,25,16,20,10,10))

        print("Initialized dense tensor...")
        tensor = PyDenseTensor(hyperspectral_data)
        return tensor


if __name__ == "__main__":
    dataset = "pavia"
    pavia_data = get_load_datasets(dataset)
    print(f"Dataset shape: {pavia_data.shape}")
