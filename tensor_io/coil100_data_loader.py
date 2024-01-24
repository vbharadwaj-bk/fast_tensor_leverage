import requests
import os
import sys
from zipfile import ZipFile
import numpy as np
from PIL import Image
import glob
import imageio.v2 as imageio
from pathlib import Path
current_script_path = os.path.dirname(os.path.abspath(__file__))
tensors_path = os.path.join(current_script_path, 'tensors')

# print("Adding to path:", tensors_path)
# print("Path exists:", os.path.exists(tensors_path))

sys.path.append(tensors_path)
from dense_tensor import *

def get_coil_dataset(image_path):
    file_url = "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip"
    filename = file_url.split('/')[-1]

    if not os.path.isfile(filename):
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(filename, 'wb') as file:
                file.write(response.content)
        else:
            raise Exception(f"Failed to download the file: Status code {response.status_code}")

    if filename.endswith('.zip') and not os.path.isdir(image_path):
        with ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(image_path)

    image_subpath = os.path.join(image_path, 'coil-100')
    image_files = sorted(glob.glob(os.path.join(image_subpath, '*.png')))

    image_files = [f for f in image_files if os.path.isfile(f)]

    if not image_files:
        raise Exception("No image files found. Check if the dataset is downloaded and extracted correctly.")

    images = []
    labels = []
    for img_file in image_files:
        with Image.open(img_file) as img:
            img = Image.open(img_file)
            img_resized = img.resize((32, 32))
            img_float = np.array(img_resized).astype(np.float32)
            images.append(img_float)
            basename = os.path.basename(img_file)
            label = int(basename.split('__')[0][3:])
            labels.append(label)

    images_np = np.array(images)

    return images_np , np.array(labels).astype(np.float32)

def get_coil_tensor(dataset_name):
    if dataset_name.lower() != "coil-100":
        raise ValueError("Unsupported dataset. This function only supports COIL-100.")

    image_path = 'coil-100'
    images_np, labels_np = get_coil_dataset(image_path)

    tensor_dims = images_np.shape
    N = len(tensor_dims)

    print("Initialized dense tensor...")
    tensor = PyDenseTensor(images_np)

    return tensor, labels_np

# if __name__ == "__main__":
#     tensor = get_coil_tensor("coil-100")
    # print(f"Tensor shape: {tensor.shape}, Labels shape: {labels.shape}")

