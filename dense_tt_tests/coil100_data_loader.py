import numpy as np
import os
import requests
from zipfile import ZipFile
from PIL import Image
import glob
import sys

# current_script_path = os.path.dirname(os.path.abspath(__file__))
# fast_tensor_leverage_path = os.path.dirname(os.path.dirname(current_script_path))
# cpp_ext_path = os.path.join(fast_tensor_leverage_path, 'cpp_ext')
# tensors_path = os.path.join(fast_tensor_leverage_path, 'tensors')
# sys.path.append(cpp_ext_path)
# sys.path.append(tensors_path)
from dense_tensor import *

def get_coil_dataset(image_path):
    file_url = "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip"
    filename = file_url.split('/')[-1]

    if not os.path.isfile(filename):
        with requests.get(file_url, stream=True) as response:
            response.raise_for_status()  # This will raise an exception for HTTP errors
            with open(filename, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

    image_subpath = os.path.join(image_path, 'coil-100')
    if filename.endswith('.zip') and not os.path.isdir(image_subpath):
        with ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(image_path)

    image_files = sorted(glob.glob(os.path.join(image_subpath, '*.png')))
    image_files = [f for f in image_files if os.path.isfile(f)]

    if not image_files:
        raise Exception("No image files found. Check if the dataset is downloaded and extracted correctly.")

    images = []
    labels = []
    for img_file in image_files:
        with Image.open(img_file) as img:
            img_resized = img.resize((32, 32))
            img_float = np.array(img_resized).astype(np.float32) / 255.0  # Normalizing pixel values
            images.append(img_float)
            basename = os.path.basename(img_file)
            label = int(basename.split('__')[0][3:])
            labels.append(label)

    images_np = np.array(images)
    return images_np.astype(np.float32), np.array(labels).astype(np.float32)


def get_coil_tensor(dataset_name):
    if dataset_name.lower() != "coil-100":
        raise ValueError("Unsupported dataset. This function only supports COIL-100.")

    image_path = 'coil-100'
    images_np, labels_np = get_coil_dataset(image_path)

    print("Initialized dense tensor...")

    tensor = PyDenseTensor(images_np)
    return tensor, labels_np

if __name__ == "__main__":
    tensor, labels = get_coil_tensor("coil-100")
    print(f"Tensor shape: {type(tensor)}, tensor shape: {tensor.shape}")
