import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import time
import json
import pickle
import h5py
import ctypes
from PIL import Image

from common import *
from tensors import *
from als import *
from image_classification import * 

import cppimport.import_hook
from cpp_ext.efficient_krp_sampler import CP_ALS 
from cpp_ext.als_module import Tensor, LowRankTensor, SparseTensor, ALS 

if __name__=='__main__':
    print("Starting sparse tensor benchmark!")

    max_iterations = 100                 # For now, this needs to stay a multiple of 5! 
    stop_tolerance = 1e-4
    sample_counts = [2 ** 16 + 8192 * i for i in range(8)]
    R_values = [25, 75, 125]
    trial_count = 5

    tensor_name = "uber"
    samplers = ["larsen_kolda", "larsen_kolda_hybrid", "efficient"]
    results = []

    rhs = PySparseTensor(f"/pscratch/sd/v/vbharadw/tensors/{tensor_name}.tns_converted.hdf5", lookup="sort")

    for R in R_values: 
        for sampler in samplers:
            for J in sample_counts: 
                for trial in range(trial_count):
                    result = {"R": R, "J": J, "sampler": sampler}

                    lhs = PyLowRank(rhs.dims, R)
                    lhs.ten.renormalize_columns(-1)

                    start = time.time()
                    result["trace"] = als_prod(lhs, rhs, J, sampler, max_iterations, stop_tolerance)
                    elapsed = time.time() - start
                    result["elapsed"] = elapsed

                    print(f"Elapsed: {elapsed}")
                    results.append(result)

    with open(f'outputs/{tensor_name}_sparse_traces.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)