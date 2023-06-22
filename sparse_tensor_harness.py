import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import time
import json
import itertools
from mpi4py import MPI

from common import *
from tensors import *
from als import *


import cppimport.import_hook
from cpp_ext.als_module import Tensor, LowRankTensor, SparseTensor, ALS 
from cpp_ext.als_module import ALS 

class SparseTensorALSExperiment:
    def __init__(self, 
        tensor_path,
        sample_count,
        target_rank,
        method,
        preprocessing=None, 
        initialization=None):
        print("Initializing sparse tensor experiment")

        self.tensor_path = tensor_path
        self.sample_count = sample_count
        self.target_rank = target_rank
        self.preprocessing = preprocessing
        self.initialization = initialization
        self.method = method

        # Goal is to compute a tensor such that
        # approx \approxeq ground_truth 

        self.ground_truth = PySparseTensor(self.tensor_path, lookup="sort", preprocessing=self.preprocessing)
        self.approx = PyLowRank(self.ground_truth.dims, self.target_rank)
        if initialization is not None and initialization == "rrf":
            self.ground_truth.ten.execute_rrf(self.approx.ten)
        else:
            self.approx.ten.renormalize_columns(-1)

        self.iteration_count = 0
        self.update_times = []
        self.fit_computation_times = []
        self.fits = []

        self.iterations = []
        self.als = ALS(self.approx.ten, self.ground_truth.ten)

        if self.method != "exact":
            self.als.initialize_ds_als(self.sample_count, self.method)

    def setup_randomized_accuracy_estimation(self, nz_samples):
        self.ground_truth.initialize_randomized_accuracy_estimation(nz_samples)

    def change_sample_count(self, new_sample_count):
        self.sample_count = new_sample_count
        self.als = None
        self.als = ALS(self.approx.ten, self.ground_truth.ten)

        if self.method != "exact":
            self.als.initialize_ds_als(self.sample_count, self.method)

    def get_ground_truth_norm(self):
        normsq = self.ground_truth.ten.get_normsq() 
        return np.sqrt(normsq)

    def get_ground_truth_dimensions(self):
        return self.ground_truth.dims 

    def run_als_round(self):
        print(f"Starting Iteration {self.iteration_count + 1}.")
        for j in range(self.approx.N):
            start = time.time()
            if self.method == "exact":
                self.als.execute_exact_als_update(j, True, True)
            else:
                self.als.execute_ds_als_update(j, True, True)
            elapsed = time.time() - start
            self.update_times.append(elapsed)

        self.iteration_count += 1

    def compute_fit(self):
        start = time.time()
        self.iterations.append(self.iteration_count + 1)
        self.fits.append(self.approx.compute_exact_fit(self.ground_truth))
        print(f"Iteration: {self.iteration_count+1}\tFit: {self.fits[-1]}")
        elapsed = time.time() - start
        self.fit_computation_times.append(elapsed)
        return self.fits[-1]  

    def compute_approx_fit(self):
        start = time.time()
        self.iterations.append(self.iteration_count + 1)
        self.fits.append(self.approx.compute_approx_fit(self.ground_truth))
        print(f"Iteration: {self.iteration_count+1}\tFit: {self.fits[-1]}")
        elapsed = time.time() - start
        self.fit_computation_times.append(elapsed)
        return self.fits[-1] 

    def compute_approx_diff_norm(self):
        return self.approx.compute_approx_diff_norm(self.ground_truth)

if __name__=='__main__':
    experiment = SparseTensorALSExperiment(
        tensor_path="/pscratch/sd/v/vbharadw/tensors/uber.tns_converted.hdf5",
        sample_count=2 ** 16,
        target_rank=25,
        method="efficient"
    )

    for i in range(20):
        experiment.run_als_round()
        experiment.compute_fit()


