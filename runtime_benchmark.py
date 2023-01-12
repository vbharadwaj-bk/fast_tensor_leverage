import numpy as np
from datetime import datetime
import time
import subprocess
import json
import os
import gc
import sys

import cppimport.import_hook
from cpp_ext.als_module import Sampler
from experiment import *

# Benchmarks the time to construct the sampler
# and samples from the Khatri-Rao product of
# randomly initialized matrices. Tests the performance
# impact of varying I, R, N 

if __name__=='__main__':
    experiment = Experiment("outputs/runtime_bench.json") 
    data = experiment.get_data()
    data["I_trace"] = []
    data["N_trace"] = []
    data["R_trace"] = []

    J = 50000 
    trial_count = 5

    data["J"] = J
    data["trial_count"] = trial_count 

    # Eliminate cold start 
    U = [np.random.normal(size=(500, 25)) for i in range(2)]
    sample_buffer = np.zeros((J, 25), dtype=np.uint64)
    sampler = Sampler(U, J, 25, "efficient") 
    sampler.KRPDrawSamples(0, sample_buffer)
    # Eliminate cold start 

    # Benchmark the effect of increasing I
    for i in range(0, 20):
        print(i)
        base_I = 2 ** 6
        I, R, N = base_I * 2 ** i, 32, 3 
        U = [np.random.normal(size=(I, R)) for i in range(N)]
        sample_buffer = np.zeros((J, R), dtype=np.uint64)

        construction_times = []
        sampling_times = []

        for j in range(5):
            start_construct = time.time()
            sampler = Sampler(U, J, R, "efficient")
            construction_time = time.time() - start_construct

            start_sample = time.time()
            sampler.KRPDrawSamples(N+1, sample_buffer)
            sampling_time = time.time() - start_sample

            construction_times.append(construction_time)
            sampling_times.append(sampling_time)

        data["I_trace"].append({"I": I, "R": R, "N": N, 
            "construction_times": construction_times, 
            "sampling_times": sampling_times})

    for i in range(4):
        print(i)
        I, R, N = 2 ** 22, 16 * (2 ** i), 3 
        U = [np.random.normal(size=(I, R)) for i in range(N)]
        sample_buffer = np.zeros((J, R), dtype=np.uint64)

        construction_times = []
        sampling_times = []

        for j in range(trial_count):
            start_construct = time.time()
            sampler = Sampler(U, J, R, "efficient")
            construction_time = time.time() - start_construct

            start_sample = time.time()
            sampler.KRPDrawSamples(N+1, sample_buffer)
            sampling_time = time.time() - start_sample

            construction_times.append(construction_time)
            sampling_times.append(sampling_time)

        data["R_trace"].append({"I": I, "R": R, "N": N, 
            "construction_times": construction_times, 
            "sampling_times": sampling_times})

    for i in range(8):
        print(i)
        I, R, N = 2 ** 22, 32, 2 + i 
        U = [np.random.normal(size=(I, R)) for i in range(N)]
        sample_buffer = np.zeros((J, R), dtype=np.uint64)

        construction_times = []
        sampling_times = []

        for j in range(trial_count):
            start_construct = time.time()
            sampler = Sampler(U, J, R, "efficient")
            construction_time = time.time() - start_construct

            start_sample = time.time()
            sampler.KRPDrawSamples(N+1, sample_buffer)
            sampling_time = time.time() - start_sample

            construction_times.append(construction_time)
            sampling_times.append(sampling_time)

        data["N_trace"].append({"I": I, "R": R, "N": N, 
            "construction_times": construction_times, 
            "sampling_times": sampling_times})

    experiment.write_to_file()
