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
from image_classification import * 

import cppimport.import_hook
from cpp_ext.als_module import Tensor, LowRankTensor, SparseTensor, ALS 

# We will run this benchmark across multiple nodes. 

if __name__=='__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_ranks = comm.Get_size() 

    max_iterations = 80 # For now, this needs to stay a multiple of 5! 
    stop_tolerance = 1e-4
 
    #sample_counts = [2 ** 16 + 8192 * 2 * i for i in range(4)]
    sample_counts = [2 ** 16] 
    R_values = [125]
    #samplers = ["larsen_kolda", "larsen_kolda_hybrid", "efficient"]
    #samplers = ["larsen_kolda_hybrid", "efficient"]
    samplers = ["efficient"]
    trial_count = 8

    # Test Configuration ==============================
    #sample_counts = [2 ** 16] 
    #R_values = [25]
    #samplers = ["larsen_kolda"] 
    #trial_count = 5
    # =================================================

    trial_list = [trial_count // num_ranks] * num_ranks
    for i in range(trial_count % num_ranks):
        trial_list[i] += 1

    tensor_name = "reddit-2015"
    preprocessing="log_count" 
    results = []

    rhs = PySparseTensor(f"/pscratch/sd/v/vbharadw/tensors/{tensor_name}.tns_converted.hdf5", lookup="sort", preprocessing=preprocessing)

    for R in R_values: 
        for sampler in samplers:
            for J in sample_counts:
                local_result = [] 
                for trial in range(trial_list[rank]):
                    print(f"Starting trial on rank {rank}")
                    result = {"R": R, "J": J, "sampler": sampler}

                    lhs = PyLowRank(rhs.dims, R)
                    lhs.ten.renormalize_columns(-1)

                    start = time.time()
                    result["trace"] = als_prod(lhs, rhs, J, sampler, max_iterations, stop_tolerance, verbose=True)
                    elapsed = time.time() - start
                    result["elapsed"] = elapsed

                    #print(f"Elapsed: {elapsed}")
                    local_result.append(result)

                nested_list = comm.allgather(local_result)
                chained_list = list(itertools.chain.from_iterable(nested_list))
                results.extend(chained_list)

                if rank == 0:
                    print(f"Length of Result List: {len(results)}")
                    with open(f'outputs/{tensor_name}_sparse_traces_extended.json', 'w') as outfile:
                        json.dump(results, outfile, indent=4)

