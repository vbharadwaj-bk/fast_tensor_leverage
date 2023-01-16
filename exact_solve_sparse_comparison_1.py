import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from common import *
from tensors import *
from als import *
from experiment import *

# For a specified tensor, runs the ALS algorithm twice 
# and 

if __name__=='__main__':
    tensor_name = "amazon-reviews"
    experiment = Experiment(f"outputs/{tensor_name}_exact_solve_comp_1.json")
    data = experiment.data
    max_iterations = 10   # For now, this needs to stay a multiple of 5! 

    trial_count = 5
    J = 2 ** 16
    R= 50
    samplers = ["larsen_kolda_hybrid", "efficient"]
    rhs = PySparseTensor(f"/pscratch/sd/v/vbharadw/tensors/{tensor_name}.tns_converted.hdf5", lookup="sort")
 
    for sampler in samplers:
        lhs = PyLowRank(rhs.dims, R, seed=5872343)
        lhs.ten.renormalize_columns(-1)
        data[sampler] = []
        for trial in range(trial_count):
            data[sampler].append(als_exact_comparison(lhs, rhs, J, sampler, max_iterations))

    experiment.write_to_file()


