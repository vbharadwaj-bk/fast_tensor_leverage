import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import time
import json
import itertools
import argparse
from mpi4py import MPI

from common import *
from tensors import *
from als import *
import itertools

import cppimport.import_hook
from cpp_ext.als_module import Tensor, LowRankTensor, SparseTensor, ALS 

# We will run this benchmark across multiple nodes. 

tensor_configurations = {
    "uber": {
        "preprocesing": None,
        "initialization": None,
        "max_iterations": 40,
        "stop_tolerance": 1e-4
    },
    "enron": {
        "preprocesing": "log_count",
        "initialization": "rrf",
        "max_iterations": 40,
        "stop_tolerance": 1e-4
    },
    "nell-2": {
        "preprocesing": "log_count",
        "initialization": None,
        "max_iterations": 40,
        "stop_tolerance": 1e-4
    },
    "amazon-reviews": {
        "preprocesing": None, 
        "initialization": None,
        "max_iterations": 40,
        "stop_tolerance": 1e-4
    },
    "reddit-2015": {
        "preprocesing": "log_count", 
        "initialization": None,
        "max_iterations": 80,
        "stop_tolerance": 1e-4
    }
}


def create_tasks(tensor_name): 
    parameter_dict = {
        "tensor_name": [tensor_name],
        "sampler": ["efficient"],
        "R": [25],
        "J": [2 ** 16],
        "trial": 4
    }

    # ==============================
    # Create a list of tasks to run.
    # ==============================
    trials = list(range(parameter_dict["trial"]))
    parameter_dict["trial"] = trials

    iteration_order = ["tensor_name", "R", "J", "sample_count", "trial"]
    parameter_space = [parameter_dict[key] for key in iteration_order]

    entries = []

    for element in itertools.product(*parameter_space):
        entry = {}
        for i in range(len(iteration_order)):
            entry[iteration_order[i]] = element[i]

        configuration = tensor_configurations[entry["tensor_name"]]

        for key in configuration:
            entry[key] = configuration[key]

    return entries

def execute_task(task, rhs, folder):
    max_iterations = task["max_iterations"]
    stop_tolerance = task["stop_tolerance"]
    J = task["sample_count"] 
    R = task["R"]
    sampler = task["sampler"]
    tensor_name = task["tensor_name"]
    preprocessing = task["preprocessing"]
    initialization = task["initialization"]
    trial_number = task["trial"]

    result = task
    result["tensor_order"] = rhs.N
                    
    lhs = PyLowRank(rhs.dims, R)
    if initialization is not None and initialization == "rrf":
        lhs.ten.initialize_rrf(rhs.ten)
    else:
        lhs.ten.renormalize_columns(-1)

    start = time.time()
    result["trace"] = als_prod(lhs, rhs, J, sampler, max_iterations, stop_tolerance, verbose=True)
    elapsed = time.time() - start
    result["elapsed"] = elapsed

    # Concatenate tensor name, sampler, R, J, trial number
    filename = f"{tensor_name}_{sampler}_{R}_{J}_{trial_number}.json"
    with open(os.path.join(folder, filename), "w") as outfile:
        json.dump(result, outfile, indent=4)


if __name__=='__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_ranks = comm.Get_size() 

    parser = argparse.ArgumentParser(description='Run sparse tensor ALS benchmarks.')    
    parser.add_argument('--tensor', type=str, help='Name of tensor to run ALS on.')
    parser.add_argument('--folder', type=str, help='Name of folder to store run.')

    args = parser.parse_args()
    tasks = create_tasks(args.tensor)
    folder = f'outputs/{args.folder}/{args.tensor}' 

    # Check if folder exists, if not, create it.
    if rank == 0:
        if not os.path.exists(folder):
            os.makedirs(folder)

    data = []
    # Read every file in folder, each of which contains a single json.
    # Compare each json to the tasks list to find which tasks have not ben completed.
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            with open(os.path.join(folder, filename), "r") as infile:
                data.append(json.load(infile)) 

    complete_tasks = []
    incomplete_tasks = []
    for task in tasks:
        task_match = false
        for datum in data:
            task_match = True
            for key in task:
                if task[key] != datum[key]:
                    task_match = False

        if task_match:
            complete_tasks.append(task)
        else:
            incomplete_tasks.append(task)

    if rank == 0:
        print(f"Found {len(complete_tasks)} complete tasks and {len(incomplete_tasks)} incomplete tasks.")

    print("Rank {rank} has {len(incomplete_tasks)} tasks to complete.")

    MPI.COMM_WORLD.barrier()
    if rank == 0:
        print("Starting execution of incomplete tasks.")

    node_tasks = []
    for i in range(len(incomplete_tasks)):
        if i % num_ranks == rank:
            node_tasks.append(incomplete_tasks[i])

    preprocessing = tensor_configurations[args.tensor]["preprocessing"]
    rhs = PySparseTensor(f"/pscratch/sd/v/vbharadw/tensors/{tensor_name}.tns_converted.hdf5", lookup="sort", preprocessing=preprocessing)

    for i in range(len(node_tasks)): 
        execute_task(node_tasks[i], rhs, folder)
        print(f"Rank {rank} completed task {i} of {len(node_tasks)}.")  
