import numpy as np
from mpi4py import MPI
import tensorly.contrib.sparse as stl
import sparse, h5py, argparse, json, time, os, itertools
from tensorly.contrib.sparse.decomposition import parafac as sparse_parafac

# We will run this benchmark across multiple nodes. 

def create_tasks(tensor_name): 
    parameter_dict = {
        "tensor_name": [tensor_name],
        "R": [25],
        "trial": 4
    }

    # ==============================
    # Create a list of tasks to run.
    # ==============================
    trials = list(range(parameter_dict["trial"]))
    parameter_dict["trial"] = trials

    iteration_order = ["tensor_name", "R", "trial"]
    parameter_space = [parameter_dict[key] for key in iteration_order]

    entries = []

    for element in itertools.product(*parameter_space):
        entry = {}
        for i in range(len(iteration_order)):
            entry[iteration_order[i]] = element[i]

        configuration = tensor_configurations[entry["tensor_name"]]

        for key in configuration:
            entry[key] = configuration[key]

        entries.append(entry)

    return entries

tensor_configurations = {
    "uber": {
        "preprocessing": None,
        "max_iterations": 1
    },
    "enron": {
        "preprocessing": "log_count",
        "max_iterations": 1
    },
    "nell-2": {
        "preprocessing": "log_count",
        "max_iterations": 1
    },
    "amazon-reviews": {
        "preprocessing": None, 
        "max_iterations": 1
    },
    "reddit-2015": {
        "preprocessing": "log_count", 
        "max_iterations": 1
    }
}

def execute_task(task, tensor, folder):
    max_iterations = task["max_iterations"]
    R = task["R"]
    tensor_name = task["tensor_name"]
    preprocessing = task["preprocessing"]
    trial_number = task["trial"]

    result = task
                    
    start = time.time()
    print(tensor)
    sparse_cp = sparse_parafac(tensor, R, init='random', n_iter_max=max_iterations) 
    elapsed = time.time() - start

    result["elapsed"] = elapsed

    filename = f"{tensor_name}_{R}_{trial_number}.json"
    with open(os.path.join(folder, filename), "w") as outfile:
        json.dump(result, outfile, indent=4)


def load_sparse_tensor(filename, preprocessing=None):
    print("Loading sparse tensor...")
    f = h5py.File(filename, 'r')

    max_idxs = f['MAX_MODE_SET'][:]
    min_idxs = f['MIN_MODE_SET'][:]
    N = len(max_idxs)
    dims = max_idxs[:]

    # The tensor must have at least one mode
    nnz = len(f['MODE_0']) 

    tensor_idxs = np.zeros((N, nnz), dtype=np.int64) 

    for i in range(N): 
        tensor_idxs[i, :] = f[f'MODE_{i}'][:] - min_idxs[i]

    values = f['VALUES'][:]
    print("Loaded tensor values from disk...")

    if preprocessing is not None:
        if preprocessing == "log_count":
            values = np.log(values + 1.0)
        else:
            print(f"Unknown preprocessing option '{preprocessing}' specified!")

    tensor = sparse.COO(tensor_idxs, values, shape=dims)
    print("Finished loading sparse tensor...")

    return tensor 

if __name__=='__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_ranks = comm.Get_size() 

    parser = argparse.ArgumentParser(description='Run sparse tensor ALS benchmarks.')    
    parser.add_argument('--tensor', type=str, help='Name of tensor to run ALS on.')
    parser.add_argument('--folder', type=str, help='Name of folder to store run.')

    args = parser.parse_args()
    folder = args.folder

    tasks = create_tasks(args.tensor)

    # Check if folder exists, if not, create it.
    if rank == 0:
        if not os.path.exists(folder):
            os.makedirs(folder)

    MPI.COMM_WORLD.barrier()

    data = []
    # Read every file in folder, each of which contains a single json.
    # Compare each json to the tasks list to find which tasks have not ben completed.
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            with open(os.path.join(folder, filename), "r") as infile:
                try:
                    data.append(json.load(infile))
                except:
                    if rank == 0:
                        print(f"Error loading file {filename}.")
                    continue

    complete_tasks = []
    incomplete_tasks = []
    for task in tasks:
        task_match = False
        for datum in data:
            task_match = True
            for key in task:
                if task[key] != datum[key]:
                    task_match = False

            if task_match:
                break 

        if task_match:
            complete_tasks.append(task)
        else:
            incomplete_tasks.append(task)

    if rank == 0:
        print(f"Found {len(complete_tasks)} complete tasks and {len(incomplete_tasks)} incomplete tasks.")

    MPI.COMM_WORLD.barrier()

    if rank == 0:
        print("Starting execution of incomplete tasks.")

    node_tasks = []
    for i in range(len(incomplete_tasks)):
        if i % num_ranks == rank:
            node_tasks.append(incomplete_tasks[i])

    print(f"Rank {rank} has {len(node_tasks)} tasks to complete.")
    MPI.COMM_WORLD.barrier()

    tensor_name = args.tensor
    preprocessing = tensor_configurations[tensor_name]["preprocessing"]
    tensor = load_sparse_tensor(f"/pscratch/sd/v/vbharadw/tensors/{tensor_name}.tns_converted.hdf5", preprocessing=preprocessing)

    for i in range(len(node_tasks)): 
        print(f"Rank {rank} started task {i+1} of {len(node_tasks)}.")  
        execute_task(node_tasks[i], tensor, folder)
        print(f"Rank {rank} completed task {i+1} of {len(node_tasks)}.")  

    MPI.COMM_WORLD.barrier()
    if rank == 0:
        print("All tasks completed.")
