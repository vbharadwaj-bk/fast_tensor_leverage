import numpy as np
import sparse, h5py, argparse, json, time
from tensorly.contrib.sparse.decomposition import parafac as sparse_parafac

def load_sparse_tensor(filename, preprocessing=None):
    print("Loading sparse tensor...")
    f = h5py.File(filename, 'r')

    max_idxs = f['MAX_MODE_SET'][:]
    min_idxs = f['MIN_MODE_SET'][:]
    N = len(max_idxs)
    dims = max_idxs[:]

    # The tensor must have at least one mode
    nnz = len(f['MODE_0']) 

    tensor_idxs = np.zeros((N, nnz), dtype=np.uint32) 

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

def decompose_and_time(tensor, num_iterations, target_rank, output_filename):
    for i in range(num_iterations):
        f = open(output_filename, 'a')
        # If the file is not empty, load the json, otherwise create list
        if os.stat(output_filename).st_size != 0:
            times = json.load(f)
        else:
            times = [] 

        if i >= len(times): 
            t = time.time() 
            sparse_cp = sparse_parafac(tensor, target_rank, init='random')
            elapsed = time.time() - t
            times.append(elapsed)

        json.dump(times, f)
        f.close()

if __name__=='__main__':
    # Create a command line parser
    parser = argparse.ArgumentParser(description='Decompose a sparse tensor via Tensorly.') 
    parser.add_argument('filename', type=str, help='The filename of the sparse tensor to decompose.')
    parser.add_argument('--preprocessing', type=str, default=None, help='The preprocessing to apply to the tensor before decomposition.')
    target_rank = parser.add_argument('--target_rank', type=int, help='The target rank of the decomposition.')
    output_filename = parser.add_argument('--output_filename', type=str, help='The filename to write the decomposition to.')
    num_repetitions = parser.add_argument('--num_repetitions', type=int, default=1, help='The number of times to repeat the decomposition.')

    args = parser.parse_args()

    tensor = load_sparse_tensor(args.filename, args.preprocessing)
    decompose_and_time(tensor, args.num_repetitions, args.target_rank, args.output_filename)
