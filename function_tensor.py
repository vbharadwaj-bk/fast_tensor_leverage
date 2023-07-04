import numpy as np
import numpy.linalg as la

import cppimport
import cppimport.import_hook
from cpp_ext.tt_module import quantize_indices 

class Power2Quantization:
    def __init__(self, dims, ordering):
        qdim_lists = []
        qdims = []
        for dim in dims:
            qdim = 0
            while dim % 2 == 0 and dim > 1:
                dim = dim // 2
                qdim += 1
            if dim != 1:
                raise ValueError("All dimensions must be powers of 2")

            qdim_lists.append([2 for i in range(qdim)])
            qdims.append(qdim)

        self.quantization_dimensions = np.zeros((len(dims), max(qdims)), dtype=np.uint64)
        for i in range(len(dims)):
            self.quantization_dimensions[i, :qdims[i]] = qdim_lists[i]

        self.qdim_sum = np.sum(qdims)

        if ordering == "canonical":
            self.permutation = np.arange(self.qdim_sum, dtype=np.uint64)

    def quantize_indices(self, indices):
        J = indices.shape[0]
        quantized_indices = np.zeros((J, self.qdim_sum), dtype=np.uint64)

        quantize_indices(
            indices, 
            self.quantization_dimensions, 
            self.permutation, 
            quantized_indices)

        return quantized_indices


    def unquantize_indices(self, indices):
        pass


class FunctionTensor:
    def __init__(self, grid_bounds, dims, func, quantization=None): 
        self.grid_bounds = np.array(grid_bounds, dtype=np.double)
        self.dims = np.array(dims, dtype=np.uint64)
        self.func = func
        self.N = len(dims)
        self.dx = [(grid_bounds[i, 1] - grid_bounds[i, 0]) / dims[i] for i in range(self.N)] 

        # Validation set for fit computation
        self.validation_samples = None
        self.validation_values = None
        self.quantization = quantization


    def initialize_accuracy_estimation(self, method="randomized", rsample_count=10000):
        if method != "randomized":
            raise NotImplementedError("Only randomized validation set generation is supported")

        self.validation_samples_int = np.zeros((rsample_count, self.N), dtype=np.uint64)
        for i in range(self.N):
            self.validation_samples_int[:, i] = np.random.randint(0, self.dims[i], size=rsample_count)

        self.validation_samples = self.validation_samples_int.astype(np.double) * self.dx + self.grid_bounds[:, 0]
        self.validation_values = self.func(self.validation_samples)

    def compute_observation_matrix(self, idxs_int, j):
        '''
        j is the index to ignore in the sample array (the column dimension)
        of the observation matrix. Only works for the one-site version. 
        '''
        result = np.zeros((idxs_int.shape[0], self.dims[j]), dtype=np.double)
        idxs = idxs_int.astype(np.double) * self.dx + self.grid_bounds[:, 0]

        for i in range(self.dims[j]):
            idxs[:, j] = i * self.dx[j] + self.grid_bounds[j, 0]
            result[:, i] = self.func(idxs)

        return result

    def execute_sampled_spmm(self, samples, design, j, result):
        observation = self.compute_observation_matrix(samples, j)
        result[:] = observation.T @ design  
    
    def compute_approx_tt_fit(self, tt_approx):
        '''
        Compute the fit of a TT approximation to the function tensor
        '''
        if self.validation_samples is None:
            raise ValueError("Must set validation samples before computing fit")

        validation_approx = tt_approx.evaluate_partial_fast(
                self.validation_samples_int,
                self.N, "left").squeeze()

        return 1.0 - la.norm(self.validation_values - validation_approx) / la.norm(self.validation_values)

