import numpy as np
import numpy.linalg as la

from quantization import *
from cpp_ext.tt_module import reproducible_noise 

class FunctionTensor:
    def __init__(self, grid_bounds, dims, func, quantization=None, track_evals=False, noise_params=None, noisy=False): 
        self.grid_bounds = np.array(grid_bounds, dtype=np.double)
        self.dims = np.array(dims, dtype=np.uint64)
        self.func = func
        self.N = len(dims)
        self.dx = [(grid_bounds[i, 1] - grid_bounds[i, 0]) / dims[i] for i in range(self.N)] 

        # Validation set for fit computation
        self.validation_samples = None
        self.validation_values = None
        self.quantization = quantization
        self.track_evals = track_evals

        if self.track_evals:
            self.evals = []

        # Noise param is a tuple (type, param1,...). If type is "gaussian",
        # Then param1 is the standard deviation of the noise. 
        self.noise_params = noise_params

        if noisy:
            self.noise_seed1 = np.random.randint(0, 2**64 - 1, dtype=np.uint64)
            self.noise_seed2 = np.random.randint(0, 2**64 - 1, dtype=np.uint64)

    def generate_reproducible_noise(self, indices):
        #if noise_params[0] is not "gaussian":
        #    raise NotImplementedError("Only gaussian noise is supported")
        #else:

        noise_vector = np.zeros((indices.shape[0]), dtype=np.double)
        reproducible_noise(
                indices,
                noise_vector,
                self.noise_seed1,
                self.noise_seed2)
        
        return noise_vector * 0.3

    def initialize_accuracy_estimation(self, method="randomized", rsample_count=10000):
        if method != "randomized":
            raise NotImplementedError("Only randomized validation set generation is supported")

        self.validation_samples_int = np.zeros((rsample_count, self.N), dtype=np.uint64)
        for i in range(self.N):
            self.validation_samples_int[:, i] = np.random.randint(0, self.dims[i], size=rsample_count)

        self.validation_samples = self.validation_samples_int.astype(np.double) * self.dx + self.grid_bounds[:, 0]
        self.validation_values = self.func(self.validation_samples)

        if self.quantization is not None:
            self.validation_samples_int = self.quantization.quantize_indices(self.validation_samples_int)

    def indices_to_spatial_points(self, idxs_int):  
        return idxs_int.astype(np.double) * self.dx + self.grid_bounds[:, 0]

    def compute_observation_matrix(self, idxs_int, j):
        '''
        j is the index to ignore in the sample array (the column dimension)
        of the observation matrix. This function is specific to the one-site
        version of the code. 
        '''
        if self.quantization is not None:
            dims = self.quantization.qdim_sizes
        else:
            dims = self.dims

        ncol = dims[j] 
        result = np.zeros((idxs_int.shape[0], ncol), dtype=np.double)

        for i in range(ncol):
            idxs_int[:, j] = i
            idxs_unquant = self.quantization.unquantize_indices(idxs_int) 
            idxs = idxs_unquant.astype(np.double) * self.dx + self.grid_bounds[:, 0]
            result[:, i] = self.func(idxs) 
            
            if noisy:
                result[:, i] += self.generate_reproducible_noise(idxs_int)

            if self.track_evals:
                self.evals.append(idxs_unquant.copy())

        return result

    def evaluate(self, samples):
        idxs = None
        if self.quantization is not None:
            idxs_unquant = self.quantization.unquantize_indices(samples) 
            idxs = idxs_unquant.astype(np.double) * self.dx + self.grid_bounds[:, 0]
        else:
            idxs_unquant = idxs_int

        idxs = idxs_unquant.astype(np.double) * self.dx + self.grid_bounds[:, 0]
        result = self.func(idxs)

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
                tt_approx.N, "left").squeeze()

        return 1.0 - la.norm(self.validation_values - validation_approx) / la.norm(self.validation_values)

