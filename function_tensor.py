import numpy as np
import numpy.linalg as la

class FunctionTensor:
    def __init__(self, grid_bounds, dims, func): 
        self.grid_bounds = np.array(grid_bounds, dtype=np.double)
        self.dims = np.array(dims, dtype=np.uint64)
        self.func = func
        self.N = len(dims)
        self.dx = [(grid_bounds[i, 1] - grid_bounds[i, 0]) / dims[i] for i in range(self.N)] 

        # Validation set for fit computation
        self.validation_samples = None
        self.validation_values = None

    def initialize_accuracy_estimation(self, method="randomized", rsample_count=10000):
        if method != "randomized":
            raise NotImplementedError("Only randomized validation set generation is supported")

        validation_samples_int = np.zeros((rsample_count, self.N), dtype=np.uint64)
        for i in range(self.N):
            validation_samples_int[:, i] = np.random.randint(0, self.dims[i], size=rsample_count)

        self.validation_samples = validation_samples_int.astype(np.double) * self.dx[j] + self.grid_bounds[:, 0]
        self.validation_values = self.func(self.validation_samples)

    def compute_observation_matrix(self, idxs_int, j):
        '''
        j is the index to ignore in the sample array (the column dimension)
        of the observation matrix. Only works for the one-site version. 
        '''
        result = np.zeros((idxs_int.shape[0], self.dims[j]), dtype=np.double)
        idxs = idxs_int.astype(np.double) * self.dx[j] + self.grid_bounds[:, 0]

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

        validation_approx = tt_approx.evaluate(self.validation_samples)
        return 1.0 - la.norm(self.validation_values - validation_approx) / la.norm(self.validation_values)

