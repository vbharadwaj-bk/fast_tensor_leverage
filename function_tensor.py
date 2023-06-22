import numpy as np
import numpy.linalg as la

class FunctionTensor:
    def __init__(self, grid_bounds, dims, func): 
        self.grid_bounds = np.array(grid_bounds, dtype=np.double)
        self.dims = np.array(dims, dtype=np.uint64)
        self.func = func
        self.N = len(dims)
        self.dx = [(grid_bounds[i, 1] - grid_bounds[i, 0]) / dims[i] for i in range(self.N)] 

    def compute_observation_matrix(self, idxs_int, j):
        '''
        j is the index to ignore in the sample array (the column dimension)
        of the observation matrix. Only works for the one-site version. 
        '''
        result = np.zeros((idxs_int.shape[0], self.dims[j]), dtype=np.double)
        idxs = idxs.astype(np.double) * self.dx[j] + self.grid_bounds[:, 0]

        for i in range(self.dims[j]):
            idxs[:, j] = i * self.dx[j] + self.grid_bounds[j, 0]
            result[:, i] = self.func(idxs)
            print(idxs)

        return result