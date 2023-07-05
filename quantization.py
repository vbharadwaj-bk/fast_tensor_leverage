import numpy as np

import cppimport
import cppimport.import_hook
from cpp_ext.tt_module import quantize_indices, unquantize_indices

class Quantization:
    def quantize_indices(self, indices):
        raise NotImplementedError()

    def unquantize_indices(self, quantized_indices):
        raise NotImplementedError()

class Power2Quantization(Quantization):
    def __init__(self, dims, ordering):
        self.dims = dims
        self.ordering = ordering

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


    def unquantize_indices(self, quantized_indices):
        J = quantized_indices.shape[0]
        indices = np.zeros((J, len(self.dims)), dtype=np.uint64)

        unquantize_indices(
            quantized_indices, 
            self.quantization_dimensions, 
            self.permutation, 
            indices)

        return indices