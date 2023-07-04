#pragma once

#include <iostream>
#include <string>
#include "common.h"
#include "cblas.h"
#include "lapacke.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;

/*
* indices is a J x N array of indices to quantize
* quantization_dims is an array of length N x (max quantization_dims). Each
* row is a list of quantization dimensions in order, followed by trailing -1s for ragged entries.
* Let NQ be the total number of quantization dimensions. 
* permutation is a length NQ array of indices into the quantization_dims array.
* quantized_indices is a J x NQ array of quantized indices, the output parameter. 
*/

void quantize_indices(
    py::array_t<uint64_t> indices_py,
    py::array_t<uint64_t> quantization_dims_py,
    py::array_t<uint64_t> permutation_py, 
    py::array_t<uint64_t> quantized_indices_py, 
    ) {

    Buffer<uint64_t> indices(indices_py);
    Buffer<uint64_t> quantization_dims(quantization_dims_py);
    Buffer<uint64_t> permutation(permutation_py);
    Buffer<uint64_t> quantized_indices(quantized_indices_py);

    uint64_t J = indices.shape[0];
    uint64_t N = indices.shape[1];
    uint64_t mqd = quantization_dims.shape[1];

    Buffer<uint64_t> quant_dim_nums({N});
    std::fill(quant_dim_nums(), quant_dim_nums(N), 0);
    Buffer<uint64_t> prefix_prods({N}, max_q_dims);

    for(uint64_t i = 0; i < N; i++) {
        prefix_prods[i * mqd] = 1;
        quant_dim_nums[i] = 0;
        for(uint64_t j = 1; j < mqd ; j++) {
            if(quantization_dims[i * mqd + j - 1] == -1) {
                break;
            }
            else {
                prefix_prods[i * mqd + j] = prefix_prods[i * mqd + j - 1] * quantization_dims[i * mqd + j - 1];
                quant_dim_nums[i]++;
            }
        }
    }

    uint64_t NQ = std::accumulate(quant_dim_nums(), quant_dim_nums(N), 0);

    Buffer<uint64_t> unpermuted({NQ});

    for(uint64_t i = 0; i < J; i++) {
        uint64_t current_position = 0;
        for(uint64_t j = 0; j < N; j++) {
            for(uint64_t k = 0; k < quant_dim_nums[j]; k++) {
                unpermuted[current_position] = 
                    (indices[i * N + j] / prefix_prods[j * mqd + k]) % quantization_dims[j * mqd + k];
                current_position++;
            }
        }
        for(uint64_t j = 0; j < NQ; j++) {
            quantized_indices[i * NQ + j] = unpermuted[permutation[j]];
        }
    }
}


