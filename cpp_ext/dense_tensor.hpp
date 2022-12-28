#pragma once

#include <iostream>
#include <string>
#include "common.h"
#include "cblas.h"
#include "lapacke.h"
#include "black_box_tensor.hpp"

using namespace std;

template<typename VAL_T>
class __attribute__((visibility("hidden"))) DenseTensor : public BlackBoxTensor {
public:
    Buffer<VAL_T> tensor;
    Buffer<uint64_t> prefix_prod;
    DenseTensor(py::array_t<VAL_T> tensor_py, uint64_t max_rhs_rows) 
    :   tensor(tensor_py),
        dims(tensor.shape)
        prefix_prod({dims.size()})
    {
        this->max_rhs_rows = max_rhs_rows;

        prefix_prod[dims.size() - 1] = 1;
        for(int i = dims.size() - 2; i >= 0; i--) {
            prefix_prod[i] = prefix_prod[i+1] * dims[i+1];
        }
    }

    void preprocess(Buffer<uint64_t> &samples, uint64_t j) {
        // Empty - use to initialize data structures
        // before repeatedly calling ``materialize_rhs" 
    }

    void materialize_rhs(Buffer<uint64_t> &samples_transpose, 
            uint64_t j, 
            Buffer<double> &rhs_buf) {

        uint64_t num_samples = samples_transpose.shape[0];
        uint64_t Ij = dims[j];
        uint64_t N = dims.size();
        uint64_t ldt = prefix_prod[j];

        #pragma omp parallel for 
        for(uint64_t i = 0; i < num_samples; i++) {
            uint64_t offset = 0;
            for(int u = 0; u < N; u++) {
                if(u != j) {
                    offset += samples_transpose[i * N + u] * prefix_prod[u];
                }
            }

            for(uint64_t k = 0; k < Ij; k++) {
                rhs_buf[i * Ij + k] = tensor[offset + k * ldt];                
            }
        }
    };
};