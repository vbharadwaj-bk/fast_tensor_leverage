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
    DenseTensor(py::array_t<VAL_T> tensor_py, uint64_t max_rhs_rows) 
    : tensor(tensor_py) 
    {
        this->max_rhs_rows = max_rhs_rows;
    }

    void preprocess(Buffer<uint64_t> &samples, uint64_t j) {
        // Empty - use to initialize data structures
        // before repeatedly calling ``materialize_rhs" 
    }

    void materialize_rhs(Buffer<uint64_t> &samples_transpose, uint64_t j, uint64_t row_pos) {
        Buffer<double> &temp_buf = (*rhs_buf);
        uint64_t max_range = min(row_pos + max_rhs_rows, samples_transpose.shape[0]);
        uint32_t M = (uint32_t) (max_range - row_pos);
        uint32_t Ij = (uint32_t) tensor.shape[j]; 
        uint32_t tensor_dim = tensor.shape.size(); 

    };
};