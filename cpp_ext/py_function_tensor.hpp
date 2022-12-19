#pragma once

#include <iostream>
#include <string>
#include "common.h"
#include "cblas.h"
#include "lapacke.h"
#include "black_box_tensor.hpp"

using namespace std;

class __attribute__((visibility("hidden"))) PyFunctionTensor : public BlackBoxTensor {
public:
    py::function fiber_evaluator;

    PyFunctionTensor(py::function fiber_evaluator, 
            uint64_t J,
            uint64_t max_rhs_rows) {
        this->max_rhs_rows = max_rhs_rows;
        this->J = J;
        this->fiber_evaluator = fiber_evaluator;
    }

    void preprocess(Buffer<uint64_t> &samples, uint64_t j) {
        // Empty - use to initialize data structures
        // before repeatedly calling ``materialize_rhs" 
    }

    void materialize_rhs(Buffer<uint64_t> &samples, uint64_t j, uint64_t row_pos) {
        Buffer<double> &temp_buf = (*rhs_buf);
        uint64_t max_range = min(row_pos + max_rhs_rows, J);
        uint32_t M = (uint32_t) (max_range - row_pos);
        uint32_t Ij = (uint32_t) dims[j]; 
        uint32_t tensor_dim = dims.size();

        fiber_evaluator(temp_buf(), samples(), j, row_pos, M, Ij, tensor_dim);
    };

};