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
    //py::function fiber_evaluator;

    PyFunctionTensor(py::array_t<uint64_t> dims_py,
            uint64_t J,
            uint64_t max_rhs_rows) {
        this->max_rhs_rows = max_rhs_rows;
        this->J = J;
        //this->fiber_evaluator = fiber_evaluator;

        Buffer<uint64_t> dims_in(dims_py);

        for(uint32_t i = 0; i < dims_in.shape[0]; i++) {
            dims.push_back(dims_in[i]);
        }
    }

    void preprocess(Buffer<uint64_t> &samples, uint64_t j) {
        // Empty - use to initialize data structures
        // before repeatedly calling ``materialize_rhs" 
    }

    void test_fiber_evaluator() {
        Buffer<double> temp({J});

        //fiber_evaluator((uint64_t) temp());
        cout << temp[0] << endl;
        cout << temp[1] << endl;
        cout << temp[50] << endl;
        cout << temp[100] << endl;
    }

    void materialize_rhs(Buffer<uint64_t> &samples_transpose, uint64_t j, uint64_t row_pos) {
        Buffer<double> &temp_buf = (*rhs_buf);
        uint64_t max_range = min(row_pos + max_rhs_rows, J);
        uint32_t M = (uint32_t) (max_range - row_pos);
        uint32_t Ij = (uint32_t) dims[j]; 
        uint32_t tensor_dim = dims.size();

        //fiber_evaluator((uint64_t) (temp_buf()), (uint64_t) (samples()), (uint32_t) j, (uint32_t) row_pos, M, Ij, tensor_dim);

        // As a test, we will directly compute the integral of the sine function here.

        #pragma omp parallel for
        for(uint64_t i = 0; i < M; i++) {
            uint64_t offset = (row_pos + i) * tensor_dim;
            samples_transpose[offset + j] = 0;
            double partial = std::accumulate(samples_transpose(offset), samples_transpose(offset + tensor_dim), 0);

            for(uint64_t k = 0; k < Ij; k++) {
                //temp_buf[i * Ij + k] = sin((partial + k) * 0.01);
                temp_buf[i * Ij + k] = 1.0; 
            }
        }
    };
};