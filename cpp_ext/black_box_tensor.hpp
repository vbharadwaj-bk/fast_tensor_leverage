#pragma once

#include <iostream>
#include <string>
#include "common.h"
#include "cblas.h"
#include "lapacke.h"
#include "tensor.hpp"

using namespace std;

class __attribute__((visibility("hidden"))) BlackBoxTensor : public Tensor {
public:
    vector<uint64_t> dims;
    uint64_t J, max_rhs_rows;
    Buffer<double>* rhs_buf; 

    virtual void preprocess(Buffer<uint64_t> &samples, uint64_t j) = 0;
    virtual void materialize_rhs(Buffer<uint64_t> &samples, uint64_t j, uint64_t row_pos) = 0; 

    void execute_downsampled_mttkrp(
            Buffer<uint64_t> &samples, 
            Buffer<double> &lhs,
            uint64_t j,
            Buffer<double> &result
            ) {
        
        rhs_buf = new Buffer<double>({max_rhs_rows, dims[j]});
        Buffer<double> &temp_buf = (*rhs_buf);
        preprocess(samples, j);

        // Result is a dims[j] x R matrix
        std::fill(result(), result(dims[j], 0), 0.0);

        for(uint64_t i = 0; i < J; i += max_rhs_rows) {
            uint64_t max_range = min(i + max_rhs_rows, J);
            uint32_t rows = (uint32_t) (max_range - i);

            materialize_rhs(samples, j, i);

            cblas_dgemm(
                CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                (uint32_t) dims[j],
                (uint32_t) lhs.shape[1],
                (uint32_t) rows,
                1.0,
                temp_buf(),
                (uint32_t) dims[j],
                lhs(i, 0),
                (uint32_t) lhs.shape[1],
                1.0,
                result(),
                (uint32_t) lhs.shape[1]
            );
        }
        delete rhs_buf; 
    }
};