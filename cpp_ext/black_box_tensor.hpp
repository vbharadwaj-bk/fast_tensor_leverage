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
    uint64_t max_rhs_rows;

    virtual void preprocess(Buffer<uint64_t> &samples, uint64_t j) = 0;
    virtual void materialize_rhs(Buffer<uint64_t> &samples, uint64_t j, Buffer<double> &rhs_buf) = 0; 

    void execute_downsampled_mttkrp(
            Buffer<uint64_t> &samples_transpose, 
            Buffer<double> &lhs,
            uint64_t j,
            Buffer<double> &result
            ) {

        uint64_t N = dims.size();
        uint64_t num_samples = samples_transpose.shape[0];

        Buffer<double> rhs_buf({max_rhs_rows, dims[j]});
        preprocess(samples_transpose, j);

        if(result.shape[0] != dims[j] || result.shape[1] != lhs.shape[1]) {
            throw runtime_error("Result buffer has incorrect shape");
        }

        // Result is a dims[j] x R matrix
        std::fill(result(), result(dims[j], 0), 0.0);


        for(uint64_t i = 0; i < num_samples; i += max_rhs_rows) {
            uint64_t max_range = min(i + max_rhs_rows, num_samples);
            uint64_t rows = (uint64_t) (max_range - i);

            // Create a view into the sample matrix
            Buffer<uint64_t> sample_view(
                {rows, N},
                samples_transpose(i * N));


            auto t = start_clock();
            materialize_rhs(sample_view, j, rhs_buf);
            double elapsed = stop_clock_get_elapsed(t);

            cout << "Time spent materializing RHS: " << elapsed << endl;

            t = start_clock();
            cblas_dgemm(
                CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                (uint32_t) dims[j],
                (uint32_t) lhs.shape[1],
                (uint32_t) rows,
                1.0,
                rhs_buf(),
                (uint32_t) dims[j],
                lhs(i, 0),
                (uint32_t) lhs.shape[1],
                1.0,
                result(),
                (uint32_t) lhs.shape[1]
            );
            elapsed = stop_clock_get_elapsed(t);
            cout << "Elapsed on DGEMM: " << elapsed << endl;
        }
    }
};