#pragma once

#include <iostream>
#include <string>
#include "common.h"
#include "cblas.h"
#include "lapacke.h"

using namespace std;

void ATB_chain_prod(
        vector<Buffer<double>> &A,
        vector<Buffer<double>> &B,
        Buffer<double> &sigma_A, 
        Buffer<double> &sigma_B,
        Buffer<double> &result) {

        uint64_t N = A.size();
        uint64_t R_A = A[0].shape[1];
        uint64_t R_B = B[0].shape[1];

        vector<unique_ptr<Buffer<double>>> ATB;
        for(uint64_t i = 0; i < A.size(); i++) {
                ATB.emplace_back();
                ATB[i].reset(new Buffer<double>({R_A, R_B}));
        }

        for(uint64_t i = 0; i < R_A; i++) {
                for(uint64_t j = 0; j < R_B; j++) {
                        result[i * R_B + j] = sigma_A[i] * sigma_B[j];
                }
        }

        // Can replace with a batch DGEMM call
        for(uint64_t i = 0; i < N; i++) {
                uint64_t K = A[i].shape[0];
                cblas_dgemm(
                        CblasRowMajor,
                        CblasTrans,
                        CblasNoTrans,
                        R_A,
                        R_B,
                        K,
                        1.0,
                        A[i](),
                        R_A,
                        B[i](),
                        R_B,
                        0.0,
                        (*(ATB[i]))(),
                        R_B
                );
        }

        for(uint64_t k = 0; k < N; k++) {
                for(uint64_t i = 0; i < R_A; i++) {
                        for(uint64_t j = 0; j < R_B; j++) {
                                result[i * R_B + j] *= (*(ATB[k]))[i * R_B + j];
                        }
                }
        }
}

class __attribute__((visibility("hidden"))) Tensor {
public:
    virtual void 
    execute_downsampled_mttkrp(
            Buffer<uint64_t> &samples, 
            Buffer<double> &lhs,
            uint64_t j,
            Buffer<double> &result
            ) = 0;

    void execute_downsampled_mttkrp_py(
            py::array_t<uint64_t> &samples_py, 
            py::array_t<double> &lhs_py,
            uint64_t j,
            py::array_t<double> &result_py
            ) {
        
        Buffer<uint64_t> samples(samples_py);
        Buffer<double> lhs(lhs_py);
        Buffer<double> result(result_py);

        execute_downsampled_mttkrp(
                samples, 
                lhs,
                j,
                result 
                );
    }
};
