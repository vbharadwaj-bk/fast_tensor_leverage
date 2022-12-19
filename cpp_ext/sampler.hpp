#pragma once

#include <iostream>
#include <vector>
#include <cassert>
#include <random>
#include "common.h"
#include "omp.h"
#include "cblas.h"
#include "lapacke.h"

using namespace std;

class __attribute__((visibility("hidden"))) Sampler {
public:
    uint64_t N, J, R, R2;
    vector<Buffer<double>> &U;
    Buffer<double> h;
    Buffer<double> weights;

    // Related to random number generation 
    std::random_device rd;  
    std::mt19937 gen;

    Sampler(uint64_t J, 
            uint64_t R, 
            vector<Buffer<double>> &U_matrices) : 
        U(U_matrices),
        h({J, R}),
        weights({J}),
        rd(),
        gen(rd())
        {
        this->N = U.size();
        this->J = J;
        this->R = R;
        R2 = R * R;
    }

    virtual void update_sampler(uint64_t j) = 0;
    virtual void KRPDrawSamples(uint32_t j, Buffer<uint64_t> &samples, Buffer<double> *random_draws) = 0;

    /*
    * Fills the h matrix based on an array of samples. Can be bypassed if KRPDrawSamples computes
    * h during its execution.
    */
    void fill_h_by_samples(Buffer<uint64_t> &samples, uint64_t j) {
        std::fill(h(), h(J * R), 1.0);
        for(uint32_t k = 0; k < N; k++) {
            if(k != j) {
                Buffer<uint64_t> row_buffer({J}, samples(k, 0)); // View into a row of the samples array

                #pragma omp parallel for 
                for(uint64_t i = 0; i < J; i++) {
                    uint64_t sample = row_buffer[i];
                    for(uint64_t u = 0; u < R; u++) {
                        h[i * R + u] *= U[k][sample * R + u];
                    }
                }
            }
        }
    } 
};

void compute_DAGAT(double* A, double* G, 
        double* res, uint64_t J, uint64_t R) {

    Buffer<double> temp({J, R});

    cblas_dsymm(
        CblasRowMajor,
        CblasRight,
        CblasUpper,
        (uint32_t) J,
        (uint32_t) R,
        1.0,
        G,
        R,
        A,
        R,
        0.0,
        temp(),
        R
    );

    #pragma omp parallel for 
    for(uint32_t i = 0; i < J; i++) {
        res[i] = 0.0;
        for(uint32_t j = 0; j < R; j++) {
            res[i] += A[i * R + j] * temp[i * R + j];
        }
    }
}
