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

/*
* Samples uniformly at random from all rows of the KRP. 
*/
class __attribute__((visibility("hidden"))) UniformSampler: public Sampler {
public:
    UniformSampler(
            uint64_t J, 
            uint64_t R, 
            vector<Buffer<double>> &U_matrices)
    :       
            Sampler(J, R, U_matrices)
    {
        // Empty
    }

    void update_sampler(uint64_t j) {
        // Empty
    }

    void KRPDrawSamples(uint32_t j, Buffer<uint64_t> &samples, Buffer<double> *random_draws) {
        double logsum = 0.0;
        for(uint32_t k = 0; k < N; k++) {
            if(k != j) {
                uint64_t Ij = U[k].shape[0];
                logsum += log((double) Ij);
                Buffer<uint64_t> row_buffer({J}, samples(k, 0)); // View into a row of the samples array
                std::uniform_int_distribution<uint64_t> dis(0, Ij);

                // Random number generation not parallelized on CPU, since it is cheap 
                for(uint64_t i = 0; i < J; i++) {
                    row_buffer[i] = dis(gen);
                }
            }
        }
        logsum -= log((double) J);
        double weight = exp(logsum);

        #pragma omp parallel for
        for(uint64_t i = 0; i < J; i++) {
            weights[i] = weight; 
        }
    }
};