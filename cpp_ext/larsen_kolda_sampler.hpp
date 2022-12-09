#pragma once

#include <iostream>
#include <vector>
#include <cassert>
#include <random>
#include <memory>
#include "common.h"
#include "omp.h"
#include "cblas.h"
#include "lapacke.h"

using namespace std;

/*
* Larsen / Kolda leverage score sampler 
*/
class __attribute__((visibility("hidden"))) LarsenKoldaSampler : public Sampler {
public:

    vector<Buffer<double>> factor_leverage;
    Buffer<double> leverage_sums;
    vector<unique_ptr<std::discrete_distribution<uint64_t>>> distributions;

    LarsenKoldaSampler(
            uint64_t J, 
            uint64_t R, 
            vector<Buffer<double>> &U_matrices)
    :       
            Sampler(J, R, U_matrices),
            leverage_sums({N})
    {
        for(uint32_t k = 0; k < N; k++) {
            uint64_t Ij = U[k].shape[0];
            factor_leverage.emplace_back(initializer_list<uint64_t>{Ij});
            distributions.emplace_back();
            update_sampler(k);
        } 
    }

    void update_sampler(uint64_t j) {
        Buffer<double> pinv({R, R});
        compute_pinv(U[j], pinv);
        compute_DAGAT(U[j](), pinv(), factor_leverage[j](), U[j].shape[0], R);
        distributions[j].reset(new discrete_distribution<uint64_t>(factor_leverage[j](), 
                                                factor_leverage[j](U[j].shape[0])));

        leverage_sums[j] = 0.0;

        double total = 0.0;
        #pragma omp parallel for reduction(+: total)
        for(uint64_t i = 0; i < U[j].shape[0]; i++) {
            total += factor_leverage[j][i];
        }
        leverage_sums[j] = total;
    }

    void KRPDrawSamples(uint32_t j, Buffer<uint64_t> &samples, Buffer<double> *random_draws) {
        std::fill(weights(), weights(J), 0.0-log((double) J));
        for(uint32_t k = 0; k < N; k++) {
            if(k != j) {
                Buffer<uint64_t> row_buffer({J}, samples(k, 0)); // View into a row of the samples array
                std::discrete_distribution<uint64_t> &dist = (*(distributions[k]));
                // Random number generation not parallelized on CPU, since it is cheap 
                for(uint64_t i = 0; i < J; i++) {
                    uint64_t sample = dist(gen);
                    row_buffer[i] = sample;
                    weights[i] += log(factor_leverage[k][sample]) - log(leverage_sums[k]);
                }
            }
        }

        #pragma omp parallel for
        for(uint64_t i = 0; i < J; i++) {
            weights[i] = exp(weights[i]); 
        }

        fill_h_by_samples(samples, j);
    }
};