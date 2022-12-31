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
class __attribute__((visibility("hidden"))) LarsenKoldaHybrid : public Sampler {
public:

    vector<unique_ptr<Buffer<double>>> factor_leverage;
    Buffer<double> leverage_sums;
    vector<unique_ptr<std::discrete_distribution<uint64_t>>> distributions;

    LarsenKoldaHybrid(
            uint64_t J, 
            uint64_t R, 
            vector<Buffer<double>> &U_matrices)
    :       
            Sampler(J, R, U_matrices),
            leverage_sums({N})
    {
        for(uint32_t k = 0; k < N; k++) {
            uint64_t Ik = U[k].shape[0];
            factor_leverage.emplace_back();
            factor_leverage[k].reset(new Buffer<double>({Ik}));
            distributions.emplace_back();
            update_sampler(k);
        } 
    }

    void update_sampler(uint64_t j) {
        uint64_t Ij = U[j].shape[0];
        uint64_t rank = min(Ij, R);

        Buffer<double> Q({Ij, R});
        Buffer<double> tau({rank});

        std::copy(U[j](), U[j](Ij * R), Q());

        LAPACKE_dgeqrf(
            CblasRowMajor,
            Ij,
            R,
            Q(),
            R,
            tau()
        );

        LAPACKE_dorgqr(CblasRowMajor,
            Ij,
            rank,
            rank, 
            Q(),
            R,
            tau()
        );

        Buffer<double> &leverage = *(factor_leverage[j]);
        double total = 0.0;

        #pragma omp parallel for reduction(+: total)
        for(uint64_t i = 0; i < Ij; i++) {
            leverage[i] = 0.0;
            for(uint64_t k = 0; k < R; k++) {
                leverage[i] += Q[i * R + k] * Q[i * R + k];
            }
            total += leverage[i]; 
        }

        distributions[j].reset(new discrete_distribution<uint64_t>(leverage(), 
            leverage(Ij)));
 
        leverage_sums[j] = total;
    }

    void KRPDrawSamples(uint32_t j, Buffer<uint64_t> &samples, Buffer<double> *random_draws) {
        std::fill(weights(), weights(J), 0.0-log((double) J));
        double sum = 0.0;
        for(uint32_t k = 0; k < N; k++) {
            if(k != j) {
                Buffer<double> &leverage = *(factor_leverage[k]);
                Buffer<uint64_t> row_buffer({J}, samples(k, 0)); // View into a row of the samples array
                std::discrete_distribution<uint64_t> &dist = (*(distributions[k]));

                for(uint64_t i = 0; i < J; i++) {
                    uint64_t sample = dist(gen);
                    sum += sample;
                    row_buffer[i] = sample;
                    weights[i] += log(leverage_sums[k]) - log(leverage[sample]); 
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