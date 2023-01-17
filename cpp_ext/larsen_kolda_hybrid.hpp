#pragma once

#include <iostream>
#include <vector>
#include <cassert>
#include <random>
#include <memory>
#include <execution>
#include <algorithm>
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
    vector<unique_ptr<Buffer<uint64_t>>> sort_idxs;
    Buffer<double> leverage_sums;
    Buffer<double> alpha;
    double tau;
    vector<unique_ptr<std::discrete_distribution<uint64_t>>> distributions;

    LarsenKoldaHybrid(
            uint64_t J, 
            uint64_t R, 
            vector<Buffer<double>> &U_matrices)
    :       
            Sampler(J, R, U_matrices),
            leverage_sums({N}),
            alpha({N})
    {
        tau = 1.0 / J; // For now, this is hardcoded 

        for(uint32_t k = 0; k < N; k++) {
            uint64_t Ik = U[k].shape[0];
            factor_leverage.emplace_back();
            sort_idxs.emplace_back();
            factor_leverage[k].reset(new Buffer<double>({Ik}));
            sort_idxs[k].reset(new Buffer<uint64_t>({Ik}));

            #pragma omp parallel for
            for(uint64_t i = 0; i < Ik; i++) {
                (*(sort_idxs[k]))[i] = i;
            }

            distributions.emplace_back();
            update_sampler(k);
        } 
    }

    void update_sampler(uint64_t j) {
        uint64_t Ij = U[j].shape[0];
        double alpha_j = 0.0;
        /*
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

        #pragma omp parallel for reduction(+: total) reduction(max: alpha_j)
        for(uint64_t i = 0; i < Ij; i++) {
            leverage[i] = 0.0;
            for(uint64_t k = 0; k < R; k++) {
                leverage[i] += Q[i * R + k] * Q[i * R + k];
            }
            total += leverage[i];
            alpha_j = max(alpha_j, leverage[i]);
        }
        */
        Buffer<double> pinv({R, R});
        compute_pinv(U[j], pinv);

        Buffer<double> &leverage = *(factor_leverage[j]);
        compute_DAGAT(U[j](), pinv(), leverage(), Ij, R);

        distributions[j].reset(new discrete_distribution<uint64_t>(leverage(), 
            leverage(Ij)));

        double total = 0.0;

        #pragma omp parallel for reduction(+: total) reduction(max: alpha_j)
        for(uint64_t i = 0; i < U[j].shape[0]; i++) {
            total += leverage[i];
            alpha_j = max(alpha_j, leverage[i]);
        }
        leverage_sums[j] = total;

        distributions[j].reset(new discrete_distribution<uint64_t>(leverage(), 
            leverage(Ij)));
 
        leverage_sums[j] = total;
        alpha[j] = alpha_j / total;

        Buffer<uint64_t> &sort_idx = *(sort_idxs[j]);
        double* leverage_ptr = leverage();

        std::sort(std::execution::par_unseq, 
            sort_idx(), 
            sort_idx(Ij),
            [leverage_ptr](uint64_t a, uint64_t b) {
                return leverage_ptr[a] < leverage_ptr[b];
            });
    }

    void KRPDrawSamples(uint32_t j, Buffer<uint64_t> &samples, Buffer<double> *random_draws) {
        double alpha_star = 1.0;

        for(uint64_t k = 0; k < N; k++) {
            if(k != j) {
                alpha_star *= alpha[k];
            }
        }

        Buffer<uint64_t> start_ranges({N}); 
        Buffer<uint64_t> intervals({N}); 

        uint64_t candidates = 1;
        double log_candidates = 0.0;

        // Use the sorted arrays to get bounds on high-leverage candidates
        for(uint64_t k = 0; k < N; k++) {
            if(k != j) {
                double threshold = tau * alpha[k] / alpha_star;

                uint64_t Ik = U[k].shape[0];
                Buffer<uint64_t> &sort_idx = *(sort_idxs[k]);
                Buffer<double> &leverage = *(factor_leverage[k]);
                double* leverage_ptr = leverage();            
                uint64_t* lb_ptr = std::lower_bound(sort_idx(), sort_idx(Ik), threshold * leverage_sums[k],
                    [leverage_ptr](uint64_t sort_val, double thresh) {
                        return leverage_ptr[sort_val] < thresh;
                    } 
                );
                start_ranges[k] = lb_ptr - sort_idx();
                intervals[k] = Ik - start_ranges[k];
                candidates *= intervals[k];
                log_candidates += log(intervals[k]);
            }
        }

        uint64_t num_deterministic = 0;
        double p_det = 0.0;

        if(log_candidates >= 62 * log(2)) {
            cout << "Warning: Hybrid sampler candidate list is too large. Switching to deterministic algorithm." << endl;
        }
        else {
            uint64_t sample_pos = 0;

            #pragma omp parallel
{
            Buffer<uint64_t> sample({N});

            #pragma omp for reduction(+: num_deterministic, p_det)
            for(uint64_t i = 0; i < candidates; i++) {
                double weight = 0;
                uint64_t c_idx = i;
                for(uint64_t k = 0; k < N; k++) {
                    if(k != j) {
                        uint64_t idx_k = c_idx % intervals[k]; 
                        c_idx /= intervals[k];
                        uint64_t sample_k = (*(sort_idxs[k]))[start_ranges[k] + idx_k];
                        sample[k] = sample_k;
                        weight += log((*(factor_leverage[k]))[sample_k]) - log(leverage_sums[k]);
                    }
                }

                if(exp(weight) >= tau) {
                    uint64_t pos;
                    #pragma omp atomic capture 
                    pos = sample_pos++;

                    for(uint64_t k = 0; k < N; k++) {
                        samples[k * J + pos] = sample[k]; 
                    }

                    weights[pos] = 1.0;

                    num_deterministic++;
                    p_det += exp(weight); 
                }
            }
}
        }
        std::fill(weights(num_deterministic), weights(J), 0.0-log((double) J));

        #pragma omp parallel
{
        int thread_id = omp_get_thread_num();
        auto &local_gen = par_gen[thread_id];

        #pragma omp for
        for(uint64_t i = num_deterministic; i < J; i++) {
            double weight;
            bool resample = true;

            while(resample) {
                weight = 0.0;
                for(uint32_t k = 0; k < N; k++) {
                    if(k != j) {
                        std::discrete_distribution<uint64_t> &dist = (*(distributions[k]));
                        Buffer<double> &leverage = *(factor_leverage[k]);
                        uint64_t sample = dist(local_gen);
                        samples[k * J + i] = sample;
                        weight += log(leverage[sample]) - log(leverage_sums[k]); 
                    }
                }

                if(exp(weight) < tau) {
                    resample = false;
                }
            } 
            weights[i] -= weight;
            weights[i] = (1 - p_det) * exp(weights[i]); 
        }
}

        fill_h_by_samples(samples, j);
    }
};