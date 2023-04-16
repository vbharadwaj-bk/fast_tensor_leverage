#pragma once

#include <iostream>
#include <string>
#include "common.h"
#include "cblas.h"
#include "lapacke.h"
#include "sampler.hpp"
#include "tensor.hpp"
#include "low_rank_tensor.hpp"
#include "json.hpp"
#include <execution>
#include <algorithm>

using namespace std;
using json = nlohmann::json;

class __attribute__((visibility("hidden"))) ALS {
public:
    LowRankTensor &cp_decomp;
    Tensor &ground_truth;

    // These fields used only for sampling 
    uint64_t J;
    unique_ptr<Sampler> sampler;
    unique_ptr<Buffer<uint64_t>> samples;
    unique_ptr<Buffer<uint64_t>> samples_transpose;

    json statistics;

    ALS(LowRankTensor &cp_dec, Tensor &gt) : 
        cp_decomp(cp_dec),
        ground_truth(gt)    
    {
        statistics["test"] = 1.0;
        // Empty
    }

    void initialize_ds_als(uint64_t J, std::string sampler_type) {
        if(sampler_type == "efficient") {
            sampler.reset(new EfficientKRPSampler(J, cp_decomp.R, cp_decomp.U));
        }
        else if(sampler_type == "larsen_kolda") {
            sampler.reset(new LarsenKoldaSampler(J, cp_decomp.R, cp_decomp.U));
        }
        else if(sampler_type == "larsen_kolda_hybrid") {
            sampler.reset(new LarsenKoldaHybrid(J, cp_decomp.R, cp_decomp.U));
        }
        else if(sampler_type == "uniform") {
            sampler.reset(new UniformSampler(J, cp_decomp.R, cp_decomp.U));
        }

        samples.reset(new Buffer<uint64_t>({cp_decomp.N, J}));
        samples_transpose.reset(new Buffer<uint64_t>({J, cp_decomp.N}));
    }

    void execute_ds_als_update(uint32_t j, 
            bool renormalize,
            bool update_sampler
            ) {

        uint64_t Ij = cp_decomp.U[j].shape[0];
        uint64_t R = cp_decomp.R; 
        uint64_t J = sampler->J;
        uint64_t N = cp_decomp.N;

        Buffer<double> mttkrp_res({Ij, R});
        Buffer<double> pinv({R, R});

        //auto start = start_clock();
        sampler->KRPDrawSamples(j, *samples, nullptr);
        //double sampling_draw_time = stop_clock_get_elapsed(start);
        //cout << "Time to Draw Samples: " << sampling_draw_time << endl;

        // This step is unecessary, but we keep it in anyway
        #pragma omp parallel for collapse(2)
        for(uint64_t i = 0; i < J; i++) {
            for(uint64_t k = 0; k < N; k++) {
                (*samples_transpose)[i * N + k] = (uint32_t) (*samples)[k * J + i]; 
            }
        }

        // Use a sort to deduplicate the list of samples 
        Buffer<uint64_t*> sort_idxs({J});
        Buffer<uint64_t*> dedup_idxs({J});

        #pragma omp parallel for
        for(uint64_t i = 0; i < J; i++) {
            sort_idxs[i] = (*samples_transpose)(i * N);
        }

        std::sort(std::execution::par_unseq, 
            sort_idxs(), 
            sort_idxs(J),
            [j, N](uint64_t* a, uint64_t* b) {
                for(uint32_t i = 0; i < N; i++) {
                    if(i != j && a[i] != b[i]) {
                        return a[i] < b[i];
                    }
                }
                return false;  
            });

        uint64_t** end_range = 
            std::unique_copy(std::execution::par_unseq,
                sort_idxs(),
                sort_idxs(J),
                dedup_idxs(),
                [j, N](uint64_t* a, uint64_t* b) {
                    for(uint32_t i = 0; i < N; i++) {
                        if(i != j && a[i] != b[i]) {
                            return false;
                        }
                    }
                    return true; 
                });

        uint64_t num_unique = end_range - dedup_idxs();

        Buffer<uint64_t> samples_dedup({num_unique, N});
        Buffer<double> weights_dedup({num_unique});
        Buffer<double> h_dedup({num_unique, R});

        #pragma omp parallel for
        for(uint64_t i = 0; i < num_unique; i++) {
            uint64_t* buf = dedup_idxs[i];
            uint64_t offset = (buf - (*samples_transpose)()) / N;

            std::pair<uint64_t**, uint64_t**> bounds = std::equal_range(
                sort_idxs(),
                sort_idxs(J),
                buf, 
                [j, N](uint64_t* a, uint64_t* b) {
                    for(uint32_t i = 0; i < N; i++) {
                        if(i != j && a[i] != b[i]) {
                            return a[i] < b[i];
                        }
                    }
                    return false; 
                });

            uint64_t num_copies = bounds.second - bounds.first; 

            weights_dedup[i] = sampler->weights[offset] * num_copies; 

            for(uint64_t k = 0; k < N; k++) {
                samples_dedup[i * N + k] = buf[k];
            }
            for(uint64_t k = 0; k < R; k++) {
                h_dedup[i * R + k] = sampler->h[offset * R + k]; 
            }
        }

        #pragma omp parallel for collapse(2) 
        for(uint32_t i = 0; i < num_unique; i++) {
            for(uint32_t t = 0; t < R; t++) {
                h_dedup[i * R + t] *= sqrt(weights_dedup[i]); 
            }
        }


        //start = start_clock();
        std::fill(pinv(), pinv(R * R), 0.0);
        compute_pinv(h_dedup, pinv);  
        //double pinv_computation_time = stop_clock_get_elapsed(start);
        //cout << "PINV Computation Time: " << pinv_computation_time << endl;

        #pragma omp parallel for collapse(2)
        for(uint32_t i = 0; i < num_unique; i++) {
            for(uint32_t t = 0; t < R; t++) {
                h_dedup[i * R + t] *= sqrt(weights_dedup[i]); 
            }
        }

        //start = start_clock();
        ground_truth.execute_downsampled_mttkrp(
                samples_dedup,
                h_dedup,
                j,
                mttkrp_res 
                );
        //double sampling_time = stop_clock_get_elapsed(start);
        //cout << "DMTTKRP Time: " << sampling_time << endl;

        // Multiply gram matrix result by the pseudo-inverse

        //start = start_clock();
        cblas_dsymm(
            CblasRowMajor,
            CblasRight,
            CblasUpper,
            (uint32_t) Ij,
            (uint32_t) R,
            1.0,
            pinv(),
            R,
            mttkrp_res(),
            R,
            0.0,
            cp_decomp.U[j](),
            R
        );
        //double pinv_time = stop_clock_get_elapsed(start);
        //cout << "PINV Time: " << pinv_time << endl;

        if(renormalize) {
            //start = start_clock();
            cp_decomp.renormalize_columns(j);
            //double renormalization_time = stop_clock_get_elapsed(start);
            //cout << "Renormalization Time: " << renormalization_time << endl;
        }
        if(update_sampler) {
            //start = start_clock();
            sampler->update_sampler(j);
            //double update_time = stop_clock_get_elapsed(start);
            //cout << "Sampler Update Time: " << update_time << endl;
        }

    }

    void execute_exact_als_update(uint32_t j, 
            bool renormalize,
            bool update_sampler
            ) {

        uint64_t Ij = cp_decomp.U[j].shape[0];
        uint64_t R = cp_decomp.R; 

        Buffer<double> mttkrp_res({Ij, R});
        Buffer<double> gram({R, R});
        Buffer<double> gram_pinv({R, R});

        Buffer<double> ones({R});
        std::fill(ones(), ones(R), 1.0);

        ATB_chain_prod(
                cp_decomp.U,
                cp_decomp.U,
                ones, 
                ones,
                gram,
                j);        

        compute_pinv_square(gram, gram_pinv, R);
        ground_truth.execute_exact_mttkrp(cp_decomp.U, j, mttkrp_res);

        // Multiply gram matrix result by the pseudo-inverse
        cblas_dsymm(
            CblasRowMajor,
            CblasRight,
            CblasUpper,
            (uint32_t) Ij,
            (uint32_t) R,
            1.0,
            gram_pinv(),
            R,
            mttkrp_res(),
            R,
            0.0,
            cp_decomp.U[j](),
            R
        );

        cp_decomp.get_sigma(cp_decomp.sigma, j);

        if(renormalize) {
            cp_decomp.renormalize_columns(j);
        }
    }
};
