#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include "common.h"
#include "partition_tree.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;

class __attribute__((visibility("hidden"))) TTSampler {
    /*
    * This class is closely integrated with a version in Python. 
    */
public:
    vector<unique_ptr<Buffer<double>>> matricizations;
    vector<int64_t> dimensions;
    vector<unique_ptr<PartitionTree>> tree_samplers;
    uint64_t N, J, R_max;
    ScratchBuffer scratch;

    // Related to random number generation 
    int thread_count;
    std::random_device rd; 
    std::mt19937 gen;
    vector<std::mt19937> par_gen; 
    std::uniform_real_distribution<> dis;

    TTSampler(uint64_t N, uint64_t J, uint64_t R_max, 
            py::array_t<uint64_t> dimensions_py) 
    :
    N(N),
    J(J),
    R_max(R_max),
    scratch(R_max, J, R_max),
    rd(),
    gen(rd()) {
        Buffer<uint64_t> dim_wrapper(dimensions_py);
        for(uint64_t i = 0; i < N; i++) {
            matricizations.emplace_back();
            tree_samplers.emplace_back();
            dimensions.push_back(dim_wrapper[i]);
        }

        // Set up independent random streams for different threads.
        // As written, might be more complicated than it needs to be. 
        #pragma omp parallel
        {
            #pragma omp single 
            {
                thread_count = omp_get_num_threads();
            }
        }

        vector<uint32_t> biased_seeds(thread_count, 0);
        vector<uint32_t> seeds(thread_count, 0);

        for(int i = 0; i < thread_count; i++) {
            biased_seeds[i] = rd();
        }
        std::seed_seq seq(biased_seeds.begin(), biased_seeds.end());
        seq.generate(seeds.begin(), seeds.end());

        for(int i = 0; i < thread_count; i++) {
            par_gen.emplace_back(seeds[i]);
        }
    }

    void fill_buffer_random_draws(double* data, uint64_t len) {
        #pragma omp parallel
{
        int thread_id = omp_get_thread_num();
        auto &local_gen = par_gen[thread_id];

        #pragma omp for
        for(uint64_t i = 0; i < len; i++) {
            data[i] = dis(local_gen);
        }
}
    }

    void update_matricization(py::array_t<double> &matricization, uint64_t i) {
        matricizations[i].reset(new Buffer<double>(matricization));
        tree_samplers[i].reset(
            new PartitionTree(
                matricizations[i]->shape[0],
                (uint32_t) matricizations[i]->shape[1],
                (uint32_t) J,
                (uint32_t) matricizations[i]->shape[1],
                scratch
        ));
        tree_samplers[i]->build_tree(*(matricizations[i]));
    }

    void draw_samples(uint64_t i, 
            py::array_t<double> h_old_py, 
            py::array_t<uint64_t> samples_py, 
            py::array_t<double> h_new_py) {

        Buffer<double> h_old(h_old_py);
        Buffer<double> h_new(h_new_py);
        Buffer<uint64_t> samples(samples_py);
        Buffer<uint64_t> row_buffer({J}, samples(i, 0));
        Buffer<double> dummy({h_old.shape[0], h_old.shape[1]}); 

        fill_buffer_random_draws(scratch.random_draws(), J); 

        tree_samplers[i]->PTSample_internal(
                *(matricizations[i]), 
                dummy,
                h_old,
                row_buffer,
                scratch.random_draws);

/*        
        #pragma omp parallel
{
        #pragma omp for
        for(uint64_t i = 0; i < J; i++) {
            row_buffer[i] %= previous_dim;
        }

}
*/
    }
};

