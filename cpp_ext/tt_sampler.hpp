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
    vector<unique_ptr<PartitionTree>> tree_samplers;

    // 0 is right-orthogonal
    // 1 is left-orthogonal
    // -1 is undefined
    vector<int> orthogonality;

    vector<int64_t> dimensions;
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
            orthogonality.push_back(-1);
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

    void update_matricization(
        py::array_t<double> &matricization, 
        uint64_t i,
        int core_orthogonality,
        bool build_tree 
        ) {

        orthogonality[i] = core_orthogonality;
        matricizations[i].reset(new Buffer<double>(matricization));

        if(build_tree) {
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
        else {
            tree_samplers[i].reset(nullptr);
        }
    }

    void sample(int64_t exclude,
            uint64_t J,
            py::array_t<uint64_t> samples_py,
            int orth 
            ) {
        Buffer<uint64_t> samples(samples_py);

        unique_ptr<Buffer<double>> h_old;
        unique_ptr<Buffer<double>> h_new;

        int64_t start, stop, offset;
        if(orth == 1) {
            start = exclude - 1;
            stop = -1;
            offset = -1;
        }
        else if(orth == 0) {
            start = exclude + 1;
            stop = N;
            offset = 1;
        }
        else {
            throw std::invalid_argument("orthogonality must be either 0 or 1");
        }

        for(int64_t i = start; i != stop; i += offset) { 
            Buffer<double> &mat = *(matricizations[i]); 
            uint64_t left_rank = mat.shape[0] / dimensions[i];
            uint64_t right_rank = mat.shape[1];

            if(orthogonality[i] != orth) {
                throw std::invalid_argument("orthogonality of matricization does not match");
            }

            h_new.reset(new Buffer<double>({J, left_rank}));

            int64_t sample_start_idx;
            if(orth == 1) {
                sample_start_idx = i;
            }
            else {
                sample_start_idx = i - start;
            }
            Buffer<uint64_t> row_buffer({J}, samples(sample_start_idx, 0));

            if(i == start) {
                Buffer<double> squared_mat({mat.shape[1], mat.shape[0]});
                vector<std::discrete_distribution<int>> distributions;

                std::uniform_int_distribution<int> col_selector(0, right_rank-1);

                #pragma omp parallel for collapse(2)
                for(uint64_t j = 0; j < mat.shape[0]; j++) {
                    for(uint64_t k = 0; k < mat.shape[1]; k++) {
                        squared_mat[k * mat.shape[0] + j] = mat[j * mat.shape[1] + k] * mat[j * mat.shape[1] + k]; 
                    }
                }

                for(uint64_t k = 0; k < squared_mat.shape[0] * squared_mat.shape[1]; k+= squared_mat.shape[1]) {
                    distributions.emplace_back(squared_mat(k), squared_mat(k + squared_mat.shape[1]));
                }

                Buffer<double> &h_new_buf = *h_new;

                #pragma omp parallel
{
                int thread_id = omp_get_thread_num();

                #pragma omp for 
                for(uint64_t j = 0; j < J; j++) {
                    int random_col = col_selector(par_gen[thread_id]);
                    uint64_t row_idx = distributions[random_col](par_gen[thread_id]);
                    row_idx /= right_rank;
                    row_buffer[j] = row_idx;

                    uint64_t offset = row_idx * left_rank * right_rank;
                    for(uint64_t k = 0; k < left_rank; k++) {
                        h_new_buf[j * left_rank + k] = mat[offset + k * right_rank + random_col];
                    } 
                }
}
            }
            else {
                draw_samples_internal(i, *h_old, row_buffer, *h_new);
            }

            h_old = std::move(h_new);
        }
    }

    void draw_samples(uint64_t i, 
            py::array_t<double> h_old_py, 
            py::array_t<uint64_t> samples_py, 
            py::array_t<double> h_new_py) {

        Buffer<double> h_old(h_old_py);
        Buffer<double> h_new(h_new_py);
        Buffer<uint64_t> samples(samples_py);

        // The row buffer is off, but this function
        // is deprecated anyway.
        Buffer<uint64_t> row_buffer({J}, samples(i, 0));
        draw_samples_internal(i, h_old, row_buffer, h_new);
    }

    void draw_samples_internal(uint64_t i, 
            Buffer<double> &h_old, 
            Buffer<uint64_t> &row_buffer, 
            Buffer<double> &h_new) {

        Buffer<double> dummy({h_old.shape[0], h_old.shape[1]}); 

        fill_buffer_random_draws(scratch.random_draws(), J); 

        tree_samplers[i]->PTSample_internal(
                *(matricizations[i]), 
                dummy,
                h_old,
                row_buffer,
                scratch.random_draws);

        uint64_t left_rank = matricizations[i]->shape[0] / dimensions[i];
        uint64_t right_rank = matricizations[i]->shape[1];
        uint64_t mat_size = left_rank * right_rank;

        #pragma omp parallel
{
        #pragma omp for
        for(uint64_t j = 0; j < J; j++) {
            row_buffer[j] /= left_rank;
            double* mat_ptr = (*(matricizations[i]))(row_buffer[j] * mat_size);

            // Matrix-vector multiply via cblas_dgemv
            cblas_dgemv(CblasRowMajor, CblasNoTrans, 
                    left_rank, right_rank, 1.0, 
                    mat_ptr, right_rank, 
                    h_old(j * right_rank), 1, 
                    0.0, 
                    h_new(j * left_rank), 1);
        }
}
    }

    void evaluate_indices_partial(
            py::array_t<uint64_t> indices_py,
            uint32_t j, 
            int direction,
            py::array_t<double> result_py
            ) {

        Buffer<uint64_t> indices(indices_py);
        Buffer<double> result(result_py);

        uint64_t J = indices.shape[0];

        int start, stop, offset;
        if(direction == 1) {
            start = 0;
            stop = j;
            offset = 1;
        }
        else if(direction == 0){
            start = N - 1;
            stop = j;
            offset = -1;
        }
        else {
            throw std::invalid_argument("direction must be either 0 or 1");
        }

        Buffer<double> h_old;
        Buffer<double> h_new;

        h_old.reset_to_shape({J, 1});
        std::fill(h_old(), h_old(J), 1.0);

        #pragma omp parallel
{
        for(int64_t i = start; i != stop; i += offset) {
            uint64_t left_rank = matricizations[i]->shape[0] / dimensions[i];
            uint64_t right_rank = matricizations[i]->shape[1];
            uint64_t mat_size = left_rank * right_rank;

            CBLAS_TRANSPOSE transposition;

            uint64_t h_old_col_count = h_old.shape[1];
            uint64_t h_new_col_count;
            if(direction == orthogonality[i]) {
                transposition = CblasTrans;
                h_new_col_count = right_rank; 
            }
            else {
                transposition = CblasNoTrans;
                h_new_col_count = left_rank; 
            } 
            
            #pragma omp single
{ 
            h_new.reset_to_shape({J, h_new_col_count});
}

            #pragma omp for
            for(uint64_t j = 0; j < J; j++) {
                uint64_t core_idx = *(indices(j, i));
                double* mat_ptr = (*(matricizations[i]))(core_idx * mat_size);

                cblas_dgemv(CblasRowMajor, transposition, 
                        left_rank, right_rank, 1.0, 
                        mat_ptr, right_rank, 
                        h_old(j * h_old_col_count), 1, 
                        0.0, 
                        h_new(j * h_new_col_count), 1);
            }
            #pragma omp single
{
            h_old.steal_resources(h_new); 
}
        }
}
        if(result.shape[0] != J || result.shape[1] != h_old.shape[1]) {
            cout << "Provided Buffer Shape: " << result.shape[0] << " " << result.shape[1] << endl;
            cout << "Internal Buffer Shape: " << h_old.shape[0] << " " << h_old.shape[1] << endl;
            throw std::invalid_argument("Incorrect result shape!");
        }

        std::copy(h_old(), 
                h_old(h_old.shape[0] * h_old.shape[1]), 
                result());
    }

};

