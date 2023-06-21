#pragma once

#include <iostream>
#include <string>
#include <cstring>
#include <random>
#include <execution>
#include <algorithm>
#include <numeric>

#include "common.h"
#include "cblas.h"
#include "lapacke.h"
#include "tensor.hpp"
#include "hashing.hpp"
#include "random_util.hpp"

using namespace std;

template<typename IDX_T, typename VAL_T>
class __attribute__((visibility("hidden"))) SortIdxLookup : public IdxLookup<IDX_T, VAL_T> {
public:
  int N;
  int mode_to_leave;

  uint64_t nnz;
  IDX_T* idx_ptr;
  VAL_T* val_ptr;

  Buffer<IDX_T*> sort_idxs;
  Multistream_RNG ms_rng;

  SortIdxLookup(int N, int mode_to_leave, IDX_T* idx_ptr, VAL_T* val_ptr, uint64_t nnz) 
  :
  sort_idxs({nnz}),
  ms_rng()
  {
    this->N = N;
    this->mode_to_leave = mode_to_leave;
    this->nnz = nnz;
    this->idx_ptr = idx_ptr;
    this->val_ptr = val_ptr;

    #pragma omp parallel for 
    for(uint64_t i = 0; i < nnz; i++) {
        sort_idxs[i] = idx_ptr + (i * N);
    }

    std::sort(std::execution::par_unseq, 
    //std::sort(
        sort_idxs(), 
        sort_idxs(nnz),
        [mode_to_leave, N](IDX_T* a, IDX_T* b) {
            for(int i = 0; i < N; i++) {
                if(i != mode_to_leave && a[i] != b[i]) {
                    return a[i] < b[i];
                }
            }
            return false;  
        });
  }

  /*
  * Executes an SpMM with the tensor matricization tracked by this
  * lookup table.
  *
  * This function assumes that the output buffer has already been
  * initialized to zero. 
  */
  void execute_spmm(
      Buffer<IDX_T> &indices, 
      Buffer<double> &input,
      Buffer<double> &output
      ) {

    uint64_t J = indices.shape[0];
    uint64_t R = output.shape[1];

    int mode = this->mode_to_leave;
    int Nval = this->N;
    auto lambda_fcn = [mode, Nval](IDX_T* a, IDX_T* b) {
                for(int i = 0; i < Nval; i++) {
                    if(i != mode && a[i] != b[i]) {
                        return a[i] < b[i];
                    }
                }
                return false;  
            };

    uint64_t found_count = 0;

    #pragma omp parallel for reduction(+:found_count)
    for(uint64_t j = 0; j < J; j++) {
      uint64_t input_offset = j * R;
      IDX_T* buf = indices(j * N);

      std::pair<IDX_T**, IDX_T**> bounds = 
        std::equal_range(
            sort_idxs(), 
            sort_idxs(nnz),
            buf,
            lambda_fcn);

      bool found = false;
      if(bounds.first != sort_idxs(nnz)) {
        found = true;
        IDX_T* start = *(bounds.first);

        for(int i = 0; i < N; i++) {
            if(i != mode_to_leave && buf[i] != start[i]) {
                found = false;
            }
        }
      }

      if(found) {
        for(IDX_T** i = bounds.first; i < bounds.second; i++) {
          found_count++;
          IDX_T* nonzero = *i;
          uint64_t diff = (uint64_t) (nonzero - idx_ptr) / N;
          double value = val_ptr[diff];
          uint64_t output_offset = (nonzero[mode_to_leave]) * R;

          for(uint64_t k = 0; k < R; k++) {
            #pragma omp atomic 
            output[output_offset + k] += input[input_offset + k] * value; 
          }
        }
      }
    }
  }

  double compute_residual_normsq(
      Buffer<double> &sigma, 
      vector<Buffer<double>> &U) {

      uint64_t R = U[0].shape[1];
      uint64_t j = mode_to_leave;

      double residual_normsq = 0.0;
      double value_sum = 0.0;
      #pragma omp parallel reduction(+: residual_normsq, value_sum)
{
      int thread_num = omp_get_thread_num();
      int total_threads = omp_get_num_threads();      

      uint64_t chunksize = (nnz + total_threads - 1) / total_threads;
      uint64_t lower_bound = min(chunksize * thread_num, nnz);
      uint64_t upper_bound = min(chunksize * (thread_num + 1), nnz);

      Buffer<double> partial_prod({R});

      for(uint64_t i = lower_bound; i < upper_bound; i++) {
        IDX_T* index = sort_idxs[i];

        uint64_t offset = (index - idx_ptr) / N; 
        bool recompute_partial_prod = false;
        if(i == lower_bound) {
          recompute_partial_prod = true;
        }
        else {
          IDX_T* prev_index = sort_idxs[i-1];
          for(uint64_t k = 0; k < (uint64_t) N; k++) {
            if((k != j) && (index[k] != prev_index[k])) {
              recompute_partial_prod = true;
            }
          }
        }

        if(recompute_partial_prod) {
          std::copy(sigma(), sigma(R), partial_prod()); 

          for(uint64_t k = 0; k < (uint64_t) N; k++) {
            if(k != j) {
              for(uint64_t u = 0; u < R; u++) {
                partial_prod[u] *= U[k][index[k] * R + u];
              }
            }
          }
        }

        double tensor_value = val_ptr[offset];

        double value = 0.0; 
        for(uint64_t u = 0; u < R; u++) {        
          value += partial_prod[u] * U[j][index[j] * R + u]; 
        }
        value_sum += value;

        residual_normsq += tensor_value * tensor_value - 2 * value * tensor_value;
      }
}

      residual_normsq += ATB_chain_prod_sum(U, U, sigma, sigma);
      return residual_normsq;
  }

  void execute_exact_mttkrp(
      vector<Buffer<double>> &U, 
      Buffer<double> &mttkrp_res) {

      uint64_t R = U[0].shape[1];
      uint64_t j = mode_to_leave;

      #pragma omp parallel 
{
      int thread_num = omp_get_thread_num();
      int total_threads = omp_get_num_threads();      

      uint64_t chunksize = (nnz + total_threads - 1) / total_threads;
      uint64_t lower_bound = min(chunksize * thread_num, nnz);
      uint64_t upper_bound = min(chunksize * (thread_num + 1), nnz);

      Buffer<double> partial_prod({R});

      for(uint64_t i = lower_bound; i < upper_bound; i++) {
        IDX_T* index = sort_idxs[i];

        uint64_t offset = (index - idx_ptr) / N; 
        bool recompute_partial_prod = false;
        if(i == lower_bound) {
          recompute_partial_prod = true;
        }
        else {
          IDX_T* prev_index = sort_idxs[i-1];
          for(uint64_t k = 0; k < (uint64_t) N; k++) {
            if((k != j) && (index[k] != prev_index[k])) {
              recompute_partial_prod = true;
            }
          }
        }

        if(recompute_partial_prod) {
          std::fill(partial_prod(), partial_prod(R), 1.0); 

          for(uint64_t k = 0; k < (uint64_t) N; k++) {
            if(k != j) {
              for(uint64_t u = 0; u < R; u++) {
                partial_prod[u] *= U[k][index[k] * R + u];
              }
            }
          }
        }

        double tensor_value = val_ptr[offset];

        for(uint64_t u = 0; u < R; u++) {
          #pragma omp atomic 
          mttkrp_res[index[j] * R + u] += partial_prod[u] * tensor_value;
        }
      }
}
  }

  /*
  * Executes a full randomized range finder algorithm. This is very
  * similar to the exact MTTKRP update algorithm and the algorithm to
  * compute the exact norm^2 of residual with a low rank tensor. They should
  * probably be combined.
  */
  void execute_rrf(
      Buffer<double> &mttkrp_res) {

      uint64_t R = mttkrp_res.shape[1];
      uint64_t j = mode_to_leave;

      uint64_t nz_processed = 0;

      #pragma omp parallel reduction(+: nz_processed) 
{
      int thread_num = omp_get_thread_num();
      int total_threads = omp_get_num_threads();      

      uint64_t chunksize = (nnz + total_threads - 1) / total_threads;
      uint64_t lower_bound = min(chunksize * thread_num, nnz);
      uint64_t upper_bound = min(chunksize * (thread_num + 1), nnz);

      // Update the bounds to fall along fiber boundaries
      bool continue_loop = true; 
      while(continue_loop) {
        if(lower_bound == 0 || lower_bound >= nnz) {
          continue_loop = false;
        }
        else {
          IDX_T* index = sort_idxs[lower_bound];
          IDX_T* prev_index = sort_idxs[lower_bound-1];
          for(uint64_t k = 0; k < (uint64_t) N; k++) {
            if((k != j) && (index[k] != prev_index[k])) {
              continue_loop = false;
            }
          }
        }
        if(continue_loop) {
          lower_bound++;
        }
      }

      continue_loop = true; 
      while(continue_loop) {
        if(upper_bound >= nnz) {
          continue_loop = false;
        }
        else {
          IDX_T* index = sort_idxs[upper_bound - 1];
          IDX_T* next_index = sort_idxs[upper_bound];
          for(uint64_t k = 0; k < (uint64_t) N; k++) {
            if((k != j) && (index[k] != next_index[k])) {
              continue_loop = false;
            }
          }
        }
        if(continue_loop) {
          upper_bound++;
        }
      }

      Buffer<double> partial_prod({R});
      std::normal_distribution<double> std_normal;

      for(uint64_t i = lower_bound; i < upper_bound; i++) {
        IDX_T* index = sort_idxs[i];

        uint64_t offset = (index - idx_ptr) / N; 
        bool recompute_partial_prod = false;
        if(i == lower_bound) {
          recompute_partial_prod = true;
        }
        else {
          IDX_T* prev_index = sort_idxs[i-1];
          for(uint64_t k = 0; k < (uint64_t) N; k++) {
            if((k != j) && (index[k] != prev_index[k])) {
              recompute_partial_prod = true;
            }
          }
        }

        if(recompute_partial_prod) {
          for(uint64_t u = 0; u < R; u++) {
            partial_prod[u] = std_normal(ms_rng.par_gen[thread_num]); 
          }
        }

        double tensor_value = val_ptr[offset];

        for(uint64_t u = 0; u < R; u++) {
          #pragma omp atomic 
          mttkrp_res[index[j] * R + u] += partial_prod[u] * tensor_value;
        }
        nz_processed++;
      }
}
    cout << "Nonzeros processed: " << nz_processed << endl;
  }
};