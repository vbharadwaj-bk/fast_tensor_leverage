#pragma once

#include <iostream>
#include <string>
#include <memory>
#include <string>
#include <random>
#include "common.h"
#include "cblas.h"
#include "lapacke.h"
#include "tensor.hpp"
#include "idx_lookup.hpp"
#include "sort_lookup.hpp"
#include "index_filter.hpp"
#include "low_rank_tensor.hpp"
#include "random_util.hpp"

using namespace std;
namespace py = pybind11;

/*
* By default, we assume this is a tensor with uint32_t index variables
* and double-precision values. 
*/
class __attribute__((visibility("hidden"))) SparseTensor : public Tensor {
public:
    Buffer<uint32_t> indices;
    Buffer<double> values;
    vector<unique_ptr<IdxLookup<uint32_t, double>>> lookups;

    uint64_t N, nnz;
    double normsq;

    // Related to randomized accuracy estimation
    unique_ptr<IndexFilter> idx_filter;
    Multistream_RNG rng;

    SparseTensor(py::array_t<uint32_t> indices_py, 
        py::array_t<double> values_py,
        std::string method)
    :
    indices(indices_py),
    values(values_py)
    {
      nnz = indices.shape[0]; 
      N = indices.shape[1];
      idx_filter.reset(nullptr);

      for(uint64_t j = 0; j < N; j++) {
        if(method == "sort") {
          lookups.emplace_back(new SortIdxLookup(N, j, 
              indices(), 
              values(), 
              nnz));
        }
        else {
          cout << "Unknown lookup method passed, terminating..." << endl;
          exit(1);
        }
      }

      normsq = 0.0;
      #pragma omp parallel for reduction(+:normsq)
      for(uint64_t i = 0; i < nnz; i++) {
        normsq += values[i] * values[i];
      }

      // Sorting nonzeros would be good here... 
    }

    void execute_exact_mttkrp(vector<Buffer<double>> &U_L, uint64_t j, Buffer<double> &mttkrp_res) { 
      std::fill(mttkrp_res(), mttkrp_res(mttkrp_res.shape[0] * mttkrp_res.shape[1]), 0.0);
      lookups[j]->execute_exact_mttkrp(U_L, mttkrp_res);
    }

    void execute_rrf(LowRankTensor &lr) {
      vector<Buffer<double>> &U = lr.U;
      for(uint64_t j = 0; j < U.size(); j++) {
        uint64_t rows = U[j].shape[0];
        uint64_t cols = U[j].shape[1];
        std::fill(U[j](), U[j](rows * cols), 0.0);
        lookups[j]->execute_rrf(U[j]);
      } 
      lr.renormalize_columns(-1);
    }

    void execute_downsampled_mttkrp(
            Buffer<uint64_t> &samples_transpose, 
            Buffer<double> &lhs,
            uint64_t j,
            Buffer<double> &result) {
      uint64_t J = samples_transpose.shape[0];
      uint64_t N = samples_transpose.shape[1];
      uint64_t Ij = result.shape[0];
      uint64_t R = result.shape[1];
      Buffer<uint32_t> samples_dcast({J, N});

      #pragma omp parallel for 
      for(uint64_t i = 0; i < J * N; i++) {
          samples_dcast[i] = (uint32_t) samples_transpose[i]; 
      } 

      // It is probably a good idea to sort and reweight samples here... 

      std::fill(result(), result(Ij * R), 0.0);
      lookups[j]->execute_spmm(samples_dcast, lhs, result);
    }

    double compute_residual_normsq(Buffer<double> &sigma, vector<Buffer<double>> &U) {
        return lookups[0]->compute_residual_normsq(sigma, U);
    }

    void initialize_randomized_accuracy_estimation(double fp_tol) {
      idx_filter.reset(new IndexFilter(indices, fp_tol));
    }

    double compute_residual_normsq_estimated(LowRankTensor &lr) {
      uint64_t nz_sample_count = nnz;
      uint64_t zero_sample_count = nnz;

      if(idx_filter.get() == nullptr) {
        throw std::runtime_error("Randomized accuracy estimation not initialized.");
      }

      // Currently implemented as a 50-50 split between
      // zero and nonzero values 

      Buffer<uint32_t> nonzero_samples({nz_sample_count, N});
      Buffer<double> nonzero_evals({nz_sample_count}); 
      Buffer<double> nonzero_values({nz_sample_count}); 

      std::uniform_int_distribution<uint32_t> nz_dist(0, nnz -1); 

      #pragma omp parallel
      {
        int thread_num = omp_get_thread_num();

        #pragma omp for
        for(uint64_t i = 0; i < nz_sample_count; i++) {
          uint32_t idx = nz_dist(rng.par_gen[thread_num]);
          for(uint64_t j = 0; j < N; j++) {
            nonzero_samples[i * N + j] = indices[idx * N + j];
            nonzero_values[i] = values[idx];
          }
        }
      }

      lr.evaluate_indices(nonzero_samples, nonzero_evals);
      //lr.evaluate_indices(indices, nonzero_evals);
      double nonzero_loss = 0.0;
      double zero_loss = 0.0;

      #pragma omp parallel for reduction(+:nonzero_loss)
      for(uint64_t i = 0; i < nz_sample_count; i++) {
        double diff = nonzero_values[i] - nonzero_evals[i];
        nonzero_loss += diff * diff; 
      }

      vector<std::uniform_int_distribution<uint32_t>> zero_dists;
      double dense_entries = 1.0;
      for(uint64_t j = 0; j < N; j++) {
        uint32_t max_idx = (uint32_t) lr.U[j].shape[0] - 1;
        zero_dists.emplace_back(0, max_idx); 
        dense_entries *= max_idx;
      }

      Buffer<uint32_t> zero_samples({zero_sample_count, N});
      Buffer<double> zero_evals({zero_sample_count});

      #pragma omp parallel
      {
        int thread_num = omp_get_thread_num();

        #pragma omp for
        for(uint64_t i = 0; i < zero_sample_count; i++) {
          for(uint64_t j = 0; j < N; j++) {
            zero_samples[i * N + j] = zero_dists[j](rng.par_gen[thread_num]);
          }
        }
      }

      lr.evaluate_indices(zero_samples, zero_evals); 

      vector<uint64_t> collisions = idx_filter->check_idxs(zero_samples);
      for(uint64_t i = 0; i < collisions.size(); i++) {
        zero_evals[collisions[i]] = 0.0;
      } 

      #pragma omp parallel for reduction(+:zero_loss)
      for(uint64_t i = 0; i < zero_sample_count; i++) {
        zero_loss += zero_evals[i] * zero_evals[i]; 
      }

      uint64_t true_zero_count = zero_sample_count - collisions.size();
      nonzero_loss *= (double) nnz / nz_sample_count; 
      zero_loss *= (dense_entries - nnz) / true_zero_count; 

      return nonzero_loss + zero_loss;
    }

    double get_normsq() {
      return normsq; 
    }
};