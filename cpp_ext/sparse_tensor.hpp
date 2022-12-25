#pragma once

#include <iostream>
#include <string>
#include "common.h"
#include "cblas.h"
#include "lapacke.h"
#include "tensor.hpp"
#include "hash_lookup.hpp"

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
    vector<HashIdxLookup<uint32_t, double>> lookups;

    uint64_t N, nnz;
    double normsq;

    SparseTensor(py::array_t<uint32_t> indices_py, py::array_t<double> values_py)
    :
    indices(indices_py),
    values(values_py)
    {
      nnz = indices.shape[0]; 
      N = indices.shape[1]; 

      for(uint64_t j = 0; j < N; j++) {
        lookups.emplace_back(N, j, 
            indices(), 
            values(), 
            nnz);
      }

      normsq = 0.0;
      #pragma omp parallel for reduction(+:normsq)
      for(uint64_t i = 0; i < nnz; i++) {
        normsq += values[i] * values[i];
      }

      // Sorting nonzeros would be good here... 
    }

    void execute_exact_mttkrp(vector<Buffer<double>> &U_L, uint64_t j, Buffer<double> &mttkrp_res) {
      uint64_t R = U_L[0].shape[1];
      std::fill(mttkrp_res(), mttkrp_res(mttkrp_res.shape[0] * mttkrp_res.shape[1]), 0.0);

      #pragma omp parallel
{
      Buffer<double> had_product({R});
      
      #pragma omp for
      for(uint64_t i = 0; i < nnz; i++) {
        std::fill(had_product(), had_product(R), values[i]);
        uint32_t* index = indices(i * N);

        for(uint64_t k = 0; k < N; k++) {
          if(k != j) {
            for(uint64_t u = 0; u < R; u++) {
              had_product[u] *= U_L[k][index[k] * R + u];
            }
          }
        }

        for(uint64_t u = 0; u < R; u++) {
          #pragma omp atomic
          mttkrp_res[index[j] * R + u] += had_product[u];
        }
      }
} 
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
      lookups[j].execute_spmm(samples_dcast, lhs, result);
    }

    double compute_residual_normsq(Buffer<double> &sigma, vector<Buffer<double>> &U) {
      // There is likely a faster way to do this computation... but okay, let's get it working first.

      uint64_t R = U[0].shape[1];
      Buffer<double> chain_had_prod({R, R});

      double residual_normsq = ATB_chain_prod_sum(U, U, sigma, sigma);

      #pragma omp parallel
{
      vector<double*> base_ptrs;
      for(uint64_t j = 0; j < N; j++) {
          base_ptrs.push_back(nullptr);
      }
      
      #pragma omp for reduction (+:residual_normsq)
      for(uint64_t i = 0; i < nnz; i++) {
          for(uint64_t j = 0; j < N; j++) {
              base_ptrs[j] = U[j](indices[i * N + j] * R); 
          } 
          double value = 0.0;
          for(uint64_t k = 0; k < R; k++) {
              double coord_buffer = sigma[k];
              for(uint64_t j = 0; j < N; j++) {
                  coord_buffer *= base_ptrs[j][k]; 
              }
              value += coord_buffer;
          }
          residual_normsq += values[i] * values[i] - 2 * value * values[i];
      }
}
        return residual_normsq;
    }
 
    double get_normsq() {
      return normsq; 
    }
};