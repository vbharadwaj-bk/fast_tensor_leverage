#pragma once

#include <iostream>
#include <string>
#include "common.h"
#include "cblas.h"
#include "lapacke.h"
#include "tensor.hpp"
#include "hashing.hpp"
#include "sparsehash/dense_hash_map"

using namespace std;
namespace py = pybind11;

template<typename IDX_T>
struct TupleHasher 
{
public:
  int mode_to_leave;
  int dim;

  TupleHasher(int mode_to_leave, int dim) {
    this->mode_to_leave = mode_to_leave;
    this->dim = dim;
  }

  uint32_t operator()(IDX_T* const &ptr) const
  {
    if(ptr != nullptr) {
      //uint32_t pre_hash = MurmurHash3_x86_32(ptr, mode_to_leave * sizeof(IDX_T), 0x9747b28c);

      uint32_t pre_hash = MurmurHash3_x86_32(ptr, sizeof(IDX_T) * mode_to_leave, 0x9747b28c);

      uint32_t post_hash = MurmurHash3_x86_32(ptr + mode_to_leave + 1, 
          sizeof(IDX_T) * (dim - mode_to_leave - 1), 0x9747b28c);

      return pre_hash + post_hash;
    }
    else {
      return 0;
    }
  }
};

template<typename IDX_T>
struct TupleEqual 
{
public:
  int dim, mode_to_leave;
  TupleEqual(int mode_to_leave, int dim) {
    this->dim = dim;
    this->mode_to_leave = mode_to_leave;
  }

  bool operator()(IDX_T* const &s1, uint32_t* const &s2) const
  {
      if(s1 == nullptr || s2 == nullptr) {
        return s1 == s2;
      }
      else {
        auto res = (! memcmp(s1, s2, sizeof(IDX_T) * mode_to_leave)) 
            && (! memcmp(
            s1 + mode_to_leave + 1,
            s2 + mode_to_leave + 1,
            sizeof(IDX_T) * (dim - mode_to_leave - 1)
        ));

        return res;
      }
  }
};

template<typename IDX_T, typename VAL_T>
class HashIdxLookup {
public:
  int dim;
  int mode_to_leave;

  unique_ptr<google::dense_hash_map< IDX_T*, 
                          uint64_t,
                          TupleHasher<IDX_T>,
                          TupleEqual<IDX_T>>> lookup_table;

  vector<vector<pair<IDX_T, VAL_T>>> storage;
  uint64_t num_buckets;

  TupleHasher<IDX_T> hasher;
  TupleEqual<IDX_T> equality_checker;

  vector<IDX_T> idx_copies;

  HashIdxLookup(int dim, int mode_to_leave, IDX_T* idx_ptr, VAL_T* val_ptr, uint64_t nnz) :
    hasher(mode_to_leave, dim), equality_checker(mode_to_leave, dim) {

    this->dim = dim;
    this->num_buckets = 0;

    lookup_table.reset(new google::dense_hash_map< IDX_T*, 
                          uint64_t,
                          TupleHasher<IDX_T>,
                          TupleEqual<IDX_T>>(nnz, hasher, equality_checker));

    lookup_table->set_empty_key(nullptr);

    for(uint64_t i = 0; i < nnz; i++) {
      IDX_T* idx = idx_ptr + i * dim;
      VAL_T val = val_ptr[i];

      auto res = lookup_table->find(idx);

      uint64_t bucket;
      if(res == lookup_table->end()) {
        bucket = num_buckets++;
        storage.emplace_back();
        lookup_table->insert(make_pair(idx, bucket));
      }
      else {
        bucket = res->second; 
      }
      storage[bucket].emplace_back(make_pair(idx[mode_to_leave], val));
    }
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
    uint64_t N = indices.shape[1];
    uint64_t R = output.shape[1];

    uint64_t found_count = 0;

    #pragma omp parallel for reduction(+:found_count)
    for(uint64_t j = 0; j < J; j++) {
      uint64_t input_offset = j * R;
      IDX_T* buf = indices(j * N); 
      auto pair_loc = lookup_table->find(buf);
      if(pair_loc != lookup_table->end()) {
        vector<pair<IDX_T, VAL_T>> &pairs = storage[pair_loc->second];
        found_count += pairs.size();
        for(auto it = pairs.begin(); it != pairs.end(); it++) {
          uint64_t output_offset = it->first * R;
          double value = it->second;
          for(uint64_t k = 0; k < R; k++) {
            #pragma omp atomic
            output[output_offset + k] += input[input_offset + k] * value; 
          }
        }  
      }
    }
  }
};

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