#pragma once

#include <iostream>
#include <string>
#include "common.h"
#include "cblas.h"
#include "lapacke.h"
#include "idx_lookup.hpp"
#include "tensor.hpp"
#include "hashing.hpp"
#include "sparsehash/dense_hash_map"

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
class HashIdxLookup : public IdxLookup<IDX_T, VAL_T> {
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

    cout << "Found nonzeros: " << found_count << endl;
  }
};
