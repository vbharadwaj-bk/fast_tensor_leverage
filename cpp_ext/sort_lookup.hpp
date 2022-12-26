#pragma once

#include <iostream>
#include <string>
#include <cstring>
#include "common.h"
#include "cblas.h"
#include "lapacke.h"
#include "tensor.hpp"
#include "hashing.hpp"

#include <execution>
#include <algorithm>
#include <numeric>
#include <bitset>

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

  SortIdxLookup(int N, int mode_to_leave, IDX_T* idx_ptr, VAL_T* val_ptr, uint64_t nnz) 
  :
  sort_idxs({nnz})
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

    auto lambda_fcn = [mode_to_leave, N](IDX_T* a, IDX_T* b) {
                for(int i = 0; i < N; i++) {
                    if(i != mode_to_leave && a[i] != b[i]) {
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

      IDX_T* start = *(bound.first);
      IDX_T* end = *(bound.second);

      bool found = true;
      for(int i = 0; i < N; i++) {
        if(i != mode_to_leave) {
          found = found && (buf[i] != start[i]);
        }
      }

      if(found) {
        for(IDX_T** i = bounds.first; i < bounds.second; i++) {
          found_count++;
          IDX_T* nonzero = *i;
          uint64_t diff = (nonzero - idx_ptr) / sizeof(IDX_T);
          double value = val_ptr[diff];
          uint64_t output_offset = ((*i)[mode_to_leave]) * R;

          for(uint64_t k = 0; k < R; k++) {
            #pragma omp atomic 
            output[output_offset + k] += input[input_offset + k] * value; 
          }
        }
      }
    }
  }
};