#pragma once

#include <iostream>
#include <string>
#include "common.h"
#include "cblas.h"
#include "lapacke.h"
#include "tensor.hpp"
#include "hashing.hpp"

using namespace std;

template<typename IDX_T, typename VAL_T>
class SortIdxLookup{
public:
  int dim;
  int mode_to_leave;

  Buffer<IDX_T> sort_idxs;

  SortIdxLookup(int dim, int mode_to_leave, IDX_T* idx_ptr, VAL_T* val_ptr, uint64_t nnz) 
  :
  sort_idxs(nnz)
  {

    this->dim = dim;
    this->mode_to_leave = mode_to_leave;


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

  }
};