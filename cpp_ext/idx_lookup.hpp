#pragma once

#include "common.h"

using namespace std;

template<typename IDX_T, typename VAL_T>
class IdxLookup {
public:
  virtual void execute_spmm(
      Buffer<IDX_T> &indices, 
      Buffer<double> &input,
      Buffer<double> &output
      ) = 0;
};