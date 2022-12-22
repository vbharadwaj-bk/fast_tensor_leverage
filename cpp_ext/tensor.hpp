#pragma once

#include <iostream>
#include <string>
#include "common.h"
#include "cblas.h"
#include "lapacke.h"

using namespace std;

class __attribute__((visibility("hidden"))) Tensor {
public:
    virtual double compute_residual_normsq(Buffer<double> &sigma, vector<Buffer<double>> &U) {
        return -1.0;
    }

    virtual double get_normsq() {
        return -1.0;
    };

    virtual void execute_exact_mttkrp(vector<Buffer<double>> &U_L, uint64_t j, Buffer<double> &mttkrp_res) {

    }

    virtual void 
    execute_downsampled_mttkrp(
            Buffer<uint64_t> &samples_transpose, 
            Buffer<double> &lhs,
            uint64_t j,
            Buffer<double> &result
            ) = 0;

    // Python bindings

    double compute_residual_normsq_py(py::array_t<double> sigma_py, py::list U_py) {
      Buffer<double> sigma(sigma_py);
      NPBufferList<double> U(U_py);
      return compute_residual_normsq(sigma, U.buffers);
    }

    void execute_downsampled_mttkrp_py(
            py::array_t<uint64_t> &samples_py, 
            py::array_t<double> &lhs_py,
            uint64_t j,
            py::array_t<double> &result_py
            ) {
        
        Buffer<uint64_t> samples(samples_py);
        Buffer<double> lhs(lhs_py);
        Buffer<double> result(result_py);

        execute_downsampled_mttkrp(
                samples, 
                lhs,
                j,
                result 
                );
    }
};
