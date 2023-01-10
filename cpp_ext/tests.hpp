#pragma once

#include <iostream>
#include "common.h"

using namespace std;

// Catchall file for test functions

void test_pinv_multithreading(py::array_t<double> U_py) {
    Buffer<double> U(U_py);

    uint64_t Ij = U[j].shape[0];
    uint64_t R = U[j].shape[1];
    Buffer<double> pinv({R, R});
    compute_pinv(U[j], pinv);

    Buffer<double> leverage({U[j].shape[0]}); 
    compute_DAGAT(U[j](), pinv(), leverage(), Ij, R);

    double total = 0.0;

    #pragma omp parallel for reduction(+: total)
    for(uint64_t i = 0; i < U[j].shape[0]; i++) {
        total += leverage[i];
    }
    cout << total << endl;
}
