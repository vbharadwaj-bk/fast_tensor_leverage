#pragma once

#include <iostream>
#include <random>
#include "common.h"

using namespace std;

// Catchall file for test functions

void test_pinv_multithreading(py::array_t<double> U_py) {
    std::mt19937 gen(1);

    Buffer<double> U(U_py);

    uint64_t Ij = U.shape[0];
    uint64_t R = U.shape[1];
    Buffer<double> pinv({R, R});
    compute_pinv(U, pinv);

    Buffer<double> leverage({U.shape[0]}); 
    compute_DAGAT(U(), pinv(), leverage(), Ij, R);

    double total = 0.0;

    #pragma omp parallel for reduction(+: total)
    for(uint64_t i = 0; i < U.shape[0]; i++) {
        total += leverage[i];
    }

    std::discrete_distribution<uint64_t> dist(leverage(), leverage(Ij));
    cout << dist(gen) << endl;
    cout << total << endl;
}
