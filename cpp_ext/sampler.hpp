#pragma once

#include <iostream>
#include <vector>
#include <cassert>
#include <random>
#include "common.h"
#include "partition_tree.hpp"
#include "omp.h"
#include "cblas.h"
#include "lapacke.h"

using namespace std;

class __attribute__((visibility("hidden"))) Sampler {
public:
    uint64_t N, J, R, R2;
    vector<Buffer<double>> &U;
    Buffer<double> h;

    Sampler(uint64_t J, 
            uint64_t R, 
            vector<Buffer<double>> &U_matrices) : 
        U(U_matrices),
        h({J, R}) {
        // Empty 
    }
};