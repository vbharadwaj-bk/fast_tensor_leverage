#pragma once

#include <cmath>
#include "hashing.hpp"
#include "common.h"

/*
* Generates Gaussian noise that is reproducible at each
* evaluation index. 
*/
void reproducible_noise(py::array_t<uint64_t> &indices_py,
    py::array_t<double> &out_py,
    uint64_t seed1, uint64_t seed2
) {
    Buffer<uint64_t> indices(indices_py);
    Buffer<double> out(out_py);
    uint64_t rows = indices.shape[0];
    uint64_t cols = indices.shape[1];

//#pragma omp parallel
{
    Buffer<uint64_t> hash_buffer({2});
    //#pragma omp for
    for(uint64_t i = 0; i < rows; i++) {
        MurmurHash3_x86_128 ( indices(i * cols), 
                                cols * sizeof(uint64_t),
                                0x9747b28c, hash_buffer());

        // Can use Random123 package instead of this.
        hash_buffer[0] ^= seed1;
        hash_buffer[1] ^= seed2;

        double f1 = (double) hash_buffer[0] / (double) UINT64_MAX;
        double f2 = (double) hash_buffer[1] / (double) UINT64_MAX;

        // Compute box-muller transform
        double r = sqrt(-2.0 * log(f1)); 
        double theta = 2.0 * M_PI * f2;

        out[i] = r * cos(theta);
    }
}

}