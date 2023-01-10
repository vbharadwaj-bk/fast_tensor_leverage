#pragma once

#include <iostream>
#include "common.h"

using namespace std;

// Catchall file for test functions

void test_dsyrk_multithreading(py::array_t<double> U_py) {
    Buffer<double> U(U_py);
    uint64_t Ij = U.shape[0];
    uint64_t R = U.shape[1];
    Buffer<double> M({R, R});
    std::fill(M(), M(R * R), 0.0);


    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        uint64_t work = (Ij + num_threads - 1) / num_threads;
        uint64_t start = min(work * thread_id, Ij);
        uint64_t end = min(work * (thread_id + 1), Ij);

        if(end - start > 0) {
            Buffer<double> local({R, R});
            cblas_dsyrk(CblasRowMajor, 
                        CblasUpper, 
                        CblasTrans,
                        R,
                        end-start, 
                        1.0, 
                        U(start * R), 
                        R, 
                        0.0, 
                        local(), 
                        R);

            for(uint64_t i = 0; i < R * R; i++) {
                #pragma omp atomic
                M[i] += local[i];
            }
        }

    }


    /*cblas_dgemm(CblasRowMajor,
        CblasTrans,
        CblasNoTrans,
        R,
        R,
        Ij,
        1.0,
        U(),
        R,
        U(),
        R,
        0.0,
        M(),
        R
    );*/
}
