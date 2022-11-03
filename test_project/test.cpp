#include <iostream>
#include <chrono>
#include "cblas.h"
#include "omp.h"

using namespace std;

int main(int argc, char** argv) {
    int N = 4000;
    double* A = new double[N * N];
    double* B = new double[N * N];

    auto start = std::chrono::system_clock::now();

    double* C = new double[N * N];
    #pragma omp parallel for
    for(int i = 0; i < 20; i++) {
        double* C = new double[N * N];
    }

    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        N, N, N,
        1.0,
        (const double*) A,
        N,
        (const double*) B,
        N,
        0.0,
        C,
        N);


    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;

    cout << "Elapsed: " << elapsed_seconds.count() << endl;
}