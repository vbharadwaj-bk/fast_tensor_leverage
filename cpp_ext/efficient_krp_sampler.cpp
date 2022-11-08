//cppimport
#include <iostream>
#include <vector>
#include <cassert>
#include "common.h"
#include "partition_tree.hpp"
#include "lapacke.h"

using namespace std;

class __attribute__((visibility("hidden"))) EfficientKRPSampler {
    uint64_t N, J, R, R2;
    vector<Buffer<double>> &U;
    ScratchBuffer scratch;
    Buffer<double> M;
    Buffer<double> lambda;

    Buffer<double> h;
    Buffer<double> scaled_h;
    vector<Buffer<double>> scaled_eigenvecs;

    vector<PartitionTree*> gram_trees;
    vector<PartitionTree*> eigen_trees;
    double eigenvalue_tolerance;

public:
    EfficientKRPSampler(
            uint64_t J, 
            uint64_t R, 
            vector<Buffer<double>> &U_matrices)
    :        
            U(U_matrices),
            scratch(R, J, R),
            M({U_matrices.size() + 1, R * R}),
            lambda({U_matrices.size() + 1, R}),
            h({J, R}),
            scaled_h({J, R})
    {    
        this->J = J;
        this->R = R;
        this->N = U.size();

        eigenvalue_tolerance = 0.0; // Tolerance of eigenvalues for symmetric PINV 
        R2 = R * R;

        for(uint32_t i = 0; i < N; i++) {
            uint32_t n = U[i].shape[0];
            assert(U[i].shape.size() == 2);
            assert(U[i].shape[1] == R);
            assert(n % R == 0);
            gram_trees.push_back(new PartitionTree(n, R, J, R, scratch));
            eigen_trees.push_back(new PartitionTree(R, R, J, R, scratch));
        }

        // Should move the data structure initialization to another routine,
        // but this is fine for now.

        for(uint32_t i = 0; i < N; i++) {
            gram_trees[i]->build_tree(U[i]); 
        }

        for(uint32_t i = 0; i < N + 1; i++) {
            scaled_eigenvecs.emplace_back(initializer_list<uint64_t>{R, R}, M(i, 0));
        }
    }

    /*
     * Simple, unoptimized square-matrix in-place transpose.
    */
    void transpose_square_in_place(double* ptr, uint64_t n) {
        for(uint64_t i = 0; i < N - 1; i++) {
            for(uint64_t j = i + 1; j < N; j++) {
                double temp = ptr[i * n + j];
                ptr[i * n + j] = ptr[j * n + i];
                ptr[j * n + i] = temp;
            }
        }
    }

    void computeM(uint32_t j) {
        // TODO: the original U matrices are freed!
        std::fill(M(N * R2), M((N + 1) * R2), 1.0);
        uint32_t last_buffer = N;
        for(int k = N - 1; k >= 0; k--) {
            if((uint32_t) k != j) {
                for(uint32_t i = 0; i < R2; i++) {
                    M[k * R2 + i] = gram_trees[k]->G[i] * M[(last_buffer * R2) + i];   
                } 

                last_buffer = k;
            }
        }

        // Handles the case where the first matrix is left out of the KRP 
        std::copy(M(last_buffer, 0), M(last_buffer, R2), M());

        // Pseudo-inverse via eigendecomposition, stored in the N+1'th slot of
        // the 2D M array.

        LAPACKE_dsyev( CblasRowMajor, 
                        'V', 
                        'U', 
                        R,
                        M(), 
                        R, 
                        lambda() );

        for(uint32_t v = 0; v < R; v++) {
            if(lambda[v] > eigenvalue_tolerance) {
                for(uint32_t u = 0; u < R; u++) {
                        M[N * R2 + u * R + v] = M[u * R + v] / sqrt(lambda[v]); 
                }
            }
            else {
                for(uint32_t u = 0; u < R; u++) {
                        M[N * R2 + u * R + v] = 0.0; 
                }
            }
        }

        cblas_dsyrk(CblasRowMajor, 
                    CblasUpper, 
                    CblasNoTrans,
                    R,
                    R, 
                    1.0, 
                    (const double*) M(N, 0), 
                    R, 
                    0.0, 
                    M(), 
                    R);

        for(uint32_t k = N - 1; k > 0; k--) {
            if(k != j) {
                for(uint32_t i = 0; i < R2; i++) {
                    M[k * R2 + i] *= M[i];   
                } 
            }
        }

        // Eigendecompose each of the gram matrices 
        for(uint32_t k = N; k > 0; k--) {
            if(k != j) {
                if(k < N) {
                    LAPACKE_dsyev( CblasRowMajor, 
                                    'V', 
                                    'U', 
                                    R,
                                    M(k, 0), 
                                    R, 
                                    lambda(k, 0) );

                    for(uint32_t v = 0; v < R; v++) { 
                        for(uint32_t u = 0; u < R; u++) {
                            M[k * R2 + u * R + v] *= sqrt(lambda[k * R + v]); 
                        }
                    }
                }
                transpose_square_in_place(M(k, 0), R);
            }
        }

        for(uint32_t k = 0; k < N; k++) {
            if(k != j) {
                int offset = (k + 1 == j) ? k + 2 : k + 1;
                eigen_trees[k]->build_tree(scaled_eigenvecs[offset]);
                eigen_trees[k]->multiply_matrices_against_provided(gram_trees[k]->G);
            }
        } 
    }

    void KRPDrawSamples(uint32_t j, Buffer<uint64_t> samples) {
        // Samples is an array of size N x J
    
        computeM(j);
        std::fill(h(), h(J, R), 1.0);

        for(uint32_t k = 0; k < N; k++) {
            if(k != j) {
                // Sample an eigenvector component of the mixture distribution 
                std::copy(h(), h(J, R), scaled_h());

                Buffer<uint64_t> row_buffer({J}, samples(k, 0));

                /*eigen_trees[k]->PTSample_internal(scaled_eigenvecs[k], 
                        scaled_h,
                        h,
                        row_buffer 
                        );

                gram_trees[k]->PTSample_internal(U[k], 
                        h,
                        scaled_h,
                        row_buffer 
                        );*/
            }
        }
    }

    ~EfficientKRPSampler() {
        for(uint32_t i = 0; i < N; i++) {
            delete gram_trees[i];
            delete eigen_trees[i];
        }
    }
}; 

class __attribute__((visibility("hidden"))) CP_ALS {
public:
    unique_ptr<NPBufferList<double>> U_py_bufs;
    EfficientKRPSampler sampler;
    CP_ALS(int64_t J, int64_t R, py::list U_py)
    :
    U_py_bufs(new NPBufferList<double>(U_py)),
    sampler(J, R, (*U_py_bufs).buffers) 
    {
    }
    void computeM(uint32_t j) {
        sampler.computeM(j);
    }
};

PYBIND11_MODULE(efficient_krp_sampler, m) {
  py::class_<CP_ALS>(m, "CP_ALS")
    .def(py::init<int64_t, int64_t, py::list>()) 
    .def("computeM", &CP_ALS::computeM)
    ;
}

/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['--std=c++2b', '-I/global/homes/v/vbharadw/OpenBLAS_install/include', '-fopenmp', '-O3']
cfg['extra_link_args'] = ['-L/global/homes/v/vbharadw/OpenBLAS_install/lib', '-lopenblas', '-fopenmp', '-O3']
cfg['dependencies'] = ['common.h', 'partition_tree.hpp'] 
%>
*/

