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
            lambda({U_matrices.size() + 1, R})
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
    }

    void computeM(uint32_t j) {
        // TODO: the original U matrices are freed!
        std::fill(M(N * R2), M((N + 1) * R2), 1.0);
        uint32_t last_buffer = N;
        for(int k = N - 1; k >= 0; k--) {
            if(k != j) {
                for(uint32_t i = 0; i < R2; i++) {
                    M[k * R2 + i] = gram_trees[k]->G[i] * M[(last_buffer * R2) + i];   
                } 

                last_buffer = k;
            }
        }

        // Pseudo-inverse via eigendecomposition, stored in the N+1'th slot of
        // the 2D M array.

        // TODO: Should actually check the result of the LAPACK call! 
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
                        M[u * R + v] *= 1.0 / sqrt(lambda[v]); 
                }
            }
            else {
                for(uint32_t u = 0; u < R; u++) {
                        M[u * R + v] = 0.0; 
                }
            }
        }

        cblas_dsyrk(CblasRowMajor, 
                    CblasUpper, 
                    CblasNoTrans,
                    R,
                    R, 
                    1.0, 
                    (const double*) M(), 
                    R, 
                    0.0, 
                    M(N, 0), 
                    R);

        for(uint32_t u = 0; u < R; u++) {
            for(uint32_t v = 0; v < R; v++) {
                cout << M[N * R2 + u * R + v] << " ";
            }
            cout << endl;
        }

    }

    ~EfficientKRPSampler() {
        for(uint32_t i = 0; i < N; i++) {
            delete gram_trees[i];
            delete eigen_trees[i];
        }
    }
}; 

class __attribute__((visibility("hidden"))) EfficientSamplerWrapper {
public:
    unique_ptr<EfficientKRPSampler> obj;
    EfficientSamplerWrapper(int64_t J, int64_t R, py::list U_py) {
        NPBufferList<double> U(U_py);
        obj.reset(new EfficientKRPSampler(J, R, U.buffers));
    }
    void computeM(uint32_t j) {
        obj->computeM(j);
    }
};

PYBIND11_MODULE(efficient_krp_sampler, m) {
  py::class_<EfficientSamplerWrapper>(m, "EfficientKRPSampler")
    .def(py::init<int64_t, int64_t, py::list>()) 
    .def("computeM", &EfficientSamplerWrapper::computeM)
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

