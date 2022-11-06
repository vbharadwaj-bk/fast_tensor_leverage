//cppimport
#include <iostream>
#include <vector>
#include <cassert>
#include "common.h"
#include "partition_tree.hpp"

using namespace std;

class __attribute__((visibility("hidden"))) EfficientKRPSampler {
    uint64_t N, J, R, R2;
    vector<Buffer<double>> &U;
    ScratchBuffer scratch;
    Buffer<double> M;

    vector<PartitionTree*> gram_trees;
    vector<PartitionTree*> eigen_trees;

public:
    EfficientKRPSampler(
            uint64_t J, 
            uint64_t R, 
            vector<Buffer<double>> &U_matrices)
    :        
            U(U_matrices),
            scratch(R, J, R),
            M({U_matrices.size() + 1, R * R})
    {    
        this->J = J;
        this->R = R;
        this->N = U.size();

        R2 = R * R;

        for(uint32_t i = 0; i < N; i++) {
            uint32_t n = U[i].shape[0];
            assert(U[i].shape.size() == 2);
            assert(U[i].shape[1] == R);
            assert(n % R == 0);
            gram_trees.push_back(new PartitionTree(n, R, J, R, scratch));
            eigen_trees.push_back(new PartitionTree(R, R, J, R, scratch));
        }

        std::fill(M(U.size(), 0), M(U.size(), R2), 1.0);

        // Should move the data structure initialization to another routine,
        // but this is fine for now.

        for(uint32_t i = 0; i < N; i++) {
            gram_trees[i]->build_tree(U[i]);
        }
    }

    void computeM(uint32_t j) {
        cout << "-----------------------------------" << endl;
        uint32_t last_buffer = N;
        for(int k = N - 1; k >= 0; k--) {
            if(k != j) {
                for(uint32_t i = 0; i < R2; i++) {
                    M[k * R2 + i] = gram_trees[k]->G[i] * M[(last_buffer * R2) + i];   
                } 

                last_buffer = k;
            }
        }

        for(uint32_t u = 0; u < R; u++) {
            for(uint32_t v = 0; v < R; v++) {
                cout << M[last_buffer * R2 + u * R + v] << " ";
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

