//cppimport
#include <iostream>
#include <vector>
#include <cassert>
#include "common.h"
#include "partition_tree.hpp"

using namespace std;

class __attribute__((visibility("hidden"))) EfficientKRPSampler {
    uint64_t N, J, R;
    vector<Buffer<double>> &U;
    ScratchBuffer scratch;

    vector<PartitionTree*> gram_trees;
    vector<PartitionTree*> eigen_trees;

public:
    EfficientKRPSampler(
            uint64_t J, 
            uint64_t R, 
            vector<Buffer<double>> &U_matrices)
    :        
            U(U_matrices),
            scratch(R, J, R)
    {    
        this->J = J;
        this->R = R;
        this->N = U.size();

        for(uint32_t i = 0; i < N; i++) {
            uint32_t n = U[i].shape[0];
            assert(U[i].shape.size() == 2);
            assert(U[i].shape[1] == R);
            assert(n % R == 0);

            gram_trees.push_back(new PartitionTree(n, R, J, R, scratch));
            eigen_trees.push_back(new PartitionTree(R, R, J, R, scratch));
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
};

PYBIND11_MODULE(efficient_krp_sampler, m) {
  py::class_<EfficientSamplerWrapper>(m, "EfficientKRPSampler")
    .def(py::init<int64_t, int64_t, py::list>());
}

/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['--std=c++2b', '-I/global/homes/v/vbharadw/OpenBLAS_install/include', '-fopenmp', '-O3']
cfg['extra_link_args'] = ['-L/global/homes/v/vbharadw/OpenBLAS_install/lib', '-lopenblas', '-fopenmp', '-O3']
cfg['dependencies'] = ['common.h', 'partition_tree.hpp'] 
%>
*/

