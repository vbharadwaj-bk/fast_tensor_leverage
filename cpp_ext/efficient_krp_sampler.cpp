//cppimport
#include <iostream>
#include <vector>
#include <cassert>
#include "common.h"
//#include "partition_tree.hpp"

using namespace std;

class __attribute__((visibility("hidden"))) EfficientKRPSampler {
    uint64_t J, R;
    vector<Buffer<double>> &U;
    //vector<PartitionTree> gram_trees;
    //vector<PartitionTree> eigen_trees;

    Buffer<int64_t> c;
    Buffer<double> temp1;
    Buffer<double> q;

    Buffer<double> m;
    Buffer<double> mL;
    Buffer<double> low;
    Buffer<double> high;
    Buffer<double> random_draws;

public:
    EfficientKRPSampler(
            uint64_t J, 
            uint64_t R, 
            vector<Buffer<double>> &U_matrices)
    :        
            U(U_matrices), 
            c({J}),
            temp1({J, R}),
            q({J, R}),
            m({J}),
            mL({J}),
            low({J}),
            high({J}),
            random_draws({J}) {
        
        this->J = J;
        this->R = R;

        for(uint32_t i = 0; i < U.size(); i++) {
            assert(U[i].shape.size() == 2);
            assert(U[i].shape[1] == R);
            assert(U[i].shape[0] % R == 0);
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
cfg['dependencies'] = ['common.h'] 
%>
*/

