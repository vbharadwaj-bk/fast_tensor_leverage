//cppimport
#include <iostream>
#include "common.h"

using namespace std;

class Tensor {
public:
    virtual void execute_mttkrp() = 0;
};

class LowRankTensor : public Tensor {
    int j;
public:
    LowRankTensor(int j) {
        this->j = j;
    }
    void execute_mttkrp() {
        cout << "Hi, I'm a low-rank tensor executing the MTTKRP!" << endl;
    }
};

class ALS {
public:
    void test(Tensor &t) {
        t.execute_mttkrp();
    }
};

PYBIND11_MODULE(als_module, m) {
    py::class_<Tensor>(m, "Tensor");
    py::class_<LowRankTensor, Tensor>(m, "LowRankTensor")
        .def(py::init<int>());
    py::class_<ALS>(m, "ALS")
        .def(py::init<>()) 
        .def("test", &ALS::test); 
}

/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['--std=c++2b', '-I/global/homes/v/vbharadw/OpenBLAS_install/include', '-fopenmp', '-O3']
cfg['extra_link_args'] = ['-L/global/homes/v/vbharadw/OpenBLAS_install/lib', '-lopenblas', '-fopenmp', '-O3']
cfg['dependencies'] = ['common.h', 'partition_tree.hpp', 'efficient_krp_sampler.hpp'] 
%>
*/
