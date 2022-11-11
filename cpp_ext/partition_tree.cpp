//cppimport
#include "partition_tree.hpp"

class __attribute__((visibility("hidden"))) PartitionTreeWrapper {
public:
    ScratchBuffer scratch;
    PartitionTree tree;

    PartitionTreeWrapper(uint32_t n, 
            uint32_t F, 
            uint64_t J, 
            uint64_t R) 
        :
        scratch(F, J, R),
        tree(n, F, J, R, scratch)
    {}

    void build_tree(py::array_t<double> U_py) {
        Buffer<double> U(U_py);
        tree.build_tree(U);
    }

    void multiply_against_numpy_buffer(py::array_t<double> mat_py) {
        tree.multiply_against_numpy_buffer(mat_py);
    }

    void PTSample(py::array_t<double> U_py, 
            py::array_t<double> h_py,  
            py::array_t<double> scaled_h_py,
            py::array_t<uint64_t> samples_py,
            py::array_t<double> random_draws_py 
            ) {
        tree.PTSample(U_py, h_py, scaled_h_py, samples_py, random_draws_py);
    }

    void get_G0(py::array_t<double> M_buffer_py) {
        tree.get_G0(M_buffer_py);
    }

};

PYBIND11_MODULE(partition_tree, m) {
  py::class_<PartitionTreeWrapper>(m, "PartitionTree")
    .def(py::init<uint32_t, uint32_t, uint64_t, uint64_t>())
    .def("build_tree", &PartitionTreeWrapper::build_tree) 
    .def("multiply_against_numpy_buffer", &PartitionTreeWrapper::multiply_against_numpy_buffer) 
    .def("PTSample", &PartitionTreeWrapper::PTSample)
    .def("get_G0", &PartitionTreeWrapper::get_G0)
    ;
}

/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['--std=c++2b', '-fopenmp', '-O3']
cfg['extra_link_args'] = ['-fopenmp', '-O3']
cfg['dependencies'] = ['common.h', 'partition_tree.hpp'] 
%>
*/
