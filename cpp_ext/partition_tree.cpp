//cppimport
#include "partition_tree.hpp"

PYBIND11_MODULE(partition_tree, m) {
  py::class_<PartitionTree>(m, "PartitionTree")
    .def(py::init<uint32_t, uint32_t, uint64_t, uint64_t>())
    .def("build_tree", &PartitionTree::build_tree) 
    .def("multiply_against_numpy_buffer", &PartitionTree::multiply_against_numpy_buffer) 
    .def("PTSample", &PartitionTree::PTSample)
    .def("get_G0", &PartitionTree::get_G0)
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
