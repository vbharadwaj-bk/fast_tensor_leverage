//cppimport
#include <iostream>
#include <string>
#include "common.h"
#include "cblas.h"
#include "lapacke.h"
#include "hashing.hpp"
#include "sampler.hpp"
#include "tensor.hpp"
#include "sparse_tensor.hpp"
#include "low_rank_tensor.hpp"
#include "dense_tensor.hpp"
//#include "py_function_tensor.hpp"
#include "uniform_sampler.hpp"
#include "larsen_kolda_sampler.hpp"
#include "larsen_kolda_hybrid.hpp"
#include "efficient_krp_sampler.hpp"
#include "als.hpp"

#include <execution>
#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(als_module, m) {
    py::class_<Tensor>(m, "Tensor")
        .def("execute_downsampled_mttkrp_py", &Tensor::execute_downsampled_mttkrp_py)
        .def("compute_residual_normsq", &Tensor::compute_residual_normsq_py) 
        .def("get_normsq", &Tensor::get_normsq);
    py::class_<BlackBoxTensor, Tensor>(m, "BlackBoxTensor");
    py::class_<LowRankTensor, BlackBoxTensor>(m, "LowRankTensor")
        .def(py::init<uint64_t, uint64_t, py::list>()) 
        .def(py::init<uint64_t, py::list>())
        .def("get_sigma", &LowRankTensor::get_sigma_py)
        .def("renormalize_columns", &LowRankTensor::renormalize_columns);
    py::class_<DenseTensor<double>, BlackBoxTensor>(m, "DenseTensor_double")
        .def(py::init<py::array_t<double>, uint64_t>());
    py::class_<DenseTensor<float>, BlackBoxTensor>(m, "DenseTensor_float")
        .def(py::init<py::array_t<float>, uint64_t>());
    py::class_<SparseTensor, Tensor>(m, "SparseTensor")
        .def(py::init<py::array_t<uint32_t>, py::array_t<double>, std::string>());
    /*py::class_<PyFunctionTensor, BlackBoxTensor>(m, "PyFunctionTensor")
        .def(py::init<py::array_t<uint64_t>, uint64_t, uint64_t, double>()) 
        .def("test_fiber_evaluator", &PyFunctionTensor::test_fiber_evaluator);*/
    py::class_<ALS>(m, "ALS")
        .def(py::init<LowRankTensor&, Tensor&>()) 
        .def("initialize_ds_als", &ALS::initialize_ds_als) 
        .def("execute_exact_als_update", &ALS::execute_exact_als_update)
        .def("execute_ds_als_update", &ALS::execute_ds_als_update);
}

/*
<%
setup_pybind11(cfg)
openblas_include='-I/global/homes/v/vbharadw/OpenBLAS_install/include'
openblas_link_location='-L/global/homes/v/vbharadw/OpenBLAS_install/lib'

libflame_include=None
liblame_link_loc=None
blis_include=None
blis_link_loc=None

tbb_include='-I/global/homes/v/vbharadw/intel/oneapi/tbb/2021.8.0/include'
tbb_link_location='-L/global/homes/v/vbharadw/intel/oneapi/tbb/2021.8.0/lib/intel64/gcc4.8'
cfg['extra_compile_args'] = [openblas_include, tbb_include, '--std=c++2a', '-fopenmp', '-Ofast', '-march=native']
cfg['extra_link_args'] = [openblas_link_location, tbb_link_location, '-lopenblas', '-fopenmp', '-Ofast', '-ltbb']
cfg['dependencies'] = [ 'common.h', 
                        'partition_tree.hpp', 
                        'efficient_krp_sampler.hpp', 
                        'sampler.hpp', 
                        'uniform_sampler.hpp', 
                        'larsen_kolda_sampler.hpp', 
                        'larsen_kolda_hybrid.hpp', 
                        'low_rank_tensor.hpp', 
                        'sparse_tensor.hpp', 
                        'black_box_tensor.hpp', 
                        'tensor.hpp', 
                        'idx_lookup.hpp',
                        'hash_lookup.hpp',
                        'sort_lookup.hpp',
                        'dense_tensor.hpp',
                        'als.hpp'] 
%>
*/

