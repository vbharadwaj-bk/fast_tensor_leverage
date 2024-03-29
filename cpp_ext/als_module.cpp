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
#include "tests.hpp"

#include <execution>
#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;

class __attribute__((visibility("hidden"))) PySampler {
public:
    unique_ptr<NPBufferList<double>> U_py_bufs;
    unique_ptr<Sampler> sampler;
    PySampler(py::list U_py, int64_t J, int64_t R, std::string type)
    :
    U_py_bufs(new NPBufferList<double>(U_py))
    {
        
        if(type == "efficient") {
            sampler.reset(new EfficientKRPSampler(
                J, R, 
                (*U_py_bufs).buffers
            ));
        } 
        else if(type == "larsen_kolda_hybrid") {
            sampler.reset(new LarsenKoldaHybrid(
                J, R, 
                (*U_py_bufs).buffers
            ));
        }
        else if(type == "larsen_kolda") {
            sampler.reset(new LarsenKoldaSampler(
                J, R, 
                (*U_py_bufs).buffers
            ));
        }
        else {
            cout << "Unsupported argument " << type
                << " passed for sampler construction" << endl;
            exit(1);
        }
    }

    void KRPDrawSamples(uint32_t j, 
            py::array_t<uint64_t> samples_py) {
        Buffer<uint64_t> samples(samples_py);
        sampler->KRPDrawSamples(j, samples, nullptr);
    }

    void KRPDrawSamples_materialize(uint32_t j, 
            py::array_t<uint64_t> samples_py, 
            py::array_t<double> h_out_py,
            py::array_t<double> weights_out_py 
            ) {
        Buffer<uint64_t> samples(samples_py);
        Buffer<double> h_out(h_out_py);
        Buffer<double> weights_out(weights_out_py);
        sampler->KRPDrawSamples(j, samples, nullptr);
        std::copy(sampler->h(), sampler->h(sampler->J * sampler->R), h_out()); 
        std::copy(sampler->weights(), sampler->weights(sampler->J), weights_out());
    } 
};

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
        .def("set_sigma", &LowRankTensor::set_sigma_py)
        .def("renormalize_columns", &LowRankTensor::renormalize_columns)
        .def("multiply_random_factor_entries", &LowRankTensor::multiply_random_factor_entries)
        .def("materialize_rhs", &LowRankTensor::materialize_rhs_py)
        .def("evaluate_indices", &LowRankTensor::evaluate_indices_py);
    py::class_<DenseTensor<double>, BlackBoxTensor>(m, "DenseTensor_double")
        .def(py::init<py::array_t<double>, uint64_t>());
    py::class_<DenseTensor<float>, BlackBoxTensor>(m, "DenseTensor_float")
        .def(py::init<py::array_t<float>, uint64_t>());
    py::class_<SparseTensor, Tensor>(m, "SparseTensor")
        .def(py::init<py::array_t<uint32_t>, py::array_t<double>, std::string>())
        .def("execute_rrf", &SparseTensor::execute_rrf)
        .def("initialize_randomized_accuracy_estimation", &SparseTensor::initialize_randomized_accuracy_estimation)
        .def("compute_residual_normsq_estimated", &SparseTensor::compute_residual_normsq_estimated);
    /*py::class_<PyFunctionTensor, BlackBoxTensor>(m, "PyFunctionTensor")
        .def(py::init<py::array_t<uint64_t>, uint64_t, uint64_t, double>()) 
        .def("test_fiber_evaluator", &PyFunctionTensor::test_fiber_evaluator);*/
    py::class_<ALS>(m, "ALS")
        .def(py::init<LowRankTensor&, Tensor&>()) 
        .def("initialize_ds_als", &ALS::initialize_ds_als) 
        .def("execute_exact_als_update", &ALS::execute_exact_als_update)
        .def("execute_ds_als_update", &ALS::execute_ds_als_update);
    py::class_<PySampler>(m, "Sampler")
        .def(py::init<py::list, int64_t, int64_t, std::string>()) 
        .def("KRPDrawSamples", &PySampler::KRPDrawSamples)
        .def("KRPDrawSamples_materialize", &PySampler::KRPDrawSamples_materialize)
        ;
    m.def("test_dsyrk_multithreading", &test_dsyrk_multithreading);
}

/*
<%
setup_pybind11(cfg)

import json
config = None
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Required compiler flags 
compile_args=config['required_compile_args']
link_args=config['required_link_args']

# Add extra flags for the BLAS and LAPACK 

blas_include_path=[f'-I{config["blas_include_path"]}']
blas_link_path=[f'-L{config["blas_link_path"]}']

tbb_include_path=[f'-I{config["tbb_include_path"]}']
tbb_link_path=[f'-L{config["tbb_link_path"]}']

rpath_options=[f'-Wl,-rpath,{config["blas_link_path"]}:{config["tbb_link_path"]}']

for lst in [blas_include_path,
            tbb_include_path,
            config["extra_compile_args"]
            ]:
    compile_args.extend(lst)


for lst in [blas_link_path,
            tbb_link_path,
            config["blas_link_flags"], 
            config["tbb_link_flags"], 
            config["extra_link_args"],
            rpath_options 
            ]:
    link_args.extend(lst)

print(f"Compiling C++ extensions with {compile_args}")
print(f"Linking C++ extensions with {link_args}")

cfg['extra_compile_args'] = compile_args 
cfg['extra_link_args'] = link_args 
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
                        'sort_lookup.hpp',
                        'dense_tensor.hpp',
                        'tests.hpp',
                        'als.hpp',
                        'random_util.hpp',
                        'index_filter.hpp',
                        '../config.json'
                        ] 
%>
*/

