//cppimport
#include <iostream>
#include <string>
#include <algorithm>

#include "cblas.h"
#include "lapacke.h"

#include "common.h"
#include "tt_sampler.hpp"
#include "tensor.hpp"
#include "black_box_tensor.hpp"
#include "dense_tensor.hpp"
#include "sparse_tensor.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(tt_module, m) {
    py::class_<TTSampler>(m, "TTSampler")
        .def(py::init<uint64_t, uint64_t, uint64_t, py::array_t<uint64_t>>())
        .def("update_matricization", &TTSampler::update_matricization)
        .def("sample", &TTSampler::sample)
        .def("evaluate_indices_partial", &TTSampler::evaluate_indices_partial);
    py::class_<Tensor>(m, "Tensor")
        .def("execute_downsampled_mttkrp", &Tensor::execute_downsampled_mttkrp_py);
    py::class_<BlackBoxTensor, Tensor>(m, "BlackBoxTensor"); 
    py::class_<DenseTensor<double>, BlackBoxTensor>(m, "DenseTensor_double")
        .def(py::init<py::array_t<double>, uint64_t>());
    py::class_<DenseTensor<float>, BlackBoxTensor>(m, "DenseTensor_float")
        .def(py::init<py::array_t<float>, uint64_t>());
    py::class_<SparseTensor, Tensor>(m, "SparseTensor")
        .def(py::init<py::array_t<uint32_t>, py::array_t<double>, std::string>());
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

cfg['parallel'] = False 
cfg['extra_compile_args'] = compile_args 
cfg['extra_link_args'] = link_args 
cfg['dependencies'] = [ 'common.h', 
                        'tt_sampler.hpp',
                        'sort_lookup.hpp',
                        'sparse_tensor.hpp',
                        'tensor.hpp',
                        'black_box_tensor.hpp',
                        'dense_tensor.hpp',
                        '../config.json'
                        ] 
%>
*/

