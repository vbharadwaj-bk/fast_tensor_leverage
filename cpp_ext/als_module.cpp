//cppimport
#include <iostream>
#include <string>
#include "common.h"
#include "cblas.h"
#include "lapacke.h"
#include "sampler.hpp"
#include "tensor.hpp"
#include "sparse_tensor.hpp"
#include "low_rank_tensor.hpp"
#include "uniform_sampler.hpp"
#include "larsen_kolda_sampler.hpp"
#include "efficient_krp_sampler.hpp"

using namespace std;

class __attribute__((visibility("hidden"))) ALS {
public:
    LowRankTensor &cp_decomp;
    Tensor &ground_truth;

    // These fields used only for sampling 
    uint64_t J;
    unique_ptr<Sampler> sampler;
    unique_ptr<Buffer<uint64_t>> samples;

    ALS(LowRankTensor &cp_dec, Tensor &gt) : 
        cp_decomp(cp_dec),
        ground_truth(gt)    
    {
        // Empty
    }

    void initialize_ds_als(uint64_t J, std::string sampler_type) {
        if(sampler_type == "efficient") {
            sampler.reset(new EfficientKRPSampler(J, cp_decomp.R, cp_decomp.U));
        }
        else if(sampler_type == "larsen_kolda") {
            sampler.reset(new LarsenKoldaSampler(J, cp_decomp.R, cp_decomp.U));
        }
        else if(sampler_type == "uniform") {
            sampler.reset(new UniformSampler(J, cp_decomp.R, cp_decomp.U));
        }

        samples.reset(new Buffer<uint64_t>({cp_decomp.N, J}));
    }

    void execute_ds_als_update(uint32_t j, 
            bool renormalize,
            bool update_sampler
            ) {

        uint64_t Ij = cp_decomp.U[j].shape[0];
        uint64_t R = cp_decomp.R; 
        uint64_t J = sampler->J;

        Buffer<double> mttkrp_res({Ij, R});
        Buffer<double> pinv({R, R});

        sampler->KRPDrawSamples(j, *samples, nullptr);

        #pragma omp parallel for collapse(2) 
        for(uint32_t i = 0; i < J; i++) {
            for(uint32_t t = 0; t < R; t++) {
                sampler->h[i * R + t] *= sqrt(sampler->weights[i]); 
            }
        }

        std::fill(pinv(), pinv(R * R), 0.0);
        compute_pinv(sampler->h, pinv);  

        #pragma omp parallel for collapse(2)
        for(uint32_t i = 0; i < J; i++) {
            for(uint32_t t = 0; t < R; t++) {
                sampler->h[i * R + t] *= sqrt(sampler->weights[i]); 
            }
        }

        ground_truth.execute_downsampled_mttkrp(
                *samples, 
                sampler->h,
                j,
                mttkrp_res 
                );

        // Multiply gram matrix result by the pseudo-inverse
        cblas_dsymm(
            CblasRowMajor,
            CblasRight,
            CblasUpper,
            (uint32_t) Ij,
            (uint32_t) R,
            1.0,
            pinv(),
            R,
            mttkrp_res(),
            R,
            0.0,
            cp_decomp.U[j](),
            R
        );

        cp_decomp.get_sigma(cp_decomp.sigma, j);

        // Multiply result by sigma^(-1) of the CP
        // decomposition. Assumes that sigma is correct
        // upon entry to this function. 
        #pragma omp parallel for collapse(2)
        for(uint32_t u = 0; u < Ij; u++) {
            for(uint32_t v = 0; v < R; v++) {
                cp_decomp.U[j][u * R + v] /= cp_decomp.sigma[v]; 
            }
        }

        if(renormalize) {
            cp_decomp.renormalize_columns(j);
        }
        if(update_sampler) {
            sampler->update_sampler(j);
        }
    }
};

PYBIND11_MODULE(als_module, m) {
    py::class_<Tensor>(m, "Tensor")
        .def("execute_downsampled_mttkrp_py", &Tensor::execute_downsampled_mttkrp_py);
    py::class_<LowRankTensor, Tensor>(m, "LowRankTensor")
        .def(py::init<uint64_t, uint64_t, uint64_t, py::list>()) 
        .def(py::init<uint64_t, py::list>())
        .def("get_sigma", &LowRankTensor::get_sigma_py)
        .def("renormalize_columns", &LowRankTensor::renormalize_columns);
    py::class_<SparseTensor, Tensor>(m, "SparseTensor")
        .def(py::init<py::array_t<uint32_t>, py::array_t<double>>());
    py::class_<ALS>(m, "ALS")
        .def(py::init<LowRankTensor&, Tensor&>()) 
        .def("initialize_ds_als", &ALS::initialize_ds_als) 
        .def("execute_ds_als_update", &ALS::execute_ds_als_update);
}

/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['--std=c++2b', '-I/global/homes/v/vbharadw/OpenBLAS_install/include', '-fopenmp', '-Ofast']
cfg['extra_link_args'] = ['-L/global/homes/v/vbharadw/OpenBLAS_install/lib', '-lopenblas', '-fopenmp', '-Ofast']
cfg['dependencies'] = ['common.h', 'partition_tree.hpp', 'efficient_krp_sampler.hpp', 'sampler.hpp', 'uniform_sampler.hpp', 'larsen_kolda_sampler.hpp', 'low_rank_tensor.hpp', 'sparse_tensor.hpp'] 
%>
*/
