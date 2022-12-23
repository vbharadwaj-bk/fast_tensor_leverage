//cppimport
#include "efficient_krp_sampler.hpp"

class __attribute__((visibility("hidden"))) CP_ALS {
public:
    unique_ptr<NPBufferList<double>> U_py_bufs;
    EfficientKRPSampler sampler;
    CP_ALS(int64_t J, int64_t R, py::list U_py)
    :
    U_py_bufs(new NPBufferList<double>(U_py)),
    sampler(J, R, (*U_py_bufs).buffers) 
    {
    }
    void computeM(uint32_t j) {
        sampler.computeM(j);
    }

    // Can expose this function for debugging
    void KRPDrawSamples_explicit_random(uint32_t j, py::array_t<uint64_t> samples_py, py::array_t<double> random_draws_py) {
        Buffer<uint64_t> samples(samples_py);
        Buffer<double> random_draws(random_draws_py);
        sampler.KRPDrawSamples(j, samples, &random_draws); 
    }

    void KRPDrawSamples(uint32_t j, 
            py::array_t<uint64_t> samples_py, 
            py::array_t<double> h_out_py) {
        Buffer<uint64_t> samples(samples_py);
        Buffer<double> h_out(h_out_py);
        sampler.KRPDrawSamples(j, samples, nullptr);
        std::copy(sampler.h(), sampler.h(sampler.J * sampler.R), h_out()); 
    }

    void get_G_pinv(py::array_t<double> G_py) {
        Buffer<double> G(G_py);
        std::copy(sampler.M(), sampler.M(sampler.R2), G());
    }
};

PYBIND11_MODULE(efficient_krp_sampler, m) {
  py::class_<CP_ALS>(m, "CP_ALS")
    .def(py::init<int64_t, int64_t, py::list>()) 
    .def("computeM", &CP_ALS::computeM)
    .def("KRPDrawSamples", &CP_ALS::KRPDrawSamples)
    .def("get_G_pinv", &CP_ALS::get_G_pinv)
    ;
}


/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['--std=c++2a', '-I/global/homes/v/vbharadw/OpenBLAS_install/include', '-fopenmp', '-g']
cfg['extra_link_args'] = ['-L/global/homes/v/vbharadw/OpenBLAS_install/lib', '-lopenblas', '-fopenmp', '-g']
cfg['dependencies'] = ['common.h', 'partition_tree.hpp', 'efficient_krp_sampler.hpp'] 
%>
*/
