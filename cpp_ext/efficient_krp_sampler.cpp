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

class __attribute__((visibility("hidden"))) CP_Decomposition {
public:
    unique_ptr<NPBufferList<double>> U_py_bufs;
    int64_t R;
    CP_Decomposition(int64_t R, py::list U_py)
    :
    U_py_bufs(new NPBufferList<double>(U_py))
    {
        this->R = R;
    }

    // Convenience method for RHS sampling 
    void materialize_partial_evaluation(py::array_t<uint64_t> samples_py, 
        uint64_t j,
        py::array_t<double> result_py 
        ) {

        // Assume sample matrix is N x J, result is J x R
        Buffer<uint64_t> samples(samples_py);
        uint64_t num_samples = samples.shape[1];
        Buffer<double> result(result_py);
        std::fill(result(), result(num_samples, 0), 1.0);

        uint32_t N = U_py_bufs->length;

        #pragma omp parallel for 
        for(uint32_t i = 0; i < num_samples; i++) {
            for(uint32_t k = 0; k < N; k++) {
                if(k != j) {
                    for(uint32_t u = 0; u < R; u++) {
                        result[i * R + u] *= U_py_bufs->buffers[k][samples[k * num_samples + i] * R + u];
                    }
                } 
            }
        }
    }
};

PYBIND11_MODULE(efficient_krp_sampler, m) {
  py::class_<CP_ALS>(m, "CP_ALS")
    .def(py::init<int64_t, int64_t, py::list>()) 
    .def("computeM", &CP_ALS::computeM)
    .def("KRPDrawSamples", &CP_ALS::KRPDrawSamples)
    .def("get_G_pinv", &CP_ALS::get_G_pinv)
    ;

  py::class_<CP_Decomposition>(m, "CP_Decomposition")
    .def(py::init<int64_t, py::list>()) 
    .def("materialize_partial_evaluation", &CP_Decomposition::materialize_partial_evaluation);
}

/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['--std=c++2b', '-I/global/homes/v/vbharadw/OpenBLAS_install/include', '-fopenmp', '-O3']
cfg['extra_link_args'] = ['-L/global/homes/v/vbharadw/OpenBLAS_install/lib', '-lopenblas', '-fopenmp', '-O3']
cfg['dependencies'] = ['common.h', 'partition_tree.hpp', 'efficient_krp_sampler.hpp'] 
%>
*/
