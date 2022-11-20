//cppimport
#include <iostream>
#include "common.h"

using namespace std;

class __attribute__((visibility("hidden"))) Tensor {
public:
    virtual void 
    execute_downsampled_mttkrp(
            Buffer<uint64_t> &samples, 
            Buffer<double> &lhs,
            uint64_t j,
            Buffer<double> &result
            ) = 0;

    void execute_downsampled_mttkrp_py(
            py::array_t<uint64_t> &samples_py, 
            py::array_t<double> &lhs_py,
            uint64_t j,
            py::array_t<double> &result_py
            ) {
        
        Buffer<uint64_t> samples(samples_py);
        Buffer<double> lhs(lhs_py);
        Buffer<double> result(result_py);

        execute_downsampled_mttkrp(
                samples, 
                lhs,
                j,
                result 
                );
    }
};

class __attribute__((visibility("hidden"))) LowRankTensor : public Tensor {
public:
    unique_ptr<NPBufferList<double>> U_py_bufs;
    uint64_t R, J;
    Buffer<double> partial_evaluation; 

    LowRankTensor(uint64_t R, uint64_t J, py::list U_py)
    :
    U_py_bufs(new NPBufferList<double>(U_py)),
    partial_evaluation({J, R})
    {
        this->J = J;
        this->R = R;
    }

    // Convenience method for RHS sampling 
    void materialize_partial_evaluation(Buffer<uint64_t> &samples, 
        uint64_t j 
        ) {

        // Assume sample matrix is N x J, result is J x R
        std::fill(partial_evaluation(), partial_evaluation(J, 0), 1.0);

        uint32_t N = U_py_bufs->length;

        #pragma omp parallel for 
        for(uint32_t i = 0; i < J; i++) {
            for(uint32_t k = 0; k < N; k++) {
                if(k != j) {
                    for(uint32_t u = 0; u < R; u++) {
                        partial_evaluation[i * R + u] *= U_py_bufs->buffers[k][samples[k * J + i] * R + u];
                    }
                } 
            }
        }
    }

    void execute_downsampled_mttkrp(
            Buffer<uint64_t> &samples, 
            Buffer<double> &lhs,
            uint64_t j,
            Buffer<double> &result
            ) {
        cout << "Hello world!" << endl;
    }
};

class __attribute__((visibility("hidden"))) ALS {
public:
    void test(Tensor &t) {
        //t.execute_downsampled_mttkrp();
    }
};

PYBIND11_MODULE(als_module, m) {
    py::class_<Tensor>(m, "Tensor");
        //.def("execute_downsampled_mttkrp_py", &Tensor::execute_downsampled_mttkrp_py);
    py::class_<LowRankTensor, Tensor>(m, "LowRankTensor")
        .def(py::init<uint64_t, uint64_t, py::list>());
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
