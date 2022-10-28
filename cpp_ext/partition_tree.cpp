//cppimport
#include <iostream>
#include "common.h"
#include "oneapi/mkl.hpp"

using namespace std;

class PartitionTree {
public:
    uint32_t n, F;
    uint32_t leaf_count, node_count;

    uint32_t lfill_level, lfill_count;
    uint32_t nodes_upto_lfill, nodes_before_lfill;
    uint32_t complete_level_offset;
    PartitionTree(uint32_t n, uint32_t F) {
        this->n = n;
        this->F = F;

        leaf_count = divide_and_roundup(n, F);
        node_count = 2 * leaf_count - 1;

        log2_round_down(leaf_count, lfill_level, lfill_count);
        nodes_upto_lfill = lfill_count * 2 - 1;
        nodes_before_lfill = lfill_count - 1;

        uint32_t nodes_at_partial_level_div2 = (node_count - nodes_upto_lfill) / 2;
        complete_level_offset = nodes_before_lfill - nodes_at_partial_level_div2;
    }

    /*void m(vector<uint32_t> &c, double* G, double* scaled_h) {

    }*/

    void batch_dot_product(
                double* A, 
                double* B, 
                double* result,
                MKL_INT J, MKL_INT R 
                ) {
        for(int i = 0; i < J; i++) {
            result[i] = 0;
            for(int j = 0; j < R; j++) {
                result[i] += A[i * R + j] * B[i * R + j];
            }
        }
    }

    void PTSample(py::array_t<double> U_py, 
            py::array_t<double> G_py,
            py::array_t<double> h_py,  
            py::array_t<double> scaled_h_py,
            py::array_t<uint64_t> samples_py,
            py::array_t<double> random_draws_py
            ) {

        NumpyArray<double> U(U_py);
        NumpyArray<double> G(G_py);
        NumpyArray<double> h(h_py);
        NumpyArray<double> scaled_h(scaled_h_py);
        NumpyArray<uint64_t> samples(samples_py);
        NumpyArray<double> random_draws(random_draws_py);

        vector<MKL_INT> c(J, 0);
        vector<double> temp1(J * R, 0);
        vector<double> m(J, 0);

        MKL_INT J = U.info.shape[0];
        MKL_INT R = U.info.shape[1];
        MKL_INT R2 = R * R;

        char trans_array[1] = {'n'};
        MKL_INT m_array[1] = {R};
        MKL_INT n_array[1] = {R};
        double alpha_array[1] = {1.0};
        vector<double*> a_array(J, nullptr);
        MKL_INT lda_array[1] = {R};
        vector<double*> x_array(J, nullptr); 
        MKL_INT incx_array[1] = {1};
        double beta_array[1] = {0.0};
        vector<double*> y_array(J, nullptr); 
        MKL_INT incy_array[1] = {1};
        MKL_INT group_count = 1;
        MKL_INT group_size[1] = {1};

        for(MKL_INT i = 0; i < J; i++) {
            a_array[i] = G.ptr + (c[i] * R2); 
            x_array[i] = scaled_h.ptr + i * R;
            y_array[i] = temp1.data() + i * R; 
        }

        dgemv_batch(trans_array, 
            m_array, 
            n_array, 
            alpha_array, 
            a_array.data(), 
            lda_array, 
            x_array.data(), 
            incx_array, 
            beta_array, 
            y_array.data(), 
            incy_array, 
            group_count, 
            group_size);

        batch_dot_product(
            scaled_h.ptr, 
            temp1.data(), 
            m.data(),
            J, R 
            );
    }

};

PYBIND11_MODULE(partition_tree, m) {
  py::class_<PartitionTree>(m, "PartitionTree")
    .def(py::init<uint32_t, uint32_t>())
    .def("PTSample", &PartitionTree::PTSample) 
    ;
}

// C++ Compiler: dpcpp 

/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['-DMKL_ILP64', '-m64', '-I"${MKLROOT}/include"']
cfg['extra_link_args'] = ['-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed', '-lmkl_intel_ilp64', '-lmkl_gnu_thread', '-lmkl_core', '-lgomp', '-lpthread', '-lm', '-ldl']
cfg['dependencies'] = ['common.h'] 
%>
*/
