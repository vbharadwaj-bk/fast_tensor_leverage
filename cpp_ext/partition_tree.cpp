//cppimport
#include <iostream>
#include "common.h"
#include "oneapi/mkl.hpp"

using namespace std;

class PartitionTree {
    uint32_t n, F;
    uint32_t leaf_count, node_count;

    uint32_t lfill_level, lfill_count;
    uint32_t nodes_upto_lfill, nodes_before_lfill;
    uint32_t complete_level_offset;

    MKL_INT J, R;
    MKL_INT R2;

    // Temporary buffers related to sampling

    // ============================================================
    Buffer<MKL_INT> c;
    Buffer<double> temp1;
    Buffer<double> q;

    Buffer<double> m;
    Buffer<double> mL;
    Buffer<double> low;
    Buffer<double> high;

    // ============================================================
    // Parameters related to DGEMV_Batched 
    Buffer<double*> a_array;
    Buffer<double*> x_array;
    Buffer<double*> y_array;

    char trans_array; 
    MKL_INT m_array;
    MKL_INT n_array;
    double alpha_array;
    MKL_INT lda_array;
    MKL_INT incx_array;
    double beta_array;
    MKL_INT incy_array;
    MKL_INT group_count;
    MKL_INT group_size;

    void execute_mkl_dgemv_batch() {
        CBLAS_TRANSPOSE trans = CblasNoTrans;
        cblas_dgemv_batch(
            CblasRowMajor,
            &trans, 
            &m_array, 
            &n_array, 
            &alpha_array, 
            (const double**) a_array(), 
            &lda_array, 
            (const double**) x_array(), 
            &incx_array, 
            &beta_array, 
            y_array(), 
            &incy_array, 
            group_count, 
            &group_size);
    }
    // ============================================================

public:
    PartitionTree(uint32_t n, uint32_t F, uint64_t J, uint64_t R) 
        :   c({J}, 0),
            temp1({J, R}, 0.0),
            q({J, F}, 0.0),
            m({J}, 0.0),
            mL({J}, 0.0),
            low({J}, 0.0),
            high({J}, 0.0),
            a_array({J}, nullptr),
            x_array({J}, nullptr),
            y_array({J}, nullptr)
        {
        this->n = n;
        this->F = F;
        this->J = J;
        this->R = R;
        R2 = R * R;

        leaf_count = divide_and_roundup(n, F);
        node_count = 2 * leaf_count - 1;

        log2_round_down(leaf_count, lfill_level, lfill_count);
        nodes_upto_lfill = lfill_count * 2 - 1;
        nodes_before_lfill = lfill_count - 1;

        uint32_t nodes_at_partial_level_div2 = (node_count - nodes_upto_lfill) / 2;
        complete_level_offset = nodes_before_lfill - nodes_at_partial_level_div2;
    }

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

        Buffer<double> U(U_py);
        Buffer<double> G(G_py);
        Buffer<double> h(h_py);
        Buffer<double> scaled_h(scaled_h_py);
        Buffer<uint64_t> samples(samples_py);
        Buffer<double> random_draws(random_draws_py);

        trans_array = 'n';
        m_array = R;
        n_array = R;
        alpha_array = 1.0;
        lda_array = R;
        incx_array = 1;
        beta_array = 0.0;
        incy_array = 1;
        group_count = 1;
        group_size = J;

        for(MKL_INT i = 0; i < J; i++) {
            x_array[i] = scaled_h(i, 0);
            y_array[i] = temp1(i, 0); 

            c[i] = 0;
            low[i] = 0.0;
            high[i] = 1.0;
            a_array[i] = G(0);
        }

        execute_mkl_dgemv_batch();

        batch_dot_product(
            scaled_h(), 
            temp1(), 
            m(),
            J, R 
            );

        // TODO: SHOULD MODIFY THIS ROUTINE SO IT
        // HANDLES THE LAST PARTIALLY COMPLETE LEVEL
        // OF THE BINARY TREE

        for(uint32_t c_level = 0; c_level < lfill_level; c_level++) {
            // Prepare to compute m(L(v)) for all v
            for(MKL_INT i = 0; i < J; i++) {
                a_array[i] = G((2 * c[i] + 1) * R2); 
            }

            execute_mkl_dgemv_batch();

            batch_dot_product(
                scaled_h(), 
                temp1(), 
                mL(),
                J, R 
                );

            for(MKL_INT i = 0; i < J; i++) {
                double cutoff = low[i] + mL[i] / m[i];
                if(random_draws[i] <= cutoff) {
                    c[i] = 2 * c[i] + 1;
                    high[i] = cutoff;
                }
                else {
                    c[i] = 2 * c[i] + 2;
                    low[i] = cutoff;
                }
            }
        }

        // We will use the m array as a buffer 
        // for the draw fractions.
        for(int i = 0; i < J; i++) {
            m[i] = (random_draws[i] - low[i]) / (high[i] - low[i]);

            MKL_INT leaf_idx;
            if(c[i] >= nodes_upto_lfill) {
                leaf_idx = c[i] - nodes_upto_lfill;
            }
            else {
                leaf_idx = c[i] - complete_level_offset; 
            }

            a_array[i] = U(leaf_idx * F, 0);
            y_array[i] = q(i, 0);
        }

        m_array = F; // TODO: NEED TO PAD EACH ARRAY SO THIS IS OKAY!

        execute_mkl_dgemv_batch();

        for(MKL_INT i = 0; i < J; i++) {
            double running_sum = 0.0;
            for(MKL_INT j = 0; j < F; j++) {
                double temp = q[i, j] * q[i, j];
                q[i, j] = running_sum;
                running_sum += temp;
            }

            for(MKL_INT j = 0; j < F; j++) {
                q[i, j] /= running_sum; 
            }

            MKL_INT res = F-1;
            for(MKL_INT j = 0; j < F - 1; j++) {
                if(m[i] < q[i, j + 1]) {
                    res = j; 
                    break;
                }
            }

            MKL_INT leaf_idx;
            if(c[i] >= nodes_upto_lfill) {
                leaf_idx = c[i] - nodes_upto_lfill;
            }
            else {
                leaf_idx = c[i] - complete_level_offset; 
            }
            samples[i] = res + leaf_idx * F;
            
            for(MKL_INT j = 0; j < R; j++) {
                h[i * R + j] *= U[res + leaf_idx * F, j]; 
            }
        }
    }
};

PYBIND11_MODULE(partition_tree, m) {
  py::class_<PartitionTree>(m, "PartitionTree")
    .def(py::init<uint32_t, uint32_t, uint64_t, uint64_t>())
    .def("PTSample", &PartitionTree::PTSample) 
    ;
}

// C++ Compiler: dpcpp 

/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['-DMKL_ILP64', '-m64', '-I"${MKLROOT}/include"', '-std=c++2b']
cfg['extra_link_args'] = ['-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed', '-lmkl_intel_ilp64', '-lmkl_gnu_thread', '-lmkl_core', '-lgomp', '-lpthread', '-lm', '-ldl']
cfg['dependencies'] = ['common.h'] 
%>
*/
