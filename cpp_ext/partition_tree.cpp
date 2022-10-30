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

    // ============================================================
    // Parameters related to DGEMV_Batched 
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
    vector<double*> a_array;
    vector<double*> x_array;
    vector<double*> y_array;

    void execute_mkl_dgemv_batch() {
        dgemv_batch(&trans_array, 
            &m_array, 
            &n_array, 
            &alpha_array, 
            (const double**) a_array.data(), 
            &lda_array, 
            (const double**) x_array.data(), 
            &incx_array, 
            &beta_array, 
            y_array.data(), 
            &incy_array, 
            &group_count, 
            &group_size);
    }

public:
    PartitionTree(uint32_t n, uint32_t F, uint64_t J, uint64_t R) {
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

        a_array.resize(J);
        x_array.resize(J);
        y_array.resize(J);
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

        NumpyArray<double> U(U_py);
        NumpyArray<double> G(G_py);
        NumpyArray<double> h(h_py);
        NumpyArray<double> scaled_h(scaled_h_py);
        NumpyArray<uint64_t> samples(samples_py);
        NumpyArray<double> random_draws(random_draws_py);

        vector<MKL_INT> c(J, 0);
        vector<double> temp1(J * R, 0);
        vector<double> q(J * F, 0);

        vector<double> m(J, 0);
        vector<double> mL(J, 0);
        vector<double> low(J, 0);
        vector<double> high(J, 1);

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

        /*vector<double*> a_array(J, nullptr);
        vector<double*> x_array(J, nullptr); 
        vector<double*> y_array(J, nullptr);*/

        for(MKL_INT i = 0; i < J; i++) {
            x_array[i] = scaled_h.ptr + i * R;
            y_array[i] = temp1.data() + i * R; 
        }

        for(MKL_INT i = 0; i < J; i++) {
            a_array[i] = G.ptr + (c[i] * R2); 
        }

        dgemv_batch(&trans_array, 
            &m_array, 
            &n_array, 
            &alpha_array, 
            (const double**) a_array.data(), 
            &lda_array, 
            (const double**) x_array.data(), 
            &incx_array, 
            &beta_array, 
            y_array.data(), 
            &incy_array, 
            &group_count, 
            &group_size);

        batch_dot_product(
            scaled_h.ptr, 
            temp1.data(), 
            m.data(),
            J, R 
            );

        // TODO: SHOULD MODIFY THIS ROUTINE SO IT
        // HANDLES THE LAST PARTIALLY COMPLETE LEVEL
        // OF THE BINARY TREE

        for(uint32_t c_level = 0; c_level < lfill_level; c_level++) {
            // Prepare to compute m(L(v)) for all v
            for(MKL_INT i = 0; i < J; i++) {
                a_array[i] = G.ptr + ((2 * c[i] + 1) * R2); 
            }

            dgemv_batch(&trans_array, 
                &m_array, 
                &n_array, 
                &alpha_array, 
                (const double**) a_array.data(), 
                &lda_array, 
                (const double**) x_array.data(), 
                &incx_array, 
                &beta_array, 
                y_array.data(), 
                &incy_array, 
                &group_count, 
                &group_size);

            batch_dot_product(
                scaled_h.ptr, 
                temp1.data(), 
                mL.data(),
                J, R 
                );

            for(MKL_INT i = 0; i < J; i++) {
                double cutoff = low[i] + mL[i] / m[i];
                if(random_draws.ptr[i] <= cutoff) {
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
            m[i] = (random_draws.ptr[i] - low[i]) / (high[i] - low[i]);

            MKL_INT leaf_idx;
            if(c[i] >= nodes_upto_lfill) {
                leaf_idx = c[i] - nodes_upto_lfill;
            }
            else {
                leaf_idx = c[i] - complete_level_offset; 
            }

            a_array[i] = U.ptr + (leaf_idx * F * R);
            y_array[i] = q.data() + i * F; 
        }

        m_array = F; // TODO: NEED TO PAD EACH ARRAY SO THIS IS OKAY!
        //lda_array[0] = R;

        CBLAS_TRANSPOSE x = CblasNoTrans;

        cblas_dgemv_batch(
            CblasRowMajor,
            &x, 
            &m_array, 
            &n_array, 
            &alpha_array, 
            (const double**) a_array.data(), 
            &lda_array, 
            (const double**) x_array.data(), 
            &incx_array, 
            &beta_array, 
            y_array.data(), 
            &incy_array, 
            group_count, 
            &group_size);

        for(MKL_INT i = 0; i < J; i++) {
            double running_sum = 0.0;
            for(MKL_INT j = 0; j < F; j++) {
                double temp = q[i * F + j] * q[i * F + j];
                q[i * F + j] = running_sum;
                running_sum += temp;
            }

            for(MKL_INT j = 0; j < F; j++) {
                q[i * F + j] /= running_sum; 
            }
        }

        for(MKL_INT i = 0; i < J; i++) {
            MKL_INT res = F-1;
            for(MKL_INT j = 0; j < F - 1; j++) {
                if(m[i] < q[i * F + j + 1]) {
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
            samples.ptr[i] = res + leaf_idx * F;
            
            for(MKL_INT j = 0; j < R; j++) {
                h.ptr[i * R + j] *= U.ptr[(res + leaf_idx * F) * R + j]; 
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
