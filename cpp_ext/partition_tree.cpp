//cppimport
#include <iostream>
#include <cassert>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <random>
#include "common.h"
#include "cblas.h"

using namespace std;

class __attribute__((visibility("hidden"))) PartitionTree {
    int64_t n, F;
    uint32_t leaf_count, node_count;

    uint32_t lfill_level, lfill_count;
    uint32_t total_levels;
    uint32_t nodes_upto_lfill, nodes_before_lfill;
    uint32_t complete_level_offset;

    int64_t J, R;
    int64_t R2;

    Buffer<double> G;
    unique_ptr<Buffer<double>> G_unmultiplied;

    // Related to random number generation 
    std::random_device rd;  
    std::mt19937 gen;
    std::uniform_real_distribution<> dis;

    // Temporary buffers related to sampling
    // ============================================================
    Buffer<int64_t> c;
    Buffer<double> temp1;
    Buffer<double> q;

    Buffer<double> m;
    Buffer<double> mL;
    Buffer<double> low;
    Buffer<double> high;
    Buffer<double> random_draws;

    // ============================================================
    // Parameters related to DGEMV_Batched 
    Buffer<double*> a_array;
    Buffer<double*> x_array;
    Buffer<double*> y_array;

    char trans_array; 
    int64_t m_array;
    int64_t n_array;
    double alpha_array;
    int64_t lda_array;
    int64_t incx_array;
    double beta_array;
    int64_t incy_array;
    int64_t group_count;
    int64_t group_size;

    void execute_mkl_dsymv_batch() {
        for(int64_t i = 0; i < J; i++) {
            cblas_dsymv(CblasRowMajor, 
                    CblasUpper, 
                    R, 
                    1.0, 
                    (const double*) a_array[i],
                    R, 
                    (const double*) x_array[i], 
                    1, 
                    0.0, 
                    y_array[i], 
                    1);
        }
    }
    // ============================================================

public:
    PartitionTree(uint32_t n, uint32_t F, uint64_t J, uint64_t R) 
        :   G({2 * divide_and_roundup(n, F) - 1, R * R}, 0.0),
            rd(),
            gen(rd()),
            dis(0.0, 1.0), 
            c({J}, 0),
            temp1({J, R}, 0.0),
            q({J, F}, 0.0),
            m({J}, 0.0),
            mL({J}, 0.0),
            low({J}, 0.0),
            high({J}, 0.0),
            random_draws({J}, 0.0),
            a_array({J}, nullptr),
            x_array({J}, nullptr),
            y_array({J}, nullptr)
        {
        assert(n % F == 0);
        this->n = n;
        this->F = F;
        this->J = J;
        this->R = R;
        R2 = R * R;

        leaf_count = divide_and_roundup(n, F);
        node_count = 2 * leaf_count - 1;

        log2_round_down(leaf_count, lfill_level, lfill_count);
        total_levels = node_count > lfill_count ? lfill_level + 1 : lfill_level;

        nodes_upto_lfill = lfill_count * 2 - 1;
        nodes_before_lfill = lfill_count - 1;

        uint32_t nodes_at_partial_level_div2 = (node_count - nodes_upto_lfill) / 2;
        complete_level_offset = nodes_before_lfill - nodes_at_partial_level_div2;
        G_unmultiplied.reset(nullptr);

    //std::random_device rd;  // Will be used to obtain a seed for the random number engine
    //std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    //std::uniform_real_distribution<> dis(1.0, 2.0);

    }

    bool is_leaf(int64_t c) {
        return 2 * c + 1 >= node_count; 
    }

    int64_t leaf_idx(int64_t c) {
        if(c >= nodes_upto_lfill) {
            return c - nodes_upto_lfill;
        }
        else {
            return c - complete_level_offset; 
        }
    }

    void build_tree(py::array_t<double> U_py) {
        G_unmultiplied.reset(nullptr);
        Buffer<double> U(U_py);
        std::fill(G(), G(node_count * R2), 0.0);

        // First leaf must always be on the lowest filled level 
        int64_t first_leaf_idx = node_count - leaf_count; 

        Buffer<double*> a_array({leaf_count}, nullptr);
        Buffer<double*> c_array({leaf_count}, nullptr);

        for(int64_t i = 0; i < leaf_count; i++) {
            uint64_t idx = leaf_idx(first_leaf_idx + i);
            a_array[i] = U(idx * F, 0);
            c_array[i] = G(first_leaf_idx + i, 0);
        }

        for(int64_t i = 0; i < leaf_count; i++) {
            cblas_dsyrk(CblasRowMajor, 
                        CblasUpper, 
                        CblasTrans,
		                R,
                        F, 
                        1.0, 
                        (const double*) a_array[i], 
                        R, 
                        0.0, 
                        c_array[i], 
                        R);
        }

        int64_t start = nodes_before_lfill; 
        int64_t end = first_leaf_idx;

        for(int c_level = lfill_level; c_level >= 0; c_level--) {
            for(int c = start; c < end; c++) {
                for(int j = 0; j < R2; j++) {
                    G[c * R2 + j] += G[(2 * c + 1) * R2 + j] + G[(2 * c + 2) * R2 + j];
                } 
            }
            end = start;
            start = ((start + 1) / 2) - 1;
        }
    }

    /*
    * Multiplies the partial gram matrices maintained by this tree against
    * the provided buffer, caching the old values for future multiplications. 
    */
    void multiply_matrices_against_provided(Buffer<double> &mat) {
        if(! G_unmultiplied) {
            G_unmultiplied.reset(new Buffer<double>({node_count, static_cast<unsigned long>(R2)}, 0.0));
            std::copy(G(), G(node_count * R2), (*G_unmultiplied)());
        }
        for(int64_t i = 0; i < node_count; i++) {
            for(int j = 0; j < R2; j++) {
                G[i * R2 + j] = (*G_unmultiplied)[i * R2 + j] * mat[j];
            }
        }
    }
 
    void multiply_against_numpy_buffer(py::array_t<double> mat_py) {
        Buffer<double> mat(mat_py);
        multiply_matrices_against_provided(mat);
    }

    void batch_dot_product(
                double* A, 
                double* B, 
                double* result,
                int64_t J, int64_t R 
                ) {
        for(int i = 0; i < J; i++) {
            result[i] = 0;
            for(int j = 0; j < R; j++) {
                result[i] += A[i * R + j] * B[i * R + j];
            }
        }
    }

    void PTSample(py::array_t<double> U_py, 
            py::array_t<double> h_py,  
            py::array_t<double> scaled_h_py,
            py::array_t<uint64_t> samples_py
            ) {

        Buffer<double> U(U_py);
        Buffer<double> h(h_py);
        Buffer<double> scaled_h(scaled_h_py);
        Buffer<uint64_t> samples(samples_py);

        // Draw random doubles
        for(int64_t i = 0; i < J; i++) {
            random_draws[i] = dis(gen);
        }

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

        for(int64_t i = 0; i < J; i++) {
            x_array[i] = scaled_h(i, 0);
            y_array[i] = temp1(i, 0); 

            c[i] = 0;
            low[i] = 0.0;
            high[i] = 1.0;
            a_array[i] = G(0);
        }

        execute_mkl_dsymv_batch();

        batch_dot_product(
            scaled_h(), 
            temp1(), 
            m(),
            J, R 
            );

        for(uint32_t c_level = 0; c_level < lfill_level; c_level++) {
            // Prepare to compute m(L(v)) for all v

            for(int64_t i = 0; i < J; i++) {
                a_array[i] = G((2 * c[i] + 1) * R2); 
            }

            execute_mkl_dsymv_batch();

            batch_dot_product(
                scaled_h(), 
                temp1(), 
                mL(),
                J, R 
                );

            for(int64_t i = 0; i < J; i++) {
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

        // Handle the tail case
        if(node_count > nodes_before_lfill) {
            for(int64_t i = 0; i < J; i++) {
                a_array[i] = is_leaf(c[i]) ? a_array[i] : G((2 * c[i] + 1) * R2); 
            }

            execute_mkl_dsymv_batch();

            batch_dot_product(
                scaled_h(), 
                temp1(), 
                mL(),
                J, R 
                );

            for(int64_t i = 0; i < J; i++) {
                double cutoff = low[i] + mL[i] / m[i];
                if((! is_leaf(c[i])) && random_draws[i] <= cutoff) {
                    c[i] = 2 * c[i] + 1;
                    high[i] = cutoff;
                }
                else if((! is_leaf(c[i])) && random_draws[i] > cutoff) {
                    c[i] = 2 * c[i] + 2;
                    low[i] = cutoff;
                }
            }
        }

        // We will use the m array as a buffer 
        // for the draw fractions.
        if(F > 1) {
            for(int i = 0; i < J; i++) {
                m[i] = (random_draws[i] - low[i]) / (high[i] - low[i]);

                int64_t leaf_idx;
                if(c[i] >= nodes_upto_lfill) {
                    leaf_idx = c[i] - nodes_upto_lfill;
                }
                else {
                    leaf_idx = c[i] - complete_level_offset; 
                }

                a_array[i] = U(leaf_idx * F, 0);
                y_array[i] = q(i, 0);
            }

            m_array = F;
            execute_mkl_dsymv_batch();
        }

        for(int64_t i = 0; i < J; i++) {
            int64_t res; 
            if(F > 1) {
                res = F - 1;
                double running_sum = 0.0;
                for(int64_t j = 0; j < F; j++) {
                    double temp = q[i * F + j] * q[i * F + j];
                    q[i * F + j] = running_sum;
                    running_sum += temp;
                }

                for(int64_t j = 0; j < F; j++) {
                    q[i * F + j] /= running_sum; 
                }

                for(int64_t j = 0; j < F - 1; j++) {
                    if(m[i] < q[i * F + j + 1]) {
                        res = j; 
                        break;
                    }
                }
            }
            else {
                res = 0;
            }

            int64_t idx = leaf_idx(c[i]);
            samples[i] = res + idx * F;
            
            for(int64_t j = 0; j < R; j++) {
                h[i * R + j] *= U[(res + idx * F) * R + j]; 
            }
        }
    }
};

PYBIND11_MODULE(partition_tree, m) {
  py::class_<PartitionTree>(m, "PartitionTree")
    .def(py::init<uint32_t, uint32_t, uint64_t, uint64_t>())
    .def("build_tree", &PartitionTree::build_tree) 
    .def("multiply_against_numpy_buffer", &PartitionTree::multiply_against_numpy_buffer) 
    .def("PTSample", &PartitionTree::PTSample) 
    ;
}

/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['--std=c++2b', '-I/global/homes/v/vbharadw/OpenBLAS_install/include', '-fopenmp']
cfg['extra_link_args'] = ['-L/global/homes/v/vbharadw/OpenBLAS_install/lib', '-lopenblas', '-fopenmp']
cfg['dependencies'] = ['common.h'] 
%>
*/
