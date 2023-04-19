#pragma once

#include <iostream>
#include <cassert>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <random>
#include <chrono>
#include "common.h"
#include "omp.h"
#include "cblas.h"

using namespace std;

/*
* Collection of temporary buffers that can be reused by all tree samplers 
*/
class __attribute__((visibility("hidden"))) ScratchBuffer {
public:
    Buffer<int64_t> c;
    Buffer<double> temp1;
    Buffer<double> q;
    Buffer<double> m;
    Buffer<double> mL;
    Buffer<double> low;
    Buffer<double> high;
    Buffer<double> random_draws;
    Buffer<double*> a_array;
    Buffer<double*> x_array;
    Buffer<double*> y_array;

    ScratchBuffer(uint32_t F, uint64_t J, uint64_t R) : 
            c({J}),
            temp1({J, R}),
            q({J, F}),
            m({J}),
            mL({J}),
            low({J}),
            high({J}),
            random_draws({J}),
            a_array({J}),
            x_array({J}),
            y_array({J})
    {}
};

class __attribute__((visibility("hidden"))) PartitionTree {
public:
    int64_t n, F;
    uint32_t leaf_count, node_count;

    uint32_t lfill_level, lfill_count;
    uint32_t total_levels;
    uint32_t nodes_upto_lfill, nodes_before_lfill;
    uint32_t complete_level_offset;

    int64_t J, R;
    int64_t R2;

    Buffer<double> G;
    ScratchBuffer &scratch;

    unique_ptr<Buffer<double>> G_unmultiplied;

    void execute_mkl_dsymv_batch() {
        #pragma omp for
        for(int64_t i = 0; i < J; i++) {
            cblas_dsymv(CblasRowMajor, 
                    CblasUpper, 
                    R, 
                    1.0, 
                    (const double*) scratch.a_array[i],
                    R, 
                    (const double*) scratch.x_array[i], 
                    1, 
                    0.0, 
                    scratch.y_array[i], 
                    1);
        }
    }

    PartitionTree(uint32_t n, uint32_t F, uint64_t J, uint64_t R, ScratchBuffer &scr)
        :   G({2 * divide_and_roundup(n, F) - 1, R * R}),
            scratch(scr)
        {
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

    void build_tree(Buffer<double> &U) {
        G_unmultiplied.reset(nullptr);

        // First leaf must always be on the lowest filled level 
        int64_t first_leaf_idx = node_count - leaf_count; 

        Buffer<double*> a_array({leaf_count});
        Buffer<double*> c_array({leaf_count});

        #pragma omp parallel
{
        #pragma omp for
        for(int64_t i = 0; i < node_count * R2; i++) {
            G[i] = 0.0;
        }

        #pragma omp for
        for(int64_t i = 0; i < leaf_count; i++) {
            uint64_t idx = leaf_idx(first_leaf_idx + i);
            uint64_t row_ct = min((uint64_t) F, U.shape[0] - idx * F);
            a_array[i] = U(idx * F, 0);
            c_array[i] = G(first_leaf_idx + i, 0);

            cblas_dsyrk(CblasRowMajor, 
                        CblasUpper, 
                        CblasTrans,
                        R,
                        row_ct, 
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
            #pragma omp for 
            for(int c = start; c < end; c++) {
                for(int j = 0; j < R2; j++) {
                    G[c * R2 + j] += G[(2 * c + 1) * R2 + j] + G[(2 * c + 2) * R2 + j];
                } 
            }
            end = start;
            start = ((start + 1) / 2) - 1;
        }
}
    }

    void get_G0(py::array_t<double> M_buffer_py) {
        Buffer<double> M_buffer(M_buffer_py);
        for(int64_t i = 0; i < R2; i++) {
            M_buffer[i] = G[i];
        } 
    }

    /*
    * Multiplies the partial gram matrices maintained by this tree against
    * the provided buffer, caching the old values for future multiplications. 
    */
    void multiply_matrices_against_provided(Buffer<double> &mat) {
        if(! G_unmultiplied) {
            G_unmultiplied.reset(new Buffer<double>({node_count, static_cast<unsigned long>(R2)}));
            std::copy(G(), G(node_count * R2), (*G_unmultiplied)());
        }
        #pragma omp parallel for
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
        #pragma omp for
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
            py::array_t<uint64_t> samples_py,
            py::array_t<double> random_draws_py 
            ) {

        Buffer<double> U(U_py);
        Buffer<double> h(h_py);
        Buffer<double> scaled_h(scaled_h_py);
        Buffer<uint64_t> samples(samples_py);
        Buffer<double> random_draws(random_draws_py);

        PTSample_internal(U, h, scaled_h, samples, random_draws);
    }

    void PTSample_internal(Buffer<double> &U, 
            Buffer<double> &h,
            Buffer<double> &scaled_h,
            Buffer<uint64_t> &samples,
            Buffer<double> &random_draws
            ) {
 
        Buffer<int64_t> &c = scratch.c;
        Buffer<double> &temp1 = scratch.temp1;
        Buffer<double> &q = scratch.q;
        Buffer<double> &m = scratch.m;
        Buffer<double> &mL = scratch.mL;
        Buffer<double> &low = scratch.low;
        Buffer<double> &high = scratch.high;
        //Buffer<double> &random_draws = scratch.random_draws;
        Buffer<double*> &a_array = scratch.a_array;
        Buffer<double*> &x_array = scratch.x_array;
        Buffer<double*> &y_array = scratch.y_array;

        Buffer<uint64_t> sort_idxs({(uint64_t) J});
        Buffer<double> random_permuted({(uint64_t) J});
        Buffer<double> scaled_h_permuted({(uint64_t) J, (uint64_t) R});

        Buffer<uint64_t> c_blocked({(uint64_t) J});
        Buffer<uint64_t> c_idxs({(uint64_t) J});
        Buffer<uint64_t> unique_c_idxs({(uint64_t) J});
        uint64_t num_unique;

        std::iota(sort_idxs(), sort_idxs(J), 0);
        std::iota(c_idxs(), c_idxs(J), 0);
        std::iota(unique_c_idxs(), unique_c_idxs(J+1), 0);

        std::sort(std::execution::par_unseq, 
            sort_idxs(), 
            sort_idxs(J),
            [&](uint64_t i, uint64_t j) {
                return random_draws[i] < random_draws[j];
            });

        #pragma omp parallel
{
        #pragma omp for
        for(int64_t i = 0; i < J; i++) {
            random_permuted[i] = random_draws[sort_idxs[i]];
            for(int64_t j = 0; j < R; j++) {
                scaled_h_permuted[i * R + j] = scaled_h[sort_idxs[i] * R + j];
            }
        }

        #pragma omp for
        for(int64_t i = 0; i < J; i++) {
            x_array[i] = scaled_h_permuted(i, 0);
            y_array[i] = temp1(i, 0); 

            c[i] = 0;
            low[i] = 0.0;
            high[i] = 1.0;
            a_array[i] = G(0);
        }

        #pragma omp single
        {
            uint64_t* end_range = 
                std::unique_copy(std::execution::par_unseq,
                    c_idxs(),
                    c_idxs(J),
                    unique_c_idxs(),
                    [&](uint64_t a, uint64_t b) {
                        return c[a] == c[b];
                    }
                    );

            num_unique = end_range - unique_c_idxs();
            unique_c_idxs[num_unique] = J;
        }

        #pragma omp for 
        for(uint64_t i = 0; i < num_unique; i++) {
            uint64_t start_bound = unique_c_idxs[i];
            uint64_t end_bound = unique_c_idxs[i+1];

            cblas_dsymm(
                CblasRowMajor,
                CblasRight,
                CblasUpper,
                end_bound - start_bound,
                R,
                1.0,
                G(0), R,
                scaled_h_permuted(start_bound, 0), R,
                0.0,
                temp1(start_bound, 0), R
                );
        }

        //execute_mkl_dsymv_batch();

        batch_dot_product(
            scaled_h_permuted(), 
            temp1(), 
            m(),
            J, R 
            );

        for(uint32_t c_level = 0; c_level < lfill_level; c_level++) {
            // Prepare to compute m(L(v)) for all v

            #pragma omp single
            {
                uint64_t* end_range = 
                    std::unique_copy(std::execution::par_unseq,
                        c_idxs(),
                        c_idxs(J),
                        unique_c_idxs(),
                        [&](uint64_t a, uint64_t b) {
                            return c[a] == c[b];
                        }
                        );

                num_unique = end_range - unique_c_idxs();
                unique_c_idxs[num_unique] = J;

                cout << num_unique << " " << node_count << endl;

                for(uint64_t i = 0; i < J-1; i++) {
                    if(c[i] > c[i+1]) {
                        throw std::runtime_error("c not sorted");
                    }
                }
            }

            #pragma omp for 
            for(uint64_t i = 0; i < num_unique; i++) {
                uint64_t start_bound = unique_c_idxs[i];
                uint64_t end_bound = unique_c_idxs[i+1];

                cblas_dsymm(
                    CblasRowMajor,
                    CblasRight,
                    CblasUpper,
                    end_bound - start_bound,
                    R,
                    1.0,
                    G((2 * c[i] + 1) * R2), R,
                    scaled_h_permuted(start_bound, 0), R,
                    0.0,
                    temp1(start_bound, 0), R
                    );
            }

            #pragma omp for
            for(int64_t i = 0; i < J; i++) {
                a_array[i] = G((2 * c[i] + 1) * R2); 
            }

            execute_mkl_dsymv_batch();

            batch_dot_product(
                scaled_h_permuted(), 
                temp1(), 
                mL(),
                J, R 
                );

            #pragma omp for
            for(int64_t i = 0; i < J; i++) {
                double cutoff = low[i] + mL[i] / m[i];
                if(random_permuted[i] <= cutoff) {
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
            #pragma omp for
            for(int64_t i = 0; i < J; i++) {
                a_array[i] = is_leaf(c[i]) ? a_array[i] : G((2 * c[i] + 1) * R2); 
            }

            execute_mkl_dsymv_batch();

            batch_dot_product(
                scaled_h_permuted(), 
                temp1(), 
                mL(),
                J, R 
                );

            #pragma omp for
            for(int64_t i = 0; i < J; i++) {
                double cutoff = low[i] + mL[i] / m[i];
                if((! is_leaf(c[i])) && random_permuted[i] <= cutoff) {
                    c[i] = 2 * c[i] + 1;
                    high[i] = cutoff;
                }
                else if((! is_leaf(c[i])) && random_permuted[i] > cutoff) {
                    c[i] = 2 * c[i] + 2;
                    low[i] = cutoff;
                }
            }
        }

        // We will use the m array as a buffer 
        // for the draw fractions.
        if(F > 1) {
            #pragma omp for
            for(int i = 0; i < J; i++) {
                m[i] = (random_permuted[i] - low[i]) / (high[i] - low[i]);

                int64_t leaf_idx;
                if(c[i] >= nodes_upto_lfill) {
                    leaf_idx = c[i] - nodes_upto_lfill;
                }
                else {
                    leaf_idx = c[i] - complete_level_offset; 
                }

                a_array[i] = U(leaf_idx * F, 0);
                y_array[i] = q(i * F);
                
                std::fill(q(i * F), q((i + 1) * F), 0.0);
                uint64_t row_count = min((uint64_t) F, U.shape[0] - leaf_idx * F);

                cblas_dgemv(CblasRowMajor, 
                        CblasNoTrans,
                        row_count,
                        R, 
                        1.0, 
                        (const double*) scratch.a_array[i],
                        R, 
                        (const double*) scratch.x_array[i], 
                        1, 
                        0.0, 
                        scratch.y_array[i], 
                        1);
            }
        }
        
        #pragma omp for
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

            int64_t idx = res + leaf_idx(c[i]) * F;
            samples[sort_idxs[i]] = idx;
            
            for(int64_t j = 0; j < R; j++) {
                h[sort_idxs[i] * R + j] *= U[idx * R + j];
            }  
        }
}
    }
};