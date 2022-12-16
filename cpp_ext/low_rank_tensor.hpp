#pragma once

#include <iostream>
#include <string>
#include "common.h"
#include "cblas.h"
#include "lapacke.h"
#include "tensor.hpp"

using namespace std;

class __attribute__((visibility("hidden"))) LowRankTensor : public Tensor {
public:
    vector<uint64_t> dims;
    unique_ptr<NPBufferList<double>> U_py_bufs;
    vector<Buffer<double>> &U;
    uint32_t N;
    uint64_t R, J;
    uint64_t max_rhs_rows;

    // This is a pointer, since it will be dynamically allocated
    // based on the tensor mode 
    Buffer<double>* rhs_buf; 
    Buffer<double> partial_evaluation;
    Buffer<double> sigma;
    Buffer<double> col_norms;

    bool is_static;
    double normsq;

    LowRankTensor(uint64_t R, uint64_t J,
        uint64_t max_rhs_rows, 
        py::list U_py
        )
    :
    U_py_bufs(new NPBufferList<double>(U_py)),
    U(U_py_bufs->buffers),
    partial_evaluation({J, R}),
    sigma({R}),
    col_norms({(uint64_t) U_py_bufs->length, R})
    {
        this->max_rhs_rows = max_rhs_rows;
        this->J = J;
        this->R = R;
        this->N = U_py_bufs->length;

        std::fill(sigma(), sigma(R), 1.0);
        std::fill(col_norms(), col_norms(N * R), 1.0);
        for(uint32_t i = 0; i < N; i++) {
            dims.push_back(U_py_bufs->buffers[i].shape[0]);
        }

        // This is a static tensor. We will compute and store the norm^2
        is_static = true;

        get_sigma(sigma, -1);
        normsq = ATB_chain_prod_sum(U, U, sigma, sigma);
    }

    LowRankTensor(uint64_t R, py::list U_py)
    :
    U_py_bufs(new NPBufferList<double>(U_py)),
    U(U_py_bufs->buffers),
    partial_evaluation({1}),
    sigma({R}),
    col_norms({(uint64_t) U_py_bufs->length, R})
    {
        this->R = R;
        this->N = U_py_bufs->length;
        std::fill(sigma(), sigma(R), 1.0);
        std::fill(col_norms(), col_norms(N * R), 1.0);
        for(uint32_t i = 0; i < N; i++) {
            dims.push_back(U_py_bufs->buffers[i].shape[0]);
        }

        is_static = false;
    }

    double get_normsq() {
        if (! is_static) { 
            get_sigma(sigma, -1); 
            normsq = ATB_chain_prod_sum(U, U, sigma, sigma);
        }
        return normsq; 
    };

    double compute_residual_normsq(Buffer<double> &sigma_other, vector<Buffer<double>> &U_other) {
        get_sigma(sigma, -1); 
        double self_normsq = get_normsq();
        double other_normsq = ATB_chain_prod_sum(U_other, U_other, sigma_other, sigma_other);
        double inner_prod = ATB_chain_prod_sum(U_other, U, sigma_other, sigma);

        return max(self_normsq + other_normsq - 2 * inner_prod, 0.0);
    }

    // Convenience method for RHS sampling 
    void materialize_partial_evaluation(Buffer<uint64_t> &samples, 
        uint64_t j) {
        get_sigma(sigma, -1);

        #pragma omp parallel for 
        for(uint32_t i = 0; i < J; i++) {
            std::copy(sigma(), sigma(R), partial_evaluation(i * R));
            for(uint32_t k = 0; k < N; k++) {
                if(k != j) {
                    for(uint32_t u = 0; u < R; u++) {
                        partial_evaluation[i * R + u] *= U_py_bufs->buffers[k][samples[k * J + i] * R + u];
                    }
                } 
            }
        }
    }

    /*
    * Fills rhs_buf with an evaluation of the tensor starting from the
    * specified index in the array of samples. 
    */
    void materialize_rhs(Buffer<uint64_t> &samples, uint64_t j, uint64_t row_pos) {
        Buffer<double> &temp_buf = (*rhs_buf);
        uint64_t max_range = min(row_pos + max_rhs_rows, J);
        uint32_t M = (uint32_t) (max_range - row_pos);
        uint32_t N = (uint32_t) dims[j]; 

        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            M,
            N,
            R,
            1.0,
            partial_evaluation(row_pos, 0),
            R,
            U_py_bufs->buffers[j](),
            R,
            0.0,
            temp_buf(),
            N
        );
    }

    void preprocess(Buffer<uint64_t> &samples, uint64_t j) {
        materialize_partial_evaluation(samples, j);
    }

    void execute_downsampled_mttkrp(
            Buffer<uint64_t> &samples, 
            Buffer<double> &lhs,
            uint64_t j,
            Buffer<double> &result
            ) {
        
        rhs_buf = new Buffer<double>({max_rhs_rows, dims[j]});
        Buffer<double> &temp_buf = (*rhs_buf);
        preprocess(samples, j);

        // Result is a dims[j] x R matrix
        std::fill(result(), result(dims[j], 0), 0.0);

        for(uint64_t i = 0; i < J; i += max_rhs_rows) {
            uint64_t max_range = min(i + max_rhs_rows, J);
            uint32_t rows = (uint32_t) (max_range - i);

            materialize_rhs(samples, j, i);

            // Need to fix this when the ranks are different! 
            cblas_dgemm(
                CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                (uint32_t) dims[j],
                (uint32_t) lhs.shape[1],
                (uint32_t) rows,
                1.0,
                temp_buf(),
                (uint32_t) dims[j],
                lhs(i, 0),
                (uint32_t) lhs.shape[1],
                1.0,
                result(),
                (uint32_t) lhs.shape[1]
            );
        }
        delete rhs_buf; 
    }

    void execute_exact_mttkrp(vector<Buffer<double>> &U_L, uint64_t j, Buffer<double> &mttkrp_res) {
        uint64_t R_L = U_L[0].shape[1];
        Buffer<double> sigma({R});
        Buffer<double> chain_had_prod({R, R_L});
        get_sigma(sigma, -1);

        Buffer<double> ones({R_L});
        std::fill(ones(), ones(R_L), 1.0);

        ATB_chain_prod(
                U,
                U_L,
                sigma,
                ones,
                chain_had_prod,
                j);

        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            (uint32_t) U[j].shape[0],
            (uint32_t) U_L[j].shape[1],
            (uint32_t) R,
            1.0,
            U[j](),
            (uint32_t) R,
            chain_had_prod(),
            R_L,
            0.0,
            mttkrp_res(),
            R_L);

        /*cout << "---------------------------------" << endl; 
        for(uint64_t i = 0; i < U[j].shape[0]; i++) {
            for(uint64_t k = 0; k < R_L; k++) {
                cout << mttkrp_res[i * R_L + k] << " ";
            }
            cout << endl;
        }
        cout << "---------------------------------" << endl;*/

    }

    /*
    * Returns the product of all column norms, except for the
    * one specified by the parameter. If the parameter is -1,
    * the product of all column norms is returned. 
    */
    void get_sigma(Buffer<double> &sigma_out, int j) {
        std::fill(sigma_out(), sigma_out(R), 1.0);
        for(uint32_t i = 0; i < N; i++) {
            if((int) i != j) {
                for(uint32_t v = 0; v < R; v++) { 
                    sigma_out[v] *= col_norms[i * R + v];
                }
            }
        }
    }

    void get_sigma_py(py::array_t<double> sigma_py_out, int j) {
        Buffer<double> sigma_py(sigma_py_out);
        get_sigma(sigma_py, j);
    }

    // Pass j = -1 to renormalize all factor matrices.
    // The array sigma is always updated 
    void renormalize_columns(int j) {
        std::fill(sigma(), sigma(R), 1.0);

        #pragma omp parallel
{
        for(int i = 0; i < (int) N; i++) {
            if(j == -1 || j == i) {

                #pragma omp critical
                {
                std::fill(col_norms(i * R), 
                    col_norms((i + 1) * R), 0.0);
                }

                Buffer<double> thread_loc_norms({R});
                std::fill(thread_loc_norms(), 
                    thread_loc_norms(R), 0.0);

                #pragma omp for 
                for(uint32_t u = 0; u < dims[i]; u++) {
                    for(uint32_t v = 0; v < R; v++) {
                        double entry = U[i][u * R + v];
                        thread_loc_norms[v] += entry * entry; 
                    }
                }

                for(uint32_t v = 0; v < R; v++) {
                    #pragma omp atomic 
                    col_norms[i * R + v] += thread_loc_norms[v]; 
                }

                #pragma omp barrier

                #pragma omp for
                for(uint32_t v = 0; v < R; v++) { 
                    col_norms[i * R + v] = sqrt(col_norms[i * R + v]);
                }

                #pragma omp for collapse(2)
                for(uint32_t u = 0; u < dims[i]; u++) {
                    for(uint32_t v = 0; v < R; v++) {
                        U[i][u * R + v] /= col_norms[i * R + v];
                    }
                }
            }
        }
}
    }
};