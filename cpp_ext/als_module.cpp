//cppimport
#include <iostream>
#include "common.h"
#include "cblas.h"
#include "lapacke.h"
#include "efficient_krp_sampler.hpp"

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
        std::fill(sigma(), sigma(R), 1.0);
        this->max_rhs_rows = max_rhs_rows;
        this->J = J;
        this->R = R;
        this->N = U_py_bufs->length;
        for(uint32_t i = 0; i < N; i++) {
            dims.push_back(U_py_bufs->buffers[i].shape[0]);
        }
    }

    LowRankTensor(uint64_t R, 
            py::list U_py)
    :
    U_py_bufs(new NPBufferList<double>(U_py)),
    U(U_py_bufs->buffers),
    partial_evaluation({1}),
    sigma({R}),
    col_norms({(uint64_t) U_py_bufs->length, R})
    {
        std::fill(sigma(), sigma(R), 1.0);
        this->R = R;
        this->N = U_py_bufs->length;
        for(uint32_t i = 0; i < N; i++) {
            dims.push_back(U_py_bufs->buffers[i].shape[0]);
        }
    }

    // Convenience method for RHS sampling 
    void materialize_partial_evaluation(Buffer<uint64_t> &samples, 
        uint64_t j) {

        // Assume sample matrix is N x J, result is J x R
        std::fill(partial_evaluation(), partial_evaluation(J, 0), 1.0);

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

            cblas_dgemm(
                CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                (uint32_t) dims[j],
                (uint32_t) R,
                (uint32_t) rows,
                1.0,
                temp_buf(),
                (uint32_t) dims[j],
                lhs(i, 0),
                (uint32_t) R,
                1.0,
                result(),
                (uint32_t) R
            );
        }
        delete rhs_buf; 
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
                    sigma[v] *= col_norms[i * R + v];
                }

                #pragma omp for collapse(2)
                for(uint32_t u = 0; u < dims[i]; u++) {
                    for(uint32_t v = 0; v < R; v++) {
                        U[i][u * R + v] /= col_norms[i * R + v];
                    }
                }

            }

            #pragma omp for
            for(uint32_t v = 0; v < R; v++) { 
                sigma[v] *= col_norms[i * R + v];
            }
        }
} 
    }
};

void compute_DAGAT(double* A, double* G, 
        double* res, uint64_t J, uint64_t R) {

    Buffer<double> temp({J, R});

    cblas_dsymm(
        CblasRowMajor,
        CblasRight,
        CblasUpper,
        (uint32_t) J,
        (uint32_t) R,
        1.0,
        G,
        R,
        A,
        R,
        0.0,
        temp(),
        R
    );

    for(uint32_t i = 0; i < J; i++) {
        res[i] = 0.0;
        for(uint32_t j = 0; j < R; j++) {
            res[i] += A[i * R + j] * temp[i * R + j];
        }
    }
}

class __attribute__((visibility("hidden"))) ALS {
public:
    LowRankTensor &cp_decomp;
    Tensor &ground_truth;

    // These fields used only for sampling 
    uint64_t J;
    unique_ptr<EfficientKRPSampler> sampler;
    unique_ptr<Buffer<uint64_t>> samples;

    ALS(LowRankTensor &cp_dec, Tensor &gt) : 
        cp_decomp(cp_dec),
        ground_truth(gt)    
    {
        // Empty
    }

    void initialize_ds_als(uint64_t J) {
        sampler.reset(new EfficientKRPSampler(J, cp_decomp.R, cp_decomp.U));
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
        Buffer<double> weights({J});

        sampler->KRPDrawSamples(j, *samples, nullptr);

        compute_DAGAT(
            sampler->h(),
            sampler->M(),
            weights(),
            J,
            R
        );

        for(uint32_t i = 0; i < J; i++) {
            weights[i] = (double) R / (weights[i] * J);
        }

        #pragma omp parallel for
        for(uint32_t i = 0; i < J; i++) {
            for(uint32_t t = 0; t < R; t++) {
                sampler->h[i * R + t] *= weights[i]; 
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
            sampler->M(),
            R,
            mttkrp_res(),
            R,
            0.0,
            cp_decomp.U[j](),
            R
        );

        // Multiply result by sigma^(-1) of the CP
        // decomposition. Assumes that sigma is correct
        // upon entry to this function. 
        #pragma omp parallel for collapse(2)
        for(uint32_t u = 0; u < Ij; u++) {
            for(uint32_t v = 0; v < R; v++) {
                cp_decomp.U[j][u * R + v] /= sigma[v]; 
            }
        }

        if(renormalize) {
            cp_decomp.renormalize_columns(j);
        }

    }
};

PYBIND11_MODULE(als_module, m) {
    py::class_<Tensor>(m, "Tensor")
        .def("execute_downsampled_mttkrp_py", &Tensor::execute_downsampled_mttkrp_py);
    py::class_<LowRankTensor, Tensor>(m, "LowRankTensor")
        .def(py::init<uint64_t, uint64_t, uint64_t, py::list>()) 
        .def(py::init<uint64_t, py::list>())
        .def("renormalize_columns", &LowRankTensor::renormalize_columns);
    py::class_<ALS>(m, "ALS")
        .def(py::init<LowRankTensor&, Tensor&>()) 
        .def("initialize_ds_als", &ALS::initialize_ds_als) 
        .def("execute_ds_als_update", &ALS::execute_ds_als_update);
}

/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['--std=c++2b', '-I/global/homes/v/vbharadw/OpenBLAS_install/include', '-fopenmp', '-O3']
cfg['extra_link_args'] = ['-L/global/homes/v/vbharadw/OpenBLAS_install/lib', '-lopenblas', '-fopenmp', '-O3']
cfg['dependencies'] = ['common.h', 'partition_tree.hpp', 'efficient_krp_sampler.hpp'] 
%>
*/
