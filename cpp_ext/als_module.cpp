//cppimport
#include <iostream>
#include "common.h"
#include "cblas.h"
#include "lapacke.h"

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
    uint32_t N;
    uint64_t R, J;
    uint64_t max_rhs_rows;

    // This is a pointer, since it will be dynamically allocated
    // based on the tensor mode 
    Buffer<double>* rhs_buf; 
    Buffer<double> partial_evaluation;

    LowRankTensor(uint64_t R, uint64_t J,
        uint64_t max_rhs_rows, 
        py::list U_py)
    :
    U_py_bufs(new NPBufferList<double>(U_py)),
    partial_evaluation({J, R})
    {
        this->max_rhs_rows = max_rhs_rows;
        this->J = J;
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

            cout << "--------------------------------" << endl;
            for(uint32_t k = 0; k < rows; k++) {
                for(uint32_t p = 0; p < dims[j]; p++) {
                    cout << temp_buf[k * dims[j] + p] << " ";
                }
                cout << endl;
            }
            cout << "--------------------------------" << endl;

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
};

class __attribute__((visibility("hidden"))) ALS {
public:
    void test(Tensor &t) {
        //t.execute_downsampled_mttkrp();
    }
};

PYBIND11_MODULE(als_module, m) {
    py::class_<Tensor>(m, "Tensor")
        .def("execute_downsampled_mttkrp_py", &Tensor::execute_downsampled_mttkrp_py);
    py::class_<LowRankTensor, Tensor>(m, "LowRankTensor")
        .def(py::init<uint64_t, uint64_t, uint64_t, py::list>());
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
