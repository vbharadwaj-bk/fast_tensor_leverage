#pragma once

#include <cassert>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <initializer_list>
#include <chrono>
#include "omp.h"
#include "cblas.h"
#include "lapacke.h"

using namespace std;
namespace py = pybind11;

inline uint32_t divide_and_roundup(uint32_t n, uint32_t m) {
    return (n + m - 1) / m;
}

inline void log2_round_down(uint32_t m, 
        uint32_t& log2_res, 
        uint32_t& lowest_power_2) {
    
    assert(m > 0);
    log2_res = 0;
    lowest_power_2 = 1;

    while(lowest_power_2 * 2 <= m) {
        log2_res++; 
        lowest_power_2 *= 2;
    }
}

//#pragma GCC visibility push(hidden)
template<typename T>
class NumpyArray {
public:
    py::buffer_info info;
    T* ptr;

    NumpyArray(py::array_t<T> arr_py) {
        info = arr_py.request();
        ptr = static_cast<T*>(info.ptr);
    }
};

//#pragma GCC visibility push(hidden)
template<typename T>
class __attribute__((visibility("hidden"))) Buffer {
    py::buffer_info info;
    T* ptr;
    bool own_memory;
    uint64_t dim0;
    uint64_t dim1;

public:
    vector<uint64_t> shape;

    Buffer(Buffer&& other)
        :   info(std::move(other.info)), 
            ptr(std::move(other.ptr)),
            own_memory(other.own_memory),
            dim0(other.dim0),
            dim1(other.dim1),
            shape(std::move(other.shape))
    {}
    Buffer& operator=(const Buffer& other) = default;

    Buffer(py::array_t<T> arr_py) {
        info = arr_py.request();
        ptr = static_cast<T*>(info.ptr);

        if(info.ndim == 2) {
            dim0 = info.shape[0];
            dim1 = info.shape[1];
        }

        for(int64_t i = 0; i < info.ndim; i++) {
            shape.push_back(info.shape[i]);
        }
        own_memory = false;
    }

    Buffer(initializer_list<uint64_t> args) {
        uint64_t buffer_size = 1;
        for(uint64_t i : args) {
            buffer_size *= i;
            shape.push_back(i);
        }

        if(args.size() == 2) {
            dim0 = shape[0];
            dim1 = shape[1];
        }

        ptr = (T*) malloc(sizeof(T) * buffer_size);
        own_memory = true;
    }

    Buffer(initializer_list<uint64_t> args, T* ptr) {
        for(uint64_t i : args) {
            shape.push_back(i);
        }

        if(args.size() == 2) {
            dim0 = shape[0];
            dim1 = shape[1];
        }

        this->ptr = ptr;

        own_memory = false;
    }

    T* operator()() {
        return ptr;
    }

    T* operator()(uint64_t offset) {
        return ptr + offset;
    }

    // Assumes that this array is a row-major matrix 
    T* operator()(uint64_t off_x, uint64_t off_y) {
        return ptr + (dim1 * off_x) + off_y;
    }

    T& operator[](uint64_t offset) {
        return ptr[offset];
    }

    ~Buffer() {
        if(own_memory) {
            free(ptr);
        }
    }
};

template<typename T>
class __attribute__((visibility("hidden"))) NPBufferList {
public:
    vector<Buffer<T>> buffers;
    int length;

    NPBufferList(py::list input_list) {
        length = py::len(input_list);
        for(int i = 0; i < length; i++) {
            py::array_t<T> casted = input_list[i].cast<py::array_t<T>>();
            buffers.emplace_back(casted);
        }
    }
};

/*
* exclude is the index of a matrix to exclude from the chain Hadamard product. Pass -1
* to include all components in the chain Hadamard product.
*/
void ATB_chain_prod(
        vector<Buffer<double>> &A,
        vector<Buffer<double>> &B,
        Buffer<double> &sigma_A, 
        Buffer<double> &sigma_B,
        Buffer<double> &result,
        int exclude) {

        uint64_t N = A.size();
        uint64_t R_A = A[0].shape[1];
        uint64_t R_B = B[0].shape[1];

        vector<unique_ptr<Buffer<double>>> ATB;
        for(uint64_t i = 0; i < A.size(); i++) {
                ATB.emplace_back();
                ATB[i].reset(new Buffer<double>({R_A, R_B}));
        }

        for(uint64_t i = 0; i < R_A; i++) {
                for(uint64_t j = 0; j < R_B; j++) {
                        result[i * R_B + j] = sigma_A[i] * sigma_B[j];
                }
        }

        // Can replace with a batch DGEMM call
        for(uint64_t i = 0; i < N; i++) {
            if((int) i != exclude) {
                uint64_t K = A[i].shape[0];
                cblas_dgemm(
                        CblasRowMajor,
                        CblasTrans,
                        CblasNoTrans,
                        R_A,
                        R_B,
                        K,
                        1.0,
                        A[i](),
                        R_A,
                        B[i](),
                        R_B,
                        0.0,
                        (*(ATB[i]))(),
                        R_B
                );
            }
        }

        for(uint64_t k = 0; k < N; k++) {
                if((int) k != exclude) {
                    for(uint64_t i = 0; i < R_A; i++) {
                            for(uint64_t j = 0; j < R_B; j++) {
                                    result[i * R_B + j] *= (*(ATB[k]))[i * R_B + j];
                            }
                    }
                }
        }
}

double ATB_chain_prod_sum(
        vector<Buffer<double>> &A,
        vector<Buffer<double>> &B,
        Buffer<double> &sigma_A, 
        Buffer<double> &sigma_B) {

    uint64_t R_A = A[0].shape[0];
    uint64_t R_B = B[0].shape[0];
    Buffer<double> result({R_A, R_B});
    ATB_chain_prod(A, B, sigma_A, sigma_B, result, -1);
    return std::accumulate(result(), result(R_A * R_B), 0.0); 
}

void compute_pinv_square(Buffer<double> &M, Buffer<double> &out) {
    uint64_t R = M.shape[0];
    double eigenvalue_tolerance = 0.0;
    Buffer<double> lambda({R});

    LAPACKE_dsyev( CblasRowMajor, 
                    'V', 
                    'U', 
                    R,
                    M(), 
                    R, 
                    lambda() );

    for(uint32_t v = 0; v < R; v++) {
        if(lambda[v] > eigenvalue_tolerance) {
            for(uint32_t u = 0; u < R; u++) {
                M[u * R + v] = M[u * R + v] / sqrt(lambda[v]); 
            }
        }
        else {
            for(uint32_t u = 0; u < R; u++) {
                M[u * R + v] = 0.0; 
            }
        }
    }

    cblas_dsyrk(CblasRowMajor, 
                CblasUpper, 
                CblasNoTrans,
                R,
                R, 
                1.0, 
                (const double*) M(), 
                R, 
                0.0, 
                out(), 
                R);

}

void compute_pinv(Buffer<double> &in, Buffer<double> &out) {
    uint64_t R = in.shape[1];
    Buffer<double> M({R, R});

    // Compute pseudo-inverse of the input matrix through dsyrk and eigendecomposition  
    cblas_dsyrk(CblasRowMajor, 
                CblasUpper, 
                CblasTrans,
                R,
                in.shape[0], 
                1.0, 
                in(), 
                R, 
                0.0, 
                M(), 
                R);

    compute_pinv_square(M, out);
}




/*typedef chrono::time_point<std::chrono::steady_clock> my_timer_t; 


my_timer_t start_clock() {
    return std::chrono::steady_clock::now();
}


double stop_clock_get_elapsed(my_timer_t &start) {
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}*/