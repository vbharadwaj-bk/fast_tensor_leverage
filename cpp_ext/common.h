#pragma once

#include <cassert>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <initializer_list>
#include <chrono>
#include "omp.h"

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

    /*NumpyArray(py::object obj, string attr_name) {
        py::array_t<T> arr_py = obj.attr(attr_name.c_str()).cast<py::array_t<T>>();
        info = arr_py.request();
        ptr = static_cast<T*>(info.ptr);
    }*/
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
        vector<uint64_t> shape;
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

    Buffer(initializer_list<uint64_t> args, T value) {
        uint64_t buffer_size = 1;
        vector<uint64_t> shape;
        for(uint64_t i : args) {
            buffer_size *= i;
            shape.push_back(i);
        }

        if(args.size() == 2) {
            dim0 = shape[0];
            dim1 = shape[1];
        }

        ptr = (T*) malloc(sizeof(T) * buffer_size);

        #pragma omp parallel for
        for(uint64_t i = 0; i < buffer_size; i++) {
            ptr[i] = value;
        }

        own_memory = true;
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


/*typedef chrono::time_point<std::chrono::steady_clock> my_timer_t; 


my_timer_t start_clock() {
    return std::chrono::steady_clock::now();
}


double stop_clock_get_elapsed(my_timer_t &start) {
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}*/