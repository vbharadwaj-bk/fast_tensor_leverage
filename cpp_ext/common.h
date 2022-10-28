#pragma once

#include <cassert>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>

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

    while(lowest_power_2 * 2 < m) {
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

template<typename T>
class NumpyList {
public:
    vector<py::buffer_info> infos;
    vector<T*> ptrs;
    int length;

    NumpyList(py::list input_list) {
        length = py::len(input_list);
        for(int i = 0; i < length; i++) {
            py::array_t<T> casted = input_list[i].cast<py::array_t<T>>();
            infos.push_back(casted.request());
            ptrs.push_back(static_cast<T*>(infos[i].ptr));
        }
    }
};