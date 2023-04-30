#pragma once

#include <iostream>
#include <vector>
#include <memory>

#include "common.h"
#include "partition_tree.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;

class __attribute__((visibility("hidden"))) TTSampler {
    /*
    * This class is closely integrated with a versioin in Python. 
    */
public:
    vector<unique_ptr<Buffer<double>>> matricizations;
    vector<unique_ptr<PartitionTree>> tree_samplers;
    uint64_t N, J, R_max;
    ScratchBuffer scratch;

    TTSampler(uint64_t N, uint64_t J, uint64_t R_max) 
    :
    N(N),
    J(J),
    R_max(R_max),
    scratch(J, R_max, R_max)
     {
        for(uint64_t i = 0; i < N; i++) {
            matricizations.emplace_back();
            tree_samplers.emplace_back();
        }
    }

    void update_matricization(py::array_t<double> &matricization, uint64_t i) {
        matricizations[i].reset(new Buffer<double>(matricization));
        /*tree_samplers[i].reset(
            new PartitionTree(
                matricizations[i]->shape[0],
                matricizations[i]->shape[1],
                J,
                matricizations[i]->shape[1],
        ));*/
    }
};

