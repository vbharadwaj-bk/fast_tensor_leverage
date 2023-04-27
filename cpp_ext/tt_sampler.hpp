#pragma once

#include <iostream>
#include <vector>
#include <memory>

using namespace std;

class __attribute__((visibility("hidden"))) TTSampler {
    /*
    * This class is closely integrated with a versioin in Python. 
    */
public:
    vector<unique_ptr<Buffer<double>> matricizations;
    uint64_t N;


    TTSampler(uint64_t N) 
    :
    N(N) {
        // Empty for now
    }
};




