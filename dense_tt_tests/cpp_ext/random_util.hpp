#pragma once

#include <iostream>
#include <vector>
#include <cassert>
#include <random>
#include "common.h"
#include "omp.h"

class Multistream_RNG {
public:
    // Related to independent random number generation on multiple
    // streams
    int thread_count;
    std::random_device rd;  
    vector<std::mt19937> par_gen; 

    Multistream_RNG() :
        rd() 
    {
        // Set up independent random streams for different threads.
        // As written, might be more complicated than it needs to be. 
        #pragma omp parallel
        {
            #pragma omp single 
            {
                thread_count = omp_get_num_threads();
            }
        }

        vector<uint32_t> biased_seeds(thread_count, 0);
        vector<uint32_t> seeds(thread_count, 0);

        for(int i = 0; i < thread_count; i++) {
            biased_seeds[i] = rd();
        }
        std::seed_seq seq(biased_seeds.begin(), biased_seeds.end());
        seq.generate(seeds.begin(), seeds.end());

        for(int i = 0; i < thread_count; i++) {
            par_gen.emplace_back(seeds[i]);
        }
    }
};