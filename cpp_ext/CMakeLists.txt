cmake_minimum_required(VERSION 3.14)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CPP_COMPILER g++)
project(efficient_krp_sampler) 

add_executable(sampler_test main.cpp)

#common.h
#partition_tree.hpp
#efficient_krp_sampler.hpp
#sampler.hpp 
#uniform_sampler.hpp
#larsen_kolda_sampler.hpp
#larsen_kolda_hybrid.hpp
#low_rank_tensor.hpp
#sparse_tensor.hpp
#black_box_tensor.hpp
#tensor.hpp
#idx_lookup.hpp
#hash_lookup.hpp
#sort_lookup.hpp
#dense_tensor.hpp
#tests.hpp
#als.hpp

#target_include_directories(test PRIVATE /global/homes/v/vbharadw/intel/oneapi/tbb/2021.8.0/include)
#target_link_directories(test PRIVATE /global/homes/v/vbharadw/intel/oneapi/tbb/2021.8.0/lib/intel64/gcc4.8)
#target_link_libraries(test tbb)
#target_link_libraries(test openblas OpenMP::OpenMP_CXX)
#target_compile_options(test PRIVATE -std=c++17 -O3 -fopenmp -ltbb)

