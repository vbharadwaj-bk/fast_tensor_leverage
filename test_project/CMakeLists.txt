cmake_minimum_required(VERSION 3.14)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CPP_COMPILER g++)
project(test) 

add_executable(test test.cpp)
target_include_directories(test PRIVATE /global/homes/v/vbharadw/intel/oneapi/tbb/2021.8.0/include)
target_link_directories(test PRIVATE /global/homes/v/vbharadw/intel/oneapi/tbb/2021.8.0/lib/intel64/gcc4.8)
target_link_libraries(test tbb)
#target_link_libraries(test openblas OpenMP::OpenMP_CXX)
target_compile_options(test PRIVATE -std=c++17 -O3 -fopenmp -ltbb)

