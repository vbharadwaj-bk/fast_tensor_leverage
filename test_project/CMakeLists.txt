cmake_minimum_required(VERSION 3.14)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CPP_COMPILER g++)
project(test) 

find_package(OpenMP REQUIRED)
add_executable(test test.cpp)
target_include_directories(test PRIVATE /global/homes/v/vbharadw/OpenBLAS_install/include)
target_link_directories(test PRIVATE /global/homes/v/vbharadw/OpenBLAS_install/lib)
target_link_libraries(test openblas OpenMP::OpenMP_CXX)
target_compile_options(test PRIVATE -std=c++2b)

