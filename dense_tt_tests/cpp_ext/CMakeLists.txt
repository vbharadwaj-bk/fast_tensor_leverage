cmake_minimum_required(VERSION 3.14)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CPP_COMPILER g++)

project(efficient_krp_sampler) 

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_FIND_LIBRARY_PREFIXES lib)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
find_package(OpenMP REQUIRED)

add_executable(sampler_test 
    main.cpp
    common.h
)

#message("Hello!")
#message("${pybind11_INCLUDE_DIR}")
#message("Hello!")

target_include_directories(sampler_test
    PRIVATE
    ${Python_INCLUDE_DIRS}
    /opt/OpenBLAS/include/
)

target_link_directories(sampler_test
    PRIVATE
    pybind11::pybind11
    pybind11::headers
    #${Python_LIBRARIES}
    #${Python_LIBRARY_DIRS}
    #${Python_RUNTIME_LIBRARY_DIRS}
    /opt/OpenBLAS/lib/
)

target_link_libraries(sampler_test 
    openblas
    OpenMP::OpenMP_CXX
    pybind11::pybind11
    pybind11::headers
)

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

