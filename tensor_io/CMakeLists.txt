cmake_minimum_required(VERSION 3.16)

project(tensor_io)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

add_executable(log_values log_values.cpp)
target_compile_options(log_values PRIVATE -g) 