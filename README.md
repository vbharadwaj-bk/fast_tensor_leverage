# Fast Exact Leverage Score Sampling from Khatri-Rao Products with Applications to Tensor Decomposition
This repository contains the code for
the paper ``Fast Exact Leverage Score Sampling
from Khatri-Rao Products". 

This repository contains two implementations of the
data structure in the paper. The first,
available in the folder `reference_implementation`,
is written entirely in Python with Numpy. It is
slow, but the structure of the code matches
the pseudocode in our paper almost line-for-line. You
can use it to verify the correctness of our algorithm
or pick apart how the data structure works.

The second implementation is a fast version written
in C++, with Pybind11 Python bindings 
and compatibility with Numpy for easy 
testing. All benchmarks were conducted
with the fast version, and all instructions detail
how to build the fast version.

This repository contains a copy of `json.hpp`
from Niels Lohmann's repository 
<https://github.com/nlohmann/json>. It is published
under an MIT license.

## Requirements 
You need GCC 11.2+, OpenMP, and an install of the BLAS
and LAPACK. This code has been tested with OpenBLAS.
It is strongly recommended that you install Intel
Thread Building Blocks (TBB), but this is not
required. We rely on Intel TBB for fast parallel 
sorting during sparse tensor decomposition.

Our C++ code could stand alone, but is currently built
to be used within a Python extension in conjunction
with Numpy. As a result, you also need Python.

## Building our code 

## Step 0: Clone the Repository and  

