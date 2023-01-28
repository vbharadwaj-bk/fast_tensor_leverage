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
We strongly recommended that you install Intel
Thread Building Blocks (TBB), but this is not
required. We rely on Intel TBB for fast parallel 
sorting during sparse tensor decomposition.

Our C++ code could stand alone, but right now, you
need Python as well. 

## Building our code
You can build our code in two easy steps (after
cloning the repository): installing Python
dependencies, then configuring the C++ extension.

### Step 0: Clone the Repository
Clone the repoitory and `cd` into it. The exact
name of the repository has been omitted below to
preserve anonymity during the reviewing process.
```
git clone <FULL PATH HERE>/fast_tensor_leverage.git
cd fast_tensor_leverage
```

### Step 1: Install Python packages
Install Python dependencies with the following command:
```
pip install -r requirements.txt
```
We rely on the Pybind11 and cppimport packages. We
use the HDF5 format to store sparse tensors, so
you need the h5py package if you want to perform
sparse tensor decomposition. 

### Step 2: Configure the compile and runtime environments 
Within the repository, run the following command:
```
python configure.py
```
This will create two files in the repository root:
`config.json` and `env.sh`. Edit the configuration
JSON file with include / link flags for your LAPACK
install. If you have TBB installed (strongly
recommended for good performance). If you do not 
have TBB installed, set these JSON entries to
`null`.

The file `env.sh` sets up the runtime environment,
and must be called every time you start a new shell 
to run our code. First, set the variables CC
and CXX to your C and C++ compilers. The C++ extension
module is compiled with
these when it is 
imported by Python
at runtime. 

Next, update your LD_LIBRARY_PATH 
to include the BLAS and TBB library folders using
`env.sh`. You should also set the number of
OpenMP threads based on your system configuration. 

### Step 3: Test the code 
You're ready to test! The C++ extension
compiles automatically the first time you run
the code, and is not compiled subsequently.
