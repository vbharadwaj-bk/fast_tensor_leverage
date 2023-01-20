export CC=gcc
export CXX=g++

OPENBLAS_LIB=/home/vbharadw/OpenBLAS_install/lib
TBB_LIB=/home/vbharadw/intel/oneapi/tbb/2021.8.0/lib/intel64/gcc4.8

export LD_LIBRARY_PATH=$OPENBLAS_LIB:$TBB_LIB:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=$BLIS_LIB:$FLAME_LIB:$TBB_LIB:$LD_LIBRARY_PATH

export OMP_MAX_ACTIVE_LEVELS=2 
export OMP_NUM_THREADS=128
#export KMP_BLOCKTIME=0

export TORCH_DATASET_FOLDER=/pscratch/sd/v/vbharadw/torch_datasets

