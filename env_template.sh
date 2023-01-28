export CC=gcc
export CXX=g++

# Paths to BLAS and TBB libraries 
BLAS_LIB=<PATH_TO_OPENBLAS_LIB>
TBB_LIB=<PATH_TO_TBB_LIB>

export LD_LIBRARY_PATH=$BLAS_LIB:$TBB_LIB:$LD_LIBRARY_PATH

# Set based on your system configuration 
export OMP_NUM_THREADS=16      
export OMP_MAX_ACTIVE_LEVELS=2 

# This flag might improve performance 
#export KMP_BLOCKTIME=0