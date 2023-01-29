export CC=gcc
export CXX=g++

# Paths to BLAS and TBB libraries 
BLAS_LIB=<PATH_TO_OPENBLAS_LIB>
TBB_LIB=<PATH_TO_TBB_LIB>

# Change to PATH on MacOS
export LD_LIBRARY_PATH=$BLAS_LIB:$TBB_LIB:$LD_LIBRARY_PATH

# Set based on your system configuration 
export OMP_NUM_THREADS=16      
export OMP_MAX_ACTIVE_LEVELS=2

# Avoids oversubscription of resources due to nested
# parallelism. Critical for good performance!
export OPENBLAS_NUM_THREADS=1

# This flag might improve performance 
#export KMP_BLOCKTIME=0