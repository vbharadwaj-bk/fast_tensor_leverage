export CC=gcc
export CXX=g++

conda deactivate
module load python

# Both of these flags appear to degrade performance...
#export OMP_PLACES=cores
#export OMP_PROC_BIND=close

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/homes/v/vbharadw/OpenBLAS_install/lib

export OMP_MAX_ACTIVE_LEVELS=2 
export OMP_NUM_THREADS=128
export KMP_BLOCKTIME=0