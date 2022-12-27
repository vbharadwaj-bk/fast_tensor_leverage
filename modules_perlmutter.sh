export CC=gcc
export CXX=g++

conda deactivate
module load python

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/homes/v/vbharadw/OpenBLAS_install/lib

export OMP_MAX_ACTIVE_LEVELS=2 
export OMP_NUM_THREADS=128