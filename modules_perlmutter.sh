export CC=gcc
export CXX=g++

conda deactivate
module load python
module load pytorch

# Both of these flags appear to degrade performance...
#export OMP_PLACES=cores
#export OMP_PROC_BIND=close

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/homes/v/vbharadw/OpenBLAS_install/lib:/global/homes/v/vbharadw/intel/oneapi/tbb/2021.8.0/lib/intel64/gcc4.8

export OMP_MAX_ACTIVE_LEVELS=2 
export OMP_NUM_THREADS=128
export KMP_BLOCKTIME=0

export TORCH_DATASET_FOLDER=/pscratch/sd/v/vbharadw/torch_datasets

