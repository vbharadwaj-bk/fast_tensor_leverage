module load python

srun -N 1 -n 1 -u python tensorly_benchmark.py --tensor nell-2 --folder /pscratch/sd/v/vbharadw/fast_tensor_leverage/outputs/tensorly_benchmarks/

