#!/bin/bash
#SBATCH -N 4
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 04:30:00

. modules_perlmutter.sh
srun -N 4 -n 4 -u python sparse_tensor_benchmark.py 
