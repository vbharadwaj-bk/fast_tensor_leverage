#!/bin/bash
#SBATCH -N 8
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -A m1266 

. modules_perlmutter.sh
srun -N 8 -n 8 -u python sparse_tensor_benchmark.py 
