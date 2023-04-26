#!/bin/bash
#SBATCH -N 8
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 11:00:00
#SBATCH -A m1266 
#SBATCH -q regular 

. modules_perlmutter.sh
srun -N 4 -n 4 -u python sparse_tensor_benchmark.py 
