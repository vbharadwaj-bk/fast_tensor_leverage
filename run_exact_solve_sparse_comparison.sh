#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 12:00:00

. modules_perlmutter.sh
srun -N 1 -n 1 -u python exact_solve_sparse_comparison_1.py
