#!/bin/bash
#SBATCH -N 4
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 04:00:00

. modules_perlmutter.sh
srun -N 4 -n 4 python als_test.py 
