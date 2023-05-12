#!/bin/bash
#SBATCH -N 24
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH -A m1266 
#SBATCH -q regular 

. modules_perlmutter.sh
FOLDER=may_10

#TENSOR=uber
#TENSOR=enron
#TENSOR=nell-2
#TENSOR=amazon-reviews
TENSOR=reddit-2015
srun -N 24 -n 24 -u python sparse_tensor_benchmark.py --tensor $TENSOR --folder $FOLDER 
