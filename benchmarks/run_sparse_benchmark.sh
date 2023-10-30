#!/bin/bash
#SBATCH -N 8
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 10:00:00
#SBATCH -A m4293
#SBATCH -q regular 

. modules_perlmutter.sh
FOLDER=may_10

#TENSOR=uber
#TENSOR=enron
#TENSOR=nell-2
#TENSOR=amazon-reviews
TENSOR=reddit-2015
srun -N 8 -n 8 -u python sparse_tensor_benchmark.py --tensor $TENSOR --folder $FOLDER 
