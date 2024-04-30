#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 4:00:00

. env.sh 
srun -N 1 -u python sparse_tensor_benchmark.py \
        -i $sparse_tensor_benchmark \
        --trank $TARGET_RANK \
        -s      $SAMPLES \
        -iter   $ITERATIONS \
        -alg    $ALG \
        -r      $TRIAL_COUNT \
        -o      outputs/sparse_tensor_train
