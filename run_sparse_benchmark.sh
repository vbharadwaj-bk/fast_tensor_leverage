#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 4:00:00

. env.sh 

export TENSOR=uber
export TARGET_RANK=5
export SAMPLES=65536
export ITERATIONS=20
export ALG=random
export TRIAL_COUNT=3

#srun -N 1 -u 

python tt_test.py \
        -i $TENSOR \
        --trank $TARGET_RANK \
        -s      $SAMPLES \
        -iter   $ITERATIONS \
        -alg    $ALG \
        -r      $TRIAL_COUNT \
        -o      outputs/sparse_tensor_train
