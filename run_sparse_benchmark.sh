#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 7:00:00

. env.sh 

export SAMPLES=65536
export ITERATIONS=40
export TRIAL_COUNT=5
export EPOCH_INTERVAL=2

for TENSOR in uber enron nell-2
do
   for TARGET_RANK in 4 6 8 10 12
   do
      for ALG in exact random
      do
        python tt_test.py \
                -i $TENSOR \
                --trank $TARGET_RANK \
                -s      $SAMPLES \
                -iter   $ITERATIONS \
                -alg    $ALG \
                -r      $TRIAL_COUNT \
                -e      $EPOCH_INTERVAL \
                -o      outputs/sparse_tensor_train
      done
   done
done
