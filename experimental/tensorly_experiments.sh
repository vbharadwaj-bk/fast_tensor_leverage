module load python

python tensorly_benchmark.py $SCRATCH/tensors/uber.tns_converted.hdf5 \
    # --preprocessing \
    --target_rank 25 \
    --output_filename ../outputs/tensorly_benchmarks/uber_25.json \
    --num_repetitions 5

