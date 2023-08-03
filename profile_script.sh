TENSOR_LOC=/pscratch/sd/v/vbharadw/tensors
SPLATT_LOC=/global/cfs/projectdirs/m1982/vbharadw/splatt/build/Linux-x86_64/bin
export PAT_RT_EXPDIR_NAME=perf_measurements

TENSOR=amazon-reviews-spl-binary.bin
RANK=25
MAX_ITER=5
TOL=1e-5

export PAT_RT_PERFCTR="PAPI_L2_DCM"
SPLATT_OUTPUT=~/splatt_cache_measurements


rm -r $PAT_RT_EXPDIR_NAME
export OMP_NUM_THREADS=4
srun -n 32 -u $SPLATT_LOC/splatt+pat cpd $TENSOR_LOC/$TENSOR -r $RANK \
    --nowrite -i $MAX_ITER --tol $TOL > $SPLATT_OUTPUT/splatt_output-32proc.txt
pat_report perf_measurements/ > $SPLATT_OUTPUT/report_32proc.txt
