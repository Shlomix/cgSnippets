#!/usr/bin/env bash

# ==============================================
# Configuration for a single-node system with 8 GPUs
# ==============================================

## DL parameters
export BATCHSIZE=32
export NUMEPOCHS=${NUMEPOCHS:-8}
export DATASET_DIR="/datasets/open-images-v6-mlperf"
export EXTRA_PARAMS='--lr 0.0001 --output-dir=/results'

## System configuration (single-node)
# No need for DGX-specific variables, but we will use 8 GPUs locally
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ==============================================
# Run training script
# ==============================================

set -e

# Start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# Set training parameters
BATCHSIZE=${BATCHSIZE:-32}
EVALBATCHSIZE=${EVALBATCHSIZE:-${BATCHSIZE}}
NUMEPOCHS=${NUMEPOCHS:-8}
LOG_INTERVAL=${LOG_INTERVAL:-20}
DATASET_DIR=${DATASET_DIR:-"/datasets/open-images-v6-mlperf"}
TORCH_HOME=${TORCH_HOME:-"$(pwd)/torch-model-cache"}

# Setup torchrun command for local multi-GPU
CMD=( "torchrun" "--standalone" "--nnodes=1" "--nproc_per_node=8" )

PARAMS=(
    --batch-size              "${BATCHSIZE}"
    --eval-batch-size         "${EVALBATCHSIZE}"
    --epochs                  "${NUMEPOCHS}"
    --print-freq              "${LOG_INTERVAL}"
    --data-path               "${DATASET_DIR}"
)

# Run the training
"${CMD[@]}" train.py "${PARAMS[@]}" ${EXTRA_PARAMS} ; ret_code=$?

# End timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# Report result
result=$(( end - start ))
result_name="SINGLE_NODE_8GPU_TRAINING"

echo "RESULT,$result_name,,$result,nvidia,$start_fmt"

# Exit with the correct code
if [[ $ret_code != 0 ]]; then exit $ret_code; fi
