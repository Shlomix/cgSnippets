#!/usr/bin/env bash

# ==============================================
# Configuration for a multi-node system with 32 nodes and 256 GPUs
# ==============================================

## DL parameters
export BATCHSIZE=32
export NUMEPOCHS=${NUMEPOCHS:-35}
export DATASET_DIR="/datasets/open-images-v6-mlperf"
export EXTRA_PARAMS='--lr 0.0001 --output-dir=/results'

## System configuration (32-node, 8 GPUs per node)
# No need for DGX-specific variables here, as we handle it with torchrun

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
NUMEPOCHS=${NUMEPOCHS:-35}
LOG_INTERVAL=${LOG_INTERVAL:-20}
DATASET_DIR=${DATASET_DIR:-"/datasets/open-images-v6-mlperf"}
TORCH_HOME=${TORCH_HOME:-"$(pwd)/torch-model-cache"}

# Setup torchrun command for multi-node training
CMD=( "torchrun" "--nnodes=32" "--nproc_per_node=8" "--rdzv_backend=c10d" "--rdzv_endpoint=<MASTER_NODE_IP>:29500" )

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
result_name="MULTI_NODE_256GPU_TRAINING"

echo "RESULT,$result_name,,$result,nvidia,$start_fmt"

# Exit with the correct code
if [[ $ret_code != 0 ]]; then exit $ret_code; fi
