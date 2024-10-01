#!/usr/bin/env bash

# ==============================================
# Distributed Training Script with Dry-Run and Help Options
# ==============================================

# Default values for DL params
BATCHSIZE=32
NUMEPOCHS=8
DATASET_DIR="/datasets/open-images-v6-mlperf"
LR=0.0001
OUTPUT_DIR="/results"
LOG_INTERVAL=20
TORCH_HOME="$(pwd)/torch-model-cache"

# Default system params
NNODES=1        # Number of nodes
NPROC_PER_NODE=4  # Number of workers (GPUs) per node
DRY_RUN=false    # By default, we execute the command

# Function to print usage
usage() {
    cat << USAGE
Usage: $0 --hosts 'host1 host2' [OPTIONS]

Distributed training script using torchrun for multi-node and multi-GPU setups.

Arguments:
  --hosts 'HOSTS'            A space-separated list of hostnames. The first host is the master.

Optional parameters:
  --batch-size BATCHSIZE     The batch size for training (default: 32).
  --epochs NUMEPOCHS         The number of epochs to run (default: 8).
  --data-dir DATASET_DIR     The path to the dataset (default: /datasets/open-images-v6-mlperf).
  --lr LR                    Learning rate for training (default: 0.0001).
  --output-dir OUTPUT_DIR    The directory where output will be saved (default: /results).
  --log-interval INTERVAL    Frequency (in iterations) to log training info (default: 20).
  --nproc_per_node N         Number of workers (GPUs) per node (default: 4).
  --dry-run                  Show the command that would be executed without running it.

Help:
  --help                     Show this help message and exit.
USAGE
    exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    --hosts)
    HOSTS="$2"
    shift 2
    ;;
    --batch-size)
    BATCHSIZE="$2"
    shift 2
    ;;
    --epochs)
    NUMEPOCHS="$2"
    shift 2
    ;;
    --data-dir)
    DATASET_DIR="$2"
    shift 2
    ;;
    --lr)
    LR="$2"
    shift 2
    ;;
    --output-dir)
    OUTPUT_DIR="$2"
    shift 2
    ;;
    --log-interval)
    LOG_INTERVAL="$2"
    shift 2
    ;;
    --nproc_per_node)
    NPROC_PER_NODE="$2"
    shift 2
    ;;
    --dry-run)
    DRY_RUN=true
    shift
    ;;
    --help)
    usage
    ;;
    *)
    echo "Invalid argument: $key"
    usage
    ;;
esac
done

# Ensure hosts are provided
if [ -z "$HOSTS" ]; then
  echo "Error: --hosts argument is required"
  usage
fi

# Split hosts into an array
IFS=' ' read -r -a HOST_ARRAY <<< "$HOSTS"
NNODES=${#HOST_ARRAY[@]}  # Number of nodes is the length of the host list

# Determine this machine's hostname
CURRENT_HOST=$(hostname)

# Determine rank based on host
RANK=-1
for i in "${!HOST_ARRAY[@]}"; do
  if [ "${HOST_ARRAY[$i]}" == "$CURRENT_HOST" ]; then
    RANK=$i
    break
  fi
done

# Ensure the current host is in the provided list
if [ $RANK -eq -1 ]; then
  echo "Error: This machine's hostname ($CURRENT_HOST) is not in the host list."
  exit 1
fi

# Set master IP and rank based on the host list
MASTER_HOST=${HOST_ARRAY[0]}
MASTER_IP=$(getent hosts "$MASTER_HOST" | awk '{ print $1 }')

if [ "$RANK" == "0" ]; then
  echo "This machine ($CURRENT_HOST) is the master."
else
  echo "This machine ($CURRENT_HOST) is a worker with rank $RANK."
fi

# ==============================================
# Prepare training command for torchrun
# ==============================================

CMD=(
  "torchrun"
  "--nnodes=$NNODES"
  "--nproc_per_node=$NPROC_PER_NODE"
  "--rdzv_backend=c10d"
  "--rdzv_endpoint=$MASTER_IP:29500"
  "--node_rank=$RANK"
)

PARAMS=(
    --batch-size              "${BATCHSIZE}"
    --eval-batch-size         "${BATCHSIZE}"  # Assuming eval batch size same as train batch size
    --epochs                  "${NUMEPOCHS}"
    --print-freq              "${LOG_INTERVAL}"
    --data-path               "${DATASET_DIR}"
    --lr                      "${LR}"
    --output-dir              "${OUTPUT_DIR}"
)

# If dry-run is set, print the command that would be run and exit
if [ "$DRY_RUN" == "true" ]; then
  echo "Dry run mode: The following command would be executed:"
  echo "${CMD[@]} train.py ${PARAMS[@]}"
  exit 0
fi

# ==============================================
# Start training with torchrun
# ==============================================

set -e

# Start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt on $CURRENT_HOST (Rank: $RANK)"

# Run the training script
"${CMD[@]}" train.py "${PARAMS[@]}" ; ret_code=$?

# End timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt on $CURRENT_HOST (Rank: $RANK)"

# Report result
result=$(( end - start ))
result_name="DISTRIBUTED_TRAINING_RANK_${RANK}"

echo "RESULT,$result_name,,$result,nvidia,$start_fmt"

# Exit with the correct code
if [[ $ret_code != 0 ]]; then exit $ret_code; fi
