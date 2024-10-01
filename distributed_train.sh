#!/usr/bin/env bash

# ==============================================
# Distributed Training Script with Dry-Run Option
# ==============================================

# Default values for DL params
BATCHSIZE=32
NUMEPOCHS=8
DATASET_DIR="/datasets/open-images-v6-mlperf"
EXTRA_PARAMS="--lr 0.0001 --output-dir=/results"
LOG_INTERVAL=20
TORCH_HOME="$(pwd)/torch-model-cache"

# Default system params
NNODES=1        # Number of nodes
NPROC_PER_NODE=4  # Number of workers (GPUs) per node
DRY_RUN=false    # By default, we execute the command

# Function to print usage
usage() {
  echo "Usage: $0 --hosts 'host1 host2' [--batch-size BATCHSIZE] [--epochs NUMEPOCHS] [--data-dir DATASET_DIR] [--extra-params EXTRA_PARAMS] [--dry-run]"
  exit 1
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
    --extra-params)
    EXTRA_PARAMS="$2"
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
    *)
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
)

# If dry-run is set, print the command that would be run and exit
if [ "$DRY_RUN" == "true" ]; then
  echo "Dry run mode: The following command would be executed:"
  echo "${CMD[@]} train.py ${PARAMS[@]} ${EXTRA_PARAMS}"
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
"${CMD[@]}" train.py "${PARAMS[@]}" ${EXTRA_PARAMS} ; ret_code=$?

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
