#!/bin/bash
# scripts/run_ddp_train_em.sh
# DDP training for EM pipeline using torchrun
#
# Usage:
#   bash scripts/run_ddp_train_em.sh [NUM_GPUS] [EXPERIMENT]
#   bash scripts/run_ddp_train_em.sh 4 MTI_EM_Pipeline
#   bash scripts/run_ddp_train_em.sh 8 MTI_EM_Pipeline

set -euo pipefail

NUM_GPUS=${1:-4}
EXPERIMENT=${2:-MTI_EM_Pipeline}

# Check if torchrun is available
if ! command -v torchrun &> /dev/null; then
    echo "ERROR: torchrun not found. Please install PyTorch with DDP support."
    exit 1
fi

echo "========================================"
echo "DDP EM Pipeline Training"
echo "NUM_GPUS: $NUM_GPUS"
echo "EXPERIMENT: $EXPERIMENT"
echo "========================================"

torchrun --nproc_per_node=$NUM_GPUS \
    -m src.launch.train_em \
    experiment=$EXPERIMENT \
    "${@:3}"

echo "========================================"
echo "Training completed"
echo "========================================"
