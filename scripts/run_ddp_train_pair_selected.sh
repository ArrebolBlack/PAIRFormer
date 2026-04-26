#!/bin/bash
# scripts/run_ddp_train_pair_selected.sh
# DDP training for PairSelectedTrainer using torchrun
#
# Usage:
#   bash scripts/run_ddp_train_pair_selected.sh [NUM_GPUS] [EXPERIMENT]
#   bash scripts/run_ddp_train_pair_selected.sh 4 MTI_train_selected_inst
#   bash scripts/run_ddp_train_pair_selected.sh 8 MTI_train_selected_inst_v2

set -euo pipefail

NUM_GPUS=${1:-4}
EXPERIMENT=${2:-MTI_train_selected_inst}

# Check if torchrun is available
if ! command -v torchrun &> /dev/null; then
    echo "ERROR: torchrun not found. Please install PyTorch with DDP support."
    exit 1
fi

echo "========================================"
echo "DDP PairSelected Training"
echo "NUM_GPUS: $NUM_GPUS"
echo "EXPERIMENT: $EXPERIMENT"
echo "========================================"

torchrun --nproc_per_node=$NUM_GPUS \
    -m src.launch.train_pair_selected_inst \
    experiment=$EXPERIMENT \
    "${@:3}"

echo "========================================"
echo "Training completed"
echo "========================================"
