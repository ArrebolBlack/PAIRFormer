#!/bin/bash
# scripts/run_build_cache_ddp.sh
# Multi-GPU cache builder for the EM pipeline.
#
# Builds cheap → selection → instance caches in sequence,
# with each stage sharded across all available GPUs.
#
# Usage:
#   bash scripts/run_build_cache_ddp.sh [NUM_GPUS] [EXPERIMENT] [EM_CACHE_ROOT] [TRANSFER_DEST]
#
# Examples:
#   # 4 GPUs, build to default cache path
#   bash scripts/run_build_cache_ddp.sh 4 MTI_EM_Pipeline
#
#   # 4 GPUs, build to /dev/shm (memory), then transfer to vePFS
#   bash scripts/run_build_cache_ddp.sh 4 MTI_EM_Pipeline /dev/shm/pairformer_cache /vepfs-mlp2/.../cache
#
#   # 8 GPUs
#   bash scripts/run_build_cache_ddp.sh 8 MTI_EM_Pipeline

set -euo pipefail

NUM_GPUS=${1:-4}
EXPERIMENT=${2:-MTI_EM_Pipeline}
EM_CACHE_ROOT=${3:-}
TRANSFER_DEST=${4:-}

# Check torchrun
if ! command -v torchrun &> /dev/null; then
    echo "ERROR: torchrun not found."
    exit 1
fi

# Build extra args
EXTRA_ARGS=""
if [ -n "$EM_CACHE_ROOT" ]; then
    EXTRA_ARGS="$EXTRA_ARGS em_cache_root=$EM_CACHE_ROOT"
fi
if [ -n "$TRANSFER_DEST" ]; then
    EXTRA_ARGS="$EXTRA_ARGS +cache_transfer_dest=$TRANSFER_DEST"
fi

echo "========================================"
echo " Multi-GPU Cache Build"
echo " NUM_GPUS:  $NUM_GPUS"
echo " EXPERIMENT: $EXPERIMENT"
echo " EM_CACHE_ROOT: ${EM_CACHE_ROOT:-<default>}"
echo " TRANSFER_DEST: ${TRANSFER_DEST:-<none>}"
echo "========================================"

torchrun --nproc_per_node=$NUM_GPUS \
    -m src.launch.build_cache_ddp \
    experiment=$EXPERIMENT \
    $EXTRA_ARGS

echo "========================================"
echo " Cache build complete!"
echo "========================================"
