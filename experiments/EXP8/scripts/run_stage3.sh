#!/bin/bash
# EXP8: Run Stage 3 (EM Pipeline) for a single pipeline on a single GPU
# Stage 1 & 2 already completed, using best.pt from both
#
# Usage: ./run_stage3.sh A 0 [/path/to/em_cache]
#        ./run_stage3.sh B 1 /dev/shm/em_cache_B
#
# IMPORTANT:
# - build_cheap_cache MUST use explicit data.path.* overrides (Bug 1 fix)
# - em_cache directory is NOT isolated by pipeline (Bug 2), so
#   if running both pipelines, use different +em_cache_root for each
# - On GPFS/NFS, use tmpfs (+em_cache_root=/dev/shm/...) to avoid mmap lock (Bug 4)
# - Always clean stale lock files before running (Bug 5)

set -euo pipefail

PIPELINE=${1:-A}
GPU=${2:-0}
EM_CACHE_ROOT=${3:-}  # Optional: override em_cache root (e.g., /dev/shm/em_cache_A)
SEED=${4:-2020}

export CUDA_VISIBLE_DEVICES=$GPU

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate myenv

echo "============================================================"
echo "  Pipeline $PIPELINE | Stage 3: EM Pipeline | GPU $GPU"
echo "  Started at $(date)"
echo "============================================================"

# Pipeline-specific configs
DATA_CONFIG="miRNA_MTI_pair_random"
DATA_FILE="data/MTI/pair_random_split.txt"
CHEAP_EXPERIMENT="MTI_A_CheapCTSNet"
EM_EXPERIMENT="MTI_A_EM_Pipeline"
CHEAP_CKPT="checkpoints/MTI_A_CheapCTSNet/checkpoints/best.pt"
if [ "$PIPELINE" = "B" ]; then
    DATA_CONFIG="miRNA_MTI_pair_mirna"
    DATA_FILE="data/MTI/pair_mirna_split.txt"
    CHEAP_EXPERIMENT="MTI_B_CheapCTSNet"
    EM_EXPERIMENT="MTI_B_EM_Pipeline"
    CHEAP_CKPT="checkpoints/MTI_B_CheapCTSNet/checkpoints/best.pt"
fi

# Build em_cache_root override (optional)
EM_CACHE_ARG=""
if [ -n "$EM_CACHE_ROOT" ]; then
    mkdir -p "$EM_CACHE_ROOT"
    EM_CACHE_ARG="+em_cache_root=$EM_CACHE_ROOT"
    echo "  em_cache_root: $EM_CACHE_ROOT"
fi

# Clean stale lock files
CACHE_DIR="${EM_CACHE_ROOT:-cache}"
find "$CACHE_DIR" -name "*.lock" -delete 2>/dev/null || true
echo "  Stale locks cleaned"

# ---------- Step 1: Build cheap cache ----------
# CRITICAL: Must use explicit data.path.* overrides, NOT data=<config>!
# Reason: CheapCTSNet experiment config uses @package _global_ which
# overrides data.path to CTS-level paths. CLI data=<config> group override
# is applied BEFORE @package _global_ in Hydra defaults resolution.
echo ""
echo ">>> [Pipeline $PIPELINE] Step 1: Building cheap cache..."
echo "    Data: $DATA_FILE ($DATA_CONFIG)"
echo "    Checkpoint: $CHEAP_CKPT"

python3 -u -m src.launch.build_cheap_cache \
    experiment=$CHEAP_EXPERIMENT \
    seed=$SEED \
    data.name=$DATA_CONFIG \
    data.path.train=$DATA_FILE \
    data.path.val=$DATA_FILE \
    data.path.test=$DATA_FILE \
    +data.split_map.test=test \
    run.batch_size=10240 \
    run.num_workers=$(nproc) \
    +cheap_ckpt_path="$CHEAP_CKPT" \
    +cheap_cache_splits='[train,val,test]' \
    +cheap_cache_overwrite=true \
    $EM_CACHE_ARG

echo ">>> [Pipeline $PIPELINE] Cheap cache done at $(date)"

# ---------- Step 2: Build selection cache ----------
echo ""
echo ">>> [Pipeline $PIPELINE] Step 2: Building selection cache..."
# build_selection_cache uses EM_Pipeline experiment config which does NOT
# have @package _global_ data.path overrides, so data=<config> works correctly.

python3 -u -m src.launch.build_selection_cache \
    experiment=$EM_EXPERIMENT \
    seed=$SEED \
    data=$DATA_CONFIG \
    $EM_CACHE_ARG

echo ">>> [Pipeline $PIPELINE] Selection cache done at $(date)"

# ---------- Step 3: Train EM ----------
echo ""
echo ">>> [Pipeline $PIPELINE] Step 3: Training EM pipeline..."

python3 -u -m src.launch.train_em \
    experiment=$EM_EXPERIMENT \
    seed=$SEED \
    data=$DATA_CONFIG \
    run.num_epochs=150 \
    run.batch_size=4096 \
    run.kmax=64 \
    $EM_CACHE_ARG

echo ">>> [Pipeline $PIPELINE] Training done at $(date)"

# ---------- Step 4: Evaluation ----------
echo ""
echo ">>> [Pipeline $PIPELINE] Step 4: Evaluation..."

OUTPUT_DIR=$(ls -d outputs/${EM_EXPERIMENT}*/checkpoints 2>/dev/null | tail -1)
if [ -n "$OUTPUT_DIR" ]; then
    CKPT="${OUTPUT_DIR}/best.pt"
    echo "    Using checkpoint: $CKPT"
    python3 -u -m src.launch.eval_em \
        experiment=$EM_EXPERIMENT \
        seed=$SEED \
        data=$DATA_CONFIG \
        run.checkpoint="$CKPT" \
        run.test_splits='["test"]' \
        $EM_CACHE_ARG
else
    echo "WARNING: Output directory not found for Pipeline $PIPELINE"
fi

echo ""
echo "============================================================"
echo "  Pipeline $PIPELINE | Stage 3 ALL DONE at $(date)"
echo "============================================================"
