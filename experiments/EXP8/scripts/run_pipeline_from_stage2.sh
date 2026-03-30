#!/bin/bash
# EXP8: Resume pipeline from Stage 2 (Stage 1 already completed)
# Uses best.pt from Stage 1 (overfitting: best@ep15, last@ep99)
#
# Usage: ./run_pipeline_from_stage2.sh A 0 [/path/to/em_cache]
#        ./run_pipeline_from_stage2.sh B 1 /dev/shm/em_cache_B
#
# NOTE: Stage 2 training uses CTS-level data (miRNA_MTI_CTS).
#       Cheap cache for Stage 3 uses pair-level data (explicit overrides).
#       See STAGE3_PLAN.md Bug 1 for explanation.

set -euo pipefail

PIPELINE=${1:-A}
GPU=${2:-0}
EM_CACHE_ROOT=${3:-}
SEED=${4:-2020}

export CUDA_VISIBLE_DEVICES=$GPU

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate myenv

# Pipeline-specific configs
DATA_CONFIG="miRNA_MTI_pair_random"
DATA_FILE="data/MTI/pair_random_split.txt"
CHEAP_EXPERIMENT="MTI_A_CheapCTSNet"
TARGET_EXPERIMENT="MTI_A_TargetNet_Optimized"
EM_EXPERIMENT="MTI_A_EM_Pipeline"
if [ "$PIPELINE" = "B" ]; then
    DATA_CONFIG="miRNA_MTI_pair_mirna"
    DATA_FILE="data/MTI/pair_mirna_split.txt"
    CHEAP_EXPERIMENT="MTI_B_CheapCTSNet"
    TARGET_EXPERIMENT="MTI_B_TargetNet_Optimized"
    EM_EXPERIMENT="MTI_B_EM_Pipeline"
fi

EM_CACHE_ARG=""
if [ -n "$EM_CACHE_ROOT" ]; then
    mkdir -p "$EM_CACHE_ROOT"
    EM_CACHE_ARG="+em_cache_root=$EM_CACHE_ROOT"
fi

echo "============================================================"
echo "  Pipeline $PIPELINE | GPU $GPU | Seed $SEED"
echo "  Resuming from Stage 2 (Stage 1 completed)"
echo "  Started at $(date)"
echo "============================================================"

# -------- Stage 2: CheapCTSNet (CTS-level distillation) --------
echo ""
echo ">>> [Pipeline $PIPELINE] Stage 2: CheapCTSNet (CTS-level distillation)"

python3 -u -m src.launch.train \
    experiment=$CHEAP_EXPERIMENT \
    seed=$SEED \
    data=miRNA_MTI_CTS \
    run.num_epochs=100 \
    run.batch_size=4096 \
    run.num_workers=$(nproc)

echo ">>> [Pipeline $PIPELINE] Stage 2 training done at $(date)"

# -------- Build Cheap Cache (pair-level data!) --------
# CRITICAL: Must use explicit data.path.* overrides, NOT data=<config>
# See STAGE3_PLAN.md Bug 1
echo ""
echo ">>> [Pipeline $PIPELINE] Building cheap cache (pair-level)..."

CHEAP_CKPT=$(ls -d outputs/${CHEAP_EXPERIMENT}*/checkpoints 2>/dev/null | tail -1)/best.pt
echo "    Using cheap ckpt: $CHEAP_CKPT"

# Clean stale locks
find cache/em_cache -name "*.lock" -delete 2>/dev/null || true

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

# -------- Stage 3: Selection Cache + EM Training --------
echo ""
echo ">>> [Pipeline $PIPELINE] Stage 3: Selection cache + EM training"

python3 -u -m src.launch.build_selection_cache \
    experiment=$EM_EXPERIMENT \
    seed=$SEED \
    data=$DATA_CONFIG \
    $EM_CACHE_ARG

python3 -u -m src.launch.train_em \
    experiment=$EM_EXPERIMENT \
    seed=$SEED \
    data=$DATA_CONFIG \
    run.num_epochs=150 \
    run.batch_size=4096 \
    run.kmax=64 \
    $EM_CACHE_ARG

echo ">>> [Pipeline $PIPELINE] Stage 3 done at $(date)"

# -------- Evaluation --------
echo ""
echo ">>> [Pipeline $PIPELINE] Evaluation"

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
echo "  Pipeline $PIPELINE | ALL DONE at $(date)"
echo "============================================================"
