#!/bin/bash
# EXP8: Run Stage 2 only (CheapCTSNet training) on a single GPU
# Stage 1 already completed, using best.pt as teacher
# Usage: ./run_stage2.sh A 0   (Pipeline A on GPU 0)
#        ./run_stage2.sh B 1   (Pipeline B on GPU 1)

set -euo pipefail

PIPELINE=${1:-A}
GPU=${2:-0}
SEED=${3:-2020}

export CUDA_VISIBLE_DEVICES=$GPU

echo "============================================================"
echo "  Pipeline $PIPELINE | Stage 2: CheapCTSNet | GPU $GPU"
echo "  Teacher: best.pt from Stage 1 (best@ep15)"
echo "  Started at $(date)"
echo "============================================================"

# Stage 2: Train CheapCTSNet (distillation from TargetNet teacher)
python3 -m src.launch.train \
    experiment=MTI_${PIPELINE}_CheapCTSNet \
    seed=$SEED \
    data=miRNA_MTI_CTS \
    run.num_epochs=100 \
    run.batch_size=4096 \
    train.lr=0.006 \
    run.num_workers=13

echo ">>> [Pipeline $PIPELINE] Stage 2 training done at $(date)"
echo "============================================================"
