#!/bin/bash
# run_all.sh — Relational distillation ablation
# ================================================
# Stage 2: Train CheapCTSNet with β_rel=0 (no relational distillation)
# Stage 3: Train EM pipeline using ablated cheap encoder
#
# Baseline: existing checkpoint with β_rel=1 → Stage 3: F1=0.9740
#
# Usage:
#   bash scripts/rebuttal/rel_distill_ablation/run_all.sh
#   bash scripts/rebuttal/rel_distill_ablation/run_all.sh stage3  # Skip to Stage 3

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE=${1:-"all"}
OUT_BASE="experiments/rel_distill_ablation"
LOG_DIR="scripts/rebuttal/rel_distill_ablation"
mkdir -p "$LOG_DIR"

if [ "$STAGE" = "all" ] || [ "$STAGE" = "stage2" ]; then
    echo "============================================"
    echo "Stage 2: Training CheapCTSNet with β_rel=0"
    echo "============================================"
    python -m src.launch.train \
        experiment=CheapCTSNet_noRelDistill \
        logging.wandb.enabled=false \
        run.num_workers=0 \
        hydra.run.dir="${OUT_BASE}/cheap" \
        2>&1 | tee "${LOG_DIR}/stage2.log"
    echo "Stage 2 done."
fi

CHEAP_CKPT="${OUT_BASE}/cheap/checkpoints/last.pt"

if [ ! -f "$CHEAP_CKPT" ]; then
    echo "ERROR: Stage 2 checkpoint not found at ${CHEAP_CKPT}"
    exit 1
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "stage3" ]; then
    echo ""
    echo "============================================"
    echo "Stage 3: Training EM pipeline with ablated cheap encoder"
    echo "============================================"
    python -m src.launch.train_em \
        experiment=miRAW_EM_Pipeline \
        seed=2020 \
        run.num_workers=8 \
        logging.wandb.enabled=false \
        cheap_ckpt_path="${CHEAP_CKPT}" \
        em_cache_root="cache/rel_distill_ablation" \
        hydra.run.dir="${OUT_BASE}/em" \
        2>&1 | tee "${LOG_DIR}/stage3.log"
    echo "Stage 3 done."
fi

echo ""
echo "============================================"
echo "All done! Results:"
echo "  Stage 2: ${OUT_BASE}/cheap/"
echo "  Stage 3: ${OUT_BASE}/em/"
echo "============================================"
