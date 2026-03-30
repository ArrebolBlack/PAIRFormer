#!/bin/bash
# run_k1_ratio_ablation.sh
# ================================================
# Emergency fix: k1_ratio ablation
# ALL main experiments used k1_ratio=1, degenerating STSelector to pure TopK
# This runs k1_ratio=0.5 (the intended design default) to measure impact
#
# Baseline (k1_ratio=1): F1=0.9740, PR-AUC=0.9961, ROC-AUC=0.9946
#
# Usage:
#   bash scripts/rebuttal/k1_ratio_ablation/run_all.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUT_BASE="experiments/k1_ratio_ablation"
LOG_DIR="scripts/rebuttal/k1_ratio_ablation"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "k1_ratio=0.5 ablation: Full Stage 3"
echo "============================================"

python -m src.launch.train_em \
    experiment=miRAW_EM_Pipeline_k1ratio05 \
    seed=2020 \
    run.num_workers=8 \
    logging.wandb.enabled=false \
    em_cache_root="cache/k1_ratio_ablation" \
    hydra.run.dir="${OUT_BASE}/k1ratio05_seed2020" \
    2>&1 | tee "${LOG_DIR}/k1ratio05_seed2020.log"

echo ""
echo "============================================"
echo "Done! Check results:"
echo "  ${OUT_BASE}/k1ratio05_seed2020/"
echo "============================================"
