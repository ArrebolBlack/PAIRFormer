#!/bin/bash
# exp6e_run.sh — Run all EXP6E CAMELYON16 experiments
# Matches torchmil Table 1 protocol: 5-fold CV, resnet50_bt features
#
# Usage:
#   cd PAIRFormer
#   bash scripts/rebuttal/exp6e_run.sh

set -euo pipefail

DATA_DIR="/home/yjq/workspace/rebuttal/camelyon16-torchmil/dataset"
OUTPUT_DIR="experiments/EXP6E_CAMELYON16/results"
SEED=42
EPOCHS=50
LR=1e-4

echo "========================================"
echo "EXP6E: CAMELYON16 BR-MIL Benchmark"
echo "========================================"
echo "Data: ${DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Seed: ${SEED}, Epochs: ${EPOCHS}, LR: ${LR}"
echo ""

# 1. ABMIL baseline (run once, results saved for all K to reference)
echo "[1/6] Running ABMIL baseline ..."
python -u scripts/rebuttal/exp6_camelyon16_v3.py \
    --method abmil --seed ${SEED} --epochs ${EPOCHS} --lr ${LR} \
    --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR}
echo ""

# K-sweep: 1024, 10240, 20480, 33333, 44402
K_VALUES="1024 10240 20480 33333 44402"
run_idx=2
for K in ${K_VALUES}; do
    echo "[${run_idx}/6] Running BR-MIL K=${K} ..."
    python -u scripts/rebuttal/exp6_camelyon16_v3.py \
        --method brmil_original --K ${K} --seed ${SEED} --epochs ${EPOCHS} --lr ${LR} \
        --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR}
    echo ""
    run_idx=$((run_idx + 1))
done

echo "========================================"
echo "All experiments completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================"

# Print summary table
echo ""
echo "=== Results Summary ==="
python -c "
import json, glob
from pathlib import Path
results_dir = Path('${OUTPUT_DIR}')
print(f\"{'Method':<20} {'K':>6} {'ACC':>14} {'AUC':>14} {'F1':>14}\")
print('-' * 70)
for jf in sorted(results_dir.glob('*/K*/seed_*/metrics.json')):
    d = json.load(open(jf))
    method = d['method']
    K = d.get('K', 0)
    if 'auc_mean' in d:
        acc = f\"{d['accuracy_mean']:.4f}±{d['accuracy_std']:.4f}\"
        auc = f\"{d['auc_mean']:.4f}±{d['auc_std']:.4f}\"
        f1 = f\"{d['f1_mean']:.4f}±{d['f1_std']:.4f}\"
        print(f\"{method:<20} {K:>6} {acc:>14} {auc:>14} {f1:>14}\")
    else:
        print(f\"{method:<20} {K:>6} {'ERROR':>14} {'':>14} {'':>14}\")
"
