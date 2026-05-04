#!/bin/bash
# =============================================================================
# TargetNet (Official) Baseline Evaluation — RTX 5090
# =============================================================================
# 使用官方 TargetNet 代码 (seonwoo-min/TargetNet) 和预训练 checkpoint
# 在 miRAW balanced 80/20 10-fold 上评测
#
# 官方模型配置:
#   channels=[16,16,32], blocks=[2,1,1], stem_kernel=5, block_kernel=3, pool=3
#   with_esa=True, dropout=0.5
#   ESA filter: score >= 6 (official default)
#   Aggregation: max logit per pair
#
# 预估时间 (RTX 5090):
#   单 fold (~218 pairs):  ~3-5 min (BioPython pairwise2 较慢)
#   10-fold:               ~30-50 min
#
# 用法：
#   bash plan_targetnet.sh
#   FOLD_START=5 FOLD_END=9 bash plan_targetnet.sh
# =============================================================================
set -euo pipefail

FOLD_START="${FOLD_START:-0}"
FOLD_END="${FOLD_END:-9}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/eval_targetnet.py"
RESULTS_DIR="$SCRIPT_DIR/results"
MODEL_PATH="$PROJ_ROOT/external/TargetNet/pretrained_models/TargetNet.pt"

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "  TargetNet (Official) Baseline Evaluation"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Results: $RESULTS_DIR"
echo "Folds: $FOLD_START - $FOLD_END"
echo ""

# ---- Verify prerequisites ----
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: TargetNet model not found at $MODEL_PATH" >&2
    echo "Run: cd external && git clone https://github.com/seonwoo-min/TargetNet.git" >&2
    exit 1
fi

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "ERROR: eval_targetnet.py not found at $EVAL_SCRIPT" >&2
    exit 1
fi

# ---- 10-fold Evaluation ----
for fold in $(seq $FOLD_START $FOLD_END); do
    TEST_FILE="data/rebuttal/miRAW_fold${fold}_split/miRAW_Test_nosplit.txt"
    if [ ! -f "$TEST_FILE" ]; then
        echo "  WARNING: $TEST_FILE not found, skipping fold $fold"
        continue
    fi
    echo "--- Fold $fold ---"
    python3 "$EVAL_SCRIPT" \
        --test_data "$TEST_FILE" \
        --model_path "$MODEL_PATH" \
        --output "$RESULTS_DIR/targetnet_fold${fold}.json"
    echo ""
done

# ---- Collect 10-fold results ----
echo "=== Collecting 10-fold results ==="
python3 -c "
import json, numpy as np

results = {}
for fold in range(10):
    path = '$RESULTS_DIR/targetnet_fold{fold}.json'.format(fold=fold)
    try:
        with open(path) as f:
            m = json.load(f)
        results[fold] = m
        print(f'  Fold{fold}: F1={m[\"f1\"]:.4f} PR-AUC={m[\"pr_auc\"]:.4f} ROC-AUC={m[\"roc_auc\"]:.4f} Acc={m[\"accuracy\"]:.4f}')
    except FileNotFoundError:
        print(f'  Fold{fold}: MISSING')

if results:
    metrics = ['f1', 'roc_auc', 'pr_auc', 'accuracy', 'precision', 'recall', 'specificity', 'npv']
    print()
    print(f'=== TargetNet (Official) 10-fold (N={len(results)}) ===')
    summary = {
        'model': 'TargetNet_Official',
        'n_folds': len(results),
        'per_fold': {},
    }
    for m_name in metrics:
        vals = [results[f][m_name] for f in results]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        summary[f'{m_name}_mean'] = float(mean_val)
        summary[f'{m_name}_std'] = float(std_val)
        print(f'  {m_name:12s}: {mean_val:.4f} +/- {std_val:.4f}')

    for fold in results:
        summary['per_fold'][str(fold)] = {k: results[fold][k] for k in metrics}

    print()
    print(f'  Comparison (miRAW 10-fold balanced 80/20):')
    print(f'    TargetNet (official): F1={summary[\"f1_mean\"]:.4f}+/-{summary[\"f1_std\"]:.4f}')
    print(f'    MaxPool (TN-Opt):     F1=0.798+/-0.017')
    print(f'    Mimosa:               F1=0.788+/-0.017')
    print(f'    PAIR-Former:          F1=0.840+/-0.022')

    with open('$RESULTS_DIR/targetnet_10fold_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  Saved: $RESULTS_DIR/targetnet_10fold_summary.json')
"

echo ""
echo "=========================================="
echo "  TargetNet evaluation complete"
echo "=========================================="
