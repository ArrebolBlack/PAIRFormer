#!/bin/bash
# =============================================================================
# Mimosa Baseline Evaluation — RTX 5090
# =============================================================================
# 复现 Mimosa (Bi et al., NAR 2024)，在 miRAW balanced 80/20 和 10-fold 上评测
#
# 预估时间 (RTX 5090):
#   Balanced 80/20 (218 pairs, step=1):  ~2 min
#   10-fold (218×10 pairs, step=1):      ~20 min
#   Total:                               ~25 min
#
# 用法：
#   bash plan_mimosa.sh                     # 全部执行
#   PHASE=2 bash plan_mimosa.sh             # 只跑 balanced 80/20
#   PHASE=3 bash plan_mimosa.sh             # 只跑 10-fold
#   PHASE=3 FOLD_START=5 FOLD_END=9 bash ... # 只跑 fold 5-9
# =============================================================================
set -euo pipefail

PHASE="${PHASE:-all}"
FOLD_START="${FOLD_START:-0}"
FOLD_END="${FOLD_END:-9}"
STEPSIZE="${STEPSIZE:-1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/eval_mimosa.py"
RESULTS_DIR="$SCRIPT_DIR/results"
MODEL_PATH="$PROJ_ROOT/external/Mimosa/training/model_mimosa.pth"

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "  Mimosa Baseline Evaluation"
echo "=========================================="
echo "Phase: $PHASE | Stepsize: $STEPSIZE"
echo "Model: $MODEL_PATH"
echo "Results: $RESULTS_DIR"
echo ""

# ---- Verify prerequisites ----
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Mimosa model not found at $MODEL_PATH" >&2
    echo "Run: cd external && git clone https://github.com/biyueeee/Mimosa.git" >&2
    exit 1
fi

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "ERROR: eval_mimosa.py not found at $EVAL_SCRIPT" >&2
    exit 1
fi

# =====================================================================
# Phase 1: Data Validation
# =====================================================================
if [ "$PHASE" = "all" ] || [ "$PHASE" = "1" ]; then
    echo "=== Phase 1: Data Validation ==="

    python3 -c "
import numpy as np
files = [
    'data/rebuttal/miRAW_8020_balanced/miRAW_Test.txt',
]
for f in files:
    with open(f) as fh:
        lines = fh.readlines()[1:]
    lens = [len(l.split('\t')[3]) for l in lines]
    a = np.array(lens)
    pos = sum(1 for l in lines if l.split('\t')[4] == '1')
    neg = len(lines) - pos
    print(f'{f}:')
    print(f'  {len(lines)} pairs ({pos}+ / {neg}-)')
    print(f'  mRNA: min={a.min()}, max={a.max()}, median={int(np.median(a))}')
    assert a.min() > 40, 'ERROR: mRNA too short for gene-level prediction!'
    print('  OK: gene-level data confirmed')
print()
print('All data validated.')
"

    echo "Phase 1 done."
    echo ""
fi

# =====================================================================
# Phase 2: Balanced 80/20 Test
# =====================================================================
if [ "$PHASE" = "all" ] || [ "$PHASE" = "2" ]; then
    echo "=== Phase 2: Balanced 80/20 Test (stepsize=$STEPSIZE) ==="

    python3 "$EVAL_SCRIPT" \
        --test_data data/rebuttal/miRAW_8020_balanced/miRAW_Test.txt \
        --model_path "$MODEL_PATH" \
        --stepsize $STEPSIZE \
        --output "$RESULTS_DIR/mimosa_8020_balanced.json"

    echo ""
    echo "Phase 2 done."
    echo ""
fi

# =====================================================================
# Phase 3: 10-fold Test
# =====================================================================
if [ "$PHASE" = "all" ] || [ "$PHASE" = "3" ]; then
    echo "=== Phase 3: 10-fold Test (stepsize=$STEPSIZE, fold ${FOLD_START}-${FOLD_END}) ==="

    for fold in $(seq $FOLD_START $FOLD_END); do
        TEST_FILE="data/rebuttal/miRAW_fold${fold}_split/miRAW_Test.txt"
        if [ ! -f "$TEST_FILE" ]; then
            echo "  WARNING: $TEST_FILE not found, skipping fold $fold"
            continue
        fi
        echo "--- Fold $fold ---"
        python3 "$EVAL_SCRIPT" \
            --test_data "$TEST_FILE" \
            --model_path "$MODEL_PATH" \
            --stepsize $STEPSIZE \
            --output "$RESULTS_DIR/mimosa_fold${fold}.json"
        echo ""
    done

    # Collect 10-fold results
    echo "=== Collecting 10-fold results ==="
    python3 -c "
import json, glob, numpy as np

results = {}
for fold in range(10):
    path = '$RESULTS_DIR/mimosa_fold{fold}.json'.format(fold=fold)
    try:
        with open(path) as f:
            m = json.load(f)
        results[fold] = m
        print(f'  Fold{fold}: F1={m[\"f1\"]:.4f} AUC={m[\"roc_auc\"]:.4f}')
    except FileNotFoundError:
        print(f'  Fold{fold}: MISSING')

if results:
    f1s = [m['f1'] for m in results.values()]
    aucs = [m['roc_auc'] for m in results.values()]
    prs = [m['pr_auc'] for m in results.values()]
    accs = [m['accuracy'] for m in results.values()]
    print()
    print(f'=== Mimosa 10-fold (N={len(results)}) ===')
    print(f'  F1:      {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}')
    print(f'  ROC-AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}')
    print(f'  PR-AUC:  {np.mean(prs):.4f} +/- {np.std(prs):.4f}')
    print(f'  Acc:     {np.mean(accs):.4f} +/- {np.std(accs):.4f}')
    print()
    print(f'  Comparison (miRAW balanced 80/20):')
    print(f'    Mimosa:             F1={np.mean(f1s):.4f}+/-{np.std(f1s):.4f}')
    print(f'    MaxPool baseline:   F1=0.796')
    print(f'    PAIR-Former (10-fold): F1=0.840+/-0.022')

    # Save aggregated
    agg = {
        'model': 'Mimosa',
        'stepsize': $STEPSIZE,
        'n_folds': len(results),
        'f1_mean': float(np.mean(f1s)),
        'f1_std': float(np.std(f1s)),
        'roc_auc_mean': float(np.mean(aucs)),
        'roc_auc_std': float(np.std(aucs)),
        'pr_auc_mean': float(np.mean(prs)),
        'pr_auc_std': float(np.std(prs)),
        'per_fold': {str(k): v for k, v in results.items()},
    }
    with open('$RESULTS_DIR/mimosa_10fold_summary.json', 'w') as f:
        json.dump(agg, f, indent=2)
    print(f'  Saved: $RESULTS_DIR/mimosa_10fold_summary.json')
"

    echo ""
    echo "Phase 3 done."
    echo ""
fi

echo "=========================================="
echo "  Mimosa evaluation complete"
echo "=========================================="
