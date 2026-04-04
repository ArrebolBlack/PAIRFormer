#!/bin/bash
# run_exp6d_sweep.sh — 并行编排 EXP6D 实验
#
# 用法:
#   cd PAIRFormer
#   bash scripts/rebuttal/run_exp6d_sweep.sh [OPTIONS]
#
# 选项:
#   --methods "m1 m2 ..."   要跑的方法 (默认全部)
#   --K_values "1 4 12 64"  K 值 (默认 "4 12 64")
#   --seeds "2020"           种子 (默认 "2020")
#   --epochs 200             训练轮数 (默认 200)
#   --n_folds 10             N-fold CV (0=单次划分, 默认 10)
#   --workers 4              并行进程数 (默认 4)
#   --output_dir DIR         输出目录
#   --skip_existing          跳过已有结果的配置
#
# 示例:
#   # 快速对比
#   bash scripts/rebuttal/run_exp6d_sweep.sh --workers 4 --epochs 50 --K_values "4 12 64"
#
#   # 完整运行 (10-fold CV)
#   bash scripts/rebuttal/exp6-musk2/run_sweep.sh --workers 4 --seeds "2020 2025 2026" \
#       --K_values "1 4 12 64 512 1024" --n_folds 10 --epochs 200

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/train.py"

# Defaults
METHODS="abmil brmil_original brmil_auxloss brmil_gumbel brmil_twostage brmil_fixed_l2 brmil_fixed_dist brmil_fixed_random"
K_VALUES="4 12 64"
SEEDS="2020"
EPOCHS=200
N_FOLDS=10
WORKERS=4
OUTPUT_DIR="experiments/EXP6D_Musk2/results_v3"
SKIP_EXISTING=false

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --methods)    METHODS="$2"; shift 2 ;;
        --K_values)   K_VALUES="$2"; shift 2 ;;
        --seeds)      SEEDS="$2"; shift 2 ;;
        --epochs)     EPOCHS="$2"; shift 2 ;;
        --n_folds)    N_FOLDS="$2"; shift 2 ;;
        --workers)    WORKERS="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --skip_existing) SKIP_EXISTING=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=========================================="
echo "  EXP6D Sweep Configuration"
echo "=========================================="
echo "  Methods:    $METHODS"
echo "  K values:   $K_VALUES"
echo "  Seeds:      $SEEDS"
echo "  Epochs:     $EPOCHS"
echo "  N_folds:    $N_FOLDS"
echo "  Workers:    $WORKERS"
echo "  Output:     $OUTPUT_DIR"
echo "=========================================="

# Generate task list
TASK_FILE=$(mktemp)
trap "rm -f $TASK_FILE" EXIT

for method in $METHODS; do
    if [ "$method" = "abmil" ]; then
        # ABMIL: no K, run once per seed
        for seed in $SEEDS; do
            K_tag="K0"
            result_file="$OUTPUT_DIR/$method/$K_tag/seed_$seed/metrics.json"
            if $SKIP_EXISTING && [ -f "$result_file" ]; then
                echo "[SKIP] $method/K0/seed_$seed (exists)"
                continue
            fi
            echo "python -u $SCRIPT --method $method --K 0 --seed $seed --epochs $EPOCHS --n_folds $N_FOLDS --output_dir $OUTPUT_DIR" >> "$TASK_FILE"
        done
    else
        # BRMIL variants: loop over K and seeds
        for K in $K_VALUES; do
            for seed in $SEEDS; do
                K_tag="K$K"
                result_file="$OUTPUT_DIR/$method/$K_tag/seed_$seed/metrics.json"
                if $SKIP_EXISTING && [ -f "$result_file" ]; then
                    echo "[SKIP] $method/K$K/seed_$seed (exists)"
                    continue
                fi
                echo "python -u $SCRIPT --method $method --K $K --seed $seed --epochs $EPOCHS --n_folds $N_FOLDS --output_dir $OUTPUT_DIR" >> "$TASK_FILE"
            done
        done
    fi
done

TOTAL=$(wc -l < "$TASK_FILE")
echo ""
echo "Total tasks: $TOTAL"
echo ""

if [ "$TOTAL" -eq 0 ]; then
    echo "Nothing to run."
    exit 0
fi

echo "Starting parallel execution ($WORKERS workers)..."
echo ""

# Run tasks in parallel
cat "$TASK_FILE" | xargs -I{} -P "$WORKERS" bash -c '{}'

echo ""
echo "=========================================="
echo "  All tasks completed. Collecting results..."
echo "=========================================="

# Collect results
python3 -u - "$OUTPUT_DIR" <<'PYEOF'
import sys, json, os
from pathlib import Path
import numpy as np

output_dir = Path(sys.argv[1])
results = []
for jf in sorted(output_dir.rglob("metrics.json")):
    with open(jf) as f:
        r = json.load(f)
    results.append(r)

if not results:
    print("No results found.")
    sys.exit(0)

methods_order = ["abmil", "brmil_original", "brmil_auxloss", "brmil_gumbel",
                 "brmil_twostage", "brmil_fixed_l2", "brmil_fixed_dist",
                 "brmil_fixed_random"]

print(f"\n{'Method':<22} {'K':>4} {'AUC_mean':>9} {'AUC_std':>8} {'F1_mean':>8} {'F1_std':>7} {'Folds':>5} {'Time':>6}")
print("-" * 80)

for method in methods_order:
    matching = [r for r in results if r["method"] == method]
    if not matching:
        continue
    ks = sorted(set(r["K"] for r in matching))
    for K in ks:
        km = [r for r in matching if r["K"] == K]
        times = [r.get("elapsed_sec", 0) for r in km]

        # K-fold CV results have auc_mean/auc_std
        if "auc_mean" in km[0]:
            aucs = [r["auc_mean"] for r in km]
            f1s = [r["f1_mean"] for r in km]
            folds = km[0].get("n_valid_folds", "?")
            # Aggregate across seeds
            print(f"  {method:<20} {K:>4} {np.mean(aucs):>9.4f} {np.std(aucs):>8.4f} "
                  f"{np.mean(f1s):>8.4f} {np.std(f1s):>7.4f} {folds:>5} {sum(times):>5.0f}s")
        elif "auc" in km[0]:
            aucs = [r["auc"] for r in km]
            f1s = [r["f1"] for r in km]
            print(f"  {method:<20} {K:>4} {np.mean(aucs):>9.4f} {'':>8} "
                  f"{np.mean(f1s):>8.4f} {'':>7} {'1':>5} {sum(times):>5.0f}s")
        else:
            print(f"  {method:<20} {K:>4}     N/A")

import csv
csv_path = output_dir / "summary.csv"
all_keys = sorted(set(k for r in results for k in r.keys()))
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_keys)
    writer.writeheader()
    for r in sorted(results, key=lambda x: (x.get("method",""), x.get("K",0), x.get("seed",0))):
        writer.writerow(r)
print(f"\nSummary saved to: {csv_path}")
PYEOF
