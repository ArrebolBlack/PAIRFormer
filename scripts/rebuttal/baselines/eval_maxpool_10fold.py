#!/usr/bin/env python3
"""
Run MaxPool baseline evaluation on all 10 folds.
Uses TargetNet_Optimized dp-0.1 checkpoint with test_reduction=max.
"""
import subprocess
import json
import glob
import numpy as np
import sys
import os

CKPT = "checkpoints/miRAW_TargetNet_Optimized_dp-0.1/checkpoints/best.pt"
RESULTS_DIR = "scripts/rebuttal/baselines/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

results = {}
for fold in range(10):
    test_file = f"data/rebuttal/miRAW_fold{fold}_split/miRAW_Test_nosplit.txt"
    if not os.path.exists(test_file):
        print(f"Fold {fold}: {test_file} not found, skipping")
        continue

    print(f"\n{'='*50}")
    print(f"Fold {fold}")
    print(f"{'='*50}")

    cmd = [
        sys.executable, "-m", "src.launch.eval",
        "experiment=miRAW_TargetNet_Optimized",
        "run.mode=eval",
        f"run.checkpoint={CKPT}",
        "run.test_reduction=max",
        f"data.path.test={test_file}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Find the latest metrics output
    pattern = f"outputs/miRAW_TargetNet_Optimized/*/eval/test/test/max/thr0_5/metrics.json"
    files = sorted(glob.glob(pattern))
    if files:
        with open(files[-1]) as f:
            m = json.load(f)
        results[fold] = m
        print(f"  F1={m['f1']:.4f}  ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
              f"Acc={m['accuracy']:.4f}  Spec={m['specificity']:.4f}")
        # Clean up old outputs to avoid confusion
        for old_f in files[:-1]:
            pass  # don't delete, just use latest
    else:
        print(f"  ERROR: no metrics found")
        print(result.stderr[-500:] if result.stderr else "")

if results:
    metrics = ['f1', 'roc_auc', 'pr_auc', 'accuracy', 'specificity', 'npv']
    print(f"\n{'='*60}")
    print(f"  MaxPool Baseline 10-fold Results (N={len(results)})")
    print(f"{'='*60}")
    for m_name in metrics:
        vals = [results[f][m_name] for f in results]
        print(f"  {m_name:12s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # Save
    summary = {
        'model': 'MaxPool_TN-Opt_dp0.1',
        'n_folds': len(results),
        'per_fold': {},
    }
    for m_name in metrics:
        vals = [results[f][m_name] for f in results]
        summary[f'{m_name}_mean'] = float(np.mean(vals))
        summary[f'{m_name}_std'] = float(np.std(vals))

    for fold in results:
        summary['per_fold'][str(fold)] = {k: results[fold][k] for k in metrics}

    out_path = os.path.join(RESULTS_DIR, 'maxpool_10fold_summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")
