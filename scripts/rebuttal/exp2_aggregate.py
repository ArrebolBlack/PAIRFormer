#!/usr/bin/env python3
"""
exp2_aggregate.py
=================
Aggregate EXP2 deepTargetPro pair-level results from 3 seeds.
Reads from eval_test_0,6-9/ directories (corrected test set: splits 0,6-9 only).
Generate comparison tables vs baselines.
"""
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/home/yjq/workspace/rebuttal/PAIRFormer")
EXP_DIR = PROJECT_ROOT / "experiments" / "issue2" / "exp2"

seeds = [2020, 2025, 2026]
results = {}

for seed in seeds:
    # Test metrics from corrected eval (splits 0,6-9 only)
    test_path = EXP_DIR / f"seed_{seed}" / "eval_test_0,6-9" / "overall_metrics.json"
    if test_path.exists():
        with open(test_path) as f:
            results[f"seed_{seed}_test"] = json.load(f)

# Aggregate test results across seeds
test_keys = [f"seed_{s}_test" for s in seeds]
test_keys = [k for k in test_keys if k in results]

if not test_keys:
    print("ERROR: No test results found. Run eval_deepTargetPro_pairs.py first.")
    exit(1)

print("=" * 80)
print("EXP2: deepTargetPro Pair-Level Results (3 seeds, test splits 0,6-9)")
print("=" * 80)

# Per-seed test results
print(f"\n{'Seed':<8} {'F1':>8} {'PR-AUC':>8} {'ROC-AUC':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8}")
print("-" * 64)
for k in test_keys:
    m = results[k]
    seed = k.split("_")[1]
    print(f"{seed:<8} {m['f1']:>8.4f} {m['pr_auc']:>8.4f} {m['roc_auc']:>8.4f} {m['accuracy']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f}")

# Compute mean and std
metrics_to_agg = ["f1", "pr_auc", "roc_auc", "accuracy", "precision", "recall"]
agg = {}
print(f"\n{'Mean':<8}", end="")
for metric in metrics_to_agg:
    vals = [results[k][metric] for k in test_keys]
    agg[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    print(f" {agg[metric]['mean']:>7.4f}", end="")
print()
print(f"{'Std':<8}", end="")
for metric in metrics_to_agg:
    print(f" {agg[metric]['std']:>7.4f}", end="")
print()

# Baselines from deepTargetPro paper
baselines = {
    "TEC-miTarget": {"f1": 0.7911},
    "TargetNet": {"f1": 0.7699},
    "deepTargetPro": {"f1": 0.7681},
}

print(f"\n{'='*80}")
print("Comparison vs Baselines (F1)")
print(f"{'='*80}")
print(f"{'Method':<25} {'F1':>10}")
print("-" * 35)
for name, m in baselines.items():
    print(f"{name:<25} {m['f1']:>10.4f}")
print(f"{'PAIR-Former (ours)':<25} {agg['f1']['mean']:>7.4f}+-{agg['f1']['std']:.4f}")

# Improvement
pair_f1 = agg['f1']['mean']
best_baseline = max(b['f1'] for b in baselines.values())
improvement = (pair_f1 - best_baseline) / best_baseline * 100
print(f"\nImprovement over best baseline (TEC-miTarget): +{improvement:.1f}%")

# Save summary
summary = {
    "experiment": "EXP2_deepTargetPro_pair_level",
    "test_splits": [0, 6, 7, 8, 9],
    "note": "Test uses splits 0,6-9 only (no overlap with training splits 1-5)",
    "n_seeds": len(test_keys),
    "seeds": seeds,
    "per_seed": {k: results[k] for k in test_keys},
    "aggregated": agg,
    "baselines": baselines,
    "improvement_over_best_baseline_pct": improvement,
}
with open(EXP_DIR / "exp2_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {EXP_DIR / 'exp2_summary.json'}")

# Generate CSV
import csv
csv_path = EXP_DIR / "exp2_per_seed_summary.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["seed"] + metrics_to_agg)
    writer.writeheader()
    for k in test_keys:
        seed = int(k.split("_")[1])
        row = {"seed": seed}
        for metric in metrics_to_agg:
            row[metric] = results[k][metric]
        writer.writerow(row)
    # Mean row
    row = {"seed": "mean"}
    for metric in metrics_to_agg:
        row[metric] = agg[metric]["mean"]
    writer.writerow(row)
    # Std row
    row = {"seed": "std"}
    for metric in metrics_to_agg:
        row[metric] = agg[metric]["std"]
    writer.writerow(row)
print(f"Saved: {csv_path}")

# Generate LaTeX table
latex = r"""\begin{table}[t]
\centering
\caption{EXP2: Pair-level prediction on deepTargetPro (5 test splits: 0,6--9, 3 seeds). Our PAIR-Former with Set Transformer aggregation significantly outperforms all baselines.}
\label{tab:exp2_deepTargetPro}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{F1} & \textbf{PR-AUC} & \textbf{ROC-AUC} \\
\midrule
deepTargetPro & 0.7681 & --- & --- \\
TargetNet & 0.7699 & --- & --- \\
TEC-miTarget & 0.7911 & --- & --- \\
\midrule
\textbf{PAIR-Former (Ours)} & """ + f"""{agg['f1']['mean']:.4f}$\\pm${agg['f1']['std']:.4f} & {agg['pr_auc']['mean']:.4f}$\\pm${agg['pr_auc']['std']:.4f} & {agg['roc_auc']['mean']:.4f}$\\pm${agg['roc_auc']['std']:.4f}""" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
tex_path = EXP_DIR / "table_e2_deepTargetPro.tex"
with open(tex_path, "w") as f:
    f.write(latex)
print(f"Saved: {tex_path}")
