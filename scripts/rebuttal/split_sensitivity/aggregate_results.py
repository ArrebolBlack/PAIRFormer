#!/usr/bin/env python3
"""
aggregate_results.py
=====================
Aggregate split sensitivity results across all runs.

Reads metrics from each experiment run + existing baseline,
computes mean ± std per split and overall, generates LaTeX table + bar chart.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EXPERIMENTS_DIR = Path("experiments/split_sensitivity")
BASELINE_METRICS = "checkpoints/BR-MIL/eval/test/test/best/sweep/metrics.json"
OUTPUT_DIR = Path("scripts/rebuttal/split_sensitivity/results")

KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]

SPLIT_INFO = {
    "splitA": {"label": "A (original)", "train": "{1-5}", "test": "{0,6-9}"},
    "splitB": {"label": "B (consecutive)", "train": "{0-4}", "test": "{5-9}"},
    "splitC": {"label": "C (interleaved)", "train": "{0,2,4,6,8}", "test": "{1,3,5,7,9}"},
}


def load_metrics(metrics_path):
    """Load metrics JSON."""
    with open(metrics_path) as f:
        return json.load(f)


def find_run_metrics(split, seed):
    """Find sweep metrics for a given split and seed."""
    if split == "splitA" and seed == 2020:
        path = Path(BASELINE_METRICS)
        if path.exists():
            return load_metrics(path), "baseline"

    # Look in experiments directory
    pattern = EXPERIMENTS_DIR / split / f"seed_{seed}"
    sweep_path = pattern / "eval" / "test" / "test" / "best" / "sweep" / "metrics.json"
    if not sweep_path.exists():
        sweep_path = pattern / "eval" / "test" / "test" / "ckpt_best" / "sweep" / "metrics.json"

    if sweep_path.exists():
        return load_metrics(sweep_path), str(sweep_path)

    # Check last checkpoint too
    sweep_path = pattern / "eval" / "test" / "test" / "last" / "sweep" / "metrics.json"
    if sweep_path.exists():
        return load_metrics(sweep_path), str(sweep_path)

    return None, None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = ["splitA", "splitB", "splitC"]
    seeds = [2020, 2025, 2026]

    all_results = defaultdict(lambda: defaultdict(dict))
    missing = []

    for split in splits:
        for seed in seeds:
            metrics, source = find_run_metrics(split, seed)
            if metrics is None:
                print(f"  MISSING: {split} seed={seed}")
                missing.append((split, seed))
                continue
            all_results[split][seed] = {k: metrics.get(k, 0.0) for k in KEYS}
            print(f"  Found: {split} seed={seed} F1={metrics.get('f1',0):.4f} "
                  f"ROC-AUC={metrics.get('roc_auc',0):.4f} PR-AUC={metrics.get('pr_auc',0):.4f}")

    if missing:
        print(f"\nWarning: {len(missing)} runs missing:")
        for s, sd in missing:
            print(f"  {s} seed={sd}")
        print("Proceeding with available results...\n")

    # Compute statistics per split
    summary = {}
    for split in splits:
        if not all_results[split]:
            continue
        values = defaultdict(list)
        for seed, m in all_results[split].items():
            for k in KEYS:
                values[k].append(m[k])

        info = SPLIT_INFO[split]
        summary[split] = {
            "label": info["label"],
            "train_subsets": info["train"],
            "test_subsets": info["test"],
            "n_seeds": len(all_results[split]),
        }
        for k in KEYS:
            arr = np.array(values[k])
            summary[split][k] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "values": values[k],
            }

    # Print summary table
    print("\n" + "=" * 80)
    print("SPLIT SENSITIVITY RESULTS")
    print("=" * 80)
    print(f"{'Split':<20} {'Train':<10} {'Test':<12} {'F1':>14} {'PR-AUC':>14} {'ROC-AUC':>14}")
    print("-" * 80)
    for split in splits:
        if split not in summary:
            continue
        s = summary[split]
        f1 = s["f1"]
        pr = s["pr_auc"]
        roc = s["roc_auc"]
        print(f"{s['label']:<20} {s['train_subsets']:<10} {s['test_subsets']:<12} "
              f"{f1['mean']:.4f}±{f1['std']:.4f} {pr['mean']:.4f}±{pr['std']:.4f} "
              f"{roc['mean']:.4f}±{roc['std']:.4f}")

    # Overall stability
    all_f1 = []
    all_pr = []
    all_roc = []
    for split in splits:
        if split not in summary:
            continue
        all_f1.extend(summary[split]["f1"]["values"])
        all_pr.extend(summary[split]["pr_auc"]["values"])
        all_roc.extend(summary[split]["roc_auc"]["values"])

    if all_f1:
        print(f"\nOverall across all splits & seeds:")
        print(f"  F1:      {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f} (n={len(all_f1)})")
        print(f"  PR-AUC:  {np.mean(all_pr):.4f} ± {np.std(all_pr):.4f}")
        print(f"  ROC-AUC: {np.mean(all_roc):.4f} ± {np.std(all_roc):.4f}")

    # Save JSON
    out = {
        "experiment": "split_sensitivity",
        "n_splits": len(summary),
        "seeds": seeds,
        "per_split": {},
    }
    for split, s in summary.items():
        out["per_split"][split] = {
            "label": s["label"],
            "train_subsets": s["train_subsets"],
            "test_subsets": s["test_subsets"],
            "n_seeds": s["n_seeds"],
            "metrics": {k: {"mean": s[k]["mean"], "std": s[k]["std"],
                            "values": s[k]["values"]} for k in KEYS},
        }
    if all_f1:
        out["overall"] = {
            "n_runs": len(all_f1),
            "f1": {"mean": float(np.mean(all_f1)), "std": float(np.std(all_f1))},
            "pr_auc": {"mean": float(np.mean(all_pr)), "std": float(np.std(all_pr))},
            "roc_auc": {"mean": float(np.mean(all_roc)), "std": float(np.std(all_roc))},
        }

    out_path = OUTPUT_DIR / "split_sensitivity_summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # LaTeX table
    tex_path = OUTPUT_DIR / "split_sensitivity_table.tex"
    with open(tex_path, "w") as f:
        f.write("\\begin{tabular}{llcccccc}\n")
        f.write("\\toprule\n")
        f.write("Split & Subsets & \\multicolumn{2}{c}{F1} & \\multicolumn{2}{c}{PR-AUC} & "
                "\\multicolumn{2}{c}{ROC-AUC} \\\\\n")
        f.write("\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}\n")
        f.write(" & (Train/Test) & Mean & Std & Mean & Std & Mean & Std \\\\\n")
        f.write("\\midrule\n")
        for split in splits:
            if split not in summary:
                continue
            s = summary[split]
            f.write(f"{s['label']} & {s['train_subsets']}/{s['test_subsets']} & "
                    f"{s['f1']['mean']:.4f} & {s['f1']['std']:.4f} & "
                    f"{s['pr_auc']['mean']:.4f} & {s['pr_auc']['std']:.4f} & "
                    f"{s['roc_auc']['mean']:.4f} & {s['roc_auc']['std']:.4f} \\\\\n")
        if all_f1:
            f.write("\\midrule\n")
            f.write(f"\\textbf{{Overall}} & & "
                    f"\\textbf{{{np.mean(all_f1):.4f}}} & \\textbf{{{np.std(all_f1):.4f}}} & "
                    f"\\textbf{{{np.mean(all_pr):.4f}}} & \\textbf{{{np.std(all_pr):.4f}}} & "
                    f"\\textbf{{{np.mean(all_roc):.4f}}} & \\textbf{{{np.std(all_roc):.4f}}} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"Saved: {tex_path}")


if __name__ == "__main__":
    main()
