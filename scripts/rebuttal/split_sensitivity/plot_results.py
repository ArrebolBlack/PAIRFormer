#!/usr/bin/env python3
"""
plot_results.py
===============
Bar chart comparing F1, PR-AUC, ROC-AUC across splits with error bars.
Reads from aggregate_results.py output (split_sensitivity_summary.json).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("scripts/rebuttal/split_sensitivity/results")
PLOT_DIR = Path("paper/artifacts/plots/rebuttal")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["f1", "pr_auc", "roc_auc"]
METRIC_LABELS = {"f1": "F1", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC"}
SPLIT_ORDER = ["splitA", "splitB", "splitC"]


def main():
    summary_path = RESULTS_DIR / "split_sensitivity_summary.json"
    if not summary_path.exists():
        print(f"Error: {summary_path} not found. Run aggregate_results.py first.")
        return

    with open(summary_path) as f:
        data = json.load(f)

    per_split = data["per_split"]

    # Build arrays
    splits_present = [s for s in SPLIT_ORDER if s in per_split]
    labels = [per_split[s]["label"] for s in splits_present]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    colors = ["#4C72B0", "#55A868", "#C44E52"]

    for ax, metric in zip(axes, METRICS):
        means = [per_split[s]["metrics"][metric]["mean"] for s in splits_present]
        stds = [per_split[s]["metrics"][metric]["std"] for s in splits_present]

        x = np.arange(len(splits_present))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors[:len(splits_present)],
                      edgecolor="black", linewidth=0.5, error_kw={"linewidth": 1.2})

        # Add value labels
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.002, f"{m:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(METRIC_LABELS[metric], fontsize=12, fontweight="bold")
        ax.set_ylim(0.95, 1.005)
        ax.axhline(y=means[0], color="gray", linestyle="--", alpha=0.3)
        ax.grid(axis="y", alpha=0.3)

        # Overall line
        if "overall" in data:
            ov = data["overall"][metric]
            ax.axhline(y=ov["mean"], color="red", linestyle=":", alpha=0.5, linewidth=1.5)

    axes[0].set_ylabel("Score", fontsize=11)

    fig.suptitle("Split Sensitivity: Performance Across Different Train/Test Partitions",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = RESULTS_DIR / "split_sensitivity_bar.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")

    out_path2 = PLOT_DIR / "split_sensitivity_bar.png"
    fig.savefig(out_path2, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path2}")

    plt.close()


if __name__ == "__main__":
    main()
