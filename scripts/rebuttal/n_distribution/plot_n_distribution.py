#!/usr/bin/env python3
"""
plot_n_distribution.py
=======================
Rebuttal Point 14: Publication-quality visualization of CTS count (n) distribution.

Shows heavy-tail distribution across datasets, demonstrating the necessity of
budgeted approach in PAIR-Former.

Usage:
  python scripts/rebuttal/n_distribution/plot_n_distribution.py

  # With specific result files
  python scripts/rebuttal/n_distribution/plot_n_distribution.py \
    --results_dir scripts/rebuttal/n_distribution/results
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PAPER_PLOTS_DIR = PROJECT_ROOT = Path(__file__).resolve().parents[3] / "paper" / "artifacts" / "plots" / "rebuttal"


def load_results(results_dir):
    """Load all result JSON files."""
    results = {}
    for ds in ["miRAW", "deepTargetPro", "MTI"]:
        path = os.path.join(results_dir, f"{ds}_n_distribution.json")
        if os.path.exists(path):
            with open(path) as f:
                results[ds] = json.load(f)
    return results


def plot_main_figure(results, output_dir):
    """
    Main figure: 3-panel figure showing n distribution for each dataset.

    Top row: Histogram of n (log-scale y-axis) to show heavy-tail
    Bottom: Combined CDF comparison
    """
    dataset_names = list(results.keys())
    n_datasets = len(dataset_names)

    # Color scheme
    colors = {
        "miRAW": "#2563EB",        # Blue
        "deepTargetPro": "#DC2626", # Red
        "MTI": "#059669",           # Green
    }
    nice_names = {
        "miRAW": "miRAW",
        "deepTargetPro": "deepTargetPro",
        "MTI": "MTI-mRNALevel",
    }

    fig = plt.figure(figsize=(14, 10), dpi=150)

    # ---- Top row: individual histograms ----
    for i, ds in enumerate(dataset_names):
        ax = fig.add_subplot(2, n_datasets, i + 1)
        data = results[ds]
        n_vals = np.array(data["n_values"], dtype=np.float64)
        s = data["statistics"]
        color = colors.get(ds, "#333333")

        # Histogram with log scale
        bins = np.logspace(0, np.log10(max(n_vals.max(), 2)), 60)
        ax.hist(n_vals, bins=bins, color=color, alpha=0.85, edgecolor="white",
                linewidth=0.3)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of CTS (n)", fontsize=10)
        if i == 0:
            ax.set_ylabel("Number of pairs", fontsize=10)
        ax.set_title(nice_names.get(ds, ds), fontsize=12, fontweight="bold")

        # Annotate key stats
        stats_text = (
            f"N = {s['num_pairs']:,}\n"
            f"Median = {s['n_median']:.0f}\n"
            f"P95 = {s['n_p95']:.0f}\n"
            f"Max = {s['n_max']:,}"
        )
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Mark budget K=64
        if n_vals.max() > 64:
            ax.axvline(x=64, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
            ax.text(64, ax.get_ylim()[1] * 0.5, "K=64", color="red",
                    fontsize=8, fontweight="bold", ha="left", va="center")

        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlim(left=0.8)

    # ---- Bottom: combined CDF ----
    ax_cdf = fig.add_subplot(2, 1, 2)
    for ds in dataset_names:
        data = results[ds]
        s = data["statistics"]
        n_vals = np.array(data["n_values"], dtype=np.float64)
        sorted_n = np.sort(n_vals)
        cumprob = np.arange(1, len(sorted_n) + 1) / len(sorted_n)
        color = colors.get(ds, "#333333")

        ax_cdf.plot(sorted_n, cumprob, color=color, linewidth=2,
                    label=f"{nice_names.get(ds, ds)} (N={s['num_pairs']:,}, "
                          f"median={s['n_median']:.0f})")

    # Mark K=64 budget line
    ax_cdf.axvline(x=64, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
                   label="Budget K=64")
    ax_cdf.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax_cdf.axhline(y=0.9, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax_cdf.axhline(y=0.95, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    ax_cdf.set_xlabel("Number of CTS per pair (n)", fontsize=12)
    ax_cdf.set_ylabel("Cumulative fraction of pairs", fontsize=12)
    ax_cdf.set_title("Empirical CDF: CTS Count per Pair Across Datasets",
                     fontsize=13, fontweight="bold")
    ax_cdf.legend(fontsize=10, loc="lower right")
    ax_cdf.set_xscale("log")
    ax_cdf.set_xlim(left=0.8)
    ax_cdf.set_ylim(bottom=0, top=1.02)
    ax_cdf.grid(True, alpha=0.3, which="both")

    # Add percentile annotations
    ax_cdf.text(0.02, 0.52, "P50", transform=ax_cdf.transAxes, fontsize=8,
                color="gray", va="bottom")
    ax_cdf.text(0.02, 0.92, "P95", transform=ax_cdf.transAxes, fontsize=8,
                color="gray", va="bottom")

    plt.tight_layout(h_pad=3.0)

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in ["pdf", "png"]:
        out_path = output_dir / f"n_distribution.{fmt}"
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {out_path}")

    # Also save to paper artifacts
    PAPER_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ["pdf", "png"]:
        out_path = PAPER_PLOTS_DIR / f"n_distribution.{fmt}"
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {out_path}")

    plt.close(fig)


def plot_summary_table(results, output_dir):
    """Create a LaTeX-ready summary table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tex_path = output_dir / "n_distribution_table.tex"
    nice_names = {
        "miRAW": "miRAW",
        "deepTargetPro": "deepTargetPro",
        "MTI": "MTI-mRNALevel",
    }

    with open(tex_path, "w") as f:
        f.write("\\begin{tabular}{lrrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Dataset & Pairs & Mean & Median & P95 & P99 & Max & "
                "Pairs $> K{=}64$ \\\\\n")
        f.write("\\midrule\n")

        for ds, data in results.items():
            s = data["statistics"]
            n_vals = np.array(data["n_values"])
            pct_gt_64 = 100.0 * (n_vals > 64).sum() / len(n_vals)
            f.write(
                f"{nice_names.get(ds, ds)} & "
                f"{s['num_pairs']:,} & "
                f"{s['n_mean']:.0f} & "
                f"{s['n_median']:.0f} & "
                f"{s['n_p95']:.0f} & "
                f"{s['n_p99']:.0f} & "
                f"{s['n_max']:,} & "
                f"{pct_gt_64:.1f}\\% \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    print(f"Saved: {tex_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Directory with result JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PLOTS_DIR),
        help="Directory to save plots",
    )
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No result files found in {args.results_dir}")
        print("Run compute_n_distribution.py first.")
        sys.exit(1)

    print(f"Loaded results for: {list(results.keys())}")

    plot_main_figure(results, args.output_dir)
    plot_summary_table(results, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
