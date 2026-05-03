#!/usr/bin/env python3
"""
plot_n_distribution_3ds.py
==========================
3-dataset version of the n_distribution figure.

Shows heavy-tail distribution across miRAW, deepTargetPro, and MTI datasets,
demonstrating the necessity of budgeted approach in PAIR-Former.

Usage:
  python scripts/rebuttal/n_distribution/plot_n_distribution_3ds.py

  # With specific result files
  python scripts/rebuttal/n_distribution/plot_n_distribution_3ds.py \
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
PAPER_PLOTS_DIR = Path(__file__).resolve().parents[3] / "paper" / "artifacts" / "plots" / "rebuttal"

DATASET_ORDER = ["miRAW", "deepTargetPro", "MTI_new"]
NICE_NAMES = {
    "miRAW": "miRAW",
    "deepTargetPro": "deepTargetPro",
    "MTI_new": "MTI (ours)",
}
COLORS = {
    "miRAW": "#2563EB",          # blue
    "deepTargetPro": "#DC2626",  # red
    "MTI_new": "#16A34A",        # green
}
K_COLOR = "#EF4444"
BUDGET_K = 64


def load_results(results_dir):
    """Load all result JSON files."""
    results = {}
    file_map = {
        "miRAW": "miRAW_n_distribution.json",
        "deepTargetPro": "deepTargetPro_n_distribution.json",
        "MTI_new": "MTI_n_distribution.json",
    }
    for ds, fname in file_map.items():
        path = os.path.join(results_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                results[ds] = json.load(f)
    return results


def style_axis(ax, logx=False, logy=False):
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.18, linewidth=0.7)
    ax.grid(False, which="minor")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)


def plot_main_figure(results, output_dir):
    """
    Publication-style Figure: 3 datasets.
    - top row: 3 per-dataset histograms
    - bottom row: ECDF comparison (main panel)
    """
    dataset_names = [ds for ds in DATASET_ORDER if ds in results]
    n_ds = len(dataset_names)

    fig = plt.figure(figsize=(14, 7.8), dpi=200)
    gs = fig.add_gridspec(
        2, n_ds,
        height_ratios=[1.0, 1.45],
        hspace=0.30,
        wspace=0.18,
    )

    # ── Top row: Histograms ──
    for i, ds in enumerate(dataset_names):
        ax = fig.add_subplot(gs[0, i])

        data = results[ds]
        n_vals = np.array(data["n_values"], dtype=np.float64)
        s = data["statistics"]
        color = COLORS.get(ds, "#333333")

        xmax = max(n_vals.max(), 2)
        bins = np.logspace(0, np.ceil(np.log10(xmax)), 45)

        ax.hist(n_vals, bins=bins, color=color, alpha=0.88, edgecolor="none")
        style_axis(ax, logx=True, logy=True)

        ax.set_xlim(left=0.8)
        ax.set_xlabel("CTS count per pair, n", fontsize=10)
        if i == 0:
            ax.set_ylabel("Pair count", fontsize=10)

        panel_title = f"({chr(ord('a') + i)}) {NICE_NAMES.get(ds, ds)}"
        ax.set_title(panel_title, fontsize=11, fontweight="bold", pad=6)

        # budget line
        if n_vals.max() > BUDGET_K:
            ax.axvline(x=BUDGET_K, color=K_COLOR, linestyle=(0, (4, 2)),
                       linewidth=1.5, alpha=0.95)
            ax.text(BUDGET_K * 1.03, ax.get_ylim()[1] / 2.2,
                    f"K={BUDGET_K}", color=K_COLOR, fontsize=8,
                    fontweight="bold", ha="left", va="center")

        frac_gt_k = 100.0 * (n_vals > BUDGET_K).sum() / len(n_vals)
        stats_text = (
            f"N={s['num_pairs']:,} | med={s['n_median']:.0f}\n"
            f">K={BUDGET_K}: {frac_gt_k:.1f}%"
        )
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                          edgecolor="#BBBBBB", linewidth=0.6, alpha=0.92))

    # ── Bottom row: ECDF (main panel, spanning all columns) ──
    ax_cdf = fig.add_subplot(gs[1, :])

    summary_lines = []
    for ds in dataset_names:
        data = results[ds]
        s = data["statistics"]
        n_vals = np.array(data["n_values"], dtype=np.float64)
        sorted_n = np.sort(n_vals)
        ecdf = np.arange(1, len(sorted_n) + 1) / len(sorted_n)
        color = COLORS.get(ds, "#333333")

        ax_cdf.plot(sorted_n, ecdf, color=color, linewidth=2.2,
                    label=NICE_NAMES.get(ds, ds))

        frac_gt_k = 100.0 * (n_vals > BUDGET_K).sum() / len(n_vals)
        summary_lines.append(
            f"{NICE_NAMES.get(ds, ds)}: median={s['n_median']:.0f}, >{BUDGET_K}={frac_gt_k:.1f}%"
        )

    # budget line
    ax_cdf.axvline(x=BUDGET_K, color=K_COLOR, linestyle=(0, (4, 2)),
                   linewidth=1.6, alpha=0.95, label=f"Budget K={BUDGET_K}")

    # percentile reference lines
    for y, label in [(0.5, "P50"), (0.95, "P95")]:
        ax_cdf.axhline(y=y, color="#888888", linestyle=":", linewidth=0.9, alpha=0.8)
        ax_cdf.text(0.01, y + 0.01, label, transform=ax_cdf.transAxes,
                    fontsize=8, color="#666666", va="bottom")

    style_axis(ax_cdf, logx=True, logy=False)
    ax_cdf.set_xlim(left=0.8)
    ax_cdf.set_ylim(0, 1.02)

    panel_idx = chr(ord('a') + n_ds)
    ax_cdf.set_title(f"({panel_idx}) ECDF across datasets", fontsize=11,
                     fontweight="bold", pad=6)
    ax_cdf.set_xlabel("CTS count per pair, n", fontsize=11)
    ax_cdf.set_ylabel("Fraction of pairs with count \u2264 n", fontsize=11)

    ax_cdf.legend(fontsize=9, loc="lower right", frameon=True,
                  framealpha=0.92, edgecolor="#CCCCCC")

    # summary box
    ax_cdf.text(0.015, 0.98, "\n".join(summary_lines),
                transform=ax_cdf.transAxes, fontsize=9, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                          edgecolor="#BBBBBB", linewidth=0.7, alpha=0.94))

    ax_cdf.text(BUDGET_K * 1.06, 0.08, f"K={BUDGET_K}",
                color=K_COLOR, fontsize=9, fontweight="bold",
                ha="left", va="bottom")

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ["pdf", "png"]:
        out_path = output_dir / f"n_distribution_3ds.{fmt}"
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")

    PAPER_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ["pdf", "png"]:
        out_path = PAPER_PLOTS_DIR / f"n_distribution_3ds.{fmt}"
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.close(fig)


def plot_summary_table(results, output_dir):
    """Create a LaTeX-ready summary table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / "n_distribution_table_3ds.tex"

    with open(tex_path, "w") as f:
        f.write("\\begin{tabular}{lrrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Dataset & Pairs & Mean & Median & P95 & P99 & Max & "
                "Pairs $> K{=}64$ \\\\\n")
        f.write("\\midrule\n")

        for ds in DATASET_ORDER:
            if ds not in results:
                continue
            data = results[ds]
            s = data["statistics"]
            n_vals = np.array(data["n_values"])
            pct_gt_64 = 100.0 * (n_vals > 64).sum() / len(n_vals)
            f.write(
                f"{NICE_NAMES.get(ds, ds)} & "
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
    parser.add_argument("--results_dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--output_dir", type=str, default=str(PLOTS_DIR))
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No result files found in {args.results_dir}")
        sys.exit(1)

    print(f"Loaded results for: {list(results.keys())}")

    plot_main_figure(results, args.output_dir)
    plot_summary_table(results, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
