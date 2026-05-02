#!/usr/bin/env python3
"""
Figure 2 (ICML): Performance vs K — 3 datasets
================================================
1×3 panel figure showing PR-AUC vs budget K for:
  (a) miRAW   — original small dataset
  (b) deepTargetPro — external validation
  (c) MTI     — large-scale dataset (420K pairs)

Usage:
  python scripts/make_fig2_3dataset.py
  python scripts/make_fig2_3dataset.py --out artifacts/plots/fig2_3dataset.pdf
  python scripts/make_fig2_3dataset.py --metric aupr --dpi 300
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, MultipleLocator
from matplotlib.lines import Line2D


# ── Helpers ──────────────────────────────────────────────────────────────

DATASET_ORDER = ["miRAW", "deepTargetPro", "MTI"]
DATASET_LABELS = {
    "miRAW": "miRAW",
    "deepTargetPro": "deepTargetPro",
    "MTI": "MTI (ours)",
}

# Okabe–Ito palette
C_TRUNC  = "#0072B2"
C_RETRAIN = "#D55E00"
C_MAXPOOL = "#7A7A7A"
GRID_C    = "#BDBDBD"

CURVE_STYLE = {
    "trunc":    dict(color=C_TRUNC,  ls="-",  mk="o", ms=5.0, lw=2.0),
    "retrain":  dict(color=C_RETRAIN, ls="--", mk="^", ms=5.2, lw=2.0),
    "maxpool":  dict(color=C_MAXPOOL, ls=":",  mk="s", ms=4.4, lw=1.4),
}

CURVE_LABEL = {
    "trunc": "truncate@Kmax",
    "retrain": "retrain@K",
    "maxpool": "maxpool baseline",
}


def _setup_rcparams() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 7.2,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 6.8,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _aggregate(df: pd.DataFrame, dataset: str, metric: str, split: str = "test") -> pd.DataFrame:
    """Aggregate multi-seed data to mean ± std per (curve, K)."""
    mask = (
        (df["dataset"] == dataset)
        & (df["split"] == split)
        & (df["metric"].str.lower() == metric.lower())
    )
    d = df[mask].copy()
    if d.empty:
        return pd.DataFrame()

    d["K"] = d["K"].astype(int)
    g = (
        d.groupby(["curve", "K"], as_index=False)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n"})
    )
    g["std"] = g["std"].fillna(0.0)
    return g


def _plot_panel(
    ax,
    agg: pd.DataFrame,
    metric: str,
    title: str,
    show_legend: bool = True,
    kstar: Optional[int] = None,
) -> None:
    """Plot one dataset panel: retrain / trunc / maxpool curves."""
    ax.set_axisbelow(True)

    curves_in_data = sorted(agg["curve"].unique(), key=lambda c: {"trunc": 0, "retrain": 1, "maxpool": 2}.get(c, 9))

    for curve in curves_in_data:
        sub = agg[agg["curve"] == curve].sort_values("K")
        if sub.empty:
            continue

        x = sub["K"].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        e = sub["std"].to_numpy(dtype=float)

        st = CURVE_STYLE.get(curve, CURVE_STYLE["retrain"])

        # line + markers
        ax.plot(
            x, y,
            color=st["color"],
            linestyle=st["ls"],
            linewidth=st["lw"],
            marker=st["mk"],
            markersize=st["ms"],
            markerfacecolor=st["color"] if st["mk"] != "s" else "white",
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=CURVE_LABEL.get(curve, curve),
            zorder=3,
        )

        # error bars
        if np.any(e > 0):
            ax.errorbar(
                x, y, yerr=e,
                fmt="none",
                ecolor=st["color"],
                elinewidth=1.0,
                capsize=2.5,
                capthick=1.0,
                alpha=0.35,
                zorder=2,
            )

    # X axis: log2
    xticks = sorted(agg["K"].unique().tolist())
    ax.set_xscale("log", base=2)
    ax.set_xticks(xticks)
    sf = ScalarFormatter()
    sf.set_scientific(False)
    ax.xaxis.set_major_formatter(sf)
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_xlabel(r"Budget $K$")

    # Y axis
    m = metric.strip().lower()
    ylabel = "PR-AUC" if m in ("aupr", "pr_auc", "prauc") else metric
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # Grid
    ax.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.35, color=GRID_C)
    ax.grid(axis="x", visible=False)

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Title (dataset name)
    ax.set_title(title, fontsize=8.5, fontweight="medium", pad=6)

    # Y-limits with padding
    y_min = float(np.nanmin((agg["mean"] - agg["std"]).to_numpy()))
    y_max = float(np.nanmax((agg["mean"] + agg["std"]).to_numpy()))
    pad = 0.08 * max(1e-6, (y_max - y_min))
    ax.set_ylim(max(0.0, y_min - pad), min(1.01, y_max + 0.5 * pad))

    # K* vertical line
    if kstar is not None:
        ax.axvline(kstar, color=C_MAXPOOL, linestyle=(0, (3, 3)), linewidth=1.2, zorder=1)
        y_top = ax.get_ylim()[1]
        ax.annotate(
            rf"$K^\star={kstar}$",
            xy=(kstar, y_top),
            xytext=(4, -2),
            textcoords="offset points",
            ha="left", va="top",
            color="0.20",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.2),
        )

    # Legend
    if show_legend:
        handles, labs = ax.get_legend_handles_labels()
        ax.legend(
            handles, labs,
            loc="lower right",
            bbox_to_anchor=(0.995, 0.02),
            borderaxespad=0.2,
            frameon=True,
            framealpha=0.95,
            edgecolor="#CFCFCF",
            fancybox=False,
            handlelength=1.25,
            handletextpad=0.45,
            borderpad=0.22,
            labelspacing=0.20,
            ncol=1,
            markerscale=0.9,
            prop={"size": 6.5},
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="paper/artifacts/data/fig2_3dataset.csv")
    ap.add_argument("--out", type=str, default="paper/artifacts/plots/fig2_3dataset.pdf")
    ap.add_argument("--metric", type=str, default="aupr")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--kstar", type=int, default=64)
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    _setup_rcparams()

    # Aggregate per dataset
    agg_by_ds: Dict[str, pd.DataFrame] = {}
    for ds in DATASET_ORDER:
        agg = _aggregate(df, ds, metric=args.metric, split=args.split)
        if agg.empty:
            print(f"[WARN] No data for dataset={ds}, metric={args.metric}")
            continue
        agg_by_ds[ds] = agg

    n_panels = len(agg_by_ds)
    if n_panels == 0:
        raise ValueError("No data to plot.")

    # 1×N layout
    fig_width = 3.35 * n_panels + 0.3 * (n_panels - 1)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 2.45), sharey=False)

    if n_panels == 1:
        axes = [axes]

    panel_labels = "(a)", "(b)", "(c)", "(d)", "(e)"

    for i, (ax, ds) in enumerate(zip(axes, agg_by_ds.keys())):
        show_leg = (i == n_panels - 1)  # legend on last panel
        _plot_panel(
            ax,
            agg_by_ds[ds],
            metric=args.metric,
            title=DATASET_LABELS.get(ds, ds),
            show_legend=show_leg,
            kstar=args.kstar,
        )
        # Panel label
        ax.text(0.02, 0.98, panel_labels[i], transform=ax.transAxes,
                ha="left", va="top", fontsize=8.5, fontweight="bold")

    fig.subplots_adjust(wspace=0.32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] Wrote: {out_path.resolve()}")

    # Print summary table
    for ds, agg in agg_by_ds.items():
        pivot = agg.pivot_table(index="K", columns="curve", values="mean")
        print(f"\nMeans (by K) | dataset={ds}:")
        print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
