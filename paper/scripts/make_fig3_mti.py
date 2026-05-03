#!/usr/bin/env python3
"""
Figure 3 (NeurIPS): Budget and pool-size analysis on MTI dataset.
===========================================================
(a) Performance vs expensive-token budget K (dual panel: PR-AUC, F1)
    - truncate@Kmax=512 and retrain@K curves
(b) Robustness to visible candidate pool size n at fixed K*=64

Usage:
  python paper/scripts/make_fig3_mti.py
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator, ScalarFormatter

# =========================================================
# Data
# =========================================================

K_ALL = np.array([1, 8, 16, 32, 64, 128, 256, 512])

# truncate@512 (from K=512 SWA best checkpoint)
TRUNC = {
    "f1":    [0.5504, 0.6531, 0.7029, 0.7396, 0.7640, 0.7799, 0.7888, 0.7925],
    "prauc": [0.5803, 0.6983, 0.7533, 0.7974, 0.8282, 0.8488, 0.8625, 0.8671],
}

# retrain@K (separately trained models for each K)
RETRAIN = {
    "f1":    [0.6854, 0.6925, 0.6963, 0.6916, 0.7708, 0.7786, 0.7790, 0.7918],
    "prauc": [0.6317, 0.6681, 0.7002, 0.5508, 0.8295, 0.8426, 0.8587, 0.8729],
}

# robustness vs n (K*=64, seed=2020, MTI test split)
N_PLOT = np.array([64, 128, 256, 512, 1024, 2048, 4096])  # 4096 = full-pool proxy
N_LABELS = ["64", "128", "256", "512", "1024", "2048", r"$\infty$"]
ROBUST = {
    "f1":    [0.3480, 0.4536, 0.5510, 0.6408, 0.7145, 0.7560, 0.7715],
    "prauc": [0.6389, 0.6761, 0.7173, 0.7568, 0.7876, 0.8052, 0.8117],
}

KSTAR = 64

# =========================================================
# Style
# =========================================================

C_BLUE   = "#0072B2"
C_ORANGE = "#D55E00"
C_GRAY   = "#7A7A7A"
GRID_C   = "#BDBDBD"


def _setup():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 7.2,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.0,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _style_ax(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", ls=":", lw=1.0, alpha=0.35, color=GRID_C)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _add_kstar(ax):
    ax.axvline(KSTAR, color=C_GRAY, ls=(0, (3, 3)), lw=1.2, zorder=1)
    yt = ax.get_ylim()[1]
    ax.annotate(
        r"$K^\star\!=\!" + f"{KSTAR}$",
        xy=(KSTAR, yt), xytext=(4, -2), textcoords="offset points",
        ha="left", va="top", color="0.20",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.2),
    )


def _save(fig, stem, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        p = out_dir / f"{stem}.{fmt}"
        fig.savefig(p, dpi=400, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved: {p}")
    plt.close(fig)


LEGEND_KW = dict(
    loc="lower right", bbox_to_anchor=(0.995, 0.02),
    borderaxespad=0.2, frameon=True, framealpha=0.95,
    edgecolor="#CFCFCF", fancybox=False,
    handlelength=1.25, handletextpad=0.45,
    borderpad=0.22, labelspacing=0.20,
    prop={"size": 6.8},
)


# =========================================================
# Figure 3a: Performance vs K (dual panel: PR-AUC + F1)
# =========================================================

def plot_fig3a(out_dir):
    _setup()
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.45), sharex=True)
    panels = [("PR-AUC", "prauc"), ("F1@0.5", "f1")]

    for ax, (ylabel, key) in zip(axes, panels):
        y_tr = np.array(TRUNC[key])
        y_re = np.array(RETRAIN[key])

        ax.plot(K_ALL, y_tr, color=C_BLUE, ls="-", lw=2.0,
                marker="o", ms=5.2, mfc=C_BLUE, mec="white", mew=0.8,
                label="truncate@Kmax", zorder=3)
        ax.plot(K_ALL, y_re, color=C_ORANGE, ls="--", lw=2.0,
                marker="^", ms=5.4, mfc=C_ORANGE, mec="white", mew=0.8,
                label="retrain@K", zorder=3)

        ax.set_xscale("log", base=2)
        ax.set_xticks(K_ALL)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.set_xlabel(r"Budget $K$")
        ax.set_ylabel(ylabel)

        _style_ax(ax)

        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        lo, hi = ax.get_ylim()
        ax.set_ylim(max(0.0, lo - 0.02), min(1.01, hi + 0.02))

        _add_kstar(ax)

    axes[0].text(0.02, 0.98, "(a)", transform=axes[0].transAxes,
                 ha="left", va="top", fontweight="bold")

    axes[1].legend(**LEGEND_KW)

    fig.subplots_adjust(wspace=0.28)
    _save(fig, "fig2_aupr_f1_MTI", out_dir)


# =========================================================
# Figure 3b: Robustness vs n (dual panel: PR-AUC + F1)
# =========================================================

def plot_fig3b(out_dir):
    _setup()
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.45), sharex=True)
    panels = [("PR-AUC", "prauc"), ("F1@0.5", "f1")]

    for ax, (ylabel, key) in zip(axes, panels):
        y = np.array(ROBUST[key])

        ax.plot(N_PLOT, y, color=C_BLUE, ls="-", lw=2.0,
                marker="o", ms=5.2, mfc=C_BLUE, mec="white", mew=0.8,
                zorder=3)

        ax.set_xscale("log", base=2)
        ax.set_xticks(N_PLOT)
        ax.set_xticklabels(N_LABELS)
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.set_xlabel(r"Visible pool size $n$")
        ax.set_ylabel(ylabel)

        _style_ax(ax)

        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        lo, hi = ax.get_ylim()
        ax.set_ylim(max(0.0, lo - 0.02), min(1.01, hi + 0.02))

    axes[0].text(0.02, 0.98, "(b)", transform=axes[0].transAxes,
                 ha="left", va="top", fontweight="bold")

    fig.subplots_adjust(wspace=0.28)
    _save(fig, "fig4_MTI", out_dir)


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "artifacts" / "figures"
    plot_fig3a(out_dir)
    plot_fig3b(out_dir)
    print("\nDone!")
