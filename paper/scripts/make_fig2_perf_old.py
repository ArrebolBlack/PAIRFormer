#!/usr/bin/env python3
"""
Figure 2 (ICML): Performance vs K
=================================
- ICML single-column friendly
- Supports long/wide CSV
- Robust CSV parsing (comma / tab / etc.)
- Main axis focuses on high-performance region (BR-MIL curves)
- Baseline (TargetNet maxpool) shown in an inset to avoid y-axis being dragged down

Usage:
  python scripts/make_fig2_perf.py --csv artifacts/data/fig2_perf.csv --out artifacts/plots/fig2_perf_vs_k.pdf
  python scripts/make_fig2_perf.py --metric aupr --split test --kstar 64
  python scripts/make_fig2_perf.py --out artifacts/plots/fig2_perf_vs_k.png
  
  python scripts/make_fig2_perf.py --with_baseline --out artifacts/plots/fig2_perf_vs_k.png

  
  python scripts/make_fig2_perf_old.py --out artifacts/plots/fig2_perf_vs_k_old.png

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# ----------------------------
# Helpers: schema/format
# ----------------------------

def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_method_name(s: str) -> str:
    """
    Normalize method strings into publication-ready labels.
    Keep naming minimal and stable for paper drafts.
    """
    s0 = str(s).strip()
    mapping = {
        "BRMIL": "BR-MIL",
        "BR-MIL": "BR-MIL",
        "PAIRFormer": "PAIR-Former",
        "PAIR-Former": "PAIR-Former",
        "TargetNet_like_maxpool": "TargetNet (maxpool)",
        "TargetNet_like": "TargetNet (maxpool)",
        "TargetNet-like": "TargetNet (maxpool)",
        "Max pooling": "TargetNet (maxpool)",
        "maxpool": "TargetNet (maxpool)",
        "TargetNet": "TargetNet (maxpool)",
        "naive": "Naive (scan-all + topK)",
    }
    if s0 in mapping:
        return mapping[s0]

    low = s0.lower()
    if "truncate" in low or "trunc" in low:
        return "BR-MIL (truncate@Kmax)"
    if "retrain" in low or "per-k" in low or "perk" in low:
        return "BR-MIL (retrain@K)"
    if "targetnet" in low:
        return "TargetNet (maxpool)"
    return s0


def _infer_curve_label(row: pd.Series) -> str:
    """
    Decide which curve a row belongs to (truncate vs retrain vs baseline).
    Priority:
      1) explicit columns (is_k_independent / curve/eval_mode/setting/mode)
      2) train_K vs K
      3) method string
    """
    # 1) explicit K-independent baseline flag (robust to bool / str)
    if "is_k_independent" in row.index:
        v = row["is_k_independent"]
        if isinstance(v, (bool, np.bool_)) and v:
            return "TargetNet (maxpool)"
        if isinstance(v, str) and v.strip().lower() in {"true", "1", "yes"}:
            return "TargetNet (maxpool)"

    # 2) explicit curve mode
    for col in ("curve", "eval_mode", "setting", "mode"):
        if col in row.index and pd.notna(row[col]):
            val = str(row[col]).lower()
            if "trunc" in val or "truncate" in val:
                return "BR-MIL (truncate@Kmax)"
            if "retrain" in val or "per-k" in val or "perk" in val:
                return "BR-MIL (retrain@K)"
            if "maxpool" in val or "targetnet" in val:
                return "TargetNet (maxpool)"

    # 3) train_K vs K heuristic (if exists)
    if "train_K" in row.index and pd.notna(row["train_K"]) and ("K" in row.index):
        try:
            tk = int(float(row["train_K"]))
            k = int(float(row["K"]))
            return "BR-MIL (truncate@Kmax)" if tk != k else "BR-MIL (retrain@K)"
        except Exception:
            pass

    # 4) fallback: method-like columns
    for c in ("method", "pipeline", "model", "name"):
        if c in row.index and pd.notna(row[c]):
            return _normalize_method_name(row[c])

    return "method"


def _ensure_long_format(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Supports:
      (A) long:  ... split, K, seed, metric, value
      (B) wide:  ... split, K, seed, aupr/auroc/...
    Output columns:
      curve_label, split, K, run_id, metric, value
    """
    k_col = _first_present(df, ["K", "k", "kmax", "budget"])
    if k_col is None:
        raise ValueError("Could not find a K/budget column. Expected one of: K, k, kmax, budget.")

    split_col = _first_present(df, ["split", "partition"])
    if split_col is None:
        df = df.copy()
        df["split"] = "test"
        split_col = "split"

    run_col = _first_present(df, ["seed", "repeat_id", "run_id", "rep", "trial"])
    if run_col is None:
        df = df.copy()
        df["repeat_id"] = 0
        run_col = "repeat_id"

    metric_col = _first_present(df, ["metric", "Metric"])
    value_col = _first_present(df, ["value", "Value", "score", "Score"])

    # Long format
    if metric_col is not None and value_col is not None:
        out = df.rename(columns={
            k_col: "K", split_col: "split", run_col: "run_id",
            metric_col: "metric", value_col: "value"
        }).copy()
        out["curve_label"] = out.apply(_infer_curve_label, axis=1)
        return out[["curve_label", "split", "K", "run_id", "metric", "value"]]

    # Wide format: find metric by aliases
    aliases = {
        "aupr": ["aupr", "pr_auc", "prauc", "pr-auc", "prAUC"],
        "auroc": ["auroc", "roc_auc", "rocauc", "roc-auc", "rocAUC"],
    }
    wide_metric_col = None
    candidates = [metric, metric.lower(), metric.upper()] + aliases.get(metric.lower(), [])
    for cand in candidates:
        if cand in df.columns:
            wide_metric_col = cand
            break
    if wide_metric_col is None:
        raise ValueError(
            f"Could not find metric column for '{metric}'. "
            f"Provide long format with [metric,value] or wide format with a '{metric}' column."
        )

    out = df.rename(columns={k_col: "K", split_col: "split", run_col: "run_id"}).copy()
    out["metric"] = metric
    out["value"] = out[wide_metric_col].astype(float)
    out["curve_label"] = out.apply(_infer_curve_label, axis=1)
    return out[["curve_label", "split", "K", "run_id", "metric", "value"]]


def _aggregate(df_long: pd.DataFrame, split: str, metric: str) -> pd.DataFrame:
    d = df_long[
        (df_long["split"].astype(str) == str(split)) &
        (df_long["metric"].astype(str).str.lower() == metric.lower())
    ].copy()
    if d.empty:
        raise ValueError(f"No rows found for split='{split}' and metric='{metric}'.")

    d["K"] = d["K"].astype(int)

    g = (
        d.groupby(["curve_label", "K"], as_index=False)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n"})
    )
    g["std"] = g["std"].fillna(0.0)
    return g


# ----------------------------
# Plotting (ICML style)
# ----------------------------

def _is_baseline(label: str) -> bool:
    l = str(label).lower()
    return ("targetnet" in l) or ("maxpool" in l)


def _rank_label(lbl: str) -> Tuple[int, str]:
    l = str(lbl).lower()
    if "truncate" in l:
        return (0, lbl)
    if "retrain" in l:
        return (1, lbl)
    if "br-mil" in l:
        return (2, lbl)
    if _is_baseline(lbl):
        return (3, lbl)
    return (9, lbl)


def _setup_rcparams() -> None:
    """ICML-friendly typography: small fonts, thin lines."""
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.3,
    })


def _expand_k_independent_baselines(agg: pd.DataFrame) -> pd.DataFrame:
    """
    If a baseline curve has only 1 K point, replicate it across all K for plotting a horizontal line.
    This allows providing a single baseline value while keeping the main axis zoomed.
    """
    all_K = sorted(agg["K"].unique().tolist())
    out = agg.copy()

    for lbl in agg["curve_label"].unique():
        if not _is_baseline(lbl):
            continue
        sub = agg[agg["curve_label"] == lbl].sort_values("K")
        if len(sub) == 1:
            y = float(sub["mean"].iloc[0])
            e = float(sub["std"].iloc[0]) if "std" in sub.columns else 0.0
            n = int(sub["n"].iloc[0]) if "n" in sub.columns else 1
            expanded = pd.DataFrame({
                "curve_label": [lbl] * len(all_K),
                "K": all_K,
                "mean": [y] * len(all_K),
                "std": [e] * len(all_K),
                "n": [n] * len(all_K),
            })
            out = pd.concat([out, expanded], ignore_index=True)

    out = out.drop_duplicates(subset=["curve_label", "K"], keep="last")
    return out


def _make_tick_formatter(xticks: List[int]):
    """Stable tick formatter for log2 axis: float-safe matching."""
    xt = np.asarray(xticks, dtype=float)

    def _fmt(v, pos):
        if np.any(np.isclose(v, xt, rtol=0, atol=1e-8)):
            return f"{int(round(v))}"
        return ""
    return FuncFormatter(_fmt)


def _plot(
    agg: pd.DataFrame,
    metric: str,
    out_path: Path,
    kstar: Optional[int],
    title: Optional[str],
    show_inset: bool = True,
) -> None:
    # One place to tune layout/typography later
    STYLE: Dict[str, object] = dict(
        figsize=(3.35, 2.35),
        margins=dict(left=0.18, right=0.98, bottom=0.22, top=0.95),
        legend=dict(
            loc="lower right",
            frameon=True,
            fancybox=False,
            borderpad=0.25,
            labelspacing=0.25,
            handlelength=1.5,
            handletextpad=0.6,
            framealpha=0.95,
        ),
        kstar=dict(line_color="0.55", line_width=0.9, text_color="0.20", text_offset=(3, -2)),
        inset=dict(width="42%", height="42%", loc="lower left", borderpad=0.9),
    )

    _setup_rcparams()

    fig = plt.figure(figsize=STYLE["figsize"])
    ax = plt.gca()
    ax.set_axisbelow(True)

    labels_all = sorted(agg["curve_label"].unique().tolist(), key=_rank_label)
    labels_base = [l for l in labels_all if _is_baseline(l)]
    labels_focus = [l for l in labels_all if not _is_baseline(l)]

    # Stable color assignment (Matplotlib default palette)
    color_map = {
        "BR-MIL (truncate@Kmax)": "C0",
        "BR-MIL (retrain@K)": "C1",
        "TargetNet (maxpool)": "C2",
    }

    def _linestyle(lbl: str) -> str:
        ll = str(lbl).lower()
        if "truncate" in ll:
            return "-"
        if "retrain" in ll:
            return "--"
        if _is_baseline(lbl):
            return ":"
        return "-"

    def _plot_one_axis(ax0, labels, with_band: bool, lw: float, ms: float, cap: float):
        for lbl in labels:
            sub = agg[agg["curve_label"] == lbl].sort_values("K")
            x = sub["K"].to_numpy()
            y = sub["mean"].to_numpy()
            e = sub["std"].to_numpy()

            ax0.errorbar(
                x, y, yerr=e,
                marker="o",
                markersize=ms,
                linestyle=_linestyle(lbl),
                linewidth=lw,
                capsize=cap,
                elinewidth=max(0.8, lw * 0.9),
                label=lbl,
                color=color_map.get(lbl, None),
                zorder=3,
            )
            if with_band and np.any(e > 0):
                ax0.fill_between(
                    x, y - e, y + e,
                    alpha=0.10,
                    color=color_map.get(lbl, None),
                    zorder=2,
                )

    # Main axis: focus region (only BR-MIL unless user disables inset)
    if show_inset and labels_base:
        _plot_one_axis(ax, labels_focus, with_band=True, lw=1.4, ms=4.5, cap=2.0)
    else:
        _plot_one_axis(ax, labels_all, with_band=True, lw=1.4, ms=4.5, cap=2.0)

    # X axis (log2)
    ax.set_xscale("log", base=2)
    xticks = sorted(agg["K"].unique().tolist())
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(_make_tick_formatter(xticks))
    ax.set_xlabel(r"Budget $K$ (log2 scale)")

    # Y label
    ylab = {
        "aupr": "PR-AUC",
        "pr_auc": "PR-AUC",
        "prauc": "PR-AUC",
        "auroc": "ROC-AUC",
        "roc_auc": "ROC-AUC",
    }.get(metric.lower(), metric)
    ax.set_ylabel(ylab)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # Subtle grid
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)

    if title:
        ax.set_title(title)

    # K* marker (main axis)
    if kstar is not None:
        ks = STYLE["kstar"]
        ax.axvline(
            kstar,
            linewidth=ks["line_width"],
            linestyle="--",
            color=ks["line_color"],
            zorder=1,
        )
        y_top = float(np.nanmax(agg["mean"] + agg["std"]))
        ax.annotate(
            rf"$K^*={kstar}$",
            xy=(kstar, y_top),
            xytext=ks["text_offset"],
            textcoords="offset points",
            ha="left",
            va="top",
            color=ks["text_color"],
        )

    # Y-limits: zoom if baseline is in inset
    if show_inset and labels_base:
        focus_mask = agg["curve_label"].apply(lambda s: not _is_baseline(s))
        y_min = float(np.nanmin((agg.loc[focus_mask, "mean"] - agg.loc[focus_mask, "std"]).to_numpy()))
        y_max = float(np.nanmax((agg.loc[focus_mask, "mean"] + agg.loc[focus_mask, "std"]).to_numpy()))
    else:
        y_min = float(np.nanmin((agg["mean"] - agg["std"]).to_numpy()))
        y_max = float(np.nanmax((agg["mean"] + agg["std"]).to_numpy()))

    pad = 0.08 * max(1e-6, (y_max - y_min))
    ax.set_ylim(y_min - pad, y_max + 0.5 * pad)

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend: if baseline only appears in inset, add a proxy handle
    handles, labs = ax.get_legend_handles_labels()
    if show_inset and labels_base:
        base_lbl = "TargetNet (maxpool)"
        handles.append(Line2D([0], [0],
                              linestyle=":", marker="o",
                              markersize=4.5, linewidth=1.2,
                              color=color_map.get(base_lbl, "C2"),
                              label=base_lbl))
        labs.append(base_lbl)

    ax.legend(**STYLE["legend"])

    # Inset: show global scale including baseline (context only)
    if show_inset and labels_base:
        ins = STYLE["inset"]
        axins = inset_axes(
            ax,
            width=ins["width"],
            height=ins["height"],
            loc=ins["loc"],
            borderpad=ins["borderpad"],
        )
        axins.set_axisbelow(True)

        _plot_one_axis(axins, labels_all, with_band=False, lw=1.0, ms=3.2, cap=1.6)

        axins.set_xscale("log", base=2)
        # fewer ticks in inset
        pick = [xticks[0], kstar if (kstar in xticks) else xticks[len(xticks)//2], xticks[-1]]
        pick = sorted({int(x) for x in pick if x is not None})
        axins.set_xticks(pick)
        axins.xaxis.set_major_formatter(_make_tick_formatter(pick))

        gy_min = float(np.nanmin((agg["mean"] - agg["std"]).to_numpy()))
        gy_max = float(np.nanmax((agg["mean"] + agg["std"]).to_numpy()))
        gpad = 0.10 * max(1e-6, (gy_max - gy_min))
        axins.set_ylim(gy_min - gpad, gy_max + 0.4 * gpad)

        axins.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
        axins.tick_params(axis="both", which="major", labelsize=6)

        if kstar is not None:
            axins.axvline(kstar, linewidth=0.8, linestyle="--", color="0.65")

        axins.spines["top"].set_visible(False)
        axins.spines["right"].set_visible(False)

    # Save (avoid tight_layout/bbox_inches tight with inset_axes for stability)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(**STYLE["margins"])
    fig.canvas.draw()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="artifacts/data/fig2_perf.csv")
    ap.add_argument("--out", type=str, default="artifacts/plots/fig2_perf_vs_k.pdf")
    ap.add_argument("--metric", type=str, default="aupr")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--kstar", type=int, default=64)
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--no_inset", action="store_true", help="Disable inset (baseline context window).")

    # NEW: 默认不画 baseline；需要时显式打开
    ap.add_argument(
        "--with_baseline",
        action="store_true",
        help="Include TargetNet baseline (and inset if not disabled).",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    # Robust read: auto-detect delimiter (comma/tab/etc.) + handle UTF-8 BOM
    df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    df_long = _ensure_long_format(df, args.metric)
    agg = _aggregate(df_long, split=args.split, metric=args.metric)

    # If baseline is enabled, allow providing a single baseline value and expand to all K
    if args.with_baseline:
        agg = _expand_k_independent_baselines(agg)

    # Plot-time: optionally drop baseline curves entirely
    if not args.with_baseline:
        agg_plot = agg[~agg["curve_label"].apply(_is_baseline)].copy()
    else:
        agg_plot = agg

    _plot(
        agg=agg_plot,
        metric=args.metric,
        out_path=Path(args.out),
        kstar=args.kstar,
        title=args.title,
        # inset only makes sense when baseline is included
        show_inset=(not args.no_inset) and args.with_baseline,
    )

    print(f"[OK] Wrote: {Path(args.out).resolve()}")

    pivot = agg_plot.pivot_table(index="K", columns="curve_label", values="mean")
    print("\nMeans (by K):")
    print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
