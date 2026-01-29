#!/usr/bin/env python3
"""
Figure 2 (ICML): Performance vs K (ICML-friendly)
=================================================
Changes (ICML reviewer-friendly):
- Split line/markers (plot) vs errorbars (errorbar fmt="none" + alpha) to reduce errorbar dominance.
- Force y-ticks to be evenly spaced (PR/ROC: 0.05, F1@0.5: 0.02).
- Clamp y-limits to [0, 1.01] for bounded metrics.
- Add (a)(b) panel labels for 1x2 plots.

Usage (single panel):
  python scripts/make_fig2_perf.py --csv artifacts/data/fig2_perf.csv --out artifacts/plots/fig2_perf_vs_k.pdf
  python scripts/make_fig2_perf.py --metric aupr --split test --out artifacts/plots/fig2_aupr.pdf
  python scripts/make_fig2_perf.py --metric "f1@0.5" --split test --out artifacts/plots/fig2_f1.pdf

Usage (two panels, 1x2):
  python scripts/make_fig2_perf.py --metrics aupr,f1@0.5 --split test --out artifacts/plots/fig2_aupr_f1.pdf

Enable K* marker (optional):
  python scripts/make_fig2_perf.py --metrics aupr,f1@0.5 --kstar 64 --out fig2_with_kstar.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, MultipleLocator
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
    }
    if s0 in mapping:
        return mapping[s0]

    low = s0.lower()
    if "truncate" in low or "trunc" in low:
        return "BR-MIL (truncate@Kmax)"
    if "retrain" in low or "per-k" in low or "perk" in low:
        return "BR-MIL (retrain@K)"
    if "targetnet" in low or "maxpool" in low:
        return "TargetNet (maxpool)"
    return s0


def _infer_curve_label(row: pd.Series) -> str:
    # 1) explicit K-independent baseline flag
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

    # 3) train_K vs K heuristic
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


def _resolve_wide_metric_column(df: pd.DataFrame, metric: str) -> str:
    aliases = {
        "aupr": ["aupr", "pr_auc", "prauc", "pr-auc", "prAUC", "AUPR"],
        "auroc": ["auroc", "roc_auc", "rocauc", "roc-auc", "rocAUC", "AUROC"],
        "f1@0.5": ["f1@0.5", "F1@0.5", "f1_at_0.5", "F1_at_0.5", "f1_0.5", "F1_0.5", "f1-0.5", "F1-0.5"],
    }
    m = metric.strip()
    candidates = [m, m.lower(), m.upper()] + aliases.get(m.lower(), [])
    for cand in candidates:
        if cand in df.columns:
            return cand
    raise ValueError(
        f"Could not find metric column for '{metric}'. "
        f"Provide long format with [metric,value] or wide format with a '{metric}' (or alias) column."
    )


def _ensure_long_format_multi(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """
    Supports:
      (A) long:  ... split, K, seed, metric, value
      (B) wide:  ... split, K, seed, <metric columns...>

    Output:
      curve_label, split, K, run_id, metric, value
    """
    k_col = _first_present(df, ["K", "k", "kmax", "budget"])
    if k_col is None:
        raise ValueError("Missing K/budget column. Expected one of: K, k, kmax, budget.")

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

    # Long format: keep all metrics as-is
    if metric_col is not None and value_col is not None:
        out = df.rename(columns={
            k_col: "K", split_col: "split", run_col: "run_id",
            metric_col: "metric", value_col: "value"
        }).copy()
        out["curve_label"] = out.apply(_infer_curve_label, axis=1)
        return out[["curve_label", "split", "K", "run_id", "metric", "value"]]

    # Wide format: stack requested metric columns into long
    base = df.rename(columns={k_col: "K", split_col: "split", run_col: "run_id"}).copy()
    outs = []
    for m in metrics:
        col = _resolve_wide_metric_column(base, m)
        tmp = base.copy()
        tmp["metric"] = m
        tmp["value"] = tmp[col].astype(float)
        tmp["curve_label"] = tmp.apply(_infer_curve_label, axis=1)
        outs.append(tmp[["curve_label", "split", "K", "run_id", "metric", "value"]])
    return pd.concat(outs, ignore_index=True)


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
    if _is_baseline(lbl):
        return (2, lbl)
    return (9, lbl)


def _setup_rcparams() -> None:
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


def _expand_k_independent_baselines(agg: pd.DataFrame) -> pd.DataFrame:
    all_K = sorted(agg["K"].unique().tolist())
    out = agg.copy()

    for lbl in agg["curve_label"].unique():
        if not _is_baseline(lbl):
            continue
        sub = agg[agg["curve_label"] == lbl].sort_values("K")
        if len(sub) == 1:
            y = float(sub["mean"].iloc[0])
            e = float(sub["std"].iloc[0])
            n = int(sub["n"].iloc[0])
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


def _metric_ylabel(metric: str) -> str:
    m = metric.strip().lower()
    if m in {"aupr", "pr_auc", "prauc", "pr-auc"}:
        return "PR-AUC"
    if m in {"auroc", "roc_auc", "rocauc", "roc-auc"}:
        return "ROC-AUC"
    if m == "f1@0.5":
        return "F1@0.5"
    if m.startswith("f1@"):
        return "F1@" + metric.split("@", 1)[1]
    return metric


def _apply_yaxis_rules(ax, metric: str) -> None:
    m = metric.strip().lower()
    if m in {"aupr", "pr_auc", "prauc", "pr-auc", "auroc", "roc_auc", "rocauc", "roc-auc"}:
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
    elif m == "f1@0.5" or m.startswith("f1@"):
        ax.yaxis.set_major_locator(MultipleLocator(0.02))

    # format + clamp (bounded metrics)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    lo, hi = ax.get_ylim()
    ax.set_ylim(max(0.0, lo), min(1.01, hi))


def _plot_on_ax(
    ax,
    agg: pd.DataFrame,
    metric: str,
    kstar: Optional[int],
    title: Optional[str],
    with_band: bool,
    show_inset: bool,
    show_legend: bool,
) -> None:
    # --- Spec constants (Okabe–Ito + ICML-friendly grays)
    C_BLUE   = "#0072B2"
    C_ORANGE = "#D55E00"
    C_GRAY   = "#7A7A7A"
    GRID_C   = "#BDBDBD"

    STYLE = dict(
        lw=2.0,
        ms_o=5.2,
        ms_t=5.4,
        m_ec="white",
        m_ew=0.8,
        eb_eline=1.0,
        eb_capsize=2.5,
        eb_capthick=1.0,
        eb_alpha=0.35,
        band_alpha=0.14,
        legend=dict(
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
        ),
    )

    ax.set_axisbelow(True)

    labels_all = sorted(agg["curve_label"].unique().tolist(), key=_rank_label)
    labels_base = [l for l in labels_all if _is_baseline(l)]
    labels_focus = [l for l in labels_all if not _is_baseline(l)]

    # Stable color + marker + linestyle mapping (BW-friendly)
    def _style(lbl: str):
        ll = lbl.lower()
        if "truncate" in ll:
            return dict(color=C_BLUE,   ls="-",  mk="o", ms=STYLE["ms_o"])
        if "retrain" in ll:
            return dict(color=C_ORANGE, ls="--", mk="^", ms=STYLE["ms_t"])
        if _is_baseline(lbl):
            return dict(color="0.35",   ls=":",  mk="s", ms=4.6)
        return dict(color="0.2", ls="-", mk="o", ms=STYLE["ms_o"])

    def _plot_one(ax0, labels, include_label=True, z=3):
        for lbl in labels:
            sub = agg[agg["curve_label"] == lbl].sort_values("K")
            x = sub["K"].to_numpy(dtype=float)
            y = sub["mean"].to_numpy(dtype=float)
            e = sub["std"].to_numpy(dtype=float)

            st = _style(lbl)

            # (i) line + markers (primary visual)
            ax0.plot(
                x, y,
                color=st["color"],
                linestyle=st["ls"],
                linewidth=STYLE["lw"] if not _is_baseline(lbl) else 1.4,
                marker=st["mk"],
                markersize=st["ms"],
                markerfacecolor=st["color"] if st["mk"] != "s" else "white",
                markeredgecolor=STYLE["m_ec"],
                markeredgewidth=STYLE["m_ew"],
                label=lbl if include_label else None,
                zorder=z,
            )

            # (ii) errorbars only (secondary visual)
            if np.any(e > 0):
                ax0.errorbar(
                    x, y, yerr=e,
                    fmt="none",
                    ecolor=st["color"],
                    elinewidth=STYLE["eb_eline"],
                    capsize=STYLE["eb_capsize"],
                    capthick=STYLE["eb_capthick"],
                    alpha=STYLE["eb_alpha"],
                    zorder=z - 1,
                )

            if with_band and np.any(e > 0):
                ax0.fill_between(
                    x, y - e, y + e,
                    color=st["color"],
                    alpha=STYLE["band_alpha"],
                    linewidth=0.0,
                    zorder=z - 2,
                )

    # Main axis: focus curves only if inset is used for baseline
    if show_inset and labels_base:
        _plot_one(ax, labels_focus, include_label=True, z=3)
    else:
        _plot_one(ax, labels_all, include_label=True, z=3)

    # X axis: log2 scale
    xticks = sorted(agg["K"].unique().tolist())
    ax.set_xscale("log", base=2)
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_xlabel(r"Budget $K$")

    # Y label
    ax.set_ylabel(_metric_ylabel(metric))

    # Grid: horizontal only
    ax.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.35, color=GRID_C)
    ax.grid(axis="x", visible=False)

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title:
        ax.set_title(title)

    # Y-limits (computed from plotted curves; baseline excluded if inset is used)
    if show_inset and labels_base:
        mask = ~agg["curve_label"].apply(_is_baseline)
        y_min = float(np.nanmin((agg.loc[mask, "mean"] - agg.loc[mask, "std"]).to_numpy()))
        y_max = float(np.nanmax((agg.loc[mask, "mean"] + agg.loc[mask, "std"]).to_numpy()))
    else:
        y_min = float(np.nanmin((agg["mean"] - agg["std"]).to_numpy()))
        y_max = float(np.nanmax((agg["mean"] + agg["std"]).to_numpy()))

    pad = 0.08 * max(1e-6, (y_max - y_min))
    ax.set_ylim(y_min - pad, y_max + 0.5 * pad)

    # Apply y-axis rules (even ticks + clamp + formatter)
    _apply_yaxis_rules(ax, metric)

    # K* marker + label (OFF unless kstar is provided)
    if kstar is not None:
        ax.axvline(kstar, color=C_GRAY, linestyle=(0, (3, 3)), linewidth=1.2, zorder=1)
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

    # Legend (optional; usually show only on right panel in 1x2)
    if show_legend:
        handles, labs = ax.get_legend_handles_labels()
        if show_inset and labels_base:
            base_lbl = labels_base[0] if labels_base else "TargetNet (maxpool)"
            handles.append(Line2D([0], [0],
                                  linestyle=":", marker="s",
                                  markersize=4.6, linewidth=1.4,
                                  color="0.35", markerfacecolor="white",
                                  markeredgecolor="white", markeredgewidth=0.8,
                                  label=base_lbl))
            labs.append(base_lbl)

        def _legend_short(lab: str) -> str:
            if lab.startswith("BR-MIL (") and lab.endswith(")"):
                return lab[len("BR-MIL ("):-1]
            return lab

        labs2 = [_legend_short(x) for x in labs]
        has_brmil = any(x in {"truncate@Kmax", "retrain@K"} for x in labs2)
        legend_title = "BR-MIL" if has_brmil else None

        ax.legend(
            handles, labs2,
            title=legend_title,
            **STYLE["legend"],
            ncol=1,
            markerscale=0.9,
            prop={"size": 6.8},
            title_fontsize=6.8,
        )

    # Inset (baseline context)
    if show_inset and labels_base:
        axins = inset_axes(ax, width="42%", height="42%", loc="lower left", borderpad=0.9)
        axins.set_axisbelow(True)
        _plot_one(axins, labels_all, include_label=False, z=3)

        axins.set_xscale("log", base=2)
        pick = [xticks[0], (kstar if (kstar in xticks) else xticks[len(xticks)//2]), xticks[-1]]
        pick = sorted({int(x) for x in pick if x is not None})
        axins.set_xticks(pick)
        axins.xaxis.set_major_formatter(ScalarFormatter())
        axins.tick_params(axis="x", which="minor", bottom=False)

        gy_min = float(np.nanmin((agg["mean"] - agg["std"]).to_numpy()))
        gy_max = float(np.nanmax((agg["mean"] + agg["std"]).to_numpy()))
        gpad = 0.10 * max(1e-6, (gy_max - gy_min))
        axins.set_ylim(gy_min - gpad, gy_max + 0.4 * gpad)
        _apply_yaxis_rules(axins, metric)

        axins.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35, color=GRID_C)
        axins.grid(axis="x", visible=False)
        axins.tick_params(axis="both", which="major", labelsize=6)

        if kstar is not None:
            axins.axvline(kstar, color="0.65", linestyle=(0, (3, 3)), linewidth=1.0)

        axins.spines["top"].set_visible(False)
        axins.spines["right"].set_visible(False)


def _plot_single(
    agg: pd.DataFrame,
    metric: str,
    out_path: Path,
    kstar: Optional[int],
    title: Optional[str],
    with_band: bool,
    show_inset: bool,
    dpi: int,
) -> None:
    _setup_rcparams()
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    _plot_on_ax(ax, agg, metric, kstar, title, with_band, show_inset, show_legend=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _plot_two_panel(
    agg_left: pd.DataFrame,
    metric_left: str,
    agg_right: pd.DataFrame,
    metric_right: str,
    out_path: Path,
    kstar: Optional[int],
    title: Optional[str],
    with_band: bool,
    show_inset: bool,
    dpi: int,
) -> None:
    _setup_rcparams()
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.45), sharex=True)
    ax0, ax1 = axes

    if title:
        fig.suptitle(title, y=0.995)

    _plot_on_ax(ax0, agg_left, metric_left, kstar, None, with_band, show_inset, show_legend=False)
    _plot_on_ax(ax1, agg_right, metric_right, kstar, None, with_band, show_inset, show_legend=True)

    # panel labels
    ax0.text(0.02, 0.98, "(a)", transform=ax0.transAxes, ha="left", va="top")
    ax1.text(0.02, 0.98, "(b)", transform=ax1.transAxes, ha="left", va="top")

    fig.subplots_adjust(wspace=0.28)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="artifacts/data/fig2_perf.csv")
    ap.add_argument("--out", type=str, default="artifacts/plots/fig2_perf_vs_k.pdf")

    ap.add_argument("--metric", type=str, default="aupr")
    ap.add_argument("--metrics", type=str, default=None,
                    help="Comma-separated metrics for 1x2 panel, e.g. aupr,f1@0.5. If set, overrides --metric.")

    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--kstar", type=int, default=None, help="If set (e.g. 64), draw K* vertical line + label.")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--no_inset", action="store_true", help="Disable inset (even if baseline is included).")
    ap.add_argument("--with_baseline", action="store_true", help="Include TargetNet baseline (and inset unless disabled).")
    ap.add_argument("--with_band", action="store_true", help="Also draw a light mean±std band (default: OFF, errorbar-only).")
    ap.add_argument("--dpi", type=int, default=400, help="Raster DPI (png). pdf ignores DPI mostly but safe.")

    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    if args.metrics is not None:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
        if len(metrics) != 2:
            raise ValueError(f"--metrics expects exactly 2 metrics (for 1x2). Got: {metrics}")
    else:
        metrics = [args.metric]

    df_long = _ensure_long_format_multi(df, metrics)

    agg_by_metric: Dict[str, pd.DataFrame] = {}
    for m in metrics:
        agg = _aggregate(df_long, split=args.split, metric=m)

        if args.with_baseline:
            agg = _expand_k_independent_baselines(agg)

        if not args.with_baseline:
            agg = agg[~agg["curve_label"].apply(_is_baseline)].copy()

        agg_by_metric[m] = agg

    out_path = Path(args.out)

    if len(metrics) == 1:
        m = metrics[0]
        _plot_single(
            agg=agg_by_metric[m],
            metric=m,
            out_path=out_path,
            kstar=args.kstar,
            title=args.title,
            with_band=args.with_band,
            show_inset=(args.with_baseline and (not args.no_inset)),
            dpi=args.dpi,
        )
    else:
        m0, m1 = metrics
        _plot_two_panel(
            agg_left=agg_by_metric[m0],
            metric_left=m0,
            agg_right=agg_by_metric[m1],
            metric_right=m1,
            out_path=out_path,
            kstar=args.kstar,
            title=args.title,
            with_band=args.with_band,
            show_inset=(args.with_baseline and (not args.no_inset)),
            dpi=args.dpi,
        )

    print(f"[OK] Wrote: {out_path.resolve()}")

    for m in metrics:
        pivot = agg_by_metric[m].pivot_table(index="K", columns="curve_label", values="mean")
        print(f"\nMeans (by K) | metric={m}:")
        print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
