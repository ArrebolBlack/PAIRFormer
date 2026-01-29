#!/usr/bin/env python3
# paper/scripts/make_fig3_compute.py
#
# Implements:
#   Figure-A: 2x3 K-sweep curves (6 metrics) with errorbars = std over repeats
#            - NO per-subplot legends
#            - ONE global legend at bottom (horizontal)
#            - NO grey dashed k* line
#            - reference-like: solid line + clear dot marker; only color differs by pipeline
#            - wider horizontal spacing; y-label closer to axis
#
#   Figure-B: 1x3 stage breakdown bars at fixed K (default K=64), shared y-axis across subplots
#            - y-axis in log scale (to avoid domination by very large stages)
#            - bar colors use the SAME global pipeline colors as Figure-A
#            - errorbar color follows pipeline (darker shade of same hue for visibility)
#            - ensure errorbars are not occluded by bars (higher zorder + drawn after bars)
#
# Inputs: three CSVs (each may contain pipeline,K,repeat_id plus metric columns).
# Output: PDF + PNG for both figures.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator, LogLocator, NullLocator
from matplotlib.lines import Line2D


# ----------------------------
# Paths (paper/...)
# ----------------------------
PAPER_DIR = Path(__file__).resolve().parents[1]  # paper/
ARTIFACTS_DIR = PAPER_DIR / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
PLOTS_DIR = ARTIFACTS_DIR / "plots"


# ----------------------------
# ICML rcParams (Fig2-like base)
# ----------------------------
def set_icml_rcparams():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 6.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 7.2,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.3,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


GRID_C = "#BDBDBD"

# Canonical pipeline names required by spec:
PIPELINES_CANON = ["BR-MIL_online", "Naive_online", "TargetNet_like_online"]


# ============================================================
# GLOBAL COLOR TABLE (used by BOTH FigA and FigB)  ✅
# (Reference-like palette you already chose for FigA)
# ============================================================
PIPE_COLORS: Dict[str, str] = {
    "BR-MIL_online": "#00A087",           # teal
    "Naive_online": "#E64B35",            # red/orange
    "TargetNet_like_online": "#3C5488",   # blue
}

# FigA curve style (reference vibe): solid + clear dot, only color differs
PSTYLE_A: Dict[str, Dict] = {
    p: dict(color=PIPE_COLORS[p], ls="-", mk="o", lw=1.6, ms=3.3)
    for p in PIPELINES_CANON
}

# FigB bar style: SAME colors as FigA
PSTYLE_B: Dict[str, Dict] = {
    p: dict(color=PIPE_COLORS[p])
    for p in PIPELINES_CANON
}

# FigA global legend at bottom, horizontal (reference-like)
FIGA_LEGEND_STYLE = dict(
    loc="lower center",
    bbox_to_anchor=(0.5, 0.02),  # uses enlarged bottom margin
    frameon=False,
    ncol=3,
    columnspacing=1.4,
    handlelength=2.0,
    handletextpad=0.6,
    borderaxespad=0.0,
)


# ----------------------------
# Robust CSV loading
# ----------------------------
def _resolve_under_paper(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = (PAPER_DIR / path).resolve()
    return path


def read_csv_auto(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    for c in ["pipeline", "K"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if "repeat_id" not in df.columns:
        if "seed" in df.columns:
            df["repeat_id"] = df["seed"]
        else:
            df["repeat_id"] = 0

    df["pipeline"] = df["pipeline"].astype(str)

    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    df = df[df["K"].notna()].copy()
    df["K"] = df["K"].astype(int)

    df["repeat_id"] = pd.to_numeric(df["repeat_id"], errors="coerce").fillna(0).astype(int)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df


# ----------------------------
# Pipeline canonicalization
# ----------------------------
def canonicalize_pipeline(p: str) -> str:
    s = str(p).strip()
    low = s.lower()

    if ("br-mil" in low) or ("brmil" in low) or ("pair-former" in low) or ("pairformer" in low) or ("pair" in low):
        return "BR-MIL_online"
    if "naive" in low:
        return "Naive_online"
    if ("targetnet" in low) or ("maxpool" in low) or ("max_pool" in low) or ("max pooling" in low):
        return "TargetNet_like_online"

    return s


# ----------------------------
# Filters / de-mix runs
# ----------------------------
def maybe_filter_basic(df: pd.DataFrame, split: Optional[str], seed: Optional[int]) -> pd.DataFrame:
    out = df
    if split is not None and "split" in out.columns:
        out = out[out["split"].astype(str) == str(split)]
    if seed is not None and "seed" in out.columns:
        out = out[pd.to_numeric(out["seed"], errors="coerce").fillna(-1).astype(int) == int(seed)]
    return out


def apply_where(df: pd.DataFrame, where: str) -> pd.DataFrame:
    if not where:
        return df
    try:
        return df.query(where)
    except Exception as e:
        raise ValueError(f"Invalid --where query: {where}\nError: {e}") from e


def keep_latest_run_per_group(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns or df["timestamp"].isna().all():
        return df
    df = df.copy()
    latest = (df.groupby(["pipeline", "K"], as_index=False)["timestamp"]
              .max()
              .rename(columns={"timestamp": "_ts_max"}))
    df = df.merge(latest, on=["pipeline", "K"], how="left")
    df = df[df["timestamp"] == df["_ts_max"]].drop(columns=["_ts_max"])
    return df


# ----------------------------
# JSON parsing: segment_time_mean_json -> t_next_batch_ms
# ----------------------------
def _parse_json_cell(x) -> dict:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return {}
        try:
            return json.loads(s)
        except Exception:
            return {}
    return {}


def add_t_next_batch_ms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    col = "segment_time_mean_json"
    if col not in df.columns:
        df["t_next_batch_ms"] = np.nan
        return df

    vals = []
    for v in df[col].tolist():
        obj = _parse_json_cell(v)
        t = obj.get("t_next_batch", None)
        try:
            vals.append(np.nan if t is None else 1000.0 * float(t))
        except Exception:
            vals.append(np.nan)

    df["t_next_batch_ms"] = np.array(vals, dtype=float)
    return df


# ----------------------------
# Aggregation over repeats
# ----------------------------
def agg_over_repeats(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    d = df[["pipeline", "K", "repeat_id", value_col]].copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[value_col])
    if d.empty:
        return pd.DataFrame(columns=["pipeline", "K", "mean", "std", "n"])

    g = (d.groupby(["pipeline", "K"], as_index=False)[value_col]
         .agg(["mean", "std", "count"])
         .reset_index()
         .rename(columns={"count": "n"}))
    g["std"] = g["std"].fillna(0.0)
    return g


# ----------------------------
# Plot helpers
# ----------------------------
def format_x_as_log2(ax, ticks: List[int]):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False)


def _style_for_pipeline(p: str, style_map: Dict[str, Dict]) -> Dict:
    if p not in style_map:
        return dict(color="0.2", ls="-", mk="o", lw=1.6, ms=3.0)
    return style_map[p]


def _common_axes_style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", linewidth=0.9, alpha=0.35, color=GRID_C)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _derive_out_paths(out_arg: str) -> Tuple[Path, Path]:
    out_path = _resolve_under_paper(out_arg)
    if out_path.suffix.lower() in {".png", ".pdf"}:
        out_path = out_path.with_suffix("")
    pdf_path = out_path.with_suffix(".pdf")
    png_path = out_path.with_suffix(".png")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    return pdf_path, png_path


def _hex_to_rgb01(h: str) -> Tuple[float, float, float]:
    h = h.strip().lstrip("#")
    if len(h) != 6:
        return (0.2, 0.2, 0.2)
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return (r, g, b)


def _darken(hex_color: str, factor: float = 0.78) -> Tuple[float, float, float]:
    """
    Darken a hex color by multiplying RGB by factor (<1).
    Returns RGB tuple in [0,1].
    """
    r, g, b = _hex_to_rgb01(hex_color)
    return (max(0.0, r * factor), max(0.0, g * factor), max(0.0, b * factor))


# ----------------------------
# Figure-A: 2x3 K-sweep curves (6 metrics)
# ----------------------------
FIGA_METRICS = [
    ("t_total_mean", "t_total_mean", "End-to-end total", "Time (s)"),
    ("throughput_pairs_per_s_mean", "throughput_pairs_per_s_mean", "Throughput", "pairs/s"),
    ("peak_vram_gb_mean", "peak_vram_gb_mean", "Peak VRAM", "GB"),
    ("profile_B_gather_all_cpu_ms", "profile_B_gather_all_cpu_ms", "Profile: gather_all_cpu", "ms"),
    ("profile_A_pair_slices_ms", "profile_A_pair_slices_ms", "Profile: pair_slices", "ms"),
    ("t_next_batch_ms", "t_next_batch_ms", "Segment: next_batch", "ms"),
]


def _errorbar_figA(ax, x, y, e, label: str, st: Dict):
    # reference-like: clean markers + solid line
    ax.errorbar(
        x, y, yerr=e,
        color=st["color"],
        linestyle=st["ls"],
        linewidth=st["lw"],
        marker=st["mk"],
        markersize=st["ms"],
        markerfacecolor=st["color"],
        markeredgecolor=st["color"],
        markeredgewidth=0.0,
        elinewidth=0.9,
        capsize=2.0,
        capthick=0.9,
        label=label,
        zorder=3,
    )


def plot_figA_k_sweep(
    df: pd.DataFrame,
    ticks: List[int],
    kstar: Optional[int],   # kept for compatibility, but NOT drawn
    out_stem: str,
    dpi: int,
):
    df = add_t_next_batch_ms(df)

    # Layout: wider horizontal spacing + enlarged bottom margin for global legend
    nrows, ncols = 2, 3
    fig_w, fig_h = 6.9, 3.35
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=True, constrained_layout=False)
    axes = np.array(axes).reshape(-1)

    fig.subplots_adjust(
        left=0.07, right=0.995, top=0.93, bottom=0.20,
        wspace=0.38, hspace=0.42   # ↑ 增大上下间距
    )


    for i, (_, col, title, yunit) in enumerate(FIGA_METRICS):
        ax = axes[i]
        _common_axes_style(ax)
        ax.set_title(title)

        if col in df.columns:
            agg = agg_over_repeats(df, col)
        else:
            agg = pd.DataFrame()

        if not agg.empty:
            for p in PIPELINES_CANON:
                sub = agg[agg["pipeline"] == p].sort_values("K")
                if sub.empty:
                    continue
                st = _style_for_pipeline(p, PSTYLE_A)
                _errorbar_figA(
                    ax,
                    sub["K"].to_numpy(dtype=float),
                    sub["mean"].to_numpy(dtype=float),
                    sub["std"].to_numpy(dtype=float),
                    label=p,
                    st=st,
                )

        format_x_as_log2(ax, ticks)
        ax.set_xlabel(r"$K$")
        ax.set_ylabel(yunit, labelpad=1.2)  # y-label closer
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        # Per your command: NO kstar / grey dashed line -> do nothing

    # ---- ONE global legend at bottom (no per-subplot legends)
    handles = []
    labels = []
    for p in PIPELINES_CANON:
        st = _style_for_pipeline(p, PSTYLE_A)
        handles.append(
            Line2D([0], [0],
                   color=st["color"], linestyle=st["ls"], linewidth=st["lw"],
                   marker=st["mk"], markersize=st["ms"],
                   markerfacecolor=st["color"], markeredgecolor=st["color"])
        )
        labels.append(p)

    fig.legend(handles, labels, **FIGA_LEGEND_STYLE)

    pdf_path, png_path = _derive_out_paths(out_stem)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=int(dpi), facecolor="white")
    plt.close(fig)
    print(f"[OK] Figure-A saved: {pdf_path}")
    print(f"[OK] Figure-A saved: {png_path}")


# ----------------------------
# Figure-B: 1x3 stage breakdown bars at fixed K (log y-axis)
# ----------------------------
STAGES = [
    ("A_pair_slices", "A_pair_slices"),
    ("B_gather_all_cpu", "B_gather_all_cpu"),
    ("C_scan_or_cheap_forward", "C_scan_or_cheap_forward"),
    ("D_selector", "D_selector"),
    ("E_build_or_pad_K_batch", "E_build_or_pad_K_batch"),
    ("F_instance_encode_or_assemble", "F_instance_encode_or_assemble"),
    ("G_assemble_or_agg_forward", "G_assemble_or_agg_forward"),
    ("H_agg_forward", "H_agg_forward"),
]

STAGE_MAP: Dict[str, Dict[str, Optional[str]]] = {
    "BR-MIL_online": {
        "A_pair_slices": "profile_A_pair_slices_ms",
        "B_gather_all_cpu": "profile_B_gather_all_cpu_ms",
        "C_scan_or_cheap_forward": "profile_C_cheap_forward_ms",
        "D_selector": "profile_D_selector_ms",
        "E_build_or_pad_K_batch": "profile_E_build_batch_cpu_K_ms",
        "F_instance_encode_or_assemble": "profile_F_instance_encode_ms",
        "G_assemble_or_agg_forward": "profile_G_assemble_tokens_ms",
        "H_agg_forward": "profile_H_agg_forward_ms",
    },
    "Naive_online": {
        "A_pair_slices": "profile_A_pair_slices_ms",
        "B_gather_all_cpu": "profile_B_gather_all_cpu_ms",
        "C_scan_or_cheap_forward": "profile_C_expensive_scan_all_ms",
        "D_selector": "profile_D_selector_ms",
        "E_build_or_pad_K_batch": "profile_E_gather_K_to_padded_ms",
        "F_instance_encode_or_assemble": "profile_F_assemble_tokens_ms",
        "G_assemble_or_agg_forward": "profile_G_agg_forward_ms",
        "H_agg_forward": None,
    },
    "TargetNet_like_online": {
        "A_pair_slices": "profile_A_pair_slices_ms",
        "B_gather_all_cpu": "profile_B_gather_all_cpu_ms",
        "C_scan_or_cheap_forward": "profile_C_expensive_scan_all_ms",
        "D_selector": None,
        "E_build_or_pad_K_batch": None,
        "F_instance_encode_or_assemble": None,
        "G_assemble_or_agg_forward": None,
        "H_agg_forward": None,
    },
}


def _choose_k_for_bar(df: pd.DataFrame, k_default: int) -> int:
    avail = {}
    for p in PIPELINES_CANON:
        ks = sorted(df[df["pipeline"] == p]["K"].unique().tolist())
        avail[p] = set(ks)

    if all(k_default in avail[p] for p in PIPELINES_CANON):
        return k_default

    common = set.intersection(*[avail[p] for p in PIPELINES_CANON if len(avail[p]) > 0])
    if common:
        return int(max(common))

    return int(k_default)


def _stage_stats_at_k(df: pd.DataFrame, pipeline: str, kbar: int) -> Tuple[np.ndarray, np.ndarray, float]:
    sub = df[(df["pipeline"] == pipeline) & (df["K"] == int(kbar))].copy()

    means, stds = [], []
    mapping = STAGE_MAP[pipeline]

    for stage_key, _ in STAGES:
        col = mapping.get(stage_key, None)
        if col is None or col not in sub.columns:
            means.append(0.0)
            stds.append(0.0)
            continue

        v = pd.to_numeric(sub[col], errors="coerce").dropna()
        if v.empty:
            means.append(0.0)
            stds.append(0.0)
        else:
            means.append(float(v.mean()))
            stds.append(float(v.std(ddof=1)) if len(v) >= 2 else 0.0)

    total_ms = np.nan
    if "t_total_mean" in sub.columns:
        vt = pd.to_numeric(sub["t_total_mean"], errors="coerce").dropna()
        if not vt.empty:
            total_ms = float(vt.mean()) * 1000.0

    return np.array(means, dtype=float), np.array(stds, dtype=float), total_ms


def plot_figB_stage_breakdown(
    df: pd.DataFrame,
    k_default: int,
    out_stem: str,
    dpi: int,
):
    kbar = _choose_k_for_bar(df, k_default)

    stage_means, stage_stds, totals_ms = {}, {}, {}
    for p in PIPELINES_CANON:
        m, s, tms = _stage_stats_at_k(df, p, kbar)
        stage_means[p] = m
        stage_stds[p] = s
        totals_ms[p] = tms

    # ---- log-scale safety: replace zeros with epsilon so log axis works
    all_vals = np.concatenate([stage_means[p] for p in PIPELINES_CANON], axis=0)
    pos = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
    eps = float(max(1e-3, (np.min(pos) * 0.25) if pos.size > 0 else 1e-3))

    # global y max on log axis (shared across subplots)
    all_vals_plot = np.where(np.isfinite(all_vals) & (all_vals > 0), all_vals, eps)
    global_y_max = float(np.max(all_vals_plot)) if all_vals_plot.size > 0 else 1.0
    ylim_top = global_y_max * 1.10

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.45), sharey=True, constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.90, bottom=0.33, wspace=0.28)

    x = np.arange(len(STAGES), dtype=float)
    xticklabels = [lbl for _, lbl in STAGES]

    for ax, p in zip(axes, PIPELINES_CANON):
        _common_axes_style(ax)

        base_hex = _style_for_pipeline(p, PSTYLE_B)["color"]
        bar_color = base_hex
        err_color = _darken(base_hex, factor=0.78)  # same hue, darker -> visible on bar

        y_raw = stage_means[p]
        e_raw = stage_stds[p]

        # log-scale plotting values
        y = np.where(y_raw > 0, y_raw, eps)

        # For log-axis, ensure lower whisker stays > 0: clip errorbar to < y
        e = np.where(y_raw > 0, e_raw, 0.0)
        e = np.minimum(e, 0.95 * y)

        # Bars first (lower zorder)
        ax.bar(
            x, y,
            color=bar_color,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )

        # Errorbars AFTER bars + higher zorder => NOT occluded ✅
        ax.errorbar(
            x, y, yerr=e,
            fmt="none",
            ecolor=err_color,     # follows pipeline (same hue, darker)
            elinewidth=1.15,
            capsize=2.2,
            capthick=1.15,
            zorder=5,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, rotation=40, ha="right")
        ax.set_title(p)

        # log y-axis
        ax.set_yscale("log")
        ax.set_ylim(eps, ylim_top)

        # log ticks: sparse + readable
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_locator(NullLocator())

        # Total annotation
        tms = totals_ms[p]
        txt = f"Total≈{tms:.0f} ms" if np.isfinite(tms) else "Total≈N/A"
        ax.text(
            0.98, 0.98, txt,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=6.5,
            color="0.20",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2),
        )

    axes[0].set_ylabel("Time (ms, log)")

    pdf_path, png_path = _derive_out_paths(out_stem)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=int(dpi), facecolor="white")
    plt.close(fig)

    print(f"[OK] Figure-B saved: {pdf_path}")
    print(f"[OK] Figure-B saved: {png_path}")
    print(f"[OK] Figure-B used K={kbar} (default request was K={k_default}); log-eps={eps:g}")

    # ---- Visibility check note (for you):
    # Bars are zorder=2; errorbars are drawn after bars with zorder=5,
    # so they will not be occluded by the bar patches.


# ----------------------------
# Main
# ----------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Fig3: (A) K-sweep curves + (B) stage breakdown bars.")

    ap.add_argument("--csv", type=str, default="",
                    help="Single CSV path OR comma-separated CSVs. If empty, use default 3 CSVs under paper/artifacts/data/.")
    ap.add_argument("--csvs", type=str, nargs="*", default=None,
                    help="Multiple CSV paths (space-separated). Overrides --csv if provided.")

    ap.add_argument("--outA", type=str, default=str(PLOTS_DIR / "fig3A_k_sweep"),
                    help="Output stem for Figure-A (writes .pdf + .png).")
    ap.add_argument("--outB", type=str, default=str(PLOTS_DIR / "fig3B_stage_breakdown"),
                    help="Output stem for Figure-B (writes .pdf + .png).")
    ap.add_argument("--dpi", type=int, default=400, help="PNG DPI (default: 400)")

    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--where", type=str, default="",
                    help='Optional pandas query filter, e.g. "amp==False and batch_size==64"')
    ap.add_argument("--no_latest", action="store_true",
                    help="Disable keep-latest timestamp per (pipeline,K).")

    ap.add_argument("--ticks", type=str, default="8,16,32,64,128,256,512")
    ap.add_argument("--kstar", type=int, default=64,
                    help="Kept for CLI compatibility; Figure-A no longer draws kstar line per request.")

    ap.add_argument("--kbar", type=int, default=64,
                    help="Fixed K for Figure-B stage breakdown (default: 64). "
                         "If any pipeline missing this K, auto-choose max common K.")

    return ap


def main():
    args = build_argparser().parse_args()
    set_icml_rcparams()

    if args.csvs is not None and len(args.csvs) > 0:
        csv_list = list(args.csvs)
    elif args.csv.strip():
        csv_list = [s.strip() for s in args.csv.split(",") if s.strip()]
    else:
        csv_list = [
            str(DATA_DIR / "fig3_brmil_online.csv"),
            str(DATA_DIR / "fig3_naive_online.csv"),
            str(DATA_DIR / "fig3_targetnet_like_online.csv"),
        ]

    dfs = []
    missing = []
    for p in csv_list:
        csv_path = _resolve_under_paper(p)
        if not csv_path.exists():
            missing.append(str(csv_path))
            continue
        dfi = read_csv_auto(csv_path)
        dfi = normalize_columns(dfi)
        dfs.append(dfi)

    if missing:
        raise FileNotFoundError("CSV not found:\n" + "\n".join(missing))
    if not dfs:
        raise FileNotFoundError("No CSV loaded (all missing or empty).")

    df = pd.concat(dfs, ignore_index=True)

    df = maybe_filter_basic(df, split=args.split, seed=args.seed)
    df = apply_where(df, args.where)

    df["pipeline"] = df["pipeline"].map(canonicalize_pipeline)
    df = df[df["pipeline"].isin(PIPELINES_CANON)].copy()
    if df.empty:
        raise ValueError(f"After canonicalization, no rows match required pipelines: {PIPELINES_CANON}")

    if not args.no_latest:
        df = keep_latest_run_per_group(df)

    ticks = [int(t.strip()) for t in args.ticks.split(",") if t.strip()]

    plot_figA_k_sweep(
        df=df,
        ticks=ticks,
        kstar=None,          # explicitly disabled
        out_stem=args.outA,
        dpi=int(args.dpi),
    )

    plot_figB_stage_breakdown(
        df=df,
        k_default=int(args.kbar),
        out_stem=args.outB,
        dpi=int(args.dpi),
    )


if __name__ == "__main__":
    main()
