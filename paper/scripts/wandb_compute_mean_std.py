#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper/scripts/wandb_compute_mean_std.py

用途
- 读取 paper/artifacts/data/group={safe_group}.csv
- 用户指定：
    * 分类列（group-by keys）
    * 指标列（需要按 seed 聚合 mean/std）
    * seed 列名（默认 "seed"）
    * 可选：限定使用的 seed 集合（例如 2020,2025,2026）
- 输出：原文件名追加后缀（默认 ".computed.csv"），写回到同一目录

注意（更通用/更稳健）
- 若同一个 (keys, seed) 下存在多行（例如多次重复运行），脚本会先在 seed 内做一次 mean，
  然后再对 seed 维度计算 mean/std。

  
  
python paper/scripts/wandb_compute_mean_std.py \
  --group "loss ablation" \
  --by "Row,kmax" \
  --metrics "test/test/best_thr0.5/pr_auc,test/test/best_thr0.5/f1" \
  --seeds "2020,2025,2026" \
  --sort "kmax,Row"



"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_DATA_DIR = Path("paper/artifacts/data")
DEFAULT_SUFFIX = "computed"
DEFAULT_SEED_COL = "seed"


def safeify_group(group: str) -> str:
    """
    将 group 转为文件名安全的 safe_group：
    - 空白 -> "_"
    - 非 [A-Za-z0-9_.-] -> "_"
    """
    s = group.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(
        description="Compute mean/std over seeds for specified metrics from a W&B-exported CSV."
    )
    ap.add_argument(
        "--group",
        type=str,
        required=True,
        help="W&B group 名称（可包含空格）；脚本会自动 safeify 并读取 group={safe_group}.csv",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"数据目录（默认: {DEFAULT_DATA_DIR}）",
    )
    ap.add_argument(
        "--by",
        type=str,
        required=True,
        help="用于分类/分组的列名列表，逗号分隔。例如：Row,kmax",
    )
    ap.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="用于计算 mean/std 的指标列名列表，逗号分隔。"
             "例如：test/test/best_thr0.5/pr_auc,test/test/best_thr0.5/f1",
    )
    ap.add_argument(
        "--seed-col",
        type=str,
        default=DEFAULT_SEED_COL,
        help=f"seed 列名（默认: {DEFAULT_SEED_COL}）",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default="",
        help="仅使用这些 seed（逗号分隔，例如 2020,2025,2026）。不填则使用文件中所有 seed。",
    )
    ap.add_argument(
        "--suffix",
        type=str,
        default=DEFAULT_SUFFIX,
        help=f"输出文件后缀（默认: {DEFAULT_SUFFIX}）",
    )
    ap.add_argument(
        "--ddof",
        type=int,
        default=1,
        help="std 的 ddof（默认 1：样本标准差；0：总体标准差）",
    )
    ap.add_argument(
        "--round",
        type=int,
        default=6,
        help="输出小数位数（默认 6）",
    )
    ap.add_argument(
        "--sort",
        type=str,
        default="",
        help="可选：输出排序列，逗号分隔。默认不特别排序；若包含 kmax，可自行指定 kmax,Row。",
    )

    args = ap.parse_args()

    safe_group = safeify_group(args.group)
    in_path = args.data_dir / f"group={safe_group}.csv"
    if not in_path.exists():
        raise SystemExit(f"[ERROR] input not found: {in_path}")

    by_cols = parse_csv_list(args.by)
    metric_cols = parse_csv_list(args.metrics)
    seed_col = args.seed_col

    df = pd.read_csv(in_path, skipinitialspace=True)

    # 校验列是否存在
    missing = [c for c in by_cols + [seed_col] + metric_cols if c not in df.columns]
    if missing:
        raise SystemExit(
            "[ERROR] missing columns in CSV:\n"
            f"  missing: {missing}\n"
            f"  existing: {list(df.columns)}"
        )

    # 类型转换：seed/metrics 转 numeric（metrics 允许 NaN）
    df[seed_col] = pd.to_numeric(df[seed_col], errors="coerce")
    for m in metric_cols:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    # 可选：过滤 seeds
    if args.seeds.strip():
        seed_list = [int(x) for x in parse_csv_list(args.seeds)]
        df = df[df[seed_col].isin(seed_list)].copy()

    if df.empty:
        raise SystemExit("[ERROR] after seed filtering, dataframe is empty.")

    # Step 1: 先在 seed 内聚合（防止同 seed 多条记录）
    df_seed = (
        df.groupby(by_cols + [seed_col], dropna=False, as_index=False)[metric_cols]
          .mean()
    )

    # Step 2: 对 seed 维度做 mean/std
    agg_dict = {
        "n_seeds": (seed_col, "nunique"),
    }
    for m in metric_cols:
        agg_dict[f"{m}_mean"] = (m, "mean")
        agg_dict[f"{m}_std"] = (m, lambda x, ddof=args.ddof: x.std(ddof=ddof))

    out = (
        df_seed.groupby(by_cols, dropna=False)
               .agg(**agg_dict)
               .reset_index()
    )

    # 友好：kmax 若存在，放最前（仅调整展示顺序）
    cols = list(out.columns)
    if "kmax" in by_cols:
        front = ["kmax"] + [c for c in by_cols if c != "kmax"]
    else:
        front = by_cols
    tail = [c for c in cols if c not in front]
    out = out[front + tail]

    # 可选排序
    if args.sort.strip():
        sort_cols = parse_csv_list(args.sort)
        sort_missing = [c for c in sort_cols if c not in out.columns]
        if sort_missing:
            raise SystemExit(f"[ERROR] sort columns missing: {sort_missing}")
        out = out.sort_values(sort_cols, ascending=True)

    # round
    num_cols = [c for c in out.columns if c.endswith("_mean") or c.endswith("_std")]
    out[num_cols] = out[num_cols].round(args.round)

    # 输出路径：原文件名 + ".{suffix}.csv"
    out_path = in_path.with_name(f"{in_path.stem}.{args.suffix}.csv")
    out.to_csv(out_path, index=False)

    print(f"[OK] input : {in_path}")
    print(f"[OK] output: {out_path}")
    print(f"[OK] rows  : {len(out)}")


if __name__ == "__main__":
    main()
