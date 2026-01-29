#!/usr/bin/env python3
import argparse
from io import StringIO
import pandas as pd

DATA_INLINE = """Row,kmax,seed,pr_auc,f1@0.5
TopK,64,2020,0.99553,0.97406
TopK,64,2025,0.99585,0.97464
TopK,64,2026,0.99107,0.97662
TopK,32,2020,0.99519,0.97897
TopK,32,2025,0.99471,0.97377
TopK,32,2026,0.9936,0.97021
PosDiv,64,2020,0.99681,0.97793
PosDiv,64,2025,0.99352,0.96669
PosDiv,64,2026,0.99509,0.97908
PosDiv,32,2020,0.99601,0.97211
PosDiv,32,2025,0.99424,0.96908
PosDiv,32,2026,0.98128,0.96957
PosEmbDiv,64,2020,0.99613,0.97397
PosEmbDiv,64,2025,0.99066,0.97566
PosEmbDiv,64,2026,0.995,0.97347
PosEmbDiv,32,2020,0.99521,0.97289
PosEmbDiv,32,2025,0.99478,0.9725
PosEmbDiv,32,2026,0.99503,0.97619
"""

VARIANT_LABEL = {
    "TopK": "S0 (TopK)",
    "PosDiv": "S1 (+PosDiv)",
    "PosEmbDiv": "S2 (+PosDiv+EmbDiv)",
}

def fmt_pm(mean: float, std: float, decimals: int) -> str:
    f = f"{{:.{decimals}f}}"
    return f"{f.format(mean)}$\\pm${f.format(std)}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="", help="CSV 文件路径（可选）。不提供则使用代码内置 DATA_INLINE。")
    ap.add_argument("--metric", type=str, default="pr_auc", choices=["pr_auc", "f1@0.5"])
    ap.add_argument("--decimals", type=int, default=4, help="输出小数位数（默认 4，跟你表格 XX.XXXX 对齐）")
    ap.add_argument("--bold_final", action="store_true", help="给 S2 的 mean 加粗（匹配你表格的 \\textbf{}）")
    args = ap.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        df = pd.read_csv(StringIO(DATA_INLINE))

    # 计算 mean/std（样本标准差 ddof=1 是 pandas 默认）
    stats = (
        df.groupby(["Row", "kmax"])[args.metric]
          .agg(mean="mean", std="std")
          .reset_index()
    )

    # 取出 K=32 和 K=64 两列
    def get_ms(row_name: str, k: int):
        sub = stats[(stats["Row"] == row_name) & (stats["kmax"] == k)]
        if sub.empty:
            raise ValueError(f"Missing stats for Row={row_name}, kmax={k}")
        return float(sub["mean"].iloc[0]), float(sub["std"].iloc[0])

    rows_order = ["TopK", "PosDiv", "PosEmbDiv"]
    print("\n=== Values to paste into LaTeX table (mean$\\pm$std) ===")
    for r in rows_order:
        m32, s32 = get_ms(r, 32)
        m64, s64 = get_ms(r, 64)

        cell32 = fmt_pm(m32, s32, args.decimals)
        cell64 = fmt_pm(m64, s64, args.decimals)

        if args.bold_final and r == "PosEmbDiv":
            # 只加粗 mean（按你的表格写法）
            mean_fmt = f"{{:.{args.decimals}f}}".format
            cell32 = f"\\textbf{{{mean_fmt(m32)}}}$\\pm${mean_fmt(s32)}"
            cell64 = f"\\textbf{{{mean_fmt(m64)}}}$\\pm${mean_fmt(s64)}"

        print(f"{VARIANT_LABEL[r]} & {cell32} & {cell64} \\\\")

if __name__ == "__main__":
    main()
