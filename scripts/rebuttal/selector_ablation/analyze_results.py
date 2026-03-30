#!/usr/bin/env python3
"""
分析 Selector 消融实验结果
读取所有实验的 metrics.json，生成汇总表格和统计检验
"""
import json
import pandas as pd
from pathlib import Path
from scipy import stats
import numpy as np

# 实验根目录
EXP_ROOT = Path("experiments/selector_ablation")

# 实验配置
DATASETS = ["miRAW", "deepTargetPro"]
K_VALUES = [8, 16]
SELECTORS = ["S0", "S1", "S2"]
SEEDS = [2020]  # 单种子快速验证

def load_metrics(dataset, K, selector, seed):
    """加载单个实验的 metrics.json"""
    metrics_file = EXP_ROOT / f"{dataset}_{selector}_K{K}" / f"seed{seed}" / "eval" / "test" / "test" / "best" / "sweep" / "metrics.json"

    if not metrics_file.exists():
        print(f"  ⚠ Missing: {metrics_file}")
        return None

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    return {
        "dataset": dataset,
        "K": K,
        "selector": selector,
        "seed": seed,
        "PR-AUC": metrics.get("pr_auc", None),
        "ROC-AUC": metrics.get("roc_auc", None),
        "F1": metrics.get("f1", None),
        "Precision": metrics.get("precision", None),
        "Recall": metrics.get("recall", None),
        "Accuracy": metrics.get("accuracy", None),
    }

def main():
    print("=" * 80)
    print("Selector Ablation Results Analysis")
    print("=" * 80)

    # 收集所有结果
    results = []
    for dataset in DATASETS:
        for K in K_VALUES:
            for selector in SELECTORS:
                for seed in SEEDS:
                    metrics = load_metrics(dataset, K, selector, seed)
                    if metrics:
                        results.append(metrics)

    if not results:
        print("\n❌ No results found!")
        return

    df = pd.DataFrame(results)

    # 保存原始数据
    output_file = EXP_ROOT / "results_summary.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved raw results to: {output_file}")

    # 按 (dataset, K) 分组汇总
    print("\n" + "=" * 80)
    print("Results Summary (mean ± std)")
    print("=" * 80)

    for dataset in DATASETS:
        print(f"\n[{dataset}]")
        for K in K_VALUES:
            print(f"\n  K={K}:")
            subset = df[(df["dataset"] == dataset) & (df["K"] == K)]

            if subset.empty:
                print("    No data")
                continue

            for selector in SELECTORS:
                sel_data = subset[subset["selector"] == selector]
                if sel_data.empty:
                    continue

                pr_auc = sel_data["PR-AUC"].values
                roc_auc = sel_data["ROC-AUC"].values
                f1 = sel_data["F1"].values

                print(f"    {selector}: PR-AUC={pr_auc[0]:.4f}, ROC-AUC={roc_auc[0]:.4f}, F1={f1[0]:.4f}")

    # 计算改进
    print("\n" + "=" * 80)
    print("Improvements: S2 vs S0 (Δ PR-AUC)")
    print("=" * 80)

    for dataset in DATASETS:
        print(f"\n[{dataset}]")
        for K in K_VALUES:
            subset = df[(df["dataset"] == dataset) & (df["K"] == K)]

            s0_prauc = subset[subset["selector"] == "S0"]["PR-AUC"].values
            s2_prauc = subset[subset["selector"] == "S2"]["PR-AUC"].values

            if len(s0_prauc) > 0 and len(s2_prauc) > 0:
                improvement = s2_prauc[0] - s0_prauc[0]
                rel_improvement = (improvement / s0_prauc[0]) * 100
                print(f"  K={K}: Δ={improvement:+.4f} ({rel_improvement:+.2f}%)")

    # 生成 LaTeX 表格
    print("\n" + "=" * 80)
    print("LaTeX Table")
    print("=" * 80)

    latex_lines = []
    latex_lines.append("\\begin{table}[t]")
    latex_lines.append("  \\centering")
    latex_lines.append("  \\caption{Selector ablation at small budgets (seed=2020).}")
    latex_lines.append("  \\label{tab:selector_ablation_small_k}")
    latex_lines.append("  \\small")
    latex_lines.append("  \\begin{tabular}{llccc}")
    latex_lines.append("    \\toprule")
    latex_lines.append("    \\textbf{Dataset} & \\textbf{K} & \\textbf{S0 (TopK)} & \\textbf{S1 (+PosDiv)} & \\textbf{S2 (+EmbDiv)} \\\\")
    latex_lines.append("    \\midrule")

    for dataset in DATASETS:
        for K in K_VALUES:
            subset = df[(df["dataset"] == dataset) & (df["K"] == K)]

            s0_val = subset[subset["selector"] == "S0"]["PR-AUC"].values
            s1_val = subset[subset["selector"] == "S1"]["PR-AUC"].values
            s2_val = subset[subset["selector"] == "S2"]["PR-AUC"].values

            if len(s0_val) > 0 and len(s1_val) > 0 and len(s2_val) > 0:
                # 找最大值加粗
                vals = [s0_val[0], s1_val[0], s2_val[0]]
                max_idx = np.argmax(vals)

                s0_str = f"\\textbf{{{s0_val[0]:.4f}}}" if max_idx == 0 else f"{s0_val[0]:.4f}"
                s1_str = f"\\textbf{{{s1_val[0]:.4f}}}" if max_idx == 1 else f"{s1_val[0]:.4f}"
                s2_str = f"\\textbf{{{s2_val[0]:.4f}}}" if max_idx == 2 else f"{s2_val[0]:.4f}"

                latex_lines.append(f"    {dataset} & {K} & {s0_str} & {s1_str} & {s2_str} \\\\")

    latex_lines.append("    \\bottomrule")
    latex_lines.append("  \\end{tabular}")
    latex_lines.append("\\end{table}")

    latex_output = "\n".join(latex_lines)
    print("\n" + latex_output)

    latex_file = EXP_ROOT / "table_selector_ablation.tex"
    latex_file.write_text(latex_output)
    print(f"\n✓ Saved LaTeX table to: {latex_file}")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
