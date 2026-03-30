#!/usr/bin/env python3
"""
可视化 Selector 消融实验结果
生成 PR-AUC vs K 曲线图
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

# 实验根目录
EXP_ROOT = Path("experiments/selector_ablation")

# 实验配置
DATASETS = ["miRAW", "deepTargetPro"]
K_VALUES = [8, 16]
SELECTORS = ["S0", "S1", "S2"]
SEEDS = [2020]

SELECTOR_LABELS = {
    "S0": "S0 (TopK)",
    "S1": "S1 (+PosDiv)",
    "S2": "S2 (+EmbDiv)",
}

SELECTOR_COLORS = {
    "S0": "#E74C3C",  # Red
    "S1": "#F39C12",  # Orange
    "S2": "#27AE60",  # Green
}

def load_results():
    """加载所有实验结果"""
    results = []
    for dataset in DATASETS:
        for K in K_VALUES:
            for selector in SELECTORS:
                for seed in SEEDS:
                    metrics_file = EXP_ROOT / f"{dataset}_{selector}_K{K}" / f"seed{seed}" / "eval" / "test" / "test" / "best" / "sweep" / "metrics.json"

                    if not metrics_file.exists():
                        continue

                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)

                    results.append({
                        "dataset": dataset,
                        "K": K,
                        "selector": selector,
                        "seed": seed,
                        "PR-AUC": metrics.get("pr_auc", None),
                        "ROC-AUC": metrics.get("roc_auc", None),
                        "F1": metrics.get("f1", None),
                    })

    return pd.DataFrame(results)

def plot_pr_auc_vs_k(df, output_file):
    """绘制 PR-AUC vs K 曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, dataset in enumerate(DATASETS):
        ax = axes[i]
        subset = df[df["dataset"] == dataset]

        for selector in SELECTORS:
            sel_data = subset[subset["selector"] == selector]
            if sel_data.empty:
                continue

            # 按 K 分组计算均值和标准差
            grouped = sel_data.groupby("K")["PR-AUC"].agg(["mean", "std"])

            ax.errorbar(
                grouped.index,
                grouped["mean"],
                yerr=grouped["std"],
                label=SELECTOR_LABELS[selector],
                marker='o',
                markersize=8,
                linewidth=2,
                capsize=5,
                color=SELECTOR_COLORS[selector],
            )

        ax.set_xlabel("Budget K", fontsize=14)
        ax.set_ylabel("PR-AUC", fontsize=14)
        ax.set_title(f"{dataset}", fontsize=16, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(alpha=0.3)
        ax.set_xticks(K_VALUES)

        # 设置 y 轴范围
        if dataset == "miRAW":
            ax.set_ylim([0.95, 1.0])
        else:
            ax.set_ylim([0.75, 0.95])

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_file}")

def plot_improvement_heatmap(df, output_file):
    """绘制改进热力图 (S2 vs S0)"""
    improvements = []

    for dataset in DATASETS:
        for K in K_VALUES:
            subset = df[(df["dataset"] == dataset) & (df["K"] == K)]

            s0_prauc = subset[subset["selector"] == "S0"]["PR-AUC"].values
            s2_prauc = subset[subset["selector"] == "S2"]["PR-AUC"].values

            if len(s0_prauc) > 0 and len(s2_prauc) > 0:
                improvement = s2_prauc[0] - s0_prauc[0]
                improvements.append({
                    "Dataset": dataset,
                    "K": K,
                    "Improvement": improvement * 100,  # 转为百分比
                })

    imp_df = pd.DataFrame(improvements)
    pivot = imp_df.pivot(index="Dataset", columns="K", values="Improvement")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        cbar_kws={"label": "Δ PR-AUC (%)"},
        ax=ax,
    )
    ax.set_title("S2 vs S0 Improvement", fontsize=16, fontweight='bold')
    ax.set_xlabel("Budget K", fontsize=14)
    ax.set_ylabel("Dataset", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap to: {output_file}")

def main():
    print("=" * 80)
    print("Plotting Selector Ablation Results")
    print("=" * 80)

    # 加载结果
    df = load_results()

    if df.empty:
        print("\n❌ No results found!")
        return

    print(f"\nLoaded {len(df)} results")

    # 绘制 PR-AUC vs K 曲线
    plot_file = EXP_ROOT / "figure_selector_ablation_pr_auc.png"
    plot_pr_auc_vs_k(df, plot_file)

    # 绘制改进热力图
    heatmap_file = EXP_ROOT / "figure_selector_ablation_improvement.png"
    plot_improvement_heatmap(df, heatmap_file)

    print("\n" + "=" * 80)
    print("Plotting complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
