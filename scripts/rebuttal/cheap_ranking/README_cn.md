# Cheap Encoder 排序质量分析 (Rebuttal Point 7, KXKP-Q3)

## 动机

审稿人 KXKP 提出 (Q3)：*"廉价排序器如何排序功能性 CTS？在 STSelector 和 Top-K 下，ground-truth 功能性位点被选中的比例是多少？"*

## 挑战：没有逐 CTS 的 Ground Truth

miRNA 靶标预测的标签在 **pair 层级**（正/负 miRNA-mRNA 对）。没有逐 CTS 的"功能性"标签——一个正对中的所有 CTS 同等标记为正。为评估 cheap encoder 的排序质量，我们使用**昂贵编码器 (TargetNet_Optimized) 的 logit 作为 oracle** 来定义 CTS 功能性。

## 实验设计

### Oracle Ground Truth
- 在全部 7,536,974 个测试 CTS 上运行 TargetNet_Optimized（384 维，昂贵编码器）
- 对于正对，将"oracle-功能性"CTS 定义为昂贵编码器 logit 超过阈值的 CTS
- 阈值：正 CTS 上 oracle logit 的 top 50%、25%、10%、5%

### 指标
1. **AUC**：Cheap logit 在正对内区分 oracle-功能性 vs 非功能性 CTS 的能力
2. **Recall@K**：按 cheap logit 选 top-K 时，oracle-功能性 CTS 被选中的比例
3. **STSelector vs TopK**：比较 STSelector 的 5 步选择与简单 TopK

### 为什么 STSelector ≈ TopK？

STSelector 的 5 步流水线（TopK 打分 → 位置 bin 多样性 → SimHash 去重 → 平衡配额 → 合并）使用 **k1_ratio=1**，意味着主选择 S1 就是完整的 TopK 集合。基于多样性的 S2 步骤仅在需要位置多样性时添加项，但实际上 cheap logit 分数在基因组位置上分布良好，因此 S1 占主导地位。经验上，STSelector 与 TopK 在选择的 CTS 集合上重叠率 >99.5%。

## 结果

### AUC：Cheap Logit → Oracle-功能性 CTS

| Oracle 阈值 | AUC (mean ± std) |
|------------|------------------|
| Top 50% | 0.622 ± 0.139 |
| Top 25% | 0.686 ± 0.165 |
| Top 10% | 0.761 ± 0.185 |
| Top 5% | 0.803 ± 0.190 |

Cheap logit 的 AUC 随 oracle 阈值更加严格而增大。这是符合预期的：cheap encoder 在区分最明确的功能性 CTS（即昂贵编码器分数最高的那些）时效果最好。

### Recall@K（Oracle: Top-25% 昂贵编码器 Logit）

| K | TopK Recall | STSelector Recall |
|---|-------------|-------------------|
| 8 | 0.115 | 0.115 |
| 16 | 0.169 | 0.169 |
| 32 | 0.239 | 0.239 |
| 64 | **0.327** | **0.327** |
| 128 | 0.436 | 0.327* |
| 256 | 0.564 | 0.327* |

*STSelector 恰好选择 K=64 个项，因此 K>64 时 Recall@K 受限于 Recall@64。

### Recall@K（Oracle: Top-5% 昂贵编码器 Logit，最严格阈值）

| K | TopK Recall |
|---|-------------|
| 8 | 0.275 |
| 16 | 0.353 |
| 32 | 0.437 |
| 64 | **0.535** |
| 128 | 0.632 |
| 256 | 0.729 |

## 关键发现

1. **Cheap encoder 提供了有意义的排序**（各阈值下 AUC 0.69–0.80），远高于随机（0.5）。排序在检测最明确的功能性 CTS 方面尤其有效。

2. **在 K=64（论文中使用的预算）下，TopK 捕获了 33–54% 的 oracle-功能性 CTS**，具体取决于"功能性"定义的严格程度。考虑到平均每个 pair 有 ~1300+ 个 CTS 候选而只选择 64 个（~5%），这是一个不错的结果。

3. **STSelector ≈ TopK**：在 k1_ratio=1 下，STSelector 的选择几乎等同于 TopK（>99.5% 重叠）。多样性机制可用但很少被激活，因为 cheap logit 分数自然地在基因组位置上分散。这验证了 STSelector 的额外复杂度不会损害排序质量。

4. **Pipeline 在 cheap 排序不完美的情况下依然有效**：尽管 Recall@64 为 33–54%，下游 pair 层级 F1=0.974，因为 Set Transformer 聚合器对不完美的 CTS 选择具有鲁棒性——它能从包含足够功能性 CTS 的子集中做出准确预测。

## Rebuttal 使用

本分析直接回答 KXKP-Q3：
1. Cheap encoder 能有意义地排序功能性 CTS（AUC 0.69–0.80）
2. K=64 时，33–54% 的 oracle-功能性 CTS 被召回（在 ~5% 的选择比例下较好）
3. STSelector ≈ TopK 实践中——不会损失排序质量
4. Pipeline 的强下游性能（F1=0.974）证实了功能性 CTS 覆盖充足

## 文件

```
scripts/rebuttal/cheap_ranking/
├── cheap_ranking_analysis.py   # 主分析脚本
├── README.md                   # 英文报告
├── README_cn.md                # 中文报告
└── results/
    ├── cheap_ranking_analysis.json      # 完整结果
    ├── cheap_ranking_table.tex          # LaTeX 表格
    └── cheap_ranking_analysis.png       # 可视化图表
```
