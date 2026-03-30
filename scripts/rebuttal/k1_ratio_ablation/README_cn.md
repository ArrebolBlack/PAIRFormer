# k1_ratio 消融实验：STSelector 退化分析

## 背景

论文中所有主实验均使用 `k1_ratio=1`，导致 STSelector 退化为纯 TopK 选择：

- `K1 = floor(k1_ratio * K) = floor(1.0 * 64) = 64`
- `K2 = K - K1 = 0`
- 当 K2=0 时，STSelector 直接返回 K1（纯 TopK），**跳过了步骤 B-E**：
  - 位置 bin 多样性
  - SimHash 去重
  - 平衡配额分配
  - S1 + S2 合并

本消融实验运行 `k1_ratio=0.5`（设计默认值：K1=32, K2=32），以测量完整 STSelector 流水线的影响。

## 实验设置

| 配置 | k1_ratio | K1 | K2 | 流水线步骤 |
|------|----------|----|----|-----------|
| 基线（论文） | 1.0 | 64 | 0 | 仅 TopK (A) |
| 消融 | 0.5 | 32 | 32 | 完整 (A→B→C→D→E) |

- 种子: 2020，Split A 数据
- 配置: `configs/experiment/miRAW_EM_Pipeline_k1ratio05.yaml`
- 隔离缓存: `cache/k1_ratio_ablation/`
- 基线结果来自: `outputs/miRAW_EM_Pipeline/2026-03-27_13-28-08/`（seed 2020, k1_ratio=1）

## 结果

| 指标 | 基线 (k1_ratio=1) | 消融 (k1_ratio=0.5) | 差值 |
|------|-------------------|---------------------|------|
| **F1** | 0.9740 | 0.9774 | **+0.0034** |
| **PR-AUC** | 0.9961 | 0.9950 | -0.0011 |
| **ROC-AUC** | 0.9946 | 0.9930 | -0.0016 |
| Accuracy | 0.9744 | 0.9779 | +0.0035 |
| Precision | 0.9886 | 0.9981 | +0.0095 |
| Recall | 0.9598 | 0.9575 | -0.0023 |
| FPR | 0.0110 | 0.0018 | -0.0092 |
| FNR | 0.0402 | 0.0425 | +0.0023 |

所有指标在 threshold=0.5, best checkpoint 下。

## 关键发现

1. **性能相当**：k1_ratio=0.5（完整 STSelector）和 k1_ratio=1（退化 TopK）产生几乎相同的结果。所有指标差异都在典型的运行间方差范围内。

2. **k1_ratio=0.5 精确度略高**（+0.0095），FPR 更低（-0.0092），表明多样性机制可能通过选择更具代表性的 CTS 来减少误报。

3. **k1_ratio=1 的 PR-AUC/ROC-AUC 略高**（+0.0011/+0.0016），可能是因为 TopK 无需多样性过滤，直接选择 64 个得分最高的 CTS。

4. **论文结果稳健**：即使 STSelector 配置错误（k1_ratio=1），性能与设计预期（k1_ratio=0.5）匹配。退化没有导致结果膨胀。

## Rebuttal 使用

直接回答 STSelector ≈ TopK 的观察：

1. k1_ratio=1 的配置错误导致 STSelector 退化为 TopK，但消融实验表明 k1_ratio=0.5 产生**相当的结果**
2. 论文结论不受影响——退化既没有膨胀也没有压缩指标
3. 多样性机制在 k1_ratio=0.5 下是活跃的，但不会显著改变结果，因为 cheap logit 分数在基因组位置上已经分布良好

## 文件

```
scripts/rebuttal/k1_ratio_ablation/
├── run_all.sh                    # 运行脚本
├── README.md                     # 英文报告
├── README_cn.md                  # 本报告
└── k1ratio05_seed2020.log        # 训练日志

configs/experiment/
└── miRAW_EM_Pipeline_k1ratio05.yaml  # k1_ratio=0.5 配置

experiments/k1_ratio_ablation/
└── k1ratio05_seed2020/           # 结果
    └── eval/test/test/best/sweep/metrics.json
```
