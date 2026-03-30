# EXP8 实验方案：MTI (mirCLASH) 全量训练

> **目标**: 在第三个外部数据集 MTI (mirCLASH) 上全量训练 PAIR-Former，进一步证明方法泛化性
> **数据集**: MTI (miRNA-Target Interaction) 来自 mirCLASH 数据库，仅含 human 正类数据
> **核心挑战**: MTI 无负类样本，需生成合成负类

---

## 1. 数据概况

### 1.1 原始数据

| 文件 | 规模 | 列 | 说明 |
|------|------|-----|------|
| `MTI_siteLevel_U.csv` | 188,333 对 (686 miRNA, 10,978 gene) | miRNA名, 基因名, 靶点序列(~47nt), miRNA序列(~22nt), label=1 | 短序列，CTS 级别 |
| `MTI_mRNALevel.csv` | 237,893 对 (1,392 miRNA, 51,591 mRNA) | 基因名, mRNA ID, 全长序列(~1922nt), miRNA名, miRNA序列, label=1 | 全长序列，pair 级别 |

两个文件均为纯 human 数据（所有 miRNA 以 `hsa-` 开头）。

### 1.2 Site-level 序列长度分布

| 长度范围 | 数量 | 占比 | 可生成 CTS 窗口数 |
|----------|------|------|------------------|
| < 40nt | 86,386 | 45.9% | 0（被丢弃） |
| 40-49nt | 30,203 | 16.0% | 1-10 |
| 50-99nt | 44,128 | 23.4% | 11-60 |
| 100+nt | 5,240 | 2.8% | 60+ |
| **≥40nt 合计** | **101,947** | **54.1%** | — |

### 1.3 与已有数据集对比

| 维度 | miRAW | deepTargetPro | MTI |
|------|-------|-------------|-----|
| 实验体系 | PAR-CLIP | CLASH-seq | mirCLASH |
| CTS 训练正类 | 29,188 | 32,651 | 237,893 (mRNA) / 101,947 (site≥40nt) |
| CTS 训练负类 | 29,605 | 31,775 | **需生成** |
| Pair 测试正类 | 2,740 | 3,893 | ~35,684 (15%) |
| Pair 测试负类 | 2,740 | 4,180 | **需生成** |
| 负类来源 | 实验验证 | 实验验证 | **随机配对** |

---

## 2. 负类生成策略

### 2.1 方法：随机配对 (Random Pairing)

与 TDC/MTI 官方 benchmark 一致 (`neg_sample(frac=1)`)。

**CTS 负类 (Stage 1-2)**：
- 对每个正样本 (miRNA_i, mRNA_j)，随机选择一个 mRNA_k ∉ targets(miRNA_i)
- 构成负样本 (miRNA_i, mRNA_k, label=0)
- mRNA_k 的序列完整保留（pipeline 滑窗生成负窗口）

**Pair 负类 (Stage 3)**：
- 对每个正样本 (miRNA_i, mRNA_j)，随机选择一个 mRNA_k ∉ targets(miRNA_i)
- 构成负样本 (miRNA_i, mRNA_k, label=0)

**约束条件**：
- 正:负 = 1:1
- 负样本不在正样本集中
- 固定 seed (1234) 保证可复现

### 2.2 生成量估算

| 数据源 | 正类 | 负类(合成) | 合计 |
|--------|------|-----------|------|
| mRNA-level | 237,893 | 237,893 | 475,786 |
| site-level (≥40nt) | 101,947 | 101,947 | 203,894 |

---

## 3. 实验组设计

### 3.1 三个子实验

| 实验ID | CTS 数据源 | Pair 数据源 | Split 策略 | 说明 |
|--------|-----------|------------|-----------|------|
| **EXP8-A** | mRNA-level 滑窗 | mRNA-level pair | 随机 split | 主实验 |
| **EXP8-B** | mRNA-level 滑窗 | mRNA-level pair | miRNA 分组 split | 泛化测试 |
| **EXP8-C** | site-level (≥40nt) | mRNA-level pair | 随机 split | CTS 数据源对比 |

**EXP8-A** 是核心实验，与 EXP4 (deepTargetPro) 方法论完全一致。
**EXP8-B** 测试模型对未见过 miRNA 的泛化能力（更强验证）。
**EXP8-C** 对比短序列 CTS vs 全长 CTS 的效果。

### 3.2 数据分割

**随机 split (EXP8-A/C)**：
- 所有 pair 随机划分: train 70% / val 15% / test 15%
- CTS 训练用 train+val pair 的滑窗数据
- Pair 训练/测试用 split 后的 pair 数据
- 添加 `split` 列: `train` / `val` / `test`

**miRNA 分组 split (EXP8-B)**：
- 将 miRNA 随机分为三组: train_miRNAs (70%) / val_miRNAs (15%) / test_miRNAs (15%)
- pair 按 miRNA 归属划分
- test 集中的 miRNA 在 train 中完全未见过
- CTS 训练只用 train_miRNAs 的 pair

---

## 4. 训练配置

### 4.1 与 EXP4 保持一致

所有超参数与 EXP4 deepTargetPro Route 2 相同（确保结果可比）。

| Stage | 关键参数 |
|-------|---------|
| Stage 1 | TargetNet_Optimized (opt4_tiny), emb=384, AdamW lr=6e-4, 100 epochs, BCE+Focal, EMA |
| Stage 2 | CheapCTSNet_TinyConv, emb=64, KD from Stage 1, 100 epochs |
| Stage 3 | PairSetTransformerAggregator, d_model=256, 3 layers, budget K=64, 100 epochs + ext150 |
| Seeds | 2020, 2025, 2026 (R=3) |

### 4.2 数据配置文件（需新建）

```
configs/data/miRNA_MTI.yaml          # MTI 数据路径配置
configs/experiment/MTI_TargetNet_Optimized.yaml
configs/experiment/MTI_CheapCTSNet.yaml
configs/experiment/MTI_EM_Pipeline.yaml
```

---

## 5. 实施步骤

### Phase 1: 数据准备

1. **过滤 human-only**：site-level 已是纯 human；mRNA-level 已是纯 human ✓
2. **生成 CTS 负类**：
   - 读取 mRNA-level 正类对
   - 对每个正样本，随机选非靶标 mRNA
   - 输出 `MTI_CTS_train_with_neg.txt` (格式同 deepTargetPro)
3. **生成 Pair 负类**：
   - 同上，但保留 pair-level 格式
4. **数据分割**：
   - 随机 split → `MTI_pair_random_split.txt`
   - miRNA 分组 split → `MTI_pair_miRNA_split.txt`
5. **格式化**：输出为 pipeline 兼容格式
   ```
   mirna_id\tmirna_seq\tmrna_id\tmrna_seq\tlabel\tsplit
   ```

### Phase 2: CTS Cache 生成

- 对 mRNA-level 数据执行 pipeline 滑窗 + ESA 过滤
- 估计生成 ~5M-10M CTS 窗口 (475K pair × ~20 窗口/对)
- Site-level: ~200K-500K 窗口 (203K pair × ~2 窗口/对)

### Phase 3: 三阶段训练

按 EXP4 Route 2 的流程执行：
1. Stage 1: 训练 TargetNet_Optimized × 3 seeds
2. Stage 2: KD 蒸馏 CheapCTSNet × 3 seeds
3. Stage 3: 训练 PairSetTransformerAggregator × 3 seeds × (A/B/C 三组)
4. Stage 3-ext: 扩展训练 150 epochs (如需要)

### Phase 4: 评估

- 测试集评估: PR-AUC, ROC-AUC, F1, Precision, Recall
- 对比 EXP4 (deepTargetPro) 和原始 miRAW 结果
- 两种 split 策略的性能差异分析
- 两种 CTS 数据源 (mRNA vs site) 的性能差异分析

---

## 6. 实验执行优先级

鉴于 GPU 资源和时间约束，建议分步执行：

1. **先做 EXP8-A**（主实验）：mRNA-level + 随机 split → 最直接对标 EXP4
2. **再做 EXP8-B**（miRNA 分组 split）：复用 A 的 Stage 1-2，只需重训 Stage 3
3. **最后做 EXP8-C**（site-level CTS）：独立训练，作为补充

---

## 7. 预期结果与 Rebuttal 价值

### 7.1 预期

MTI 数据量更大但负类是合成生成的，预计：
- CTS-level PR-AUC: 0.85-0.95（取决于 ESA 覆盖率）
- Pair-level PR-AUC: 0.95-0.99（Set Transformer 聚合能力强）
- 随机 split 优于 miRNA 分组 split

### 7.2 Rebuttal 论点

> "We further validate PAIR-Former on a third external dataset, MTI (mirCLASH), comprising 237K+ miRNA-mRNA interactions from an independent experimental paradigm. Despite using computationally generated negatives (random pairing), our model achieves PR-AUC = X.XX, demonstrating robust performance across diverse data sources and negative sampling strategies."

### 7.3 对应 Reviewer 质疑

| Reviewer | 质疑 | EXP8 回应 |
|----------|------|-----------|
| KXKP-W1 | 仅单一数据集 | 第三方外部数据集验证 |
| F4i7-W3 | 需在 miRAW 以外验证 | 第三个不同实验体系的数据集 |
| ohuZ-W1 | 数据泄露 | 完全独立的 MTI 数据 + 合成负类 |
| 所有 | 性能饱和 | 不同负类生成策略下的表现 |

---

## 8. 待确认事项

- [ ] 是否需要先跑一个小规模 pilot（如只用 1 个 seed）确认 pipeline 兼容性？
- [ ] EXP8-B (miRNA 分组 split) 的 test 集可能较小，需要调整比例吗？
- [ ] 是否需要对负类生成做多种策略的对比（如方案 A vs B）？
