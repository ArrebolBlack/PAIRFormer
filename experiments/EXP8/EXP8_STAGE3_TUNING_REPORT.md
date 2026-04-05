# EXP8 Stage3 Aggregator 调优实验报告

> 日期: 2025-04-05
> 目标: 提升 PairSetTransformerAggregator (Stage3) 的 pair-level F1 至 0.8+

---

## 1. 实验环境与数据

- **数据集**: MTI (mirCLIP-derived miRNA-mRNA interaction), pair-level split
  - Train: 333,050 pairs / Val: 71,368 pairs / Test: 71,368 pairs
  - 50/50 正负样本平衡
  - 平均每个 pair ~54 个 CTS tokens, Kmax=64
- **Selector**: TopK (略优于 STSelector)
- **Token 特征**: `[inst_emb(384), inst_logit(1), esa(1), pos(1)]` = 387 维
  - inst_emb: TargetNet_Optimized 冻结提取的 384-dim embedding
  - inst_logit: TargetNet_Optimized 的 scalar logit
  - esa: Evolutionary Seed Alignment score
  - pos: Positional feature
- **Loss**: focal (α=0.4, γ=1.0) + label smoothing (neg=0.05, pos=0.95) + BCE regularizer
- **硬件**: 2x GPU (81GB each)

---

## 2. 实验结果总表

### 2.1 Batch Size 实验

| 实验 | BS | Selector | Epochs | Test F1 | Test PR-AUC |
|------|------|----------|--------|---------|-------------|
| A3 (bs=4096) | 4096 | STSelector | 40 | 0.5155 | 0.6635 |
| A6 (bs=4096) | 4096 | TopK | 40 | 0.5464 | 0.6852 |
| bs=512 TopK | **512** | TopK | 100 | **0.6542** | **0.7228** |

**结论**: BS=512 远优于 BS=4096。小 batch size 提供更频繁的梯度更新，泛化更好。

### 2.2 ISAB vs SAB + 模型 Scaling

| 模型 | d_model | layers | Attention | BS | LR | Test F1 | 状态 |
|------|---------|--------|-----------|------|------|---------|------|
| **小模型** | **256** | **2** | **SAB** | **512** | **3e-4** | **0.6417** | **最优** |
| 小模型 | 256 | 2 | SAB+norm+dropout | 512 | 3e-4 | 0.6130 | PR-AUC更好 |
| d=320 | 320 | 3 | SAB | 512 | 3e-4 | 0.5424 | 不稳定 |
| d=384 | 384 | 3 | SAB | 512 | 3e-4 | ~0.37 (衰退) | 过拟合 |
| d=512 | 512 | 4 | ISAB | 512 | 3e-4 | 0.0000 | 完全失败 |
| d=512 | 512 | 4 | ISAB | 512 | 5e-4 | 0.0000 | 完全失败 |
| d=512 | 512 | 4 | ISAB | 512 | 1e-3 | 0.0000 | 完全失败 |
| d=512 | 512 | 4 | SAB | 512 | 3e-4 | 0.0000 | 完全失败 |
| d=512 | 512 | 4 | SAB | 512 | 5e-4 | 0.0000 | 完全失败 |

**结论**:
1. **放大模型性能反而更差**，当前数据/特征复杂度只需要小模型容量
2. ISAB 的 inducing points 在大模型上是瓶颈，但小模型上 ISAB/SAB 差异不大
3. d=320 不稳定（F1 在 0.05-0.50 之间剧烈波动）
4. d≥512 无论怎么调 LR/warmup/clip 都无法学习（loss 卡在 0.1686，F1=0）

### 2.3 大模型优化失败尝试汇总

| 尝试 | LR | clip_grad | warmup | 结果 |
|------|------|-----------|--------|------|
| v3 (ISAB, d=512) | 5e-4 | 1.0 | 0 | F1=0, 14 epoch 无变化 |
| v4 (ISAB, d=512) | 1e-3 | 5.0 | 500 | F1=0, 6 epoch 无变化 |
| SAB_L (SAB, d=512) | 3e-4 | 1.0 | 0 | F1=0, 16 epoch 无变化 |
| SAB_L (SAB, d=512) | 5e-4 | 1.0 | 0 | F1=0, 16 epoch 无变化 |

**根本原因**: 大模型在当前数据规模下陷入退化解（预测全负），不是 ISAB/SAB 的问题，而是**模型规模 vs 数据/特征复杂度**的匹配问题。

### 2.4 Token Norm + Token Dropout

| 实验 | token_norm | token_dropout | Test F1 | Test PR-AUC |
|------|-----------|---------------|---------|-------------|
| SAB-base | N | 0.0 | **0.6417** | 0.7204 |
| SAB-full | Y | 0.2 | 0.6130 | **0.7269** |

**结论**: Token normalization + dropout 微幅提升 PR-AUC（排序质量），但 F1 略降。对 threshold=0.5 的分类判据不友好。

---

## 3. 关键发现

### 3.1 Early Stopping: val_loss 不是好的监控指标

| 模型 | Best (by val_loss) epoch | Test F1 (best) | Test F1 (last) |
|------|------------------------|----------------|----------------|
| SAB-base | ep23 | 0.4995 | **0.6417** |
| SAB-d320 | ep43 | 0.4766 | **0.5424** |

val_loss 在 epoch 20-30 达到最低，但 val F1 持续提升到 epoch 60-80。按 val_loss 选 best checkpoint 严重低估模型能力。

**应使用 val_f1 作为 early stopping / best checkpoint 的监控指标。**

### 3.2 模型规模与优化困难

实验证实了一个反直觉的现象: 在这个任务上，**更大的模型 = 更差的性能**。

可能的原因:
- 387-dim token 中，384 维来自冻结的 TargetNet_Optimized，信息已被压缩
- 大模型容量远超特征信息量，导致过参数化和梯度消失
- Focal loss 在大模型上更容易陷入"预测全负"的退化解（loss=0.1686 约等于"全负"策略）

### 3.3 训练速度对比

| 模型 | d | layers | it/s (bs=512) | 每 epoch 时间 | 80 epochs 总时间 |
|------|---|--------|--------------|-------------|----------------|
| 小模型 SAB | 256 | 2 | ~16 | ~42s | ~56min |
| d=320 SAB | 320 | 3 | ~9.2 | ~73s | ~97min |
| d=384 SAB | 384 | 3 | ~11.8 | ~57s | ~76min |
| d=512 ISAB | 512 | 4 | ~3.3 | ~200s | ~4.5h |
| d=512 SAB | 512 | 4 | ~4.7 | ~140s | ~3.1h |

---

## 4. 当前最优配置

```yaml
# model
model:
  arch: PairSetTransformerAggregator
  in_dim: 387
  d_model: 256
  dim_ff: 1024
  n_layers: 2
  n_heads: 8
  block_type: sab           # SAB > ISAB
  num_seeds: 1
  use_output_sab: false
  use_token_norm: false     # F1 导向不用
  dropout: 0.1

# trainer
trainer_pair_selected:
  batch_size: 512           # 远优于 4096
  lr_agg: 3e-4
  wd_agg: 1e-2
  scheduler_agg: cosine
  scheduler_t_max: 80
  warmup_steps: 0           # 小模型不需要
  clip_grad_norm: 1.0
  loss_type: focal
  focal_alpha: 0.4
  focal_gamma: 1.0
  label_smoothing: true
  smooth_neg: 0.05
  smooth_pos: 0.95
  bce_lambda: 0.01
  num_epochs: 80

# 当前最优成绩
# Test F1: 0.6417, PR-AUC: 0.7204, Precision: 0.6826, Recall: 0.6055
```

---

## 5. 下一步方向

当前 F1=0.6417 → 目标 F1>0.8，差距 ~0.16。

**已排除的方向:**
- ~~放大模型~~ (反而更差)
- ~~ISAB→SAB~~ (小模型上提升微弱)
- ~~Token norm/dropout~~ (PR-AUC微升，F1微降)
- ~~调整 LR/warmup/clip~~ (对大模型无效，小模型已近最优)

**待探索的方向:**

| 方向 | 预期收益 | 难度 | 说明 |
|------|---------|------|------|
| **预训练 Instance Model** | 高 | 中 | 在更大 CTS 数据集上预训练 TargetNet_Optimized，提升 embedding 质量 |
| **预训练 Cheap Model** | 中 | 中 | 用更好的 teacher 重新蒸馏，提升 selector 质量 |
| **阈值优化** | 低-中 | 低 | 在 val 上 sweep 最优阈值 (但用户认为难以公平比较) |
| **更多特征** | 高 | 高 | 加入 RNA 二级结构、保守性分数等 |
| **数据清洗** | 中 | 中 | 检查噪声标签，过滤低质量 pair |
| **Ensemble** | 中 | 低 | 多 seed / 多 selector 融合 |

**优先级: 预训练 Instance Model > 预训练 Cheap Model > 数据清洗 > Ensemble**

---

## 6. 文件索引

| 文件 | 说明 |
|------|------|
| `experiments/EXP8/exp8_shm_plan.md` | Agent A/B 并行执行计划 |
| `experiments/EXP8/exp8_agentA_shm.md` | Agent A (GPU 0) 执行手册 |
| `experiments/EXP8/exp8_agentB_shm.md` | Agent B (GPU 1) 执行手册 |
| `experiments/EXP8/exp8_bs512_vs_bs4096.png` | Batch size 对比曲线图 |
| `experiments/EXP8/exp8_a3_a6_curves.png` | A3/A6 训练曲线 |
| `experiments/EXP8/STAGE3_PLAN.md` | Stage3 原始计划 |
| `configs/experiment/MTI_train_selected_inst_v2.yaml` | V2 配置 (token_norm, token_dropout, warmup, clip) |
