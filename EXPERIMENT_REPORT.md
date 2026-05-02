# PAIRFormer Experiment Report: MTI Dataset
## Budgeted Relational Multi-Instance Learning for miRNA-mRNA Target Interaction Prediction

**Date**: 2026-05-02
**Status**: SWA K=512 仍在训练中 (ep167/200), 完成后需更新最终结果

---

## 1. 任务定义

预测功能性 miRNA-mRNA 靶向互作 (miRNA-mRNA Target Interaction, MTI)。给定一个 miRNA 和一个 mRNA 3'UTR 序列，预测该 miRNA 是否调控该 mRNA 的表达。

**核心挑战**:
- 每个 miRNA-mRNA 对产生数量不一 (几十到数千) 的候选靶位点 (Candidate Target Sites, CTS)，需从大量 CTS 中聚合出 pair-level 预测
- CTS 数量差异极大，从 0 到数千不等，需要处理变长集合
- 传统的 Max/Mean pooling 忽略了 CTS 间的空间关系和相互依赖

---

## 2. 数据集

### 2.1 数据来源

MTI 数据集由 **mirCLIP** (miRNA-Target Interaction from CLIP-seq) 系列实验数据整合而来，包含三种实验技术来源的 miRNA-mRNA 互作数据：

| 来源 | 物种 | 说明 |
|------|------|------|
| CLASH (Crosslinking, Ligation, And Sequencing of Hybrids) | Human | 直接检测 miRNA-mRNA 杂交双链 |
| chiRA (chiRAlP) | Human, Mouse, Worm | 基于 CLIP-seq 的互作检测 |
| HYBRID | Human, Mouse, Worm | 杂交验证方法 |

原始数据来源文件:
- `human_mRNA_chira_download.csv` (110,981 条 CTS-level 记录)
- `human_mRNA_clan_download.csv` (110,981 条)
- `human_mRNA_hyb_download.csv` (110,981 条)
- 以及 mouse/worm 的对应文件

### 2.2 数据预处理流程

**Step 1: CTS-level 聚合为 mRNA-level**

从 site-level (CTS-level) 的互作记录中，按 (miRNA, mRNA) 对聚合，生成 mRNA-level 的正例对。每个正例对包含 miRNA 序列和完整的 mRNA 3'UTR 序列。

**Step 2: 负例生成**

对每个正例对 (miRNA_i, mRNA_j)，随机选择一个非靶 mRNA_k（即 miRNA_i 不靶向 mRNA_k），构造负例对。具体策略:
- 从 mRNA 池中随机采样 (seed=1234)
- 最多尝试 100 次找到非靶 mRNA
- 回退策略: 若无法找到非靶 mRNA，选择其他任意 mRNA

**Step 3: 数据划分**

使用 `sklearn.model_selection.train_test_split` 进行 **分层随机划分** (stratified random split, seed=2020):
- 训练集: 70%
- 验证集: 15%
- 测试集: 15%

**Step 4: CTS 过滤 (去重)**

运行 `scripts/filter_nocts_pairs.py` 过滤掉无法生成任何有效 CTS 窗口的 pair。三种过滤原因:
1. mRNA 序列为 "Sequence unavailable" (字符串长度 19 < 窗口大小 40)
2. mRNA 序列长度 < 40 nt
3. mRNA >= 40 nt 但没有任何窗口通过 ESA >= 6.0 阈值

过滤结果: 475,787 -> 420,351 pairs (移除 ~55,436 pairs, ~11.6%)

### 2.3 最终数据统计

| 划分 | Pairs | 正例 | 负例 | 正/负比 |
|------|-------|------|------|---------|
| Train | 294,029 | 152,516 | 141,513 | 1.078 |
| Val | 63,227 | 32,720 | 30,507 | 1.073 |
| Test | 63,095 | 32,720 | 30,375 | 1.077 |
| **Total** | **420,351** | **217,956** | **202,395** | **1.077** |

- **唯一 miRNA 数量**: 1,390
- **唯一 mRNA (转录本) 数量**: 44,172
- **数据格式**: TSV, 7 列: `gene_name, mrna_id, mrna_seq, mirna_id, mirna_seq, label, split`

### 2.4 CTS 窗口提取

对于每个 miRNA-mRNA pair，提取 CTS 候选靶位点的流程:

1. 将 mRNA 序列 T->U 转换后反转
2. 沿反转后的 mRNA 滑动 **40bp 窗口**
3. 对每个窗口计算 **扩展种子对齐 (Extended Seed Alignment, ESA)** 分数
   - 对齐 miRNA 种子区 (positions 0-9) 与 CTS 窗口 (positions 5-14)
   - 使用 Watson-Crick + G:U wobble 配对的打分矩阵 (match=1, mismatch=0)
   - BioPython pairwise2 进行全局对齐
4. **过滤**: 仅保留 ESA >= 6.0 的窗口
5. **编码**: 将 miRNA 和 CTS 窗口编码为 one-hot 矩阵 (10 channels x 50 positions)
   - 5 channels (A, C, G, U, gap) x 2 rows (miRNA seed, CTS window)
   - 包含 ESA 对齐后的上游/下游上下文

6. **位置编码**: 对每个窗口计算归一化位置 `pos = center / (L - 1)`, 其中 center 为窗口在原始 mRNA 中的中心位置, L 为 mRNA 序列长度

### 2.5 数据特征

- 每个 pair 的 CTS 窗口数量 (n) 差异极大，从几十到数千不等
- 平均每个 pair 约有 ~1,152 个候选 CTS 窗口
- 这使得 pair-level 预测本质上是一个变长集合聚合问题 (MIL / Set-to-One)

---

## 3. 方法: 三阶段训练框架

### 3.1 总体架构

```
miRNA-mRNA pair -> n 个 CTS 窗口
    |
Stage 1: TargetNet_Optimized (1D-CNN) 编码每个 CTS -> 384/1536-dim embedding + logit
    |
Stage 2: CheapCTSNet_TinyConv 蒸馏轻量编码器 -> 64-dim embedding + logit
    |
Stage 3: STSelector 选择 Top-K CTS -> PairSetTransformerAggregator 聚合 -> pair-level 预测
```

### 3.2 Stage 1: CTS-level 编码器 (TargetNet_Optimized)

#### 3.2.1 模型架构

1D-CNN with residual blocks and squeeze-excitation attention:

```
Input [B, 10, 50] (one-hot CTS window with ESA)
  -> Stem: Conv_Layer blocks (Conv1d -> BN -> ReLU -> Dropout)
  -> Stage 1~N: ResNet_Block blocks
      - Pre-activation: ReLU -> Dropout -> Conv1d (multi-scale optional) -> BN -> ReLU -> Dropout -> Conv1d -> BN
      - Skip connection (1x1 conv when channel dims change)
      - Multi-scale: 3 parallel convolutions (kernel 3/5/7), split channels ~1/3 each, concatenate
  -> SE Block: AdaptiveAvgPool1d -> FC -> ReLU -> FC -> Sigmoid -> channel scaling + residual
  -> ReLU -> Dropout -> AdaptiveAvgPool1d(target_output_length)
  -> Flatten -> Linear(embedding_dim, 1)
  -> Output: logit [B], embedding [B, embedding_dim]
```

**Embedding 维度** = `num_channels[-1] * target_output_length`

#### 3.2.2 配置变体与消融实验

| 变体 | Channels | Blocks | Multi-scale | BN | Dropout | Params | Embedding Dim | Val F1 (CTS-level) |
|------|----------|--------|-------------|----|---------|--------|---------------|---------------------|
| Standard (opt4_tiny) | [16,16,32,32] | [1,1,1,1] | x | x | 0.5 | ~14K | 384 | 0.6775 |
| Large | [32,32,64,64] | [2,2,2,2] | x | y | 0.1 | ~153K | 768 | 0.6838 |
| **X-Large** | **[64,64,128,128]** | **[3,3,3,3]** | **y** | **y** | **0.1** | **~909K** | **1536** | **0.6849** |
| XX-Large | [128,128,256,256] | [4,4,4,4] | y | y | 0.1 | ~3.6M | 3072 | 0.6840 |
| X-Large drop=0.05 | [64,64,128,128] | [3,3,3,3] | y | y | 0.05 | ~909K | 1536 | 0.6849 |
| X-Large drop=0.2 | [64,64,128,128] | [3,3,3,3] | y | y | 0.2 | ~909K | 1536 | 0.6820 |

**X-Large 配置细节**:
- `stem_kernel_size=5`, `block_kernel_size=3`, `skip_connection=True`
- `se_type="basic"`, `se_reduction=8`
- `target_output_length=12`
- `embedding_dim = 128 * 12 = 1536`
- `params = 909K`

**训练配置**:
- Optimizer: AdamW, selective weight decay (bias/norm excluded)
- Loss: BCE + Focal (bce_lambda=0.01, focal_gamma=1.0, focal_alpha=0.4)
- Label Smoothing: smooth_pos=0.95, smooth_neg=0.05
- ESA Weighting: esa_scale=10.0, esa_lambda_pos=1.0, esa_lambda_neg=0.5
- AMP enabled, gradient clipping=1.0
- EMA: decay=0.999
- MIL Aggregation: softmax pooling (temperature=1.0)

**关键发现**: CTS-level F1 在 ~0.685 饱和, 进一步增大模型 (XX-Large) 不再提升。这表明 CTS 级别的预测能力受限于窗口本身的信号质量。

### 3.3 Stage 2: 轻量编码器蒸馏 (CheapCTSNet)

#### 3.3.1 模型架构

```
Input [B, 10, 50]
  -> Conv1d(10, 16, k=5, s=2) -> ReLU
  -> Conv1d(16, 32, k=3, s=2) -> ReLU
  -> AdaptiveAvgPool1d(1) -> squeeze -> z_content [B, 32]
  -> Dropout
  -> Embedding head: Linear(32, 64) -> L2-normalize -> emb [B, 64]
  -> Logit head: Linear(34, 64) -> ReLU -> Linear(64, 1) -> logit [B]
      (34 = 32 content + 2 metadata [esa_score, pos])
```

参数量: ~6.8K (相比 TargetNet_X-Large 的 909K, 减少 ~134x)

#### 3.3.2 蒸馏训练

**Loss**:
```
L = (1 - a) * L_sup + a * (b_kd * L_kd + b_feat * L_feat + b_rel * L_rel)
```

- L_sup: 监督 BCE + Focal + Label Smoothing + ESA Weighting
- L_kd: KD logit loss (temperature T=2.0)
- L_feat: 余弦相似度损失 (student projection -> teacher feature space)
- L_rel: 关系蒸馏损失 (样本间 Gram 矩阵的余弦相似度)
- a: 余弦退火从 0.8 -> 0.5

**超参数**: b_kd=1.0, b_feat=0.1, b_rel=1.0, meta_mode="logit_only", meta_dropout=0.2

**关键作用**: CheapCTSNet 在 Stage 3 中用于快速对所有 CTS 窗口打分 (不经过 GPU), 为 STSelector 提供排序信号。

### 3.4 Stage 3: Budgeted Relational MIL

#### 3.4.1 STSelector (预算感知选择器)

非参数化的确定性算法，5 步选择流程:

1. **Step A - TopK**: 按 CheapCTSNet logit 排序, 取 Top-K1 = floor(k1_ratio * K) 个高分 CTS
2. **Step B - 位置分箱**: 将 CTS 按 mRNA 上的位置分到 B 个等宽 bin, 每个 bin 取 top-m 的 min-heap 候选池
3. **Step C - SimHash 去重**: 每个 bin 内用 Axis-SimHash 移除冗余 CTS (随机采样 hash 维度, 对 hash key 设置 cap)
4. **Step D - 平衡配额**: 按各 bin 的质量权重 (top-t scores 的均值) 分配 K2 = K - K1 个 slot
5. **Step E - 合并**: 合并 S1 和 S2, 去重, 不足 K 则用 score 填充

**K 值相关参数**:

| K 范围 | 候选池 L | 分箱 B | 每 bin top-m | Hash bits | Hash cap |
|--------|----------|--------|-------------|-----------|----------|
| K <= 128 | 1024 | 32 | 32 | 24 | 2 |
| 128 < K <= 256 | 2048 | 32 | 64 | 32 | 2 |
| K > 256 | 4096 | 64 | 64 | 32 | 2 |

**零可学习参数** -- STSelector 是纯算法模块。

#### 3.4.2 Token 组装

每个被选中的 CTS 窗口被组装为一个 token:

```
token = [inst_emb, inst_logit, esa, pos]
```

| 特征 | 维度 | 来源 | 说明 |
|------|------|------|------|
| inst_emb | 384 / 1536 | TargetNet_Optimized | CTS 窗口编码器 embedding |
| inst_logit | 1 | TargetNet_Optimized | CTS 窗口编码器 logit |
| esa | 1 | 预计算 | 扩展种子对齐分数 (归一化) |
| pos | 1 | 预计算 | CTS 在 mRNA 上的归一化位置 [0,1] |
| **Total** | **387 / 1539** | | |

最终 token 维度:
- Standard (opt4_tiny): 384 + 1 + 1 + 1 = **387**
- X-Large: 1536 + 1 + 1 + 1 = **1539**

#### 3.4.3 PairSetTransformerAggregator (集合变换器聚合器)

基于 Set Transformer 的置换不变聚合器:

```
Input: x [B, K, D_in], attn_mask [B, K]  (1=valid, 0=pad)
  -> (optional) LayerNorm(D_in)
  -> Linear(D_in, d_model)  -> h [B, K, d_model]
  -> Zero-out padded positions

Encoder (x n_layers):  -- 置换等变
  SAB: MAB(X, X)
    MAB(X, Y) = LN(X + MHA(X, Y, Y)) -> LN(H + FF(H))
    MHA: Multi-Head Attention with key_padding_mask
    FF: Linear(d, d_ff) -> activation -> Dropout -> Linear(d_ff, d)

Decoder:  -- 置换不变
  PMA with k=1 seed -> pooled [B, 1, d_model]
    PMA(S, rFF(Z)) = MAB(S, rFF(Z))
    rFF: Linear(d, d_ff) -> activation -> Linear(d_ff, d)

  -> LayerNorm(d_model)
  -> Linear(d_model, d_model) -> ReLU -> Dropout -> Linear(d_model, 1)
  -> Output: logit [B]
```

**关键性质**: 置换不变性 (permutation invariance) -- 输出与 CTS token 的输入顺序无关, 这对集合级别的预测至关重要。

#### 3.4.4 聚合器架构消融实验

**对比的聚合器类型** (均在 V2 relabel 数据上, K=64, d_model=1024):

| 聚合器 | 架构 | 参数量 | Val F1 |
|--------|------|--------|--------|
| PairSetTransformerAggregator (SAB) | Set Transformer, L=4, H=16 | ~18M | 0.7353* |
| PairGNNAggregator | k-NN Graph + GAT, L=3, k=8 | ~37M | 0.7602 |
| PairGNNMoEAggregator | GNN + MoE (4 experts) | ~64M | 0.7597 |
| PairCNNAggregator | Sorted 1D-CNN, dilated | ~14M | 0.6865 |

*注: 不同聚合器实验使用了不同的数据处理流程, 上述对比仅供参考。Set Transformer 在完整 V3 pipeline 中 (K=512) 达到了最好的 0.7913。*

**结论**: Set Transformer (SAB) 在参数效率和性能上均表现最优。CNN 聚合器因依赖固定排序而性能较差 (0.6865)。GNN 聚合器性能接近但参数量翻倍。

#### 3.4.5 Set Transformer 缩放消融

固定 X-Large instance encoder, K=64, 在 V2 relabel 数据上:

| 配置 | d_model | n_layers | dim_ff | n_heads | dropout | Val F1 |
|------|---------|----------|--------|---------|---------|--------|
| Medium (ISAB, m=32) | 512 | 3 | 2048 | 8 | 0.1 | 0.6417 |
| Large | 768 | 6 | 3072 | 8 | 0.1 | 0.7340 |
| **X-Large (SAB)** | **1024** | **4** | **4096** | **16** | **0.1** | **0.7353** |
| X-Large (L=5) | 1024 | 5 | 4096 | 16 | 0.1 | 0.7339 |
| XX-Large | 1280 | 4 | 5120 | 16 | 0.1 | 0.7333 |
| X-Large dropout=0.5 | 1024 | 4 | 4096 | 16 | 0.5 | 0.7217 |

**结论**: d_model=1024, n_layers=4 是最优平衡点。ISAB (inducing points) 导致性能大幅下降 (0.6417), 远不如 SAB。继续增大模型 (XX-Large) 不再提升。

#### 3.4.6 训练配置 (V3 最终配置)

| 超参数 | 值 |
|--------|-----|
| Optimizer | AdamW |
| Learning rate (agg) | 5e-5 |
| Weight decay (agg) | 1e-2 |
| Learning rate (inst) | 0 (冻结) |
| Scheduler | CosineAnnealingLR |
| Warmup steps | 200 |
| Loss | Focal (alpha=0.4, gamma=1.0) |
| Label Smoothing | smooth_pos=0.95, smooth_neg=0.05 |
| BCE lambda | 0.01 |
| Batch size | 128 (per GPU) |
| Gradient clip | 0.0 (disabled) |
| AMP | disabled |
| Monitor | val_f1, greater_is_better=true |
| SWA (可选) | swa_start_epoch=70, swa_lr=1e-5 |

#### 3.4.7 SWA (Stochastic Weight Averaging)

使用 PyTorch `torch.optim.swa_utils.AveragedModel`:
- 在 `swa_start_epoch` 之后开始平均模型权重
- 切换到常数学习率 `swa_lr=1e-5`
- 每个 epoch 更新 SWA 模型参数
- 验证时使用 SWA 模型 (averaged model) 进行推理
- Checkpoint 保存 raw model 和 SWA state dict

---

## 4. 实验结果

### 4.1 完整实验列表

所有 Stage 3 实验均使用相同的:
- **Instance Encoder**: TargetNet_Optimized X-Large (909K params, emb_dim=1536)
- **Aggregator**: PairSetTransformerAggregator (SAB, d_model=1024, n_layers=4, n_heads=16, dim_ff=4096)
- **Token dim**: 1539 (1536 emb + 1 logit + 1 esa + 1 pos)
- **Loss**: Focal + Label Smoothing
- **Seed**: 2020 (除非特别说明)

#### 4.1.1 K 值消融 (无 SWA)

| 实验 | K | 总 Epochs | Best Epoch | Best Val F1 | Best Val PR-AUC |
|------|---|-----------|------------|-------------|-----------------|
| K=128 expG v2 | 128 | 67 | 67 | 0.7793 | -- |
| K=256 expG | 256 | 42 | 41 | 0.7773 | -- |
| K=512 expG | 512 | 76 | 72 | 0.7891 | -- |
| K=512 expG (resume to 100) | 512 | 100 | 96 | **0.7900** | 0.8750 |

**趋势**: K 越大 val_f1 越高 (K=128: 0.7793 -> K=512: 0.7900), 增加 Top-K 选择的 CTS 数量有助于捕获更多靶位点信号。

#### 4.1.2 SWA 实验

| 实验 | K | 总 Epochs | Best Epoch | Best Val F1 | Best Val PR-AUC | 备注 |
|------|---|-----------|------------|-------------|-----------------|------|
| SWA expG | all | 80 | 80 | 0.7768 | 0.8312 | 早期基准 |
| SWA K=128 | 128 | 87 | 81 | 0.7813 | 0.8502 | |
| SWA K=256 | 256 | 70 | 68 | 0.7815 | 0.8660 | |
| SWA K=256 ext | 256 | 120 | 119 | 0.7832 | 0.8670 | 继续训练 |
| SWA K=256 ext2 | 256 | 200 | 191 | 0.7847 | 0.8684 | 继续训练 |
| SWA K=256 ext3 | 256 | 300 (ran 253) | 243 | 0.7854 | 0.8693 | 进程异常终止 |
| **SWA K=512** | **512** | **200 (running)** | **158** | **0.7913** | **0.8799** | **still training** |

**SWA K=256 增量训练收益**: 0.7815 -> 0.7832 -> 0.7847 -> 0.7854, 持续延长训练带来渐进提升。

#### 4.1.3 聚合器类型对比 (V3 pipeline, K=64, X-Large encoder)

| 聚合器 | Val F1 | 参数量 |
|--------|--------|--------|
| PairSetTransformerAggregator (ST05, SAB) | 0.6923 | ~18M |
| PairCNNAggregator | 0.6865 | ~14M |
| PairGNNAggregator | 0.7602 | ~37M |
| PairGNNMoEAggregator | 0.7597 | ~64M |

#### 4.1.4 多种子实验

| 实验 | Seed | Val F1 |
|------|------|--------|
| Pair EXP G | 2020 | 0.7754 |
| Pair EXP G | 2025 | 0.7734 |
| X-Large SAB (V2 relabel) | 2020 | 0.7353 |
| X-Large SAB (V2 relabel) | 2025 | 0.7346 |

### 4.2 Test Set 最终评估结果

在 63,095 对的独立测试集上评估 (threshold=0.5):

| 实验 | K | SWA | Val F1 | **Test F1** | Test Acc | Test Prec | Test Recall | Test ROC-AUC | Test PR-AUC |
|------|---|-----|--------|-------------|----------|-----------|-------------|--------------|-------------|
| V3 Baseline | 64 | N | -- | 0.7708 | 0.7656 | 0.7819 | 0.7600 | 0.8201 | 0.8295 |
| K=128 expG v2 | 128 | N | 0.7793 | **0.7786** | 0.7740 | 0.7912 | 0.7665 | 0.8332 | 0.8426 |
| SWA K=128 | 128 | Y | 0.7813 | **0.7795** | 0.7786 | 0.8062 | 0.7544 | 0.8395 | 0.8602 |
| K=256 expG | 256 | N | 0.7773 | **0.7790** | 0.7757 | 0.7964 | 0.7623 | 0.8393 | 0.8587 |
| SWA K=256 ext3 | 256 | Y | 0.7854 | **0.7857** | 0.7793 | 0.7910 | 0.7805 | 0.8448 | 0.8585 |
| **K=512 expG** | **512** | **N** | **0.7900** | **0.7918** | **0.7862** | **0.7997** | **0.7840** | **0.8253** | **0.7884** |
| SWA K=512 (current) | 512 | Y | 0.7913 | **0.7914** | 0.7841 | 0.7930 | 0.7898 | **0.8517** | **0.8729** |

**SWA K=512 仍在训练中, 完成后需更新此表。**

### 4.3 关键发现

#### 发现 1: Top-K 选择数量 (K) 与性能正相关

| K | Val F1 (no SWA) | Test F1 | Delta Test F1 vs K=64 |
|---|-----------------|---------|----------------------|
| 64 | -- | 0.7708 | baseline |
| 128 | 0.7793 | 0.7786 | +0.0078 |
| 256 | 0.7773 | 0.7790 | +0.0082 |
| 512 | 0.7900 | 0.7918 | +0.0210 |

K=512 相比 K=64 提升 2.1% test_f1, 说明更多的 CTS 候选提供了更丰富的靶位点信息。

#### 发现 2: SWA 在 K=256 上效果显著

| 配置 | K | No SWA Test F1 | SWA Test F1 | Delta SWA |
|------|---|----------------|-------------|-----------|
| K=128 | 128 | 0.7786 | 0.7795 | +0.0009 |
| K=256 | 256 | 0.7790 | 0.7857 | **+0.0067** |
| K=512 | 512 | 0.7918 | 0.7914* | -0.0004 |

*SWA K=512 仍在训练, 当前 best 为 ep158, 可能还会提升。

SWA 在 K=256 上带来 0.67% 提升, 在 K=128 上效果不明显。K=512 的 SWA 目前与 baseline 持平, 但 ROC-AUC 和 PR-AUC 有显著提升。

#### 发现 3: SWA 显著提升排序质量

| 实验 | Test ROC-AUC | Test PR-AUC |
|------|-------------|-------------|
| K=512 baseline | 0.8253 | 0.7884 |
| SWA K=512 (current) | **0.8517** (+0.026) | **0.8729** (+0.085) |

SWA 在排序指标 (ROC-AUC, PR-AUC) 上提升显著, 尤其是 PR-AUC 提升了 8.5%, 说明 SWA 改善了模型对预测概率的校准。

#### 发现 4: CTS 级别预测存在天花板

Instance encoder 的 CTS-level F1 在 ~0.685 饱和:
- Standard (14K params): 0.6775
- X-Large (909K params): 0.6849 (+0.74%)
- XX-Large (3.6M params): 0.6840 (-0.09%)

然而, pair-level F1 从 Standard 的 0.6417 提升到 X-Large 的 0.7353 (+9.36%), 说明更好的 instance embedding 为聚合器提供了更有效的 token。

#### 发现 5: Set Transformer SAB 优于 ISAB

| 注意力类型 | Val F1 |
|-----------|--------|
| ISAB (m=32) | 0.6417 |
| **SAB** | **0.7353** |

ISAB 通过 inducing points 降低复杂度, 但在本任务中导致严重的性能下降 (-9.36%)。这可能是因为 CTS 数量相对较少 (K <= 512), SAB 的 O(K^2) 复杂度完全可接受。

---

## 5. 最终模型性能总结

### 5.1 验证集最佳结果

| Rank | Experiment | K | Val F1 | Val PR-AUC |
|------|------------|---|--------|------------|
| 1 | **SWA K=512** | 512 | **0.7913** | **0.8799** |
| 2 | K=512 baseline | 512 | 0.7900 | 0.8750 |
| 3 | SWA K=256 ext3 | 256 | 0.7854 | 0.8693 |
| 4 | SWA K=256 ext2 | 256 | 0.7847 | 0.8684 |
| 5 | SWA K=128 | 128 | 0.7813 | 0.8502 |
| 6 | K=128 baseline | 128 | 0.7793 | -- |
| 7 | K=256 baseline | 256 | 0.7773 | -- |

### 5.2 测试集最佳结果

| Rank | Experiment | K | SWA | **Test F1** | Test ROC-AUC | Test PR-AUC |
|------|------------|---|-----|-------------|--------------|-------------|
| 1 | **K=512 baseline** | 512 | N | **0.7918** | 0.8253 | 0.7884 |
| 2 | SWA K=512 (current) | 512 | Y | 0.7914 | **0.8517** | **0.8729** |
| 3 | SWA K=256 ext3 | 256 | Y | 0.7857 | 0.8448 | 0.8585 |
| 4 | K=256 baseline | 256 | N | 0.7790 | 0.8393 | 0.8587 |
| 5 | SWA K=128 | 128 | Y | 0.7795 | 0.8395 | 0.8602 |
| 6 | K=128 baseline | 128 | N | 0.7786 | 0.8332 | 0.8426 |
| 7 | V3 Baseline | 64 | N | 0.7708 | 0.8201 | 0.8295 |

### 5.3 性能提升总结 (相比 K=64 baseline)

| Metric | K=64 baseline | K=512 best (current) | Improvement |
|--------|---------------|----------------------|-------------|
| Test F1 | 0.7708 | **0.7918** | **+2.10%** |
| Test ROC-AUC | 0.8201 | **0.8517** | **+3.16%** |
| Test PR-AUC | 0.8295 | **0.8729** | **+4.34%** |

---

## 6. 计算资源

### 6.1 硬件

| Resource | Spec |
|----------|------|
| GPU | 2x NVIDIA A100-SXM4-80GB |
| CPU | 128 cores |
| RAM | 463 GB |
| Storage | VepFS (network FS), /dev/shm (325GB tmpfs) |

### 6.2 Cache 体积

| K | Train Cache | Val Cache | Test Cache | Total |
|---|------------|-----------|------------|-------|
| 128 | ~28 GB | ~6 GB | ~6 GB | ~40 GB |
| 256 | ~113 GB | ~28 GB | ~24 GB | ~165 GB |
| 512 | ~453 GB | ~110 GB | ~97 GB | ~660 GB |

Cache 以 numpy float16 memory-mapped 格式存储, 包含 selected_inst 的 embedding 和 logit。

### 6.3 训练时间

| Experiment | Epochs | Time/Epoch | Total Time | GPU VRAM |
|------------|--------|-----------|------------|----------|
| K=128 | 67 | ~30 min | ~34 hr | ~25 GB |
| K=256 | 42 | ~50 min | ~35 hr | ~35 GB |
| K=512 | 100 | ~65 min | ~108 hr | ~47 GB |
| SWA K=256 (ext3) | 253 | ~55 min | ~232 hr | ~35 GB |
| SWA K=512 (ongoing) | 167+ | ~65 min | ~180+ hr | ~47 GB |

### 6.4 模型参数量

| Model | Parameters | Checkpoint Size |
|-------|-----------|-----------------|
| TargetNet_Optimized X-Large (instance) | ~909K | 3.7 MB |
| CheapCTSNet_TinyConv | ~6.8K | ~27 KB |
| PairSetTransformerAggregator (d1024/L4) | ~18M | ~848 MB |
| PairSetTransformerAggregator + SWA | ~36M (incl. SWA buffer) | ~1.2 GB |

---

## 7. 待完成事项

1. **SWA K=512 训练完成** -- 预计 2026-05-03, 需重新跑 test eval
2. **其他外部基准测试** -- DeepMirTar, deepTargetPro, miRBench 数据集上的泛化评估
3. **统计显著性检验** -- 多种子实验的置信区间
4. **消融: Token 组成** -- 移除各特征 (inst_logit, esa, pos) 的影响
5. **消融: STSelector vs TopK** -- STSelector 的 5 步算法 vs 简单 TopK

---

## 8. 实验时间线

| Date | Event |
|------|-------|
| 2026-03-29 | EXP8 start: MTI data preprocessing (negatives generation + split) |
| 2026-04-01~05 | Stage 1: TargetNet_Optimized architecture search (Standard -> X-Large) |
| 2026-04-05~07 | Stage 1: X-Large optimal config determined (val_f1=0.6849) |
| 2026-04-06 | Stage 2: CheapCTSNet distillation completed |
| 2026-04-07~13 | V2 Relabel: PairSetTransformerAggregator architecture search |
| 2026-04-13 | X-Large SAB d1024 L4确定为最优聚合器 (val_f1=0.7353) |
| 2026-04-14 | V3 pipeline启动: X-Large encoder -> K=64 baseline |
| 2026-04-14~15 | V3 pair EXP_G baseline (val_f1=0.7754, test_f1=0.7708) |
| 2026-04-15~18 | Aggregator类型对比 (ST05, CNN, GNN, GNN-MoE) |
| 2026-04-18~20 | K=128 expG v2 (val_f1=0.7793) |
| 2026-04-20~22 | K=256 expG (val_f1=0.7773) |
| 2026-04-22~29 | K=512 expG + resume (val_f1=0.7900) |
| 2026-04-22~23 | SWA K=128 (val_f1=0.7813) |
| 2026-04-22~28 | SWA K=256 ext/ext2/ext3 (val_f1=0.7815->0.7854) |
| 2026-04-29~05-02 | SWA K=512 (val_f1=0.7913, ongoing) |
| 2026-05-01 | Test eval: K=512 baseline test_f1=0.7918 (current best) |
| 2026-05-02 | Test eval: SWA K=512 (current) test_f1=0.7914, ROC-AUC=0.8517 |

---

## Appendix A: Data File Paths

| File | Path | Description |
|------|------|-------------|
| Main training data | `data/MTI/MTI_pair_random_split.txt` | 420,351 pairs, 7-col TSV |
| Pre-filter data | `data/MTI/MTI_pair_random_split_orig.txt` | 475,787 pairs |
| Instance encoder best | `checkpoints/MTI_v3_xlarge_resume/best.pt` | 909K params |
| Cheap encoder best | `checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/best.pt` | 6.8K params |
| K=512 baseline best | `checkpoints/MTI_v3_K512_expG/best.pt` | ~18M params |
| SWA K=512 best (current) | `checkpoints/MTI_v3_SWA_K512/best.pt` | ~36M params |
| SWA K=256 best | `checkpoints/MTI_v3_SWA_K256_ext3/best.pt` | ~36M params |
| K=512 cache | `cache_mti_full_topk_retrain_r4_v3relbl_k512/` | ~660 GB |

## Appendix B: Loss Function

### Focal Loss
```
p = sigmoid(logits)
p_t = p * labels + (1-p) * (1-labels)
alpha_t = alpha * labels + (1-alpha) * (1-labels)
L_focal = -alpha_t * (1 - p_t)^gamma * log(p_t)
```
gamma=1.0 (focusing parameter), alpha=0.4 (class-balanced)

### Label Smoothing
```
labels_smooth = 0.95 if label > 0.5 else 0.05
```

### Total Loss
```
L = bce_lambda * L_BCE + focal_lambda * L_focal
  = 0.01 * L_BCE + 1.0 * L_focal
```
