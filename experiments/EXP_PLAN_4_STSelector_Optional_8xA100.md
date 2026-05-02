# 实验计划 4: STSelector 补充实验（可选）

**硬件**: 8×A100-80GB（闲时执行）  
**预计时间**: 5-7小时  
**优先级**: P2（低优先级，可选）  
**负责人**: [待分配]  
**状态**: ⏳ 待决策

---

## 📋 实验概述

**目的**: 验证 STSelector (S2) 在 MTI 大规模数据集上的效果

**实验性质**: ✅ **需要训练模型**

**当前状态**:
- 论文已适度降级 STSelector 声明
- MTI 所有实验均使用 TopK (S0)
- 论文采用**方案 B**（保持弱化表述）

**本实验目的**: 
- 仅供参考，了解 S2 在 MTI 上的实际效果
- 如果 S0 ≈ S2，进一步验证"TopK 足够好"的结论
- **不强制执行**，时间紧张可跳过

---

## 🔧 实验配置

### 数据集信息
- **数据集**: MTI (420,351 pairs)
- **划分**: Train 294,029 / Val 63,227 / Test 63,095

### 模型配置
- **Instance Encoder**: TargetNet_Optimized X-Large (复用已训练)
- **Cheap Encoder**: CheapCTSNet (复用已训练)
- **Aggregator**: PairSetTransformerAggregator (新训练)
- **K**: 512（与 baseline 一致）

### Selector 配置（关键差异）

**S0 (TopK, 已有结果)**:
```yaml
selector:
  type: "topk"
  k1_ratio: 1.0              # 全部用 TopK
```

**S2 (STSelector, 本实验)**:
```yaml
selector:
  type: "stselector"
  k1_ratio: 0.5              # 50% TopK + 50% diversity
  use_position_diversity: true
  use_embedding_diversity: true
  num_bins: 64               # K=512 对应配置
  bin_top_m: 64
  hash_bits: 32
  hash_cap: 2
  candidate_pool_size: 4096  # 候选池大小
```

### 训练配置（与 K=512 baseline 一致）
```yaml
run:
  kmax: 512
  batch_size: 128            # per GPU
  num_epochs: 100
  
trainer:
  optimizer: "adamw"
  lr_agg: 5e-5
  weight_decay_agg: 1e-2
  lr_inst: 0                 # 冻结
  scheduler: "cosine"
  warmup_steps: 200

loss:
  type: "focal"
  focal_alpha: 0.4
  focal_gamma: 1.0
  bce_lambda: 0.01
  label_smoothing: true
```

---

## 💻 执行命令

### 训练命令

```bash
cd /path/to/PAIRFormer

# 使用 8×A100 DDP 训练
torchrun --nproc_per_node=8 \
    --master_port=29501 \
    -m src.launch.train_pair_selected_inst \
    experiment=MTI_train_selected_inst \
    run.kmax=512 \
    selector.type="stselector" \
    selector.k1_ratio=0.5 \
    selector.use_position_diversity=true \
    selector.use_embedding_diversity=true \
    selector.num_bins=64 \
    selector.bin_top_m=64 \
    selector.hash_bits=32 \
    selector.hash_cap=2 \
    selector.candidate_pool_size=4096 \
    run.batch_size=128 \
    run.num_epochs=100 \
    trainer.lr_agg=5e-5 \
    trainer.lr_inst=0 \
    run.instance_encoder_checkpoint=checkpoints/MTI_v3_xlarge_resume/best.pt \
    run.cheap_encoder_checkpoint=checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/best.pt \
    run.freeze_instance_encoder=true \
    seed=2020 \
    run.output_dir=checkpoints/MTI_K512_STSelector \
    run.use_ddp=true \
    2>&1 | tee logs/mti_k512_stselector_train.log
```

### 评估命令

```bash
python -m src.launch.eval_pair_selected_inst \
    experiment=MTI_train_selected_inst \
    run.checkpoint=checkpoints/MTI_K512_STSelector/best.pt \
    run.kmax=512 \
    run.batch_size=256 \
    2>&1 | tee logs/mti_k512_stselector_eval.log
```

### 执行脚本

**文件**: `scripts/run_mti_stselector.sh`

```bash
#!/bin/bash

# MTI K=512 with STSelector (S2)
# Hardware: 8×A100-80GB
# Estimated time: 5-7 hours
# Priority: P2 (optional)

set -e

EXPERIMENT="MTI_train_selected_inst"
KMAX=512
BATCH_SIZE=128
NUM_EPOCHS=100
LR_AGG=5e-5
SEED=2020
OUTPUT_DIR="checkpoints/MTI_K512_STSelector"
NUM_GPUS=8

INSTANCE_CKPT="checkpoints/MTI_v3_xlarge_resume/best.pt"
CHEAP_CKPT="checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/best.pt"

echo "========================================="
echo "MTI K=512 with STSelector (S2)"
echo "========================================="
echo "Hardware: ${NUM_GPUS}×A100-80GB"
echo "K: $KMAX"
echo "Selector: STSelector (S2)"
echo "Batch size: $BATCH_SIZE per GPU"
echo "Epochs: $NUM_EPOCHS"
echo "Output: $OUTPUT_DIR"
echo ""
echo "NOTE: This is an optional experiment."
echo "      Can be skipped if time is limited."
echo ""

# Check if user wants to proceed
read -p "Proceed with training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p logs

# Start training
echo "Starting training at $(date)"

torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=29501 \
    -m src.launch.train_pair_selected_inst \
    experiment=$EXPERIMENT \
    run.kmax=$KMAX \
    selector.type="stselector" \
    selector.k1_ratio=0.5 \
    selector.use_position_diversity=true \
    selector.use_embedding_diversity=true \
    selector.num_bins=64 \
    selector.bin_top_m=64 \
    selector.hash_bits=32 \
    selector.hash_cap=2 \
    selector.candidate_pool_size=4096 \
    run.batch_size=$BATCH_SIZE \
    run.num_epochs=$NUM_EPOCHS \
    trainer.lr_agg=$LR_AGG \
    trainer.lr_inst=0 \
    run.instance_encoder_checkpoint=$INSTANCE_CKPT \
    run.cheap_encoder_checkpoint=$CHEAP_CKPT \
    run.freeze_instance_encoder=true \
    seed=$SEED \
    run.output_dir=$OUTPUT_DIR \
    run.use_ddp=true \
    2>&1 | tee logs/mti_k512_stselector_train.log

echo ""
echo "Training completed at $(date)"

# Evaluation
echo "Starting evaluation..."
python -m src.launch.eval_pair_selected_inst \
    experiment=$EXPERIMENT \
    run.checkpoint=$OUTPUT_DIR/best.pt \
    run.kmax=$KMAX \
    run.batch_size=256 \
    2>&1 | tee logs/mti_k512_stselector_eval.log

echo ""
echo "Evaluation completed"
echo "Checkpoint: $OUTPUT_DIR/best.pt"
```

---

## 📊 结果记录

### 对比表格

| Selector | Test F1 | Test Acc | Test Prec | Test Rec | Test ROC-AUC | Test PR-AUC | 备注 |
|----------|---------|----------|-----------|----------|--------------|-------------|------|
| **S0 (TopK)** | **0.7918** | 0.7862 | 0.7997 | 0.7840 | 0.8253 | 0.7884 | 已有 |
| S2 (STSelector) | | | | | | | 填写 |
| **Delta (S2-S0)** | | | | | | | 计算 |

### 预期结果

| Metric | S0 (TopK) | S2 (STSelector, 预期) | Delta (预期) |
|--------|-----------|---------------------|-------------|
| Test F1 | 0.7918 | 0.7920-0.7925 | <0.005 |
| Test PR-AUC | 0.7884 | 0.7885-0.7890 | <0.001 |

**预期结论**: S0 ≈ S2 at K=512，与 miRAW 结果一致

---

## 📈 结果分析

### 关键问题

1. **S2 是否显著优于 S0？**
   - 如果 Delta F1 < 0.005: 不显著，TopK 足够
   - 如果 Delta F1 > 0.01: 显著，STSelector 有价值

2. **与 miRAW 的一致性**
   - miRAW K=64: S0 (0.9942) vs S2 (0.9939), 差异 <0.0005
   - MTI K=512: 预期类似（差异 <0.005）

3. **计算开销**
   - STSelector 增加 CPU 计算（position binning, SimHash）
   - 但在 K=512 下，开销相对较小

### 论文影响

**如果 S0 ≈ S2**（预期）:
- 进一步验证"TopK 足够好"的结论
- 支持论文当前的弱化表述
- 可在 Discussion 中补充：
  > "We additionally validated STSelector on the large-scale MTI dataset (K=512), 
  > confirming that simple TopK selection (S0) achieves comparable performance 
  > (F1 difference <0.005), consistent with the miRAW ablation."

**如果 S2 > S0**（意外）:
- 需要重新评估 STSelector 的价值
- 可能需要调整论文表述
- 但这种情况概率较低

---

## 🎨 可视化

### 对比图

**文件**: `scripts/plot_selector_comparison.py`

```python
import matplotlib.pyplot as plt
import numpy as np

# 数据
datasets = ['miRAW\n(K=64)', 'MTI\n(K=512)']
s0_f1 = [0.9942, 0.7918]  # TopK
s2_f1 = [0.9939, ...]     # STSelector (填写)

x = np.arange(len(datasets))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, s0_f1, width, label='S0 (TopK)', color='steelblue')
bars2 = ax.bar(x + width/2, s2_f1, width, label='S2 (STSelector)', color='coral')

ax.set_ylabel('Test F1@0.5', fontsize=12)
ax.set_title('Selector Ablation: TopK vs STSelector', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend()
ax.grid(True, axis='y', alpha=0.3)

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figures/selector_comparison.pdf', dpi=300, bbox_inches='tight')
print("Figure saved to figures/selector_comparison.pdf")
```

---

## ✅ 验收标准

### 必须满足（如果执行）
- [ ] 训练完成 100 epochs
- [ ] 在 test set 上评估
- [ ] 填写结果表格
- [ ] 计算 S0 vs S2 的差异

### 质量检查
- [ ] 训练曲线正常
- [ ] Val F1 在合理范围内（0.79-0.80）
- [ ] S0 vs S2 差异符合预期（<0.005）

---

## 🚫 跳过条件

**可以跳过本实验，如果**:
1. ⏰ 时间紧张（截止日期临近）
2. 🎯 K=1 baseline 优先级更高
3. 📊 论文已有足够的 selector ablation（miRAW Table 3）
4. ✅ 用户决定采用方案 B（保持弱化表述）

**跳过后的处理**:
- 在论文中保持当前表述（方案 B）
- 在 MTI results 中说明使用 TopK
- 在 Discussion 中引用 miRAW 的 ablation 结果

---

## 📝 提交清单

**如果执行，请提交**:

1. **Checkpoint**: `checkpoints/MTI_K512_STSelector/best.pt`
2. **训练日志**: `logs/mti_k512_stselector_train.log`
3. **评估日志**: `logs/mti_k512_stselector_eval.log`
4. **结果数据**: `results/mti_stselector_results.json`
5. **对比图**: `figures/selector_comparison.pdf`
6. **填写的结果表格**（本文档中）

**如果跳过，请说明**:
- [ ] 跳过原因：_______________
- [ ] 决策人：_______________
- [ ] 决策时间：_______________

---

## 🔄 其他可选实验（闲时执行）

### 实验 4.2: 多种子实验（MTI）

**目的**: 提供 mean±std 统计

**配置**: 
- 在 MTI K=512 上运行 seed=2025, 2026
- 计算 mean±std over 3 seeds

**时间**: 10-12h (2 seeds × 5-6h)

**优先级**: P3（最低）

### 实验 4.3: Token 组成消融（MTI）

**目的**: 验证 token 各特征的贡献

**配置**:
- 移除 inst_logit: token = [emb, esa, pos]
- 移除 esa: token = [emb, logit, pos]
- 移除 pos: token = [emb, logit, esa]

**时间**: 15-18h (3 ablations × 5-6h)

**优先级**: P3（最低）

### 实验 4.4: Loss 消融（MTI）

**目的**: 验证 Focal loss 的贡献

**配置**:
- BCE only: focal_lambda=0
- Focal only: bce_lambda=0

**时间**: 10-12h (2 ablations × 5-6h)

**优先级**: P3（最低）

---

## 📞 决策流程

### 步骤 1: 评估时间预算
- [ ] 距离截止日期还有 ___ 天
- [ ] K=1 baseline 预计完成时间：___
- [ ] Robustness vs. n 预计完成时间：___
- [ ] 剩余时间：___

### 步骤 2: 决策
- [ ] **执行 STSelector 实验**（如果时间充足）
- [ ] **跳过 STSelector 实验**（如果时间紧张）

### 步骤 3: 记录
- 决策：_______________
- 理由：_______________
- 决策人：_______________
- 时间：_______________

---

## 📞 联系方式

**问题反馈**: [待填写]  
**紧急联系**: [待填写]

---

**执行人签字**: _______________  
**开始时间**: _______________  
**完成时间**: _______________  
**审核人签字**: _______________
