# 实验计划 2: K=1 Baseline on MTI

**硬件**: 8×A100-80GB  
**预计时间**: 4-6小时  
**优先级**: P0（最高优先级）  
**负责人**: [待分配]  
**状态**: ⏳ 待执行

---

## 📋 实验概述

**目的**: 证明 relational aggregation 的价值（K=1 即 single best CTS，无聚合）

**实验性质**: ✅ **需要训练模型**

**关键对比**: 
- K=1 (single best CTS) vs K=512 (relational aggregation)
- 预期差距：+3-5pp F1

**Rebuttal 承诺**: 
> "A retrained K=1 model reaches F1@0.5=0.925, while K=64 reaches 0.950"

---

## 🔧 实验配置

### 数据集信息
- **数据集**: MTI (420,351 pairs)
- **划分**: Train 294,029 / Val 63,227 / Test 63,095
- **正负比**: 1.077 (balanced)

### 模型配置
- **Instance Encoder**: TargetNet_Optimized X-Large (909K params, emb_dim=1536)
  - Checkpoint: `checkpoints/MTI_v3_xlarge_resume/best.pt`
- **Cheap Encoder**: CheapCTSNet_TinyConv (6.8K params, emb_dim=64)
  - Checkpoint: `checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/best.pt`
- **Aggregator**: PairSetTransformerAggregator (SAB, d_model=1024, n_layers=4)
- **Token dim**: 1539 (1536 emb + 1 logit + 1 esa + 1 pos)

### 训练配置
```yaml
# K=1 specific
run:
  kmax: 1                    # 固定 K=1
  batch_size: 128            # per GPU (effective 128*8=1024)
  num_epochs: 100
  
selector:
  type: "topk"               # 简单 TopK
  k1_ratio: 1.0              # TopK1 = K = 1

# Optimizer
trainer:
  optimizer: "adamw"
  lr_agg: 5e-5
  weight_decay_agg: 1e-2
  lr_inst: 0                 # 冻结 instance encoder
  scheduler: "cosine"
  warmup_steps: 200

# Loss
loss:
  type: "focal"
  focal_alpha: 0.4
  focal_gamma: 1.0
  bce_lambda: 0.01
  label_smoothing: true
  smooth_pos: 0.95
  smooth_neg: 0.05

# Monitoring
run:
  monitor: "val_f1"
  monitor_mode: "max"
  patience: 20
```

---

## 💻 执行命令

### 方法 A: 单卡训练（调试用）

```bash
cd /path/to/PAIRFormer
export CUDA_VISIBLE_DEVICES=0

python -m src.launch.train_pair_selected_inst \
    experiment=MTI_train_selected_inst \
    run.kmax=1 \
    selector.type="topk" \
    selector.k1_ratio=1.0 \
    run.batch_size=128 \
    run.num_epochs=100 \
    trainer.lr_agg=5e-5 \
    trainer.lr_inst=0 \
    run.instance_encoder_checkpoint=checkpoints/MTI_v3_xlarge_resume/best.pt \
    run.cheap_encoder_checkpoint=checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/best.pt \
    run.freeze_instance_encoder=true \
    seed=2020 \
    run.output_dir=checkpoints/MTI_K1_baseline
```

### 方法 B: 多卡 DDP 训练（推荐，8×A100）

```bash
cd /path/to/PAIRFormer

# 使用 torchrun 启动 DDP
torchrun --nproc_per_node=8 \
    --master_port=29500 \
    -m src.launch.train_pair_selected_inst \
    experiment=MTI_train_selected_inst \
    run.kmax=1 \
    selector.type="topk" \
    selector.k1_ratio=1.0 \
    run.batch_size=128 \
    run.num_epochs=100 \
    trainer.lr_agg=5e-5 \
    trainer.lr_inst=0 \
    run.instance_encoder_checkpoint=checkpoints/MTI_v3_xlarge_resume/best.pt \
    run.cheap_encoder_checkpoint=checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/best.pt \
    run.freeze_instance_encoder=true \
    seed=2020 \
    run.output_dir=checkpoints/MTI_K1_baseline \
    run.use_ddp=true
```

### 方法 C: 使用脚本（推荐）

**文件**: `scripts/run_mti_k1.sh`

```bash
#!/bin/bash

# MTI K=1 Baseline Training
# Hardware: 8×A100-80GB
# Estimated time: 4-6 hours

set -e

# Configuration
EXPERIMENT="MTI_train_selected_inst"
KMAX=1
BATCH_SIZE=128
NUM_EPOCHS=100
LR_AGG=5e-5
SEED=2020
OUTPUT_DIR="checkpoints/MTI_K1_baseline"
NUM_GPUS=8

# Checkpoints
INSTANCE_CKPT="checkpoints/MTI_v3_xlarge_resume/best.pt"
CHEAP_CKPT="checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/best.pt"

# Check checkpoints exist
if [ ! -f "$INSTANCE_CKPT" ]; then
    echo "Error: Instance encoder checkpoint not found: $INSTANCE_CKPT"
    exit 1
fi

if [ ! -f "$CHEAP_CKPT" ]; then
    echo "Error: Cheap encoder checkpoint not found: $CHEAP_CKPT"
    exit 1
fi

echo "========================================="
echo "MTI K=1 Baseline Training"
echo "========================================="
echo "Hardware: ${NUM_GPUS}×A100-80GB"
echo "K: $KMAX"
echo "Batch size: $BATCH_SIZE per GPU (effective: $((BATCH_SIZE * NUM_GPUS)))"
echo "Epochs: $NUM_EPOCHS"
echo "Learning rate: $LR_AGG"
echo "Seed: $SEED"
echo "Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p logs

# Start training
echo "Starting training at $(date)"

torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    -m src.launch.train_pair_selected_inst \
    experiment=$EXPERIMENT \
    run.kmax=$KMAX \
    selector.type="topk" \
    selector.k1_ratio=1.0 \
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
    2>&1 | tee logs/mti_k1_train.log

echo ""
echo "Training completed at $(date)"
echo "Checkpoint saved to: $OUTPUT_DIR/best.pt"
```

**执行**:
```bash
chmod +x scripts/run_mti_k1.sh
bash scripts/run_mti_k1.sh
```

---

## 📊 训练监控

### 关键指标

**训练过程**:
- Train loss (每 100 steps)
- Val F1, Val PR-AUC (每 epoch)
- Learning rate (每 step)

**预期训练曲线**:
- Epoch 1-20: 快速下降
- Epoch 20-60: 缓慢提升
- Epoch 60-100: 收敛

**预期最佳 epoch**: 70-90

### WandB 监控（如果启用）

```python
# 在配置中启用 wandb
logging:
  use_wandb: true
  wandb_project: "PAIRFormer_MTI"
  wandb_run_name: "MTI_K1_baseline_seed2020"
```

### 训练时间估算

| 阶段 | 时间/epoch | 总时间 |
|------|-----------|--------|
| Warmup (ep 1-10) | ~3 min | ~30 min |
| Main training (ep 11-100) | ~3 min | ~4.5 hr |
| **总计** | | **~5 hr** |

---

## 🧪 评估

### 评估命令

```bash
# 在 test set 上评估
python -m src.launch.eval_pair_selected_inst \
    experiment=MTI_train_selected_inst \
    run.checkpoint=checkpoints/MTI_K1_baseline/best.pt \
    run.kmax=1 \
    run.batch_size=256
```

### 评估脚本

**文件**: `scripts/eval_mti_k1.sh`

```bash
#!/bin/bash

CHECKPOINT="checkpoints/MTI_K1_baseline/best.pt"
EXPERIMENT="MTI_train_selected_inst"
KMAX=1
BATCH_SIZE=256

export CUDA_VISIBLE_DEVICES=0

echo "Evaluating MTI K=1 baseline"
echo "Checkpoint: $CHECKPOINT"
echo ""

python -m src.launch.eval_pair_selected_inst \
    experiment=$EXPERIMENT \
    run.checkpoint=$CHECKPOINT \
    run.kmax=$KMAX \
    run.batch_size=$BATCH_SIZE \
    2>&1 | tee logs/mti_k1_eval.log

echo ""
echo "Evaluation completed"
```

---

## 📊 结果记录

### 训练结果

| Metric | Value | Epoch | 备注 |
|--------|-------|-------|------|
| Best Val F1 | | | |
| Best Val PR-AUC | | | |
| Best Val ROC-AUC | | | |
| Best Val Acc | | | |
| Training time | | | 小时 |

### 测试集结果（填写此表）

| Metric | K=1 (本实验) | K=512 (已有) | Delta | 说明 |
|--------|-------------|-------------|-------|------|
| **Test F1** | | **0.7918** | | 主要指标 |
| Test Acc | | 0.7862 | | |
| Test Prec | | 0.7997 | | |
| Test Rec | | 0.7840 | | |
| Test ROC-AUC | | 0.8253 | | |
| Test PR-AUC | | 0.7884 | | |

### 预期结果

| Metric | K=1 (预期) | K=512 (已有) | Delta (预期) |
|--------|-----------|-------------|-------------|
| Test F1 | 0.74-0.76 | 0.7918 | +3-5pp |
| Test PR-AUC | 0.75-0.77 | 0.7884 | +3-4pp |

---

## 📈 结果分析

### 关键发现

1. **K=1 vs K=512 差距**: 
   - 预期 F1 差距 3-5pp
   - 证明 relational aggregation 的价值

2. **与 miRAW 对比**:
   - miRAW: K=1 (0.925) vs K=64 (0.950), +2.5pp
   - MTI: K=1 (0.75) vs K=512 (0.7918), +4pp
   - **MTI 差距更大**，说明大规模数据下 relational aggregation 更重要

3. **Single best CTS 的局限**:
   - K=1 仅选择最高分 CTS
   - 忽略了其他潜在功能位点
   - 无法捕获 CTS 间的协同/竞争关系

---

## 🎨 可视化

### 绘图数据

**保存为**: `results/mti_k1_results.json`

```json
{
  "k1": {
    "f1": ...,
    "pr_auc": ...,
    "roc_auc": ...,
    "acc": ...,
    "prec": ...,
    "rec": ...
  },
  "k512": {
    "f1": 0.7918,
    "pr_auc": 0.7884,
    "roc_auc": 0.8253,
    "acc": 0.7862,
    "prec": 0.7997,
    "rec": 0.7840
  },
  "delta": {
    "f1": ...,
    "pr_auc": ...
  }
}
```

### 添加到 K sweep 图

将 K=1 结果添加到 Figure 3 (K sweep) 的最左端

---

## ✅ 验收标准

### 必须满足
- [ ] 训练完成 100 epochs
- [ ] 生成 best checkpoint
- [ ] 在 test set 上评估
- [ ] 填写结果表格
- [ ] K=1 F1 显著低于 K=512（差距 >2pp）

### 质量检查
- [ ] 训练曲线正常（无异常波动）
- [ ] Val F1 在 0.74-0.76 范围内
- [ ] Test F1 与 Val F1 接近（差异 <2pp）
- [ ] 训练时间在预期范围内（4-6h）

---

## 📝 提交清单

完成后请提交以下文件：

1. **Checkpoint**: `checkpoints/MTI_K1_baseline/best.pt`
2. **训练日志**: `logs/mti_k1_train.log`
3. **评估日志**: `logs/mti_k1_eval.log`
4. **结果数据**: `results/mti_k1_results.json`
5. **填写的结果表格**（本文档中）
6. **训练曲线图**（如果有 wandb）

---

## 🐛 故障排查

### 问题 1: OOM (Out of Memory)
**解决**: 
- 降低 batch_size（128 → 64）
- 启用 gradient checkpointing

### 问题 2: 训练不收敛
**检查**:
- Instance encoder 是否正确冻结
- Learning rate 是否合适（5e-5）
- Loss 是否正常下降

### 问题 3: DDP 启动失败
**解决**:
- 检查 NCCL 环境变量
- 确认所有 GPU 可见
- 尝试不同的 master_port

### 问题 4: Checkpoint 加载失败
**检查**:
- Checkpoint 路径是否正确
- Checkpoint 是否完整（未损坏）
- 模型架构是否匹配

---

## 📞 联系方式

**问题反馈**: [待填写]  
**紧急联系**: [待填写]

---

**执行人签字**: _______________  
**开始时间**: _______________  
**完成时间**: _______________  
**审核人签字**: _______________
