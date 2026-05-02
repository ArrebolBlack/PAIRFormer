# 实验计划 3: K=1 Baseline on miRAW & deepTargetPro

**硬件**: 1×RTX 5090  
**预计时间**: 9-13小时（两个数据集）  
**优先级**: P0（最高优先级）  
**负责人**: [待分配]  
**状态**: ⏳ 待执行

---

## 📋 实验概述

**目的**: 在 miRAW 和 deepTargetPro 上验证 K=1 baseline

**实验性质**: ✅ **需要训练模型**（两个数据集）

**关键对比**:
- miRAW: K=1 vs K=64
- deepTargetPro: K=1 vs K=64 (transfer learning)

**Rebuttal 承诺**:
> "A retrained K=1 model reaches F1@0.5=0.925, while K=64 reaches 0.950"

---

## 🔧 实验 3.1: miRAW K=1

### 数据集信息
- **数据集**: miRAW half-split
- **Test size**: 5,480 pairs (10 subsets, 使用 0,6-9 作为 test)
- **正负比**: balanced

### 模型配置
- **Instance Encoder**: TargetNet_Optimized (标准配置，emb_dim=384)
  - Checkpoint: `checkpoints/miRAW_instance/best.pt`
- **Cheap Encoder**: CheapCTSNet
  - Checkpoint: `checkpoints/miRAW_cheap/best.pt`
- **Aggregator**: PairSetTransformerAggregator (d_model=1024, n_layers=4)
- **Token dim**: 387 (384 emb + 1 logit + 1 esa + 1 pos)

### 训练配置
```yaml
run:
  kmax: 1
  batch_size: 64              # 单卡适配
  num_epochs: 100
  
selector:
  type: "topk"
  k1_ratio: 1.0

trainer:
  optimizer: "adamw"
  lr_agg: 5e-5
  weight_decay_agg: 1e-2
  lr_inst: 0                  # 冻结
  scheduler: "cosine"
  warmup_steps: 100

loss:
  type: "focal"
  focal_alpha: 0.4
  focal_gamma: 1.0
  bce_lambda: 0.01
  label_smoothing: true
```

### 执行命令

```bash
cd /path/to/PAIRFormer
export CUDA_VISIBLE_DEVICES=0

# miRAW K=1 训练
python -m src.launch.train_pair_selected_inst \
    experiment=miRAW_train_selected_inst \
    run.kmax=1 \
    selector.type="topk" \
    selector.k1_ratio=1.0 \
    run.batch_size=64 \
    run.num_epochs=100 \
    trainer.lr_agg=5e-5 \
    trainer.lr_inst=0 \
    run.instance_encoder_checkpoint=checkpoints/miRAW_instance/best.pt \
    run.cheap_encoder_checkpoint=checkpoints/miRAW_cheap/best.pt \
    run.freeze_instance_encoder=true \
    seed=2020 \
    run.output_dir=checkpoints/miRAW_K1_baseline \
    2>&1 | tee logs/miraw_k1_train.log
```

### 评估命令

```bash
# 在 test set 上评估
python -m src.launch.eval_pair_selected_inst \
    experiment=miRAW_train_selected_inst \
    run.checkpoint=checkpoints/miRAW_K1_baseline/best.pt \
    run.kmax=1 \
    run.batch_size=128 \
    2>&1 | tee logs/miraw_k1_eval.log
```

### 预期结果

| Metric | K=1 (预期) | K=64 (已有) | Delta |
|--------|-----------|-------------|-------|
| Test F1 | 0.925±0.003 | 0.950±0.002 | +2.5pp |
| Test PR-AUC | 0.985±0.002 | 0.989±0.001 | +0.4pp |

### 训练时间估算
- **Time/epoch**: ~2 min (单卡 5090)
- **Total epochs**: 100
- **Total time**: ~3.5 hours

---

## 🔧 实验 3.2: deepTargetPro K=1 (Transfer Learning)

### 数据集信息
- **数据集**: deepTargetPro
- **Test size**: ~4,023 pairs (10-fold CV, 使用 splits 0,6-9)
- **正负比**: balanced

### 模型配置（Transfer Learning）
- **Instance Encoder**: 使用 miRAW 预训练的 encoder
  - Checkpoint: `checkpoints/miRAW_instance/best.pt`
- **Cheap Encoder**: 使用 miRAW 预训练的 encoder
  - Checkpoint: `checkpoints/miRAW_cheap/best.pt`
- **Aggregator**: PairSetTransformerAggregator (新训练)
- **Token dim**: 387

### 训练配置
```yaml
run:
  kmax: 1
  batch_size: 64
  num_epochs: 50              # Transfer learning 更快收敛
  
selector:
  type: "topk"
  k1_ratio: 1.0

trainer:
  optimizer: "adamw"
  lr_agg: 3e-5                # Transfer learning 用更小 LR
  weight_decay_agg: 1e-2
  lr_inst: 0                  # 冻结
  scheduler: "cosine"
  warmup_steps: 50

loss:
  type: "focal"
  focal_alpha: 0.4
  focal_gamma: 1.0
  bce_lambda: 0.01
```

### 执行命令

```bash
cd /path/to/PAIRFormer
export CUDA_VISIBLE_DEVICES=0

# deepTargetPro K=1 训练 (transfer learning)
python -m src.launch.train_pair_selected_inst \
    experiment=deepTargetPro_train_selected_inst \
    run.kmax=1 \
    selector.type="topk" \
    selector.k1_ratio=1.0 \
    run.batch_size=64 \
    run.num_epochs=50 \
    trainer.lr_agg=3e-5 \
    trainer.lr_inst=0 \
    run.instance_encoder_checkpoint=checkpoints/miRAW_instance/best.pt \
    run.cheap_encoder_checkpoint=checkpoints/miRAW_cheap/best.pt \
    run.freeze_instance_encoder=true \
    seed=2020 \
    run.output_dir=checkpoints/deepTargetPro_K1_transfer \
    2>&1 | tee logs/deeptargetpro_k1_train.log
```

### 评估命令

```bash
# 在 test set 上评估
python -m src.launch.eval_pair_selected_inst \
    experiment=deepTargetPro_train_selected_inst \
    run.checkpoint=checkpoints/deepTargetPro_K1_transfer/best.pt \
    run.kmax=1 \
    run.batch_size=128 \
    2>&1 | tee logs/deeptargetpro_k1_eval.log
```

### 预期结果

| Metric | K=1 (预期) | K=64 (已有) | Delta |
|--------|-----------|-------------|-------|
| Test F1 | 0.80-0.82 | 0.8521±0.58 | +3-5pp |
| Test PR-AUC | - | - | - |

### 训练时间估算
- **Time/epoch**: ~3 min (单卡 5090)
- **Total epochs**: 50
- **Total time**: ~2.5 hours

---

## 📊 完整执行脚本

**文件**: `scripts/run_k1_miraw_deeptargetpro.sh`

```bash
#!/bin/bash

# K=1 Baseline Training on miRAW and deepTargetPro
# Hardware: 1×RTX 5090
# Estimated time: 9-13 hours

set -e

export CUDA_VISIBLE_DEVICES=0

# Create directories
mkdir -p checkpoints logs results

echo "========================================="
echo "K=1 Baseline Training"
echo "========================================="
echo "Hardware: 1×RTX 5090"
echo "Datasets: miRAW + deepTargetPro"
echo "Start time: $(date)"
echo ""

# ==========================================
# Part 1: miRAW K=1
# ==========================================
echo "========================================="
echo "Part 1: miRAW K=1 Baseline"
echo "========================================="
echo "Expected time: ~3.5 hours"
echo ""

python -m src.launch.train_pair_selected_inst \
    experiment=miRAW_train_selected_inst \
    run.kmax=1 \
    selector.type="topk" \
    selector.k1_ratio=1.0 \
    run.batch_size=64 \
    run.num_epochs=100 \
    trainer.lr_agg=5e-5 \
    trainer.lr_inst=0 \
    run.instance_encoder_checkpoint=checkpoints/miRAW_instance/best.pt \
    run.cheap_encoder_checkpoint=checkpoints/miRAW_cheap/best.pt \
    run.freeze_instance_encoder=true \
    seed=2020 \
    run.output_dir=checkpoints/miRAW_K1_baseline \
    2>&1 | tee logs/miraw_k1_train.log

echo ""
echo "miRAW K=1 training completed at $(date)"
echo "Checkpoint: checkpoints/miRAW_K1_baseline/best.pt"
echo ""

# Evaluate miRAW K=1
echo "Evaluating miRAW K=1..."
python -m src.launch.eval_pair_selected_inst \
    experiment=miRAW_train_selected_inst \
    run.checkpoint=checkpoints/miRAW_K1_baseline/best.pt \
    run.kmax=1 \
    run.batch_size=128 \
    2>&1 | tee logs/miraw_k1_eval.log

echo "miRAW K=1 evaluation completed"
echo ""

# ==========================================
# Part 2: deepTargetPro K=1 (Transfer)
# ==========================================
echo "========================================="
echo "Part 2: deepTargetPro K=1 (Transfer)"
echo "========================================="
echo "Expected time: ~2.5 hours"
echo ""

python -m src.launch.train_pair_selected_inst \
    experiment=deepTargetPro_train_selected_inst \
    run.kmax=1 \
    selector.type="topk" \
    selector.k1_ratio=1.0 \
    run.batch_size=64 \
    run.num_epochs=50 \
    trainer.lr_agg=3e-5 \
    trainer.lr_inst=0 \
    run.instance_encoder_checkpoint=checkpoints/miRAW_instance/best.pt \
    run.cheap_encoder_checkpoint=checkpoints/miRAW_cheap/best.pt \
    run.freeze_instance_encoder=true \
    seed=2020 \
    run.output_dir=checkpoints/deepTargetPro_K1_transfer \
    2>&1 | tee logs/deeptargetpro_k1_train.log

echo ""
echo "deepTargetPro K=1 training completed at $(date)"
echo "Checkpoint: checkpoints/deepTargetPro_K1_transfer/best.pt"
echo ""

# Evaluate deepTargetPro K=1
echo "Evaluating deepTargetPro K=1..."
python -m src.launch.eval_pair_selected_inst \
    experiment=deepTargetPro_train_selected_inst \
    run.checkpoint=checkpoints/deepTargetPro_K1_transfer/best.pt \
    run.kmax=1 \
    run.batch_size=128 \
    2>&1 | tee logs/deeptargetpro_k1_eval.log

echo "deepTargetPro K=1 evaluation completed"
echo ""

# ==========================================
# Summary
# ==========================================
echo "========================================="
echo "All K=1 experiments completed!"
echo "========================================="
echo "End time: $(date)"
echo ""
echo "Checkpoints:"
echo "  - miRAW K=1: checkpoints/miRAW_K1_baseline/best.pt"
echo "  - deepTargetPro K=1: checkpoints/deepTargetPro_K1_transfer/best.pt"
echo ""
echo "Logs:"
echo "  - logs/miraw_k1_train.log"
echo "  - logs/miraw_k1_eval.log"
echo "  - logs/deeptargetpro_k1_train.log"
echo "  - logs/deeptargetpro_k1_eval.log"
echo ""
echo "Please fill in the results tables in the experiment plan document."
```

**执行**:
```bash
chmod +x scripts/run_k1_miraw_deeptargetpro.sh
nohup bash scripts/run_k1_miraw_deeptargetpro.sh > logs/k1_all.log 2>&1 &
```

---

## 📊 结果记录

### miRAW K=1 结果

| Metric | K=1 (填写) | K=64 (已有) | Delta | 备注 |
|--------|-----------|-------------|-------|------|
| **Test F1** | | **0.950±0.002** | | Rebuttal 目标: 0.925 |
| Test PR-AUC | | 0.989±0.001 | | |
| Test ROC-AUC | | - | | |
| Test Acc | | - | | |
| Test Prec | | - | | |
| Test Rec | | - | | |
| Training time | | - | | 小时 |
| Best epoch | | - | | |

### deepTargetPro K=1 结果

| Metric | K=1 (填写) | K=64 (已有) | Delta | 备注 |
|--------|-----------|-------------|-------|------|
| **Test F1** | | **0.8521±0.58** | | Transfer learning |
| Test PR-AUC | | - | | |
| Test ROC-AUC | | - | | |
| Test Acc | | - | | |
| Test Prec | | - | | |
| Test Rec | | - | | |
| Training time | | - | | 小时 |
| Best epoch | | - | | |

---

## 📈 结果分析

### 关键对比

| 数据集 | K=1 F1 | K=64/512 F1 | Delta | 说明 |
|--------|--------|-------------|-------|------|
| miRAW | (填写) | 0.950 | | Rebuttal 承诺 +2.5pp |
| MTI | (填写) | 0.7918 | | 预期 +3-5pp |
| deepTargetPro | (填写) | 0.8521 | | Transfer learning |

### 预期发现

1. **miRAW**: K=1 vs K=64 差距 ~2.5pp，验证 Rebuttal 承诺
2. **MTI**: K=1 vs K=512 差距更大（~4pp），大规模数据下 relational aggregation 更重要
3. **deepTargetPro**: Transfer learning 下 K=1 vs K=64 差距 ~3-5pp

---

## ✅ 验收标准

### 必须满足
- [ ] miRAW K=1 训练完成
- [ ] deepTargetPro K=1 训练完成
- [ ] 两个数据集都在 test set 上评估
- [ ] 填写结果表格
- [ ] miRAW K=1 F1 ≈ 0.925（±0.01）
- [ ] 所有 K=1 F1 显著低于对应的 K=64/512

### 质量检查
- [ ] 训练曲线正常
- [ ] Val F1 与 Test F1 接近
- [ ] 训练时间在预期范围内

---

## 📝 提交清单

完成后请提交以下文件：

1. **Checkpoints**:
   - `checkpoints/miRAW_K1_baseline/best.pt`
   - `checkpoints/deepTargetPro_K1_transfer/best.pt`

2. **日志文件**:
   - `logs/miraw_k1_train.log`
   - `logs/miraw_k1_eval.log`
   - `logs/deeptargetpro_k1_train.log`
   - `logs/deeptargetpro_k1_eval.log`

3. **结果数据**:
   - `results/miraw_k1_results.json`
   - `results/deeptargetpro_k1_results.json`

4. **填写的结果表格**（本文档中）

---

## 🐛 故障排查

### 问题 1: OOM on 5090
**解决**: 降低 batch_size（64 → 32）

### 问题 2: Checkpoint 不存在
**检查**: 
- miRAW Stage 1-2 encoders 是否已训练
- 路径是否正确

### 问题 3: 训练太慢
**优化**:
- 启用 AMP (mixed precision)
- 增加 num_workers

---

**执行人签字**: _______________  
**开始时间**: _______________  
**完成时间**: _______________  
**审核人签字**: _______________
