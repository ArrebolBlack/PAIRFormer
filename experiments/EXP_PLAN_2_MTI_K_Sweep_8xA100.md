# 实验计划 2: MTI K Sweep 完整实验（更新版）

**硬件**: 8×A100-80GB  
**预计时间**: 18-24小时  
**优先级**: P0（最高优先级）  
**负责人**: [待分配]  
**状态**: ⏳ 待执行

---

## 📋 实验概述

**目的**: 在 MTI 大规模数据集上完成完整的 K sweep 实验，用于绘制 Figure 3

**实验内容**:
1. **K=1 baseline**: 证明 relational aggregation 的价值
2. **K=8/16/32 retrain**: 补充小 K 的训练结果  
3. **K truncated**: 使用 K=512 模型 truncate 到不同 K（快速验证）

**关键发现**: MTI 数据集上 K 敏感性更显著，适合作为主图

---

## 🔧 实验 2.1-2.4: K Retrain (K=1,8,16,32)

### 通用配置
- Batch size: 128 per GPU (effective 1024)
- Epochs: 100
- Selector: TopK
- LR: 5e-5 (aggregator only, instance encoder frozen)
- Loss: Focal + Label Smoothing

### 执行命令（K=1 示例）

```bash
torchrun --nproc_per_node=8 --master_port=29500 \
    -m src.launch.train_pair_selected_inst \
    experiment=MTI_train_selected_inst \
    run.kmax=1 \
    selector.type="topk" \
    run.batch_size=128 \
    run.num_epochs=100 \
    trainer.lr_agg=5e-5 \
    trainer.lr_inst=0 \
    run.instance_encoder_checkpoint=checkpoints/MTI_v3_xlarge_resume/best.pt \
    run.cheap_encoder_checkpoint=checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/best.pt \
    run.freeze_instance_encoder=true \
    seed=2020 \
    run.output_dir=checkpoints/MTI_K1_retrain \
    run.use_ddp=true
```

**K=8/16/32**: 修改 `run.kmax` 和 `run.output_dir` 即可

---

## 🔧 实验 2.5: K Truncated (K_max=512)

### 原理
- 使用已训练的 K=512 模型
- 在评估时通过 masking 限制只使用前 K 个 tokens
- 快速评估不同 K 的性能，无需重新训练

### 代码修改

**文件**: `src/models/pair_set_transformer_aggregator.py`

```python
def forward(self, x, attn_mask=None, truncate_k=None):
    B, K, D = x.shape
    
    # Truncate if specified
    if truncate_k is not None and truncate_k < K:
        x = x[:, :truncate_k, :]
        if attn_mask is not None:
            attn_mask = attn_mask[:, :truncate_k]
    
    # Rest of forward pass...
```

### 执行命令

```bash
# 使用 K=512 checkpoint，truncate 到不同 K
for K in 1 8 16 32 64 128 256; do
    python -m src.launch.eval_pair_selected_inst \
        experiment=MTI_train_selected_inst \
        run.checkpoint=checkpoints/MTI_v3_K512_expG/best.pt \
        run.kmax=512 \
        eval.truncate_k=$K \
        run.batch_size=256 \
        2>&1 | tee logs/mti_k512_truncate_k${K}_eval.log
done
```

---

## 📊 完整结果记录表

### Retrain 结果

| K | Test F1 | Test PR-AUC | Test ROC-AUC | Training Time | Best Epoch |
|---|---------|-------------|--------------|---------------|------------|
| 1 | | | | | |
| 8 | | | | | |
| 16 | | | | | |
| 32 | | | | | |
| 64 | 0.7708 | 0.8295 | 0.8201 | - | - |
| 128 | 0.7786 | 0.8426 | 0.8332 | - | - |
| 256 | 0.7790 | 0.8587 | 0.8393 | - | - |
| 512 | 0.7918 | 0.7884 | 0.8253 | - | - |

### Truncated 结果（K_max=512）

| K (truncated) | Test F1 | Test PR-AUC | Test ROC-AUC |
|--------------|---------|-------------|--------------|
| 1 | | | |
| 8 | | | |
| 16 | | | |
| 32 | | | |
| 64 | | | |
| 128 | | | |
| 256 | | | |
| 512 | 0.7918 | 0.7884 | 0.8253 |

### 预期结果

| K | Retrain F1 (预期) | Truncated F1 (预期) | Delta |
|---|------------------|-------------------|-------|
| 1 | 0.74-0.76 | 0.73-0.75 | ~0.01 |
| 8 | 0.76-0.77 | 0.75-0.76 | ~0.01 |
| 16 | 0.77-0.78 | 0.76-0.77 | ~0.01 |
| 32 | 0.78-0.79 | 0.77-0.78 | ~0.01 |

---

## 🚀 完整执行脚本

**文件**: `scripts/run_mti_k_sweep_complete.sh`

```bash
#!/bin/bash
set -e

NUM_GPUS=8
INSTANCE_CKPT="checkpoints/MTI_v3_xlarge_resume/best.pt"
CHEAP_CKPT="checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/best.pt"

echo "MTI Complete K Sweep Experiment"
echo "Start time: $(date)"

# Part 1: Retrain K=1,8,16,32
for K in 1 8 16 32; do
    echo "Training K=$K (retrain)"
    
    torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
        -m src.launch.train_pair_selected_inst \
        experiment=MTI_train_selected_inst \
        run.kmax=$K \
        selector.type="topk" \
        run.batch_size=128 \
        run.num_epochs=100 \
        trainer.lr_agg=5e-5 \
        trainer.lr_inst=0 \
        run.instance_encoder_checkpoint=$INSTANCE_CKPT \
        run.cheap_encoder_checkpoint=$CHEAP_CKPT \
        run.freeze_instance_encoder=true \
        seed=2020 \
        run.output_dir=checkpoints/MTI_K${K}_retrain \
        run.use_ddp=true \
        2>&1 | tee logs/mti_k${K}_retrain_train.log
    
    python -m src.launch.eval_pair_selected_inst \
        experiment=MTI_train_selected_inst \
        run.checkpoint=checkpoints/MTI_K${K}_retrain/best.pt \
        run.kmax=$K \
        run.batch_size=256 \
        2>&1 | tee logs/mti_k${K}_retrain_eval.log
done

# Part 2: K Truncated
CHECKPOINT="checkpoints/MTI_v3_K512_expG/best.pt"
for K in 1 8 16 32 64 128 256; do
    python -m src.launch.eval_pair_selected_inst \
        experiment=MTI_train_selected_inst \
        run.checkpoint=$CHECKPOINT \
        run.kmax=512 \
        eval.truncate_k=$K \
        run.batch_size=256 \
        2>&1 | tee logs/mti_k512_truncate_k${K}_eval.log
done

echo "All experiments completed! End time: $(date)"
```

---

## ⏱️ 时间估算

| 实验 | 时间 |
|------|------|
| K=1,8,16,32 retrain | 16-24h |
| K truncated | 2-3h |
| **总计** | **18-27h** |

---

## ✅ 验收标准

- [ ] K=1,8,16,32 retrain 完成
- [ ] K truncated 评估完成
- [ ] 结果表格填写完整
- [ ] F1 随 K 单调递增
- [ ] Retrain ≥ Truncated（同一 K）

---

**执行人签字**: _______________  
**完成时间**: _______________
