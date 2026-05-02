# 实验计划 1: K Truncated 实验（更新版）

**硬件**: 2×A100-80GB  
**预计时间**: 3-4小时  
**优先级**: P1（推荐做）  
**负责人**: [待分配]  
**状态**: ⏳ 待执行

---

## 📋 实验概述

**目的**: 使用已有的 K=64/128/256/512 checkpoints 进行 K truncated 实验

**实验性质**: ⚠️ **仅评估，无需训练**

**关键优势**:
- 该主机已有 4 个 K checkpoints
- 可快速完成多个 K 的 truncated 评估
- 与 8×A100 的 retrain 结果对比

---

## 🔧 实验 1.1: K Truncated from K=64

### 已有 Checkpoint
- **路径**: `checkpoints/MTI_K64_baseline/best.pt`（或类似路径）
- **K_max**: 64

### 执行命令

```bash
export CUDA_VISIBLE_DEVICES=0,1

# Truncate K=64 to smaller K
for K in 1 8 16 32; do
    python -m src.launch.eval_pair_selected_inst \
        experiment=MTI_train_selected_inst \
        run.checkpoint=checkpoints/MTI_K64_baseline/best.pt \
        run.kmax=64 \
        eval.truncate_k=$K \
        run.batch_size=256 \
        2>&1 | tee logs/mti_k64_truncate_k${K}_eval.log
done

# Baseline (no truncation)
python -m src.launch.eval_pair_selected_inst \
    experiment=MTI_train_selected_inst \
    run.checkpoint=checkpoints/MTI_K64_baseline/best.pt \
    run.kmax=64 \
    run.batch_size=256 \
    2>&1 | tee logs/mti_k64_baseline_eval.log
```

---

## 🔧 实验 1.2: K Truncated from K=128

### 已有 Checkpoint
- **路径**: `checkpoints/MTI_K128_baseline/best.pt`
- **K_max**: 128

### 执行命令

```bash
# Truncate K=128 to smaller K
for K in 1 8 16 32 64; do
    python -m src.launch.eval_pair_selected_inst \
        experiment=MTI_train_selected_inst \
        run.checkpoint=checkpoints/MTI_K128_baseline/best.pt \
        run.kmax=128 \
        eval.truncate_k=$K \
        run.batch_size=256 \
        2>&1 | tee logs/mti_k128_truncate_k${K}_eval.log
done

# Baseline
python -m src.launch.eval_pair_selected_inst \
    experiment=MTI_train_selected_inst \
    run.checkpoint=checkpoints/MTI_K128_baseline/best.pt \
    run.kmax=128 \
    run.batch_size=256 \
    2>&1 | tee logs/mti_k128_baseline_eval.log
```

---

## 🔧 实验 1.3: K Truncated from K=256

### 已有 Checkpoint
- **路径**: `checkpoints/MTI_K256_baseline/best.pt`
- **K_max**: 256

### 执行命令

```bash
# Truncate K=256 to smaller K
for K in 1 8 16 32 64 128; do
    python -m src.launch.eval_pair_selected_inst \
        experiment=MTI_train_selected_inst \
        run.checkpoint=checkpoints/MTI_K256_baseline/best.pt \
        run.kmax=256 \
        eval.truncate_k=$K \
        run.batch_size=256 \
        2>&1 | tee logs/mti_k256_truncate_k${K}_eval.log
done

# Baseline
python -m src.launch.eval_pair_selected_inst \
    experiment=MTI_train_selected_inst \
    run.checkpoint=checkpoints/MTI_K256_baseline/best.pt \
    run.kmax=256 \
    run.batch_size=256 \
    2>&1 | tee logs/mti_k256_baseline_eval.log
```

---

## 🔧 实验 1.4: K Truncated from K=512

### 已有 Checkpoint
- **路径**: `checkpoints/MTI_v3_K512_expG/best.pt`
- **K_max**: 512

### 执行命令

```bash
# Truncate K=512 to smaller K
for K in 1 8 16 32 64 128 256; do
    python -m src.launch.eval_pair_selected_inst \
        experiment=MTI_train_selected_inst \
        run.checkpoint=checkpoints/MTI_v3_K512_expG/best.pt \
        run.kmax=512 \
        eval.truncate_k=$K \
        run.batch_size=256 \
        2>&1 | tee logs/mti_k512_truncate_k${K}_eval.log
done

# Baseline
python -m src.launch.eval_pair_selected_inst \
    experiment=MTI_train_selected_inst \
    run.checkpoint=checkpoints/MTI_v3_K512_expG/best.pt \
    run.kmax=512 \
    run.batch_size=256 \
    2>&1 | tee logs/mti_k512_baseline_eval.log
```

---

## 📊 结果记录表

### K=64 Truncated

| K (truncated) | Test F1 | Test PR-AUC | Test ROC-AUC |
|--------------|---------|-------------|--------------|
| 1 | | | |
| 8 | | | |
| 16 | | | |
| 32 | | | |
| 64 | 0.7708 | 0.8295 | 0.8201 |

### K=128 Truncated

| K (truncated) | Test F1 | Test PR-AUC | Test ROC-AUC |
|--------------|---------|-------------|--------------|
| 1 | | | |
| 8 | | | |
| 16 | | | |
| 32 | | | |
| 64 | | | |
| 128 | 0.7786 | 0.8426 | 0.8332 |

### K=256 Truncated

| K (truncated) | Test F1 | Test PR-AUC | Test ROC-AUC |
|--------------|---------|-------------|--------------|
| 1 | | | |
| 8 | | | |
| 16 | | | |
| 32 | | | |
| 64 | | | |
| 128 | | | |
| 256 | 0.7790 | 0.8587 | 0.8393 |

### K=512 Truncated

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

---

## 🚀 完整执行脚本

**文件**: `scripts/run_mti_k_truncated_all.sh`

```bash
#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0,1

echo "MTI K Truncated Experiment (All Checkpoints)"
echo "Start time: $(date)"

# K=64 truncated
echo "=== K=64 Truncated ==="
for K in 1 8 16 32; do
    python -m src.launch.eval_pair_selected_inst \
        experiment=MTI_train_selected_inst \
        run.checkpoint=checkpoints/MTI_K64_baseline/best.pt \
        run.kmax=64 \
        eval.truncate_k=$K \
        run.batch_size=256 \
        2>&1 | tee logs/mti_k64_truncate_k${K}_eval.log
done

# K=128 truncated
echo "=== K=128 Truncated ==="
for K in 1 8 16 32 64; do
    python -m src.launch.eval_pair_selected_inst \
        experiment=MTI_train_selected_inst \
        run.checkpoint=checkpoints/MTI_K128_baseline/best.pt \
        run.kmax=128 \
        eval.truncate_k=$K \
        run.batch_size=256 \
        2>&1 | tee logs/mti_k128_truncate_k${K}_eval.log
done

# K=256 truncated
echo "=== K=256 Truncated ==="
for K in 1 8 16 32 64 128; do
    python -m src.launch.eval_pair_selected_inst \
        experiment=MTI_train_selected_inst \
        run.checkpoint=checkpoints/MTI_K256_baseline/best.pt \
        run.kmax=256 \
        eval.truncate_k=$K \
        run.batch_size=256 \
        2>&1 | tee logs/mti_k256_truncate_k${K}_eval.log
done

# K=512 truncated
echo "=== K=512 Truncated ==="
for K in 1 8 16 32 64 128 256; do
    python -m src.launch.eval_pair_selected_inst \
        experiment=MTI_train_selected_inst \
        run.checkpoint=checkpoints/MTI_v3_K512_expG/best.pt \
        run.kmax=512 \
        eval.truncate_k=$K \
        run.batch_size=256 \
        2>&1 | tee logs/mti_k512_truncate_k${K}_eval.log
done

echo "All truncated evaluations completed!"
echo "End time: $(date)"
```

---

## ⏱️ 时间估算

| 实验 | 评估次数 | 时间/次 | 总时间 |
|------|---------|--------|--------|
| K=64 truncated | 4 | 15min | 1h |
| K=128 truncated | 5 | 15min | 1.25h |
| K=256 truncated | 6 | 15min | 1.5h |
| K=512 truncated | 7 | 15min | 1.75h |
| **总计** | **22** | | **~6h** |

**实际可能更快**（并行或优化后）

---

## 📈 结果分析

### 关键对比

1. **不同 K_max 的 truncated 结果对比**:
   - K=64 truncate to K=32 vs K=128 truncate to K=32
   - 预期：K_max 越大，truncated 性能越好

2. **Truncated vs Retrain**:
   - 对比 8×A100 的 retrain 结果
   - 预期：Retrain 略优于 Truncated

3. **最佳 truncated 来源**:
   - 哪个 K_max 的 truncated 结果最接近 retrain？
   - 预期：K=512 truncated 最接近

---

## ✅ 验收标准

- [ ] 所有 4 个 K_max 的 truncated 评估完成
- [ ] 结果表格填写完整
- [ ] 性能随 K 单调递增
- [ ] K_max 越大，truncated 性能越好

---

## 📝 提交清单

1. **日志文件**: `logs/mti_k*_truncate_k*_eval.log`
2. **结果数据**: `results/mti_k_truncated_all.json`
3. **填写的结果表格**（本文档中）

---

**执行人签字**: _______________  
**完成时间**: _______________
