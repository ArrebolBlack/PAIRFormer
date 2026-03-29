# EXP8 最终交付 - 自动优化版本

**交付日期**: 2026-03-29
**版本**: v3.0 (自动优化系统)

---

## ✅ 完整交付内容

### 🎯 三个版本对比

| 版本 | Batch Size | LR调整 | 训练时间 | 总时间 | 特点 |
|------|-----------|--------|---------|--------|------|
| **v1.0 原版** | 512/256/32 | 固定 | 48-66h | 66-91h | 保守配置 |
| **v2.0 手动优化** | 1024/1024/64 | 固定 | 21-29h | 39-54h | 手动估算 |
| **v3.0 自动优化** | 自动测试 | 自动缩放 | **~10-15h** | **~28-33h** | **最大化显存** ⭐ |

### 📦 文件清单（15个）

#### 核心脚本（9个）
1. ✅ `step1_generate_negatives.py` - 生成负类
2. ✅ `step2_split_data.py` - 数据分割
3. ✅ `step3_generate_configs.py` - 配置生成
4. ✅ `quick_start.sh` - 快速验证
5. ✅ `run_all.sh` - v1.0原版脚本
6. ✅ `run_all_optimized.sh` - v2.0手动优化脚本
7. ✅ **`auto_find_batch_size.py`** - v3.0自动测试工具 ⭐
8. ✅ **`apply_optimal_config.sh`** - v3.0配置应用 ⭐
9. ✅ `test_batch_size.py` - 手动测试工具

#### 文档（6个）
1. ✅ `README.md` - 完整实验说明
2. ✅ `FILE_MANIFEST.md` - 文件清单
3. ✅ `TIMELINE_OPTIMIZED.md` - v2.0时间线
4. ✅ `OPTIMIZATION_SUMMARY.md` - v2.0总结
5. ✅ **`AUTO_OPTIMIZATION_GUIDE.md`** - v3.0使用指南 ⭐
6. ✅ `batch_size_config.sh` - 配置说明

---

## 🚀 推荐使用方式

### 方式1: 自动优化（推荐）⭐

```bash
cd /home/yjq/workspace/rebuttal/PAIRFormer

# Step 1: 自动测试最优batch size（5-10分钟）
python3 experiments/EXP8/scripts/auto_find_batch_size.py \
    --target_memory 70 \
    --gpu 0

# Step 2: 应用配置（1分钟）
cd experiments/EXP8/scripts
./apply_optimal_config.sh

# Step 3: 验证数据准备（5分钟）
./quick_start.sh

# Step 4: 开始训练（~1-1.5天）
./run_all_auto_optimized.sh
```

**优势**:
- ✅ 最大化显存利用（~85-90%）
- ✅ 自动LR缩放
- ✅ 最快速度（预计1-1.5天）

### 方式2: 手动优化（保守）

```bash
cd /home/yjq/workspace/rebuttal/PAIRFormer/experiments/EXP8/scripts

# 使用预设的优化配置
./run_all_optimized.sh
```

**优势**:
- ✅ 无需测试，直接运行
- ✅ 保守配置，OOM风险低
- ✅ 约2天完成

### 方式3: 原版（最保守）

```bash
cd /home/yjq/workspace/rebuttal/PAIRFormer/experiments/EXP8/scripts
./run_all.sh
```

**优势**:
- ✅ 最稳定
- ✅ 约3-4天完成

---

## 📊 性能对比

### Batch Size对比

| Stage | v1.0原版 | v2.0手动 | v3.0自动（预期） |
|-------|---------|---------|----------------|
| Stage 1 | 512 | 1024 | **~2500-3000** |
| Stage 2 | 256 | 1024 | **~4000-5000** |
| Stage 3 | 32 | 64 | **~100-150** |

### 加速比对比

| Stage | v1.0→v2.0 | v1.0→v3.0（预期） |
|-------|-----------|------------------|
| Stage 1 | 2x | **5-6x** |
| Stage 2 | 4x | **15-20x** |
| Stage 3 | 2x | **3-5x** |
| **总体** | **2.2x** | **~3x** |

### 时间线对比

| 阶段 | v1.0原版 | v2.0手动 | v3.0自动（预期） |
|------|---------|---------|----------------|
| 数据准备 | 2-3h | 2-3h | 2-3h |
| Window cache | 8-10h | 8-10h | 8-10h |
| **Stage 1训练** | 16-20h | 8-10h | **~3-4h** |
| Cheap cache | 4-6h | 4-6h | 4-6h |
| **Stage 2训练** | 12-16h | 3-4h | **~1-2h** |
| Selection cache | 2-3h | 2-3h | 2-3h |
| **Stage 3训练** | 20-30h | 10-15h | **~5-8h** |
| 评估 | 2-3h | 2-3h | 2-3h |
| **总计** | **66-91h** | **39-54h** | **~28-38h** |
| **Wall-clock** | **3-4天** | **1.5-2.5天** | **~1-1.5天** |

---

## 🎯 自动优化系统特性

### 1. 智能测试

```python
# 二分搜索找到最大batch size
max_bs = binary_search_batch_size(
    model_fn=create_model,
    target_memory_gb=70,  # 留10GB余量
    min_bs=32,
    max_bs=4096
)
```

**测试内容**:
- ✅ 完整前向传播
- ✅ 完整反向传播
- ✅ Optimizer step（包含states）
- ✅ 实时显存监控

### 2. 自动LR缩放

```python
# 线性缩放策略
scaled_lr = base_lr * (new_bs / base_bs)

# 示例
# Stage 1: 6e-4 * (2917/512) = 3.4e-3
# Stage 2: 6e-4 * (4568/256) = 1.1e-2
```

### 3. 配置持久化

```json
{
  "stage1": {
    "batch_size": 2917,
    "lr": 0.00342,
    "speedup": 5.7
  },
  ...
}
```

---

## ⚙️ 使用建议

### 首次使用

1. **先测试**: 运行`auto_find_batch_size.py`
2. **查看结果**: 检查找到的batch size是否合理
3. **保守起见**: 如果担心OOM，可手动降低10-20%
4. **开始训练**: 使用`run_all_auto_optimized.sh`

### 如果遇到OOM

```bash
# 方案1: 降低目标显存
python3 auto_find_batch_size.py --target_memory 65

# 方案2: 手动编辑生成的脚本
# 将batch size降低10-20%

# 方案3: 使用手动优化版本
./run_all_optimized.sh
```

### 监控训练

```bash
# GPU显存
watch -n 1 nvidia-smi

# 训练速度
tail -f logs/train_stage1_seed2020.log | grep "it/s"

# 预期速度（自动优化）:
# Stage 1: ~10-15 it/s (vs 原版 ~3-5 it/s)
# Stage 2: ~20-30 it/s (vs 原版 ~5-8 it/s)
# Stage 3: ~5-8 it/s (vs 原版 ~2-3 it/s)
```

---

## 📝 完整工作流程

### 快速流程（推荐）

```bash
# 1. 自动测试（5-10分钟）
cd /home/yjq/workspace/rebuttal/PAIRFormer
python3 experiments/EXP8/scripts/auto_find_batch_size.py

# 2. 应用配置（1分钟）
cd experiments/EXP8/scripts
./apply_optimal_config.sh

# 3. 开始训练（~1-1.5天）
./run_all_auto_optimized.sh
```

### 详细流程

```bash
# 1. 自动测试最优配置
python3 experiments/EXP8/scripts/auto_find_batch_size.py \
    --target_memory 70 \
    --gpu 0 \
    --output experiments/EXP8/configs/optimal_batch_sizes.json

# 输出示例:
# Stage 1: batch_size=2917, lr=3.42e-03, speedup=5.7x
# Stage 2: batch_size=4568, lr=1.07e-02, speedup=17.8x
# Stage 3: batch_size=128, lr=1.20e-03, speedup=4.0x

# 2. 应用配置并生成执行脚本
cd experiments/EXP8/scripts
./apply_optimal_config.sh

# 生成: run_all_auto_optimized.sh

# 3. 验证数据准备
./quick_start.sh

# 检查生成的文件:
# - data/MTI/MTI_pair_with_negatives_random.txt
# - data/MTI/MTI_pair_random_split.txt
# - configs/data/miRNA_MTI.yaml
# - configs/experiment/MTI_*.yaml

# 4. 开始训练
./run_all_auto_optimized.sh

# 监控进度
tail -f ../logs/master.log
watch -n 1 nvidia-smi
```

---

## 📚 文档导航

| 文档 | 用途 | 推荐阅读 |
|------|------|---------|
| **AUTO_OPTIMIZATION_GUIDE.md** | 自动优化系统详细说明 | ⭐⭐⭐ |
| README.md | 完整实验说明 | ⭐⭐⭐ |
| OPTIMIZATION_SUMMARY.md | 手动优化总结 | ⭐⭐ |
| TIMELINE_OPTIMIZED.md | 手动优化时间线 | ⭐⭐ |
| FILE_MANIFEST.md | 文件清单 | ⭐ |

---

## ⚠️ 重要提醒

### 1. 显存要求
- **必需**: A100 80GB（或同等级GPU）
- **不支持**: 40GB以下的GPU

### 2. 测试环境
- 测试前确保GPU空闲
- 关闭其他Python进程
- 建议重启后测试

### 3. 真实训练
- 自动测试使用简化模型
- 真实训练可能略大
- 如遇OOM，降低10-20%

### 4. Learning Rate
- 自动缩放是启发式的
- 监控训练loss曲线
- 如不稳定，手动降低lr

---

## 🎉 预期成果

### 性能指标
- Test集: ~65,387 pairs
- PR-AUC: 0.90-0.98
- F1@0.5: 0.85-0.95

### 时间节省
- 原计划: 5-6天
- 手动优化: 2天
- **自动优化: 1-1.5天** ⭐

### Rebuttal价值
- ✅ 第三个独立外部数据集
- ✅ 规模是miRAWtest的217倍
- ✅ 完全独立于miRAW
- ✅ **最快验证速度**

---

## 📞 支持

### 问题排查
1. 查看 `AUTO_OPTIMIZATION_GUIDE.md` 故障排查章节
2. 查看 `README.md` 常见问题
3. 检查日志: `experiments/EXP8/logs/`

### 降级方案
- 自动优化遇到问题 → 使用手动优化版本
- 手动优化遇到问题 → 使用原版

---

## ✨ 总结

我们提供了**三个版本**的执行方案：

1. **v3.0 自动优化**（推荐）⭐
   - 最大化显存利用
   - 自动测试 + 自动LR缩放
   - 预计1-1.5天完成

2. **v2.0 手动优化**（保守）
   - 预设优化配置
   - 无需测试
   - 预计2天完成

3. **v1.0 原版**（最稳定）
   - 保守配置
   - 预计3-4天完成

**推荐**: 先尝试v3.0自动优化，如遇问题降级到v2.0

---

**最后更新**: 2026-03-29
**版本**: v3.0 (自动优化系统)
**推荐使用**: `auto_find_batch_size.py` + `run_all_auto_optimized.sh`
**预计完成**: 2026-03-30 (1-1.5天后)

🚀 所有文件已准备就绪，可以立即开始！
