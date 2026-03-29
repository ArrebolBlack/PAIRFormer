# EXP8 MTI数据集 - 优化后时间线 (A100 80GB)

**优化日期**: 2026-03-29
**优化目标**: 最大化利用A100 80GB显存，压缩训练时间

---

## 🚀 Batch Size优化

### 优化前 vs 优化后

| Stage | 原batch_size | 优化后 | 加速比 | 说明 |
|-------|-------------|--------|--------|------|
| Stage 1 | 512 | **1024** | 2x | TargetNet_Optimized, emb=384 |
| Stage 2 | 256 | **1024** | 4x | CheapCTSNet, emb=64 (轻量) |
| Stage 3 | 32 | **64** | 2x | Set Transformer, K=64 |

### 显存使用估算 (A100 80GB)

| Stage | Batch Size | 估算显存 | 余量 |
|-------|-----------|---------|------|
| Stage 1 | 1024 | ~55-60GB | 20-25GB |
| Stage 2 | 1024 | ~25-30GB | 50-55GB |
| Stage 3 | 64 | ~50-55GB | 25-30GB |

**安全余量**: 每个stage预留20-30GB用于optimizer states和梯度

---

## ⏱️ 时间线对比

### 原时间线 (未优化)

| 阶段 | 时间 | 累计 |
|------|------|------|
| Phase 1: 数据准备 | 2-3h | 2-3h |
| Phase 2.1: Window cache | 8-10h | 10-13h |
| Phase 3.1: Stage 1训练 | 16-20h | 26-33h |
| Phase 2.2: Cheap cache | 4-6h | 30-39h |
| Phase 3.2: Stage 2训练 | 12-16h | 42-55h |
| Phase 2.3: Selection cache | 2-3h | 44-58h |
| Phase 3.3: Stage 3训练 | 20-30h | 64-88h |
| Phase 4: 评估 | 2-3h | 66-91h |
| **总计** | **66-91h** | **约3-4天** |

### 优化后时间线 (A100 80GB)

| 阶段 | 时间 | 加速 | 累计 |
|------|------|------|------|
| Phase 1: 数据准备 | 2-3h | - | 2-3h |
| Phase 2.1: Window cache | 8-10h | - | 10-13h |
| **Phase 3.1: Stage 1训练** | **8-10h** | **2x** | **18-23h** |
| Phase 2.2: Cheap cache | 4-6h | - | 22-29h |
| **Phase 3.2: Stage 2训练** | **3-4h** | **4x** | **25-33h** |
| Phase 2.3: Selection cache | 2-3h | - | 27-36h |
| **Phase 3.3: Stage 3训练** | **10-15h** | **2x** | **37-51h** |
| Phase 4: 评估 | 2-3h | - | 39-54h |
| **总计** | **39-54h** | **1.7x** | **约1.5-2.5天** |

**关键改进**:
- 训练时间: 48-66h → 21-29h (减少27-37h)
- 总时间: 66-91h → 39-54h (减少27-37h)
- Wall-clock: 3-4天 → **1.5-2.5天**

---

## 📅 详细执行流程 (优化后)

### Day 1 (0-24h)

```
00:00 - 02:30  Phase 1: 数据准备 (2.5h)
02:30 - 12:00  Phase 2.1: Window cache (9.5h)
12:00 - 21:00  Phase 3.1: Stage 1训练 (9h) ✨ 2x加速
21:00 - 24:00  Phase 2.2: Cheap cache (3h)
```

**Day 1 完成**: Window cache + Stage 1训练 + Cheap cache启动

### Day 2 (24-48h)

```
00:00 - 02:00  Phase 2.2: Cheap cache完成 (2h)
02:00 - 05:30  Phase 3.2: Stage 2训练 (3.5h) ✨ 4x加速
05:30 - 08:00  Phase 2.3: Selection cache (2.5h)
08:00 - 20:30  Phase 3.3: Stage 3训练 (12.5h) ✨ 2x加速
20:30 - 23:00  Phase 4: 评估 (2.5h)
23:00         ✅ 完成！
```

**Day 2 完成**: 所有训练 + 评估

**总计**: **约2天** (48小时内完成)

---

## 🎯 关键里程碑

| 时间点 | 里程碑 | 状态 |
|--------|--------|------|
| Day 1 12:00 | Window cache完成 | ✓ |
| Day 1 21:00 | Stage 1完成 (9h) | ✓ 2x加速 |
| Day 2 05:30 | Stage 2完成 (3.5h) | ✓ 4x加速 |
| Day 2 08:00 | Selection cache完成 | ✓ |
| Day 2 20:30 | Stage 3完成 (12.5h) | ✓ 2x加速 |
| Day 2 23:00 | **评估完成，结果可用** | ✅ |

---

## 💡 进一步优化建议

### 如果仍需加速

1. **Gradient Accumulation** (如果单卡显存不足):
   ```bash
   run.batch_size=512
   trainer.grad_accum_steps=2  # 等效batch_size=1024
   ```

2. **Mixed Precision Training** (AMP):
   ```bash
   trainer.use_amp=true  # 可节省30-40%显存
   ```

3. **更激进的batch size** (需要测试OOM):
   - Stage 1: 1024 → 1536
   - Stage 2: 1024 → 2048
   - Stage 3: 64 → 96

4. **并行cache构建** (如果有多GPU):
   ```bash
   # 同时构建cheap cache和selection cache
   python3 -m src.launch.build_cheap_cache ... &
   python3 -m src.launch.build_selection_cache ... &
   wait
   ```

---

## 📊 性能监控

### 训练速度指标

| Stage | Batch Size | 预期it/s | 预期epoch时间 |
|-------|-----------|----------|--------------|
| Stage 1 | 1024 | ~5-8 it/s | ~5-6min |
| Stage 2 | 1024 | ~10-15 it/s | ~2-3min |
| Stage 3 | 64 | ~3-5 it/s | ~7-8min |

### GPU利用率目标

- **GPU Utilization**: >90%
- **Memory Usage**: 55-65GB (留15-25GB余量)
- **Power Usage**: >300W (A100满载)

### 监控命令

```bash
# 实时监控
watch -n 1 nvidia-smi

# 详细监控
nvidia-smi dmon -i 0 -s pucvmet -d 1

# 检查训练速度
tail -f logs/train_stage1_seed2020.log | grep "it/s"
```

---

## ⚠️ 风险与应对

### 风险1: OOM (Out of Memory)

**症状**: CUDA out of memory

**应对**:
1. 降低batch size:
   - Stage 1: 1024 → 768
   - Stage 2: 1024 → 768
   - Stage 3: 64 → 48

2. 启用gradient checkpointing (需要修改代码)

3. 使用gradient accumulation

### 风险2: 训练不稳定

**症状**: Loss爆炸或NaN

**应对**:
1. 降低learning rate (大batch size可能需要调整lr)
2. 使用gradient clipping (已有: `clip_grad_norm=1.0`)
3. 检查数据是否有异常值

### 风险3: 性能未达预期

**症状**: 加速比<2x

**应对**:
1. 检查GPU利用率 (`nvidia-smi`)
2. 检查dataloader是否成为瓶颈 (`num_workers`)
3. 检查是否有I/O瓶颈 (cache读取)

---

## 📝 执行Checklist

### 优化前准备
- [ ] 确认GPU型号: A100 80GB
- [ ] 检查CUDA版本: ≥11.0
- [ ] 检查PyTorch版本: ≥2.0
- [ ] 备份原始配置文件

### 执行阶段
- [ ] 使用 `run_all_optimized.sh` (而非 `run_all.sh`)
- [ ] 监控GPU显存使用
- [ ] 记录实际训练时间
- [ ] 对比优化前后速度

### 完成后
- [ ] 验证结果质量 (PR-AUC等)
- [ ] 记录实际加速比
- [ ] 更新文档

---

## 🚀 快速开始 (优化版)

```bash
cd /home/yjq/workspace/rebuttal/PAIRFormer/experiments/EXP8/scripts

# 使用优化版脚本
./run_all_optimized.sh
```

**预期完成时间**: 约2天 (48小时)

---

**最后更新**: 2026-03-29
**优化版本**: v2.0 (A100 80GB优化)
**预计完成**: 2026-03-31 (2天后)
