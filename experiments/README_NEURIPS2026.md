# NeurIPS 2026 实验计划索引（更新版）

**更新时间**: 2026-05-03  
**截止日期**: 2026-05-06 AOE  
**目的**: 完成 MTI K sweep + K=1 baseline 实验

---

## 📋 实验计划文档列表（更新）

### 1️⃣ K Truncated 实验（2×A100）⭐ 新增
**文件**: `EXP_PLAN_1_K_Truncated_2xA100.md`  
**硬件**: 2×A100-80GB  
**时间**: 3-4小时  
**优先级**: P1（推荐做）  
**性质**: ⚠️ 仅评估，无需训练  

**任务**: 使用已有的 K=64/128/256/512 checkpoints 进行 K truncated 实验
- 从 K=64 truncate 到 K=1/8/16/32
- 从 K=128 truncate 到 K=1/8/16/32/64
- 从 K=256 truncate 到 K=1/8/16/32/64/128
- 从 K=512 truncate 到 K=1/8/16/32/64/128/256

**关键点**:
- 快速评估，无需重新训练
- 对比不同 K_max 的 truncated 效果
- 与 retrain 结果对比

---

### 2️⃣ MTI K Sweep 完整实验（8×A100）⭐ 更新
**文件**: `EXP_PLAN_2_MTI_K_Sweep_8xA100.md`  
**硬件**: 8×A100-80GB  
**时间**: 18-24小时  
**优先级**: P0（最高优先级）  
**性质**: ✅ 需要训练模型  

**任务**: 在 MTI 数据集上完成完整的 K sweep 实验
- **K=1/8/16/32 retrain**: 补充小 K 的训练结果（16-24h）
- **K truncated**: 使用 K=512 模型 truncate 到不同 K（2-3h）

**关键点**:
- MTI 数据集上 K 敏感性更显著
- 用于绘制 Figure 3（主图）
- Retrain + Truncated 两条曲线对比

---

### 3️⃣ K=1 Baseline on miRAW & deepTargetPro（1×5090）
**文件**: `EXP_PLAN_3_K1_miRAW_deepTargetPro_5090.md`  
**硬件**: 1×RTX 5090  
**时间**: 9-13小时  
**优先级**: P0（最高优先级）  
**性质**: ✅ 需要训练模型  

**任务**: 在 miRAW 和 deepTargetPro 上训练 K=1 baseline
- miRAW K=1: 预期 F1 ≈ 0.925（Rebuttal 承诺）
- deepTargetPro K=1: Transfer learning，预期 F1 ≈ 0.80-0.82

---

### 4️⃣ STSelector 补充实验（8×A100，可选）
**文件**: `EXP_PLAN_4_STSelector_Optional_8xA100.md`  
**硬件**: 8×A100-80GB（闲时）  
**时间**: 5-7小时  
**优先级**: P2（低优先级，可选）  
**性质**: ✅ 需要训练模型  

**任务**: 验证 STSelector (S2) vs TopK (S0) 在 MTI 上的效果
- 仅供参考，时间紧张可跳过
- 预期 S0 ≈ S2（差异 <0.005）

---

## 🎯 执行优先级与时间线（更新）

### Day 1 (5月3日)
- [x] 生成实验计划文档
- [x] 更新实验计划（增加 K sweep）
- [ ] **启动**: MTI K sweep (8×A100, 18-24h) ⭐ 最高优先级
- [ ] **启动**: K=1 miRAW + deepTargetPro (1×5090, 9-13h)

### Day 2 (5月4日)
- [ ] **监控**: K sweep 和 K=1 实验进度
- [ ] **启动**: K truncated (2×A100, 3-4h)
- [ ] **收集**: 已完成实验结果

### Day 3 (5月5日)
- [ ] **整合**: 所有实验结果到论文
- [ ] **绘图**: MTI K sweep 图（Figure 3）
- [ ] **决策**: 是否执行 STSelector 实验（可选）
- [ ] **完成**: 论文修改和润色

---

## 📊 预期结果汇总（更新）

### MTI K Sweep 结果

| K | Retrain F1 (预期) | Truncated F1 (预期) | 已有结果 |
|---|------------------|-------------------|---------|
| 1 | 0.74-0.76 | 0.73-0.75 | - |
| 8 | 0.76-0.77 | 0.75-0.76 | - |
| 16 | 0.77-0.78 | 0.76-0.77 | - |
| 32 | 0.78-0.79 | 0.77-0.78 | - |
| 64 | - | ~0.77 | 0.7708 ✅ |
| 128 | - | ~0.778 | 0.7786 ✅ |
| 256 | - | ~0.779 | 0.7790 ✅ |
| 512 | - | 0.7918 | 0.7918 ✅ |

### K=1 Baseline 结果

| 数据集 | K=1 (预期) | K=64/512 (已有) | Delta |
|--------|-----------|----------------|-------|
| miRAW | F1 ≈ 0.925 | F1 = 0.950 | +2.5pp |
| MTI | F1 ≈ 0.75 | F1 = 0.7918 | +4pp |
| deepTargetPro | F1 ≈ 0.81 | F1 = 0.8521 | +4pp |

---

## 🎨 Figure 3 绘图计划

### 数据来源
- **Retrain**: K=1/8/16/32 (新实验) + K=64/128/256/512 (已有)
- **Truncated**: K=1/8/16/32/64/128/256/512 (新实验)

### 图表内容
- 两条曲线：Retrain@K (实线) + Truncate@K_max=512 (虚线)
- X 轴：Budget K (log scale)
- Y 轴：F1@0.5 / PR-AUC
- 标注：K=64 operating point

### 预期效果
- 展示 MTI 数据集上 K 敏感性
- Retrain 略优于 Truncated
- 性能随 K 单调递增并饱和

---

## ✅ 验收标准（更新）

### 必须完成
- [ ] MTI K=1/8/16/32 retrain 完成
- [ ] MTI K truncated 评估完成
- [ ] K=1 miRAW 完成
- [ ] K=1 deepTargetPro 完成
- [ ] 所有结果表格填写完整
- [ ] Figure 3 绘制完成

### 质量检查
- [ ] MTI K sweep 曲线单调递增
- [ ] Retrain ≥ Truncated（同一 K）
- [ ] miRAW K=1 F1 ≈ 0.925（±0.01）
- [ ] 所有训练曲线正常

---

## 📝 提交清单（更新）

### Checkpoints
- `checkpoints/MTI_K1_retrain/best.pt`
- `checkpoints/MTI_K8_retrain/best.pt`
- `checkpoints/MTI_K16_retrain/best.pt`
- `checkpoints/MTI_K32_retrain/best.pt`
- `checkpoints/miRAW_K1_baseline/best.pt`
- `checkpoints/deepTargetPro_K1_transfer/best.pt`

### 结果数据
- `results/mti_k_sweep_complete.json`
- `results/mti_k_truncated_all.json`
- `results/miraw_k1_results.json`
- `results/deeptargetpro_k1_results.json`

### 图表文件
- `figures/mti_k_sweep_complete.pdf` ⭐ Figure 3
- `figures/mti_k_sweep_complete.png`

---

## 🔄 与原计划的变化

### 新增实验
1. ✅ **MTI K=8/16/32 retrain**（原计划只有 K=1）
2. ✅ **K truncated 实验**（快速验证，无需训练）
3. ✅ **2×A100 利用已有 checkpoints**（原计划是 Robustness vs. n）

### 调整原因
1. **Figure 3 需要完整的 K sweep 数据**（K=1/8/16/32/64/128/256/512）
2. **MTI 数据集更适合作为主图**（K 敏感性更显著）
3. **K truncated 可快速补充数据点**（节省训练时间）

### 时间影响
- 原计划：K=1 (4-6h) + Robustness vs. n (2-3h) = 6-9h
- 新计划：K sweep (18-24h) + K truncated (3-4h) = 21-28h
- **增加时间**：15-19h

---

## 📞 联系方式

**问题反馈**: [待填写]  
**紧急联系**: [待填写]  
**项目负责人**: [待填写]

---

**最后更新**: 2026-05-03 04:00  
**状态**: ✅ 实验计划已更新，等待执行
