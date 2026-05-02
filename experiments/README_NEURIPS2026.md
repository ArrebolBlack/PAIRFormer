# NeurIPS 2026 实验计划索引

**创建时间**: 2026-05-03  
**截止日期**: 2026-05-06 AOE  
**目的**: K=1 baseline + Robustness vs. n 实验

---

## 📋 实验计划文档列表

### 1️⃣ Robustness vs. n (MTI)
**文件**: `EXP_PLAN_1_Robustness_2xA100.md`  
**硬件**: 2×A100-80GB  
**时间**: 2-3小时  
**优先级**: P1（推荐做）  
**性质**: ⚠️ 仅评估，无需训练  
**任务**: 验证 Theorem 5.2，评估不同 visible pool size n 对性能的影响

**关键点**:
- 使用已训练的 MTI K=512 checkpoint
- 需修改 selector 支持 `candidate_pool_size` 参数
- 评估 n ∈ {64, 128, 256, 512, 1024, 2048}
- 预期：性能在 n≥512 时饱和

---

### 2️⃣ K=1 Baseline on MTI
**文件**: `EXP_PLAN_2_K1_MTI_8xA100.md`  
**硬件**: 8×A100-80GB  
**时间**: 4-6小时  
**优先级**: P0（最高优先级）  
**性质**: ✅ 需要训练模型  
**任务**: 在 MTI 大规模数据集上训练 K=1 baseline

**关键点**:
- DDP 训练（8 GPUs）
- 预期 Test F1 ≈ 0.74-0.76
- 对比 K=512 (F1=0.7918)，预期差距 +3-5pp
- 证明 relational aggregation 的价值

---

### 3️⃣ K=1 Baseline on miRAW & deepTargetPro
**文件**: `EXP_PLAN_3_K1_miRAW_deepTargetPro_5090.md`  
**硬件**: 1×RTX 5090  
**时间**: 9-13小时（两个数据集）  
**优先级**: P0（最高优先级）  
**性质**: ✅ 需要训练模型  
**任务**: 在 miRAW 和 deepTargetPro 上训练 K=1 baseline

**关键点**:
- **miRAW K=1**: 预期 F1 ≈ 0.925（Rebuttal 承诺）
- **deepTargetPro K=1**: Transfer learning，预期 F1 ≈ 0.80-0.82
- 两个实验串行执行（miRAW 3.5h + deepTargetPro 2.5h）

---

### 4️⃣ STSelector 补充实验（可选）
**文件**: `EXP_PLAN_4_STSelector_Optional_8xA100.md`  
**硬件**: 8×A100-80GB（闲时）  
**时间**: 5-7小时  
**优先级**: P2（低优先级，可选）  
**性质**: ✅ 需要训练模型  
**任务**: 验证 STSelector (S2) vs TopK (S0) 在 MTI 上的效果

**关键点**:
- 仅供参考，时间紧张可跳过
- 预期 S0 ≈ S2（差异 <0.005）
- 论文已采用方案 B（保持弱化表述）

---

## 🎯 执行优先级与时间线

### Day 1 (5月3日)
- [x] 生成实验计划文档
- [ ] **启动**: K=1 MTI (8×A100, 4-6h)
- [ ] **启动**: K=1 miRAW + deepTargetPro (1×5090, 9-13h)

### Day 2 (5月4日)
- [ ] **监控**: K=1 实验进度
- [ ] **启动**: Robustness vs. n (2×A100, 2-3h)
- [ ] **收集**: K=1 实验结果

### Day 3 (5月5日)
- [ ] **整合**: 所有实验结果到论文
- [ ] **决策**: 是否执行 STSelector 实验（可选）
- [ ] **完成**: 论文修改和润色

---

## 📊 预期结果汇总

### K=1 Baseline 结果

| 数据集 | K=1 (预期) | K=64/512 (已有) | Delta | 说明 |
|--------|-----------|----------------|-------|------|
| miRAW | F1 ≈ 0.925 | F1 = 0.950 | +2.5pp | Rebuttal 承诺 |
| MTI | F1 ≈ 0.75 | F1 = 0.7918 | +4pp | 大规模验证 |
| deepTargetPro | F1 ≈ 0.81 | F1 = 0.8521 | +4pp | Transfer learning |

### Robustness vs. n 结果

| n (visible pool) | F1 (预期) | 说明 |
|-----------------|----------|------|
| 64 | 0.76-0.77 | n=K, selection quality 差 |
| 128-256 | 0.78-0.79 | 性能恢复 |
| 512-2048 | 0.791-0.792 | 饱和 |

---

## ✅ 验收标准

### 必须完成
- [ ] K=1 MTI 训练完成并评估
- [ ] K=1 miRAW 训练完成并评估
- [ ] K=1 deepTargetPro 训练完成并评估
- [ ] Robustness vs. n 评估完成
- [ ] 所有结果表格填写完整

### 质量检查
- [ ] K=1 F1 显著低于 K=64/512（差距 >2pp）
- [ ] miRAW K=1 F1 ≈ 0.925（±0.01）
- [ ] Robustness 曲线符合预期（n=64 低，n≥512 饱和）
- [ ] 所有训练曲线正常（无异常波动）

---

## 📝 提交清单

完成后需提交：

### Checkpoints
- `checkpoints/MTI_K1_baseline/best.pt`
- `checkpoints/miRAW_K1_baseline/best.pt`
- `checkpoints/deepTargetPro_K1_transfer/best.pt`

### 日志文件
- `logs/mti_k1_train.log` + `logs/mti_k1_eval.log`
- `logs/miraw_k1_train.log` + `logs/miraw_k1_eval.log`
- `logs/deeptargetpro_k1_train.log` + `logs/deeptargetpro_k1_eval.log`
- `logs/robustness_n*.log`

### 结果数据
- `results/mti_k1_results.json`
- `results/miraw_k1_results.json`
- `results/deeptargetpro_k1_results.json`
- `results/robustness_vs_n_mti.json`

### 图表文件
- `figures/robustness_vs_n_mti.pdf`
- `figures/mti_k_sweep_updated.pdf`（添加 K=1）

---

## 🐛 常见问题

### Q1: Checkpoint 文件不存在
**A**: 检查以下路径是否正确
- MTI: `checkpoints/MTI_v3_xlarge_resume/best.pt`
- miRAW: `checkpoints/miRAW_instance/best.pt`

### Q2: OOM (Out of Memory)
**A**: 降低 batch_size
- 8×A100: 128 → 64
- 1×5090: 64 → 32

### Q3: DDP 启动失败
**A**: 检查 NCCL 环境变量和 GPU 可见性

### Q4: 训练不收敛
**A**: 检查 instance encoder 是否正确冻结

---

## 📞 联系方式

**问题反馈**: [待填写]  
**紧急联系**: [待填写]  
**项目负责人**: [待填写]

---

## 📚 相关文档

- `FINAL_EXPERIMENT_PLAN.md` - 总体实验计划
- `TODO_EXPERIMENTS_FINAL.md` - 待做实验清单
- `MTI_MISSING_EXPERIMENTS.md` - MTI 缺失实验分析
- `NEURIPS_TODO.md` - NeurIPS 论文修改待办

---

**最后更新**: 2026-05-03 03:35  
**状态**: ✅ 实验计划已完成，等待执行
