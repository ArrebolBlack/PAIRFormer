# 文档整理分类

## 总览
- **总数**: 57个markdown文件
- **根目录**: 13个
- **docs/**: 7个
- **experiments/**: 12个
- **scripts/rebuttal/**: 25个

---

## 分类决策

### 第一类：根目录保留（8个）
**必须保留在根目录的核心文档**

1. **README.md** ✅ - 项目主文档
2. **CLAUDE.md** ✅ - Claude Code指南
3. **goal.md** ✅ - 重构目标（本次整理）
4. **todo.md** ✅ - 任务拆解（本次整理）
5. **progress.md** ✅ - 进度记录（本次整理）
6. **acceptance.md** ✅ - 验收标准（本次整理）
7. **launch_analysis.md** ✅ - Launch入口点分析（本次整理）
8. **config_analysis.md** ✅ - 配置文件分析（本次整理）

### 第二类：移到docs/reports/（5个）
**实验总结报告**

1. **EXPERIMENT_REPORT.md** → docs/reports/
2. **EXP4_Report.md** → docs/reports/
3. **automation_log.md** → docs/reports/
4. **experiment_scaling_log.md** → docs/reports/

### 第三类：保留在docs/（7个）
**技术文档和部署指南**

1. **docs/DDP_Multi_GPU_Plan.md** ✅
2. **docs/DDP_TRAINING.md** ✅
3. **docs/NEW_SERVER_DEPLOYMENT.md** ✅
4. **docs/MTI_pipeline_newplan.md** ✅
5. **docs/MTI_scalable_pipeline_design.md** ✅
6. **docs/MTI_stage12_scalable_design.md** ✅
7. **docs/MTI_stage12_stage3_retrain_plan.md** ✅

### 第四类：保留在experiments/（12个）
**实验计划和报告**

#### EXP8（8个）
1. **experiments/EXP8/EXP8_STAGE3_TUNING_REPORT.md** ✅
2. **experiments/EXP8/EXP8_Stage3_Report.md** ✅
3. **experiments/EXP8/STAGE3_PLAN.md** ✅
4. **experiments/EXP8/exp8_agentA_plan.md** ✅
5. **experiments/EXP8/exp8_agentA_shm.md** ✅
6. **experiments/EXP8/exp8_agentB_plan.md** ✅
7. **experiments/EXP8/exp8_agentB_shm.md** ✅
8. **experiments/EXP8/exp8_shm_plan.md** ✅

#### 实验计划（4个）
1. **experiments/EXP_PLAN_1_K_Truncated_2xA100.md** ✅
2. **experiments/EXP_PLAN_1_Robustness_2xA100.md** ✅
3. **experiments/EXP_PLAN_2_K1_MTI_8xA100.md** ✅
4. **experiments/EXP_PLAN_2_MTI_K_Sweep_8xA100.md** ✅
5. **experiments/EXP_PLAN_3_K1_miRAW_deepTargetPro_5090.md** ✅
6. **experiments/EXP_PLAN_4_STSelector_Optional_8xA100.md** ✅
7. **experiments/README_NEURIPS2026.md** ✅

### 第五类：保留在scripts/rebuttal/（25个）
**Rebuttal实验文档**

这些是论文rebuttal阶段的实验，应该保留在scripts/rebuttal/目录下，因为它们与对应的脚本紧密相关。

#### 主目录（2个）
1. **scripts/rebuttal/EXPERIMENT_PLAN_8020.md** ✅
2. **scripts/rebuttal/exp_ema_fix_plan.md** ✅

#### 子目录（23个）
- cheap_ranking/ (2个)
- cheap_ranking_v2/ (3个)
- k1_ratio_ablation/ (2个)
- k_sensitivity/ (1个)
- n_distribution/ (1个)
- rel_distill_ablation/ (1个)
- results/ (3个)
- selector_ablation/ (1个)
- split_sensitivity/ (1个)
- tuning/ (6个)

---

## 执行计划

### Step 1: 创建目录结构
```bash
mkdir -p docs/reports
```

### Step 2: 移动实验报告到docs/reports/
```bash
git mv EXPERIMENT_REPORT.md docs/reports/
git mv EXP4_Report.md docs/reports/
git mv automation_log.md docs/reports/
git mv experiment_scaling_log.md docs/reports/
```

### Step 3: 创建docs/README.md
创建文档索引，说明各目录用途

### Step 4: 更新主README.md
- 更新项目结构部分
- 添加文档链接
- 确保所有命令准确

### Step 5: 验证
- 根目录markdown文件≤8个
- docs/目录结构清晰
- 所有链接有效

---

## 最终目录结构

```
PAIRFormer/
├── README.md                    # 项目主文档
├── CLAUDE.md                    # Claude Code指南
├── goal.md                      # 重构目标
├── todo.md                      # 任务拆解
├── progress.md                  # 进度记录
├── acceptance.md                # 验收标准
├── launch_analysis.md           # Launch分析
├── config_analysis.md           # 配置分析
│
├── docs/
│   ├── README.md                # 文档索引
│   ├── DDP_TRAINING.md          # DDP训练指南
│   ├── NEW_SERVER_DEPLOYMENT.md # 服务器部署
│   ├── MTI_*.md                 # MTI相关文档（4个）
│   └── reports/
│       ├── EXPERIMENT_REPORT.md
│       ├── EXP4_Report.md
│       ├── automation_log.md
│       └── experiment_scaling_log.md
│
├── experiments/
│   ├── README_NEURIPS2026.md
│   ├── EXP_PLAN_*.md            # 实验计划（6个）
│   └── EXP8/                    # EXP8实验（8个）
│
├── scripts/rebuttal/
│   ├── EXPERIMENT_PLAN_8020.md
│   ├── exp_ema_fix_plan.md
│   └── [各子目录]/              # 23个rebuttal文档
│
└── configs/experiment/
    └── README.md                # 配置文档
```

---

## 验证清单

- [ ] 根目录markdown ≤ 8个
- [ ] docs/reports/创建并包含4个报告
- [ ] docs/README.md创建
- [ ] 主README.md更新
- [ ] 所有git mv操作完成
- [ ] 提交并推送
