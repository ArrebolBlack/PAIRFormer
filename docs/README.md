# Documentation Index

本目录包含PAIR-Former项目的技术文档、实验报告和部署指南。

## 目录结构

```
docs/
├── README.md                           # 本文件
├── reports/                            # 实验总结报告
├── DDP_TRAINING.md                     # DDP多GPU训练指南
├── NEW_SERVER_DEPLOYMENT.md            # 新服务器部署指南
└── MTI_*.md                            # MTI大规模实验相关文档
```

---

## 技术文档

### 训练和部署

#### [DDP_TRAINING.md](DDP_TRAINING.md)
分布式数据并行（DDP）训练指南
- DDP实现细节
- 多GPU训练配置
- 性能优化建议
- 故障排查

#### [DDP_Multi_GPU_Plan.md](DDP_Multi_GPU_Plan.md)
DDP多GPU训练计划
- 实现方案
- 技术选型
- 性能预期

#### [NEW_SERVER_DEPLOYMENT.md](NEW_SERVER_DEPLOYMENT.md)
新服务器部署完整指南
- 必需文件清单（785MB数据 + checkpoints）
- 快速部署命令
- 缓存构建流程（约8小时）
- 训练启动步骤

### MTI大规模实验

#### [MTI_pipeline_newplan.md](MTI_pipeline_newplan.md)
MTI pipeline新方案
- 420K pairs大规模训练
- Shard-based数据处理
- 三阶段训练流程

#### [MTI_scalable_pipeline_design.md](MTI_scalable_pipeline_design.md)
MTI可扩展pipeline设计
- 架构设计
- 扩展性考虑
- 性能优化

#### [MTI_stage12_scalable_design.md](MTI_stage12_scalable_design.md)
MTI Stage 1-2可扩展设计
- Shard训练方案
- 数据分片策略
- 并行化方案

#### [MTI_stage12_stage3_retrain_plan.md](MTI_stage12_stage3_retrain_plan.md)
MTI Stage 1-2-3重训练计划
- 完整训练流程
- Checkpoint管理
- 实验配置

---

## 实验报告

### [reports/EXPERIMENT_REPORT.md](reports/EXPERIMENT_REPORT.md)
NeurIPS 2026综合实验报告
- 所有主要实验结果
- 消融实验分析
- 性能对比
- 论文图表对应

### [reports/EXP4_Report.md](reports/EXP4_Report.md)
EXP4实验报告
- deepTargetPro外部验证
- Transfer vs Full训练对比
- 多seed结果

### [reports/automation_log.md](reports/automation_log.md)
自动化实验日志
- 实验执行记录
- 自动化脚本使用
- 问题和解决方案

### [reports/experiment_scaling_log.md](reports/experiment_scaling_log.md)
实验扩展日志
- 扩展性实验记录
- 性能测试结果
- 资源使用分析

---

## 实验计划

实验计划文档位于 `../experiments/` 目录：

- **EXP_PLAN_1_K_Truncated_2xA100.md** - K截断实验（2×A100）
- **EXP_PLAN_1_Robustness_2xA100.md** - 鲁棒性实验（2×A100）
- **EXP_PLAN_2_K1_MTI_8xA100.md** - MTI K1实验（8×A100）
- **EXP_PLAN_2_MTI_K_Sweep_8xA100.md** - MTI K扫描实验（8×A100）
- **EXP_PLAN_3_K1_miRAW_deepTargetPro_5090.md** - K1实验（RTX 5090）
- **EXP_PLAN_4_STSelector_Optional_8xA100.md** - STSelector消融（8×A100）

详见 `../experiments/README_NEURIPS2026.md`

---

## Rebuttal实验

Rebuttal实验文档位于 `../scripts/rebuttal/` 目录，包含：

- **8:2 split实验** - `EXPERIMENT_PLAN_8020.md`
- **Cheap ranking分析** - `cheap_ranking/`, `cheap_ranking_v2/`
- **K1 ratio消融** - `k1_ratio_ablation/`
- **K sensitivity分析** - `k_sensitivity/`
- **N distribution分析** - `n_distribution/`
- **Relational distillation消融** - `rel_distill_ablation/`
- **Selector消融** - `selector_ablation/`
- **Split sensitivity** - `split_sensitivity/`
- **超参数调优** - `tuning/`
- **结果汇总** - `results/`

---

## 配置文档

配置文件文档位于 `../configs/experiment/README.md`，包含：
- 77个实验配置的分类和说明
- 核心配置（27个）
- 变体配置（10个）
- 归档配置（40个）
- 论文实验对应关系

---

## 快速导航

### 我想...

**开始训练**
→ 查看主 [README.md](../README.md) 的 Quick Start 部分

**部署到新服务器**
→ 查看 [NEW_SERVER_DEPLOYMENT.md](NEW_SERVER_DEPLOYMENT.md)

**使用DDP多GPU训练**
→ 查看 [DDP_TRAINING.md](DDP_TRAINING.md)

**训练MTI大规模模型**
→ 查看 [MTI_pipeline_newplan.md](MTI_pipeline_newplan.md)

**复现论文实验**
→ 查看 [reports/EXPERIMENT_REPORT.md](reports/EXPERIMENT_REPORT.md)

**了解实验配置**
→ 查看 [../configs/experiment/README.md](../configs/experiment/README.md)

**查看Rebuttal实验**
→ 查看 [../scripts/rebuttal/](../scripts/rebuttal/)

---

## 文档维护

### 添加新文档
1. 确定文档类型（技术文档/实验报告/实验计划）
2. 放置在相应目录
3. 更新本README的相应部分
4. 添加到快速导航（如适用）

### 文档命名规范
- 技术文档：`{主题}_{类型}.md`（如 `DDP_TRAINING.md`）
- 实验报告：`{实验名}_Report.md`（如 `EXP4_Report.md`）
- 实验计划：`EXP_PLAN_{编号}_{描述}_{资源}.md`

---

## 相关链接
- 主README：[../README.md](../README.md)
- CLAUDE.md：[../CLAUDE.md](../CLAUDE.md)
- 实验目录：[../experiments/](../experiments/)
- 配置文档：[../configs/experiment/README.md](../configs/experiment/README.md)
