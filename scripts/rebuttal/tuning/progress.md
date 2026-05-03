# Progress: PAIR-Former 超参调优

## 2026-05-04

### Baseline 结果
- 10-fold 默认配置：F1=0.8151±0.0288, ROC-AUC=0.8812±0.0181
- fold1 单独：F1=0.8203, ROC-AUC=0.8951

### 已完成
- [x] 10-fold balanced 80/20 数据准备和训练
- [x] fold1 cache (kmax=64) 已构建，state=ready
- [x] 确认代码无 early stopping（跑满 num_epochs，取 best ckpt）
- [x] 确认远程前置：CheapCTSNet ckpt 512KB, TargetNet_Optimized ckpt 240KB, 数据 2.3MB

### 当前
- 准备 Round A K sweep 实验计划

### 卡点
- 无
