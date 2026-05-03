# Goal: PAIR-Former 超参调优

## 目标
在 miRAW fold1 balanced 80/20 数据上，顺序调参找到最优配置，最终在 10-fold 上验证。
**唯一指标：best checkpoint 的 test F1@thr=0.5**

## Baseline
- fold1 默认配置：F1=0.8203, ROC-AUC=0.8951（n_layers=3, d_model=256, bs=64, kmax=64, warmup=10）
- 10-fold mean：F1=0.8151±0.0288
- **目标：fold1 F1 > 0.82**

## 调参顺序（因果链）

每轮形成有效结论，指导下轮搜索方向。

### Round A: K Budget Sweep（预算探索）
- **假设**：更多 CTS 窗口提供更多信息，但存在饱和点
- **方法**：固定默认模型配置，扫 kmax={32,64,128,256,512,1024,2048}
- **产出**：F1 vs K 曲线 → 找到信息饱和点 → 确定最优 K
- **分发**：8 个实验，每个需独立 cache → 适合 8×A100 并行（1 GPU/kmax）

### Round B: Model Capacity（模型容量）
- **假设**：模型容量应匹配 K 预算提供的信息量；K 大→需要更大模型
- **方法**：固定 Round A 最优 kmax，扫 n_layers×d_model
- **产出**：确定最优模型结构
- **分发**：不改 kmax → 复用 cache，可多 GPU 并行

### Round C: Training Dynamics（训练参数）
- **假设**：小数据集上 bs/lr/warmup 影响泛化
- **方法**：固定 Round A+B 最优配置，扫 bs/warmup
- **产出**：最优训练配方
- **分发**：可多 GPU 并行

### Round D: 10-fold 验证
- 用最终配置跑全部 10 fold，报告 mean±std

## 搜索空间

| 参数 | 范围 | 触发 cache 重建？ |
|------|------|:---:|
| kmax | {8,32,64,128,256,512,1024,2048} | 是 |
| n_layers | {1, 2, 3} | 否 |
| d_model | {32, 64, 128, 256} | 否 |
| dim_ff | 随 d_model 缩放: 2×d_model | 否 |
| n_heads | = d_model/32 (head_dim=32) | 否 |
| batch_size | {1, 4, 16, 64} | 否 |
| lr_agg | 3e-4 × sqrt(bs/64) | 否 |
| warmup_epochs | {5, 10, 20} | 否 |
| num_epochs | 50（取 best ckpt） | 否 |

## 资源

| 机器 | GPU | 用途 |
|------|-----|------|
| Local RTX 5090 | 1×32GB | 调试、快速验证 |
| 2×A100 | 2×80GB | Round B/C 并行实验 |
| 8×A100 | 8×80GB | Round A K sweep（7个kmax并行）|

## 远程前置条件
- 代码仓库（含 `data/rebuttal/miRAW_fold1_split/`，2.3MB）
- Stage 2 checkpoint: `checkpoints/CheapCTSNet/checkpoints/last.pt` (512KB)
- Stage 1 checkpoint: `checkpoints/miRAW_TargetNet_Optimized_dp-0.1/checkpoints/last.pt` (240KB)
