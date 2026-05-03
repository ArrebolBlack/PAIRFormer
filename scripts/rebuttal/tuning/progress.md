# Progress: PAIR-Former 超参调优

## 2026-05-04

### Baseline
- fold1 默认配置：F1=0.8203, ROC-AUC=0.8951（n_layers=3, d_model=256, bs=64, kmax=64, warmup=10）
- 10-fold mean：F1=0.8151±0.0288

### Round A: K Budget Sweep ✅ 完成

| K    | Test F1 | 备注           |
|------|---------|---------------|
| 8    | 0.8444  | 最优，极小预算  |
| 32   | 0.8333  |               |
| 64   | 0.8341  |               |
| 128  | 0.8224  |               |
| 256  | 0.8304  |               |
| 512  | 0.8297  |               |
| 1024 | 0.8203  |               |
| 2048 | 构建中   |               |

**结论**：小K更优，K=8已接近饱和。大K反而引入噪声。固定 K=64 进入 Round B（兼顾性能和鲁棒性）。

### 当前：Round B Model Capacity Sweep
- 固定 kmax=64，搜索 n_layers × d_model
- 分发到 A100 执行
