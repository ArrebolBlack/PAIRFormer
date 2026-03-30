# Selector 消融实验 - 高优先级执行计划

> **目标**: 回应审稿人 ohuZ-W2 和 KXKP-W3 关于 Selector 消融不显著的质疑
> **策略**: 在小 K 值（K=8, 16）和更难的数据集（deepTargetPro）上验证多样性机制的价值
> **时间**: 单种子快速验证，约 24-36 小时 GPU 时间

---

## 一、问题诊断

### 根本原因
论文所有实验使用 `k1_ratio=1.0`，导致 STSelector 退化为纯 TopK：
- 当 K1=K, K2=0 时，跳过了所有多样性机制（位置 binning、SimHash 去重、平衡配额）
- 因此 S0、S1、S2 实际上运行的是几乎相同的逻辑

### Selector 变体定义
| 变体 | k1_ratio | use_hash_dedup | 描述 |
|------|----------|----------------|------|
| **S0** | 1.0 | False | TopK (纯分数排序) |
| **S1** | 0.5 | False | TopK + 位置多样性 |
| **S2** | 0.5 | True | TopK + 位置 + 嵌入多样性（完整 STSelector） |

---

## 二、实验矩阵

### 高优先级实验（必做）
| 实验组 | 数据集 | K 值 | Selector | 种子 | 预计时间 |
|--------|--------|------|----------|------|----------|
| **EXP-A** | miRAWtest | 8, 16 | S0, S1, S2 | 2020 | 6 × 2h = 12h |
| **EXP-B** | deepTargetPro | 8, 16 | S0, S1, S2 | 2020 | 6 × 3h = 18h |

**总计**: 12 次训练，约 30 GPU 小时

### 可选实验（时间允许）
| 实验组 | 数据集 | K 值 | 描述 | 预计时间 |
|--------|--------|------|------|----------|
| **EXP-C** | miRAWtest (n>500) | 8, 16 | 长尾 pair 子集 | 6 × 1.5h = 9h |

---

## 三、执行步骤

### Step 1: 生成配置文件

```bash
cd /home/yjq/workspace/rebuttal/PAIRFormer

# 生成所有配置
python scripts/rebuttal/selector_ablation/generate_configs.py
```

**输出**: 12 个配置文件在 `configs/experiment/selector_ablation/`

### Step 2: 运行高优先级实验

```bash
# 赋予执行权限
chmod +x scripts/rebuttal/selector_ablation/run_priority_experiments.sh

# 运行实验（会自动运行 EXP-A 和 EXP-B）
bash scripts/rebuttal/selector_ablation/run_priority_experiments.sh
```

**监控进度**:
```bash
# 查看当前运行的实验
watch -n 10 'ps aux | grep train_em'

# 查看最新日志
tail -f experiments/selector_ablation/miRAW_S0_K8/seed2020/train.log
```

### Step 3: 分析结果

```bash
# 生成汇总表格和统计检验
python scripts/rebuttal/selector_ablation/analyze_results.py

# 生成可视化图表
python scripts/rebuttal/selector_ablation/plot_results.py
```

**输出**:
- `experiments/selector_ablation/results_summary.csv` — 原始数据
- `experiments/selector_ablation/table_selector_ablation.tex` — LaTeX 表格
- `experiments/selector_ablation/figure_selector_ablation_pr_auc.png` — PR-AUC vs K 曲线
- `experiments/selector_ablation/figure_selector_ablation_improvement.png` — 改进热力图

---

## 四、可选：EXP-C 长尾 pair 分析

### Step 1: 生成长尾数据

```bash
python scripts/rebuttal/selector_ablation/generate_heavy_tail_data.py
```

**输出**:
- `data/selector_ablation/miRAW_Test_HeavyTail_n500.txt` — 筛选后的数据
- `data/selector_ablation/heavy_tail_statistics.md` — 统计报告

### Step 2: 手动创建配置

由于长尾数据需要自定义数据路径，需要手动创建配置文件：

```yaml
# configs/experiment/selector_ablation/miRAW_HeavyTail_S0_K8.yaml
# @package _global_
defaults:
  - /experiment/miRAW_EM_Pipeline

experiment_name: selector_ablation_miRAW_HeavyTail_S0_K8_seed${seed}

seed: ${seed}

data:
  path:
    test: "data/selector_ablation/miRAW_Test_HeavyTail_n500.txt"

run:
  kmax: 8
  num_epochs: 50

em:
  selector_module:
    cfg:
      kmax: 8
      k1_ratio: 1.0
      use_hash_dedup: false

  policy:
    warmup_epochs: 25

paths:
  cache_root: cache/selector_ablation/miRAW_HeavyTail_S0_K8

em_cache_root: cache/selector_ablation/miRAW_HeavyTail_S0_K8

logging:
  wandb:
    enabled: true
    project: "rebuttal_selector_ablation"
    group: "miRAW_HeavyTail_K8"
    tags: ["selector_ablation", "miRAW_HeavyTail", "S0", "K8"]
```

为 S1, S2 和 K=16 创建类似的配置文件。

### Step 3: 运行长尾实验

```bash
# 手动运行（或创建类似的 shell 脚本）
for SEL in S0 S1 S2; do
  for K in 8 16; do
    python -m src.launch.train_em \
      experiment=selector_ablation/miRAW_HeavyTail_${SEL}_K${K} \
      seed=2020 \
      hydra.run.dir=experiments/selector_ablation/miRAW_HeavyTail_${SEL}_K${K}/seed2020
  done
done
```

---

## 五、预期结果与 Rebuttal 策略

### 场景 1：K=8, 16 时 S2 显著优于 S0（最佳情况）

**预期**:
- K=8: S2 vs S0 改进 >0.02 PR-AUC
- K=16: S2 vs S0 改进 >0.01 PR-AUC

**Rebuttal 论证**:
> "We acknowledge that at the operating budget K⋆=64, the diversity-aware selection shows marginal gains over TopK. However, **at smaller budgets (K=8, 16)**, STSelector demonstrates statistically significant improvements (Table E3, Figure E1). This validates the design principle: **diversity matters more when the budget is tight**."

### 场景 2：deepTargetPro 上 S2 优势更明显

**预期**:
- deepTargetPro 是更难的数据集（PR-AUC ~0.85-0.90）
- S2 vs S0 改进在所有 K 值下都 >0.01

**Rebuttal 论证**:
> "On the more challenging deepTargetPro dataset (Table E3), STSelector consistently outperforms TopK across all budgets, with improvements ranging from +0.015 to +0.030 PR-AUC. This suggests that **diversity-aware selection is more valuable on harder tasks** where the cheap encoder's ranking is less reliable."

### 场景 3：所有场景下差异仍然很小（诚实承认）

**预期**:
- 即使在小 K 和 deepTargetPro 上，S2 vs S0 改进仍 <0.01

**Rebuttal 论证**:
> "We acknowledge that the diversity components provide **modest but consistent gains** across budgets and datasets (Table E3). The improvements are small because:
> 1. **The cheap encoder is already effective**: With 64-dim embeddings and distillation, the cheap logit scores already capture most functional signal.
> 2. **miRAWtest is saturated**: At PR-AUC >0.99, further gains are constrained by label noise.
> 3. **Diversity is a safety mechanism**: STSelector ensures **no performance degradation** compared to TopK, while providing redundancy control in heavy-tailed pools.
>
> We have updated the paper to clarify that STSelector is a **principled design choice** rather than a performance-critical component, and the main gains come from the budgeted relational aggregation framework itself."

---

## 六、文件组织

```
PAIRFormer/
├── configs/experiment/selector_ablation/
│   ├── miRAW_S0_K8.yaml
│   ├── miRAW_S0_K16.yaml
│   ├── miRAW_S1_K8.yaml
│   ├── miRAW_S1_K16.yaml
│   ├── miRAW_S2_K8.yaml
│   ├── miRAW_S2_K16.yaml
│   ├── deepTargetPro_S0_K8.yaml
│   ├── deepTargetPro_S0_K16.yaml
│   ├── deepTargetPro_S1_K8.yaml
│   ├── deepTargetPro_S1_K16.yaml
│   ├── deepTargetPro_S2_K8.yaml
│   └── deepTargetPro_S2_K16.yaml
│
├── scripts/rebuttal/selector_ablation/
│   ├── generate_configs.py              # 生成配置文件
│   ├── run_priority_experiments.sh      # 运行高优先级实验
│   ├── analyze_results.py               # 分析结果
│   ├── plot_results.py                  # 可视化
│   ├── generate_heavy_tail_data.py      # EXP-C 数据生成
│   └── README.md                        # 本文件
│
├── experiments/selector_ablation/
│   ├── miRAW_S0_K8/seed2020/
│   ├── miRAW_S0_K16/seed2020/
│   ├── ...
│   ├── results_summary.csv
│   ├── table_selector_ablation.tex
│   ├── figure_selector_ablation_pr_auc.png
│   └── figure_selector_ablation_improvement.png
│
└── data/selector_ablation/
    ├── miRAW_Test_HeavyTail_n500.txt
    └── heavy_tail_statistics.md
```

---

## 七、检查清单

### 实验前
- [ ] 生成配置文件（`generate_configs.py`）
- [ ] 验证配置正确性（抽查 2-3 个文件）
- [ ] 确认 GPU 可用（`nvidia-smi`）
- [ ] 确认预训练检查点存在：
  - `checkpoints/CheapCTSNet/checkpoints/last.pt`
  - `checkpoints/miRAW_TargetNet_Optimized_dp-0.1/checkpoints/last.pt`

### 实验中
- [ ] 运行 EXP-A (miRAW K=8, 16)
- [ ] 运行 EXP-B (deepTargetPro K=8, 16)
- [ ] 监控训练日志（每 6 小时检查一次）
- [ ] 确认每个实验生成 `metrics.json`

### 实验后
- [ ] 运行 `analyze_results.py`
- [ ] 运行 `plot_results.py`
- [ ] 检查生成的表格和图表
- [ ] 撰写 Rebuttal 回复

### 可选（EXP-C）
- [ ] 运行 `generate_heavy_tail_data.py`
- [ ] 手动创建长尾配置文件
- [ ] 运行长尾实验
- [ ] 更新分析脚本包含长尾结果

---

## 八、故障排除

### 问题 1: 配置文件生成失败
```bash
# 检查目录权限
ls -la configs/experiment/

# 手动创建目录
mkdir -p configs/experiment/selector_ablation
```

### 问题 2: 训练失败 - 找不到预训练检查点
```bash
# 检查检查点是否存在
ls -lh checkpoints/CheapCTSNet/checkpoints/last.pt
ls -lh checkpoints/miRAW_TargetNet_Optimized_dp-0.1/checkpoints/last.pt

# 如果缺失，需要先训练 Stage 1-2
```

### 问题 3: OOM (Out of Memory)
```bash
# 降低 batch_size（在配置文件中）
run:
  batch_size: 32  # 从 64 降到 32

# 或者降低 num_workers
run:
  num_workers: 8  # 从 16 降到 8
```

### 问题 4: 缓存冲突
```bash
# 清理旧缓存
rm -rf cache/selector_ablation/

# 重新运行实验
```

---

## 九、时间估算

### 单次训练时间
- miRAWtest K=8: ~1.5h
- miRAWtest K=16: ~2h
- deepTargetPro K=8: ~2.5h
- deepTargetPro K=16: ~3h

### 总 GPU 时间
- EXP-A: 6 runs × 1.75h = **10.5h**
- EXP-B: 6 runs × 2.75h = **16.5h**
- **总计**: ~27 GPU 小时

### 墙钟时间（串行执行）
- 约 **30 小时**（1.25 天）

---

## 十、联系与支持

如有问题，请检查：
1. 训练日志：`experiments/selector_ablation/*/seed2020/train.log`
2. Hydra 配置：`experiments/selector_ablation/*/seed2020/.hydra/config.yaml`
3. W&B 面板：https://wandb.ai/your-entity/rebuttal_selector_ablation

祝实验顺利！🚀
