# EXP8: MTI Dataset Full Training

**目标**: 在MTI (mirTarCLASH) 数据集上全量训练PAIR-Former，验证方法在第三个独立外部数据集上的泛化性

**数据集**: 237,893 miRNA-mRNA pairs (Human, CLASH-seq/CLEAR-seq/iCLIP-seq)

**时间线**: 约5-6天完成1个seed的完整pipeline

---

## 目录结构

```
EXP8/
├── scripts/              # 执行脚本
│   ├── step1_generate_negatives.py    # 生成负类样本
│   ├── step2_split_data.py            # 数据分割
│   ├── step3_generate_configs.py      # 生成配置文件
│   ├── run_all.sh                     # 主执行脚本
│   └── quick_start.sh                 # 快速启动脚本
├── configs/              # 生成的配置文件（自动生成）
├── data/                 # 处理后的数据（自动生成）
├── results/              # 实验结果（自动生成）
├── logs/                 # 日志文件（自动生成）
└── README.md             # 本文件
```

---

## 快速开始

### 前置条件

1. **数据文件**: 确保 `PAIRFormer/data/MTI/MTI_mRNALevel.csv` 存在
2. **环境**: PyTorch 2.4.1, Hydra 1.3.2, scikit-learn
3. **硬件**:
   - CPU: 26核 (用于cache构建)
   - GPU: RTX 5090 32GB (Stage 1-2) 或 A100×2 (Stage 3)

### 一键运行

```bash
cd PAIRFormer/experiments/EXP8/scripts
./run_all.sh
```

这将自动执行：
1. Phase 1: 数据准备 (生成负类、分割、配置)
2. Phase 2: Cache构建 (window/cheap/selection)
3. Phase 3: 三阶段训练 (Stage 1/2/3)
4. Phase 4: 评估

**注意**: 脚本会先运行seed=2020，完成后询问是否继续运行剩余seeds (2025, 2026)

---

## 分步执行

如果需要分步执行或调试，可以手动运行各个阶段：

### Phase 1: 数据准备

```bash
cd PAIRFormer

# Step 1: 生成负类 (随机配对)
python3 experiments/EXP8/scripts/step1_generate_negatives.py \
    --mti_path data/MTI/MTI_mRNALevel.csv \
    --output_path data/MTI/MTI_pair_with_negatives_random.txt \
    --seed 1234

# Step 2: 数据分割 (70/15/15)
python3 experiments/EXP8/scripts/step2_split_data.py \
    --input_path data/MTI/MTI_pair_with_negatives_random.txt \
    --output_path data/MTI/MTI_pair_random_split.txt \
    --split_mode random \
    --seed 2020

# Step 3: 生成配置文件
python3 experiments/EXP8/scripts/step3_generate_configs.py \
    --configs_root configs \
    --templates_root configs
```

### Phase 2: Cache构建

```bash
cd PAIRFormer

# Window cache (预计10-12小时)
python3 -m src.launch.build_cache_window \
    experiment=MTI_TargetNet_Optimized \
    seed=2020 \
    run.num_workers=22 \
    run.batch_size=8192

# Cheap cache (在Stage 2训练后运行，预计4-6小时)
python3 -m src.launch.build_cheap_cache \
    experiment=MTI_CheapCTSNet \
    seed=2020 \
    em.cheap_cache.batch_size=10240 \
    em.cheap_cache.num_workers=16 \
    em.cheap_cache.amp=true

# Selection cache (在Cheap cache后运行，预计2-4小时)
python3 -m src.launch.build_selection_cache \
    experiment=MTI_EM_Pipeline \
    seed=2020 \
    em.selection_cache.pair_batch_size=10240
```

### Phase 3: 训练

```bash
cd PAIRFormer

# Stage 1: TargetNet_Optimized (预计16-20小时)
python3 -m src.launch.train \
    experiment=MTI_TargetNet_Optimized \
    seed=2020 \
    run.num_epochs=100 \
    run.batch_size=256 \
    run.num_workers=8

# Stage 2: CheapCTSNet (预计12-16小时)
python3 -m src.launch.train \
    experiment=MTI_CheapCTSNet \
    seed=2020 \
    instance_ckpt_path="checkpoints/MTI_TargetNet_Optimized/checkpoints/last.pt" \
    run.num_epochs=100 \
    run.batch_size=256 \
    run.num_workers=8

# Stage 3: EM Pipeline (预计20-30小时)
python3 -m src.launch.train_em \
    experiment=MTI_EM_Pipeline \
    seed=2020 \
    instance_ckpt_path="checkpoints/MTI_TargetNet_Optimized/checkpoints/last.pt" \
    cheap_ckpt_path="checkpoints/MTI_CheapCTSNet/checkpoints/last.pt" \
    run.num_epochs=100 \
    run.batch_size=32 \
    run.kmax=64
```

### Phase 4: 评估

```bash
cd PAIRFormer

# 找到输出目录
OUTPUT_DIR=$(find outputs -type d -name "MTI_EM_Pipeline*seed_2020*" | head -1)

# 评估
python3 -m src.launch.eval_em \
    experiment=MTI_EM_Pipeline \
    seed=2020 \
    run.checkpoint="${OUTPUT_DIR}/checkpoints/best.pt" \
    run.test_splits=["test"]
```

---

## 配置说明

### 数据配置 (configs/data/miRNA_MTI.yaml)

```yaml
name: mirna_MTI
path:
  train: "data/MTI/MTI_pair_random_split.txt"
  val: "data/MTI/MTI_pair_random_split.txt"
  test: "data/MTI/MTI_pair_random_split.txt"
with_esa: true
split_column: 5
split_map:
  train: "train"
  val: "val"
```

### 实验配置

- `configs/experiment/MTI_TargetNet_Optimized.yaml`: Stage 1配置
- `configs/experiment/MTI_CheapCTSNet.yaml`: Stage 2配置
- `configs/experiment/MTI_EM_Pipeline.yaml`: Stage 3配置

所有超参数与EXP4 (deepTargetPro) 保持一致。

---

## 数据统计

### MTI_mRNALevel
- 总pairs: 237,893
- Unique miRNA: 1,392
- Unique mRNA: 51,591
- mRNA长度: 平均1922nt, 中位数1154nt

### CTS分布 (n-distribution)
- 平均n: 1244.5
- 中位数n: 662
- P90: 3095
- P99: 8165
- 最大n: 29,290
- n=0的pairs: 19,937 (8.4%) - **已丢弃**

### 数据分割 (随机split, seed=2020)
- Train: 70% (~166,525 pairs)
- Val: 15% (~35,684 pairs)
- Test: 15% (~35,684 pairs)

### 负类生成
- 方法: 随机配对 (Random Pairing)
- 正负比: 1:1
- Seed: 1234

---

## 预期结果

基于EXP4 (deepTargetPro) 的经验：

| 指标 | 预期范围 |
|------|---------|
| CTS-level PR-AUC | 0.85-0.95 |
| Pair-level PR-AUC | 0.90-0.98 |
| F1@0.5 | 0.85-0.95 |

**注意**: MTI负类是合成的，性能可能略低于实验验证负类的数据集

---

## 时间估算

### 单个seed (2020)

| 阶段 | 预计时间 | GPU |
|------|---------|-----|
| Phase 1: 数据准备 | 2-4h | CPU |
| Phase 2.1: Window cache | 10-12h | CPU |
| Phase 3.1: Stage 1训练 | 16-20h | RTX 5090 |
| Phase 2.2: Cheap cache | 4-6h | CPU |
| Phase 3.2: Stage 2训练 | 12-16h | RTX 5090 |
| Phase 2.3: Selection cache | 2-4h | CPU |
| Phase 3.3: Stage 3训练 | 20-30h | A100×2 |
| Phase 4: 评估 | 2-4h | A100 |
| **总计** | **~5-6天** | - |

### 3个seeds (2020, 2025, 2026)

- 总计: ~15-18天 (如果串行执行)
- 建议: 先完成seed=2020，根据结果决定是否继续

---

## 故障排查

### 问题1: Window cache构建时间过长

**解决方案**:
- 降低batch_size: `run.batch_size=4096`
- 减少num_workers: `run.num_workers=16`
- 分段构建 (需要修改脚本)

### 问题2: GPU显存不足 (Stage 3)

**解决方案**:
- 降低batch_size: `run.batch_size=16` 或 `run.batch_size=8`
- 使用A100 80GB版本
- 启用gradient checkpointing (需要修改代码)

### 问题3: n=0的pairs导致错误

**解决方案**:
- 已在数据准备阶段丢弃n=0的pairs
- 如果仍有问题，检查CTS扫描逻辑

### 问题4: 配置文件未找到

**解决方案**:
- 确保运行了step3_generate_configs.py
- 检查configs/目录下是否有MTI相关配置文件

---

## 结果文件

评估完成后，结果保存在：

```
experiments/EXP8/results/seed_2020/
├── test_metrics.json          # 评估指标
├── test_roc_curve.png         # ROC曲线
├── test_pr_curve.png          # PR曲线
└── test_report.txt            # 详细报告
```

---

## Rebuttal价值

### 回应Reviewer质疑

| Reviewer | 质疑 | EXP8回应 |
|----------|------|----------|
| KXKP-W1 | 仅单一数据集，规模小 | 第三个外部数据集，237K pairs (217×) |
| F4i7-W3 | 需在miRAW以外验证 | 完全独立的MTI数据 |
| ohuZ-W1 | 数据泄露 | 与miRAW无重叠 |
| 所有 | 性能饱和 | 不同负类生成策略下的表现 |

### Rebuttal文本模板

> "To address concerns about dataset scale and generalization, we further validate PAIR-Former on MTI (mirTarCLASH), a third independent external dataset comprising 237,893 miRNA-mRNA interactions from 73 CLASH-seq/CLEAR-seq/iCLIP-seq experiments. Despite using computationally generated negatives (random pairing), our model achieves PR-AUC = X.XX on the test set (35,684 pairs), demonstrating robust performance across diverse data sources and negative sampling strategies. This dataset is 217× larger than miRAWtest, directly addressing the 'tiny benchmark' concern raised by reviewers."

---

## 联系与支持

如有问题或需要帮助，请联系实验负责人。

**最后更新**: 2026-03-29

---

## 🚀 优化版本 (A100 80GB)

**新增日期**: 2026-03-29

### 关键优化

为了最大化利用A100 80GB显存，我们优化了batch size配置：

| Stage | 原配置 | 优化后 | 加速比 |
|-------|--------|--------|--------|
| Stage 1 | 512 | **1024** | 2x |
| Stage 2 | 256 | **1024** | 4x |
| Stage 3 | 32 | **64** | 2x |

### 时间线对比

| 版本 | 训练时间 | 总时间 | Wall-clock |
|------|---------|--------|-----------|
| 原版本 | 48-66h | 66-91h | 3-4天 |
| **优化版** | **21-29h** | **39-54h** | **1.5-2.5天** |

**时间节省**: 约27-37小时 (1-1.5天)

### 使用优化版

```bash
cd /home/yjq/workspace/rebuttal/PAIRFormer/experiments/EXP8/scripts

# 使用优化版脚本 (推荐)
./run_all_optimized.sh

# 或使用原版脚本
./run_all.sh
```

### 详细时间线

参见: [TIMELINE_OPTIMIZED.md](TIMELINE_OPTIMIZED.md)

### 注意事项

1. **显存要求**: 需要A100 80GB (或同等级GPU)
2. **OOM风险**: 如果遇到OOM，可降低batch size:
   - Stage 1: 1024 → 768
   - Stage 2: 1024 → 768
   - Stage 3: 64 → 48
3. **监控**: 使用 `nvidia-smi` 实时监控显存使用

### 性能验证

优化后的batch size已经过理论估算，预期显存使用：
- Stage 1: ~55-60GB (余量20-25GB)
- Stage 2: ~25-30GB (余量50-55GB)
- Stage 3: ~50-55GB (余量25-30GB)

如需进一步验证，可运行：
```bash
python3 scripts/test_batch_size.py --stage 1  # 测试Stage 1
python3 scripts/test_batch_size.py --stage 2  # 测试Stage 2
python3 scripts/test_batch_size.py --stage 3  # 测试Stage 3
```

---

**推荐**: 使用优化版 `run_all_optimized.sh` 以获得最快速度
