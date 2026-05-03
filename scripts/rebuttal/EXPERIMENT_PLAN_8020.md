# Table 1 & Table 2 实验计划 (80/20 Deduplicated Stratified Split)

日期: 2026-05-03
硬件: Plan A: 8×A100 (服务器1) | Plan B: 2×A100 (服务器2) | Plan C: 1×RTX 5090 (本地)
种子: {2020, 2025, 2026}

---

## 一、数据概况

| 数据集 | 用途 | 训练 | 验证 | 测试 |
|--------|------|------|------|------|
| miRAW 80/20 pair | Table 1 Stage 3 | 4,269 | 473 | 1,186 |
| miRAW CTS (Stage 1/2) | Table 1 Stage 1-2 | 58,794 pairs | - | - |
| DTP 80/20 pair | Table 2 Stage 3 | 3,053 | 338 | 849 |
| DTP CTS (Stage 1/2) | Table 2 full Stage 1-2 | 64,427 pairs | 1,001 | - |

---

## 二、已有检查点

| ID | 检查点 | 路径 | 种子 |
|----|--------|------|------|
| P1 | miRAW TargetNet (original) | `checkpoints/miRAW_TargetNet_origin/checkpoints/best.pt` | 2020 |
| P4 | miRAW TargetNet_Optimized | `checkpoints/miRAW_TargetNet_Optimized_dp-0.1/checkpoints/best.pt` | 2020 |
| P7 | miRAW CheapCTSNet | `checkpoints/CheapCTSNet/checkpoints/last.pt` | 2020 |
| T3.1 | miRAW PAIR-Former K=64 | `outputs/miRAW_8020_split_K64/2026-05-03_17-58-46/checkpoints/best.pt` | 2020 |
| T4.1 | DTP PAIR-Former transfer | `outputs/deepTargetPro_8020_split_K64/2026-05-03_17-58-46/checkpoints/best.pt` | 2020 |

**DTP Stage 1/2 检查点不存在, 需全部训练。**

---

## 三、待运行实验清单

### A. 前置检查点 (Stage 1 + Stage 2)

| ID | 任务 | 模型 | 数据 | Epochs | Est. | 依赖 |
|----|------|------|------|--------|------|------|
| P2 | miRAW TargetNet orig s=2025 | TargetNet | miRAW 58K | 50 | ~10m | - |
| P3 | miRAW TargetNet orig s=2026 | TargetNet | miRAW 58K | 50 | ~10m | - |
| P5 | miRAW TargetNet_Opt s=2025 | TargetNet_Opt | miRAW 58K | 40 | ~15m | - |
| P6 | miRAW TargetNet_Opt s=2026 | TargetNet_Opt | miRAW 58K | 40 | ~15m | - |
| P8 | miRAW CheapCTSNet s=2025 | CheapCTSNet | miRAW 58K | 100 | ~20m | P5 |
| P9 | miRAW CheapCTSNet s=2026 | CheapCTSNet | miRAW 58K | 100 | ~20m | P6 |
| D1 | DTP TargetNet_Opt s=2020 | TargetNet_Opt | DTP 64K | 100 | ~15m | - |
| D2 | DTP TargetNet_Opt s=2025 | TargetNet_Opt | DTP 64K | 100 | ~15m | - |
| D3 | DTP TargetNet_Opt s=2026 | TargetNet_Opt | DTP 64K | 100 | ~15m | - |
| D4 | DTP CheapCTSNet s=2020 | CheapCTSNet | DTP 64K | 100 | ~20m | D1 |
| D5 | DTP CheapCTSNet s=2025 | CheapCTSNet | DTP 64K | 100 | ~20m | D2 |
| D6 | DTP CheapCTSNet s=2026 | CheapCTSNet | DTP 64K | 100 | ~20m | D3 |

### B. Table 1 实验 (miRAW 80/20 split)

| ID | 行 | 任务 | 方式 | Est. | 依赖 |
|----|-----|------|------|------|------|
| T1.1 | R1 | TargetNet eval s=2020 | eval.py softmax-top3 | ~5m | P1 |
| T1.2 | R1 | TargetNet eval s=2025 | eval.py softmax-top3 | ~5m | P2 |
| T1.3 | R1 | TargetNet eval s=2026 | eval.py softmax-top3 | ~5m | P3 |
| T2.1 | R2 | MaxPool eval s=2020 | eval.py reduction=max | ~5m | P4 |
| T2.2 | R2 | MaxPool eval s=2025 | eval.py reduction=max | ~5m | P5 |
| T2.3 | R2 | MaxPool eval s=2026 | eval.py reduction=max | ~5m | P6 |
| T3.2 | R3 | PAIR-Former K=64 s=2025 | train_em.py | ~20m | P5+P8 |
| T3.3 | R3 | PAIR-Former K=64 s=2026 | train_em.py | ~20m | P6+P9 |

### C. Table 2 实验 (DTP 80/20 split)

| ID | 行 | 任务 | 方式 | Est. | 依赖 |
|----|-----|------|------|------|------|
| T4.2 | R4 | DTP transfer s=2025 | train_em (miRAW ckpt) | ~20m | P5+P8 |
| T4.3 | R4 | DTP transfer s=2026 | train_em (miRAW ckpt) | ~20m | P6+P9 |
| T5.1 | R5 | DTP full s=2020 | train_em (DTP ckpt) | ~20m | D1+D4 |
| T5.2 | R5 | DTP full s=2025 | train_em (DTP ckpt) | ~20m | D2+D5 |
| T5.3 | R5 | DTP full s=2026 | train_em (DTP ckpt) | ~20m | D3+D6 |

### D. 不需要重跑 (Quoted baselines)

Table 1: miTDS, PITA, miRDB, miRanda, TargetScan, deepTarget, miRAW
Table 2: PITA, mirSVR, miRDB, microT, TargetScan, deepTarget, deepTargetPro, TargetNet, TEC-miTarget

---

## 四、依赖关系与关键路径

```
5条独立链路 (每条 ~55 min):
  Chain A: P5(15m)→P8(20m)→T3.2(20m)    miRAW seed=2025
  Chain B: P6(15m)→P9(20m)→T3.3(20m)    miRAW seed=2026
  Chain C: D1(15m)→D4(20m)→T5.1(20m)    DTP  seed=2020
  Chain D: D2(15m)→D5(20m)→T5.2(20m)    DTP  seed=2025
  Chain E: D3(15m)→D6(20m)→T5.3(20m)    DTP  seed=2026

延迟任务 (需等 Chain A/B 的 Stage 2 完成):
  T4.2: 等 P8 完成(35m) → 20m    DTP transfer seed=2025
  T4.3: 等 P9 完成(35m) → 20m    DTP transfer seed=2026

独立 eval (无长依赖):
  T1.1, T2.1: 已有ckpt, 即可执行
  T1.2, T1.3: 等 P2/P3 (10m)
  T2.2, T2.3: 等 P5/P6 (15m)
```

---

## 五、时间估计依据

| 实验类型 | 数据 | Epochs | Est. | 依据 |
|----------|------|--------|------|------|
| Stage 1 TargetNet orig | 58K | 50 | ~10m | 小模型大batch |
| Stage 1 TargetNet_Opt | 58K | 40 | ~15m | 中等模型小batch |
| Stage 1 DTP TargetNet_Opt | 64K | 100 | ~15m | 中等模型大batch |
| Stage 2 CheapCTSNet | 58-64K | 100 | ~20m | 含teacher蒸馏 |
| Stage 3 EM K=64 | 3-5K pairs | 100 | ~20m | **实测**: 18-19m (5090) |
| Eval (CTS→pair) | ~1.2K pairs | - | ~5m | 保守估计 |

---

## 六、Plan A: 8×A100 (服务器1) — miRAW全链路 + 1条DTP链 + eval

### GPU 时间线

```
min:  0    5    10   15   20   25   30   35   40   45   50   55
      |    |    |    |    |    |    |    |    |    |    |    |
GPU0  [====P5====][======P8======][======T3.2=====]              miRAW s=2025
GPU1  [====P6====][======P9======][======T3.3=====]              miRAW s=2026
GPU2  [====D3====][======D6======][======T5.3=====]              DTP s=2026
GPU3  [P2][T1.2][T2.2][    idle     ][======T4.2=====]           mixed
GPU4  [P3][T1.3][T2.3][    idle     ][======T4.3=====]           mixed
GPU5  [T1.1][T2.1]                                                  eval
GPU6  (空闲)
GPU7  (空闲)
```

### 各GPU详细命令

**GPU 0 — Chain A: P5→P8→T3.2**
```bash
# Stage 1 (~15m)
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train \
  experiment=miRAW_TargetNet_Optimized_baseline seed=2025

# Stage 2 (~20m, P5完成后)
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train \
  experiment=CheapCTSNet seed=2025 \
  run.distill_teacher_ckpt=<P5_OUTPUT>/checkpoints/best.pt

# Stage 3 (~20m, P5+P8完成后)
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_em \
  experiment=miRAW_8020_split_K64 seed=2025 \
  instance_ckpt_path=<P5_OUTPUT>/checkpoints/best.pt \
  cheap_ckpt_path=<P8_OUTPUT>/checkpoints/best.pt \
  paths.cache_root=cache_miRAW_8020_s2025
```

**GPU 1 — Chain B: P6→P9→T3.3** (同GPU 0, seed=2026)

**GPU 2 — Chain E: D3→D6→T5.3**
```bash
# Stage 1 (~15m)
CUDA_VISIBLE_DEVICES=2 python -m src.launch.train \
  experiment=deepTargetPro_TargetNet_Optimized seed=2026

# Stage 2 (~20m)
CUDA_VISIBLE_DEVICES=2 python -m src.launch.train \
  experiment=CheapCTSNet seed=2026 \
  data.path.train=data/deepTargetPro/train_seed_1234.txt \
  data.path.val=data/deepTargetPro/valid_seed_1234.txt \
  data.path.test=data/rebuttal/deepTargetPro_8020_split/deepTargetPro_Test.txt \
  run.distill_teacher_ckpt=<D3_OUTPUT>/checkpoints/best.pt

# Stage 3 (~20m)
CUDA_VISIBLE_DEVICES=2 python -m src.launch.train_em \
  experiment=deepTargetPro_8020_split_K64 seed=2026 \
  instance_ckpt_path=<D3>/best.pt cheap_ckpt_path=<D6>/best.pt \
  paths.cache_root=cache_dtp_8020_full_s2026
```

**GPU 3 — P2→T1.2→T2.2→T4.2**
```bash
# P2: miRAW TargetNet orig s=2025 (~10m)
CUDA_VISIBLE_DEVICES=3 python -m src.launch.train \
  experiment=miRAW_TargetNet_baseline seed=2025

# T1.2: TargetNet eval s=2025 (~5m)
CUDA_VISIBLE_DEVICES=3 python -m src.launch.eval \
  experiment=miRAW_TargetNet_baseline \
  run.checkpoint=<P2_OUTPUT>/checkpoints/best.pt \
  data.path.test=data/rebuttal/miRAW_8020_split/miRAW_Test.txt

# T2.2: MaxPool eval s=2025 (~5m, P5已完成@t=15m)
CUDA_VISIBLE_DEVICES=3 python -m src.launch.eval \
  experiment=miRAW_TargetNet_Optimized_baseline \
  run.checkpoint=<P5_OUTPUT>/checkpoints/best.pt \
  data.path.test=data/rebuttal/miRAW_8020_split/miRAW_Test.txt \
  run.test_reduction=max

# [idle ~15m, 等P8完成@t=35m]

# T4.2: DTP transfer s=2025 (~20m)
CUDA_VISIBLE_DEVICES=3 python -m src.launch.train_em \
  experiment=deepTargetPro_8020_split_K64 seed=2025 \
  instance_ckpt_path=<P5>/best.pt cheap_ckpt_path=<P8>/best.pt \
  paths.cache_root=cache_dtp_8020_transfer_s2025
```

**GPU 4 — P3→T1.3→T2.3→T4.3** (同GPU 3, seed=2026)

**GPU 5 — T1.1→T2.1** (已有ckpt)
```bash
CUDA_VISIBLE_DEVICES=5 python -m src.launch.eval \
  experiment=miRAW_TargetNet_baseline \
  run.checkpoint=checkpoints/miRAW_TargetNet_origin/checkpoints/best.pt \
  data.path.test=data/rebuttal/miRAW_8020_split/miRAW_Test.txt

CUDA_VISIBLE_DEVICES=5 python -m src.launch.eval \
  experiment=miRAW_TargetNet_Optimized_baseline \
  run.checkpoint=checkpoints/miRAW_TargetNet_Optimized_dp-0.1/checkpoints/best.pt \
  data.path.test=data/rebuttal/miRAW_8020_split/miRAW_Test.txt \
  run.test_reduction=max
```

**Plan A: ~55 min | 14 任务 | ~4.2 GPU-hours**

---

## 七、Plan B: 2×A100 (服务器2) — 2条DTP全链路

### GPU 时间线

```
min:  0    5    10   15   20   25   30   35   40   45   50   55
      |    |    |    |    |    |    |    |    |    |    |    |
GPU0  [====D1====][======D4======][======T5.1=====]    DTP s=2020
GPU1  [====D2====][======D5======][======T5.2=====]    DTP s=2025
```

### 详细命令

**GPU 0 — Chain C: D1→D4→T5.1**
```bash
# Stage 1: DTP TargetNet_Opt s=2020 (~15m)
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train \
  experiment=deepTargetPro_TargetNet_Optimized seed=2020

# Stage 2: DTP CheapCTSNet s=2020 (~20m)
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train \
  experiment=CheapCTSNet seed=2020 \
  data.path.train=data/deepTargetPro/train_seed_1234.txt \
  data.path.val=data/deepTargetPro/valid_seed_1234.txt \
  data.path.test=data/rebuttal/deepTargetPro_8020_split/deepTargetPro_Test.txt \
  run.distill_teacher_ckpt=<D1_OUTPUT>/checkpoints/best.pt

# Stage 3: DTP full s=2020 (~20m)
CUDA_VISIBLE_DEVICES=0 python -m src.launch.train_em \
  experiment=deepTargetPro_8020_split_K64 seed=2020 \
  instance_ckpt_path=<D1>/best.pt cheap_ckpt_path=<D4>/best.pt \
  paths.cache_root=cache_dtp_8020_full_s2020
```

**GPU 1 — Chain D: D2→D5→T5.2** (同上, seed=2025)

### Plan B 前置条件

服务器2 需部署:
- 完整代码仓库 (`git clone` + `pip install -r requirements.txt`)
- `data/deepTargetPro/train_seed_1234.txt` (~64K行)
- `data/deepTargetPro/valid_seed_1234.txt` (~1K行)
- `data/rebuttal/deepTargetPro_8020_split/` (3个文件)

**Plan B: ~55 min | 6 任务 | ~1.8 GPU-hours | 完全自包含**

---

## 八、Plan C: 1×RTX 5090 (本地) — 快速eval

### 顺序执行

| # | 任务 | 命令 | Est. | 累计 |
|---|------|------|------|------|
| 1 | T1.1: TargetNet eval s=2020 | `python -m src.launch.eval experiment=miRAW_TargetNet_baseline run.checkpoint=checkpoints/miRAW_TargetNet_origin/checkpoints/best.pt data.path.test=data/rebuttal/miRAW_8020_split/miRAW_Test.txt` | 5m | 5m |
| 2 | T2.1: MaxPool eval s=2020 | `python -m src.launch.eval experiment=miRAW_TargetNet_Optimized_baseline run.checkpoint=checkpoints/miRAW_TargetNet_Optimized_dp-0.1/checkpoints/best.pt data.path.test=data/rebuttal/miRAW_8020_split/miRAW_Test.txt run.test_reduction=max` | 5m | 10m |

**Plan C: ~10 min | 2 任务 | 完全自包含, 无跨机器依赖**

5090 完成后可提前整理 Table 1 Row 1-2 的 seed=2020 结果。其余 eval 已分配到 Plan A。

---

## 九、三机并行总时间线

```
min:  0         10        20        30        40        50        55
      |---------|---------|---------|---------|---------|---------|

Plan A (8×A100):
GPU0  [====P5====][======P8======][======T3.2=====]
GPU1  [====P6====][======P9======][======T3.3=====]
GPU2  [====D3====][======D6======][======T5.3=====]
GPU3  [P2][T1.2][T2.2][    idle     ][======T4.2=====]
GPU4  [P3][T1.3][T2.3][    idle     ][======T4.3=====]
GPU5  [T1.1][T2.1]

Plan B (2×A100):
GPU0  [====D1====][======D4======][======T5.1=====]
GPU1  [====D2====][======D5======][======T5.2=====]

Plan C (5090):
      [T1.1][T2.1] ✓
```

### 汇总

| 机器 | 任务数 | Wall Time | GPU-hours | 跨机器依赖 |
|------|--------|-----------|-----------|-----------|
| **Plan A**: 8×A100 | 14 | ~55 min | ~4.2 h | **无** |
| **Plan B**: 2×A100 | 6 | ~55 min | ~1.8 h | **无** |
| **Plan C**: 5090 | 2 | ~10 min | ~0.2 h | **无** |
| **总计** | **22** | **~55 min** | **~6.2 h** | |

**三机并行总 Wall Time: ~55 min (~1 小时)**
**三台机器完全自包含, 零跨机器文件传输。**

---

## 十、需要新建的配置

1. **DTP CheapCTSNet** (Plan A GPU2, Plan B): 通过 CLI override `CheapCTSNet.yaml` 即可, 无需新文件
2. **各 seed 的 cache_root**: 通过 CLI `paths.cache_root=cache_xxx_s20xx` 避免冲突
3. 所有命令均在现有配置基础上通过 CLI override 完成, **无需修改代码**

---

## 十一、结果收集

训练完成后每个实验生成:
- `outputs/<exp>/.../checkpoints/best.pt`
- `outputs/<exp>/.../eval/val/metrics.json`
- `outputs/<exp>/.../eval/test/metrics.json`

三机结果汇总到同一机器后, 使用 `scripts/wandb_compute_mean_std.py` 聚合 3-seed mean±std, 更新 Table 1 和 Table 2。

---

## 十二、风险与注意事项

1. **DTP CheapCTSNet CLI override**: 需同时 override `data.path.*` 和 `run.distill_teacher_ckpt`, 命令较长
2. **缓存冲突**: 不同 seed 的 EM pipeline 必须使用不同 `paths.cache_root`
3. **数据格式**: 80/20 split 文件含 split 列 (6列), 原始 test 文件为 5 列, eval.py 需兼容
4. **Plan B 部署**: 服务器2 需同步代码 + DTP 数据 (~65MB CTS + ~8MB pair)
5. **Hydra output 目录**: 不同 seed 的 Stage 1/2 输出会自动按时间戳分目录, 需记录 best.pt 路径
6. **GPU 3/4 空闲期**: Plan A GPU 3/4 在 t=20~35m 有 ~15m 空闲 (等 P8/P9), 可插入其他短任务
