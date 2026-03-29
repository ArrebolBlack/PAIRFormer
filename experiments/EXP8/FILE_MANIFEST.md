# EXP8 MTI数据集实验 - 文件清单

**生成时间**: 2026-03-29
**目标**: MTI数据集全量训练，验证PAIR-Former在第三个独立外部数据集上的泛化性

---

## 📁 目录结构

```
PAIRFormer/experiments/EXP8/
├── README.md                          # 实验说明文档
├── scripts/                           # 执行脚本
│   ├── step1_generate_negatives.py   # 生成负类样本
│   ├── step2_split_data.py           # 数据分割
│   ├── step3_generate_configs.py     # 生成配置文件
│   ├── quick_start.sh                # 快速启动（仅数据准备）
│   └── run_all.sh                    # 完整pipeline执行脚本
├── configs/                           # 配置文件（自动生成）
├── data/                              # 处理后的数据（自动生成）
├── results/                           # 实验结果（自动生成）
└── logs/                              # 日志文件（自动生成）
```

---

## 📄 文件说明

### 1. 文档文件

| 文件 | 说明 | 位置 |
|------|------|------|
| `README.md` | 实验完整说明，包含快速开始、配置说明、故障排查 | `experiments/EXP8/` |
| `EXP8_MTI_Execution_Plan.md` | 原始详细执行计划 | 项目根目录 |
| `EXP8_MTI_Execution_Plan_Updated.md` | 更新后的执行计划（26核CPU） | 项目根目录 |

### 2. Python脚本

| 文件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `step1_generate_negatives.py` | 随机配对生成负类 | `MTI_mRNALevel.csv` | `MTI_pair_with_negatives_random.txt` |
| `step2_split_data.py` | 数据分割 (70/15/15) | `MTI_pair_with_negatives_random.txt` | `MTI_pair_random_split.txt` |
| `step3_generate_configs.py` | 生成Hydra配置文件 | 模板配置 | `configs/data/miRNA_MTI.yaml` 等 |

### 3. Shell脚本

| 文件 | 功能 | 用途 |
|------|------|------|
| `quick_start.sh` | 快速验证数据准备 | 运行Phase 1，验证脚本正确性 |
| `run_all.sh` | 完整pipeline执行 | 自动化执行所有阶段（数据准备→cache→训练→评估） |

---

## 🚀 快速开始

### Step 1: 验证数据准备（5分钟）

```bash
cd /home/yjq/workspace/rebuttal/PAIRFormer/experiments/EXP8/scripts
./quick_start.sh
```

**预期输出**:
- ✅ `data/MTI/MTI_pair_with_negatives_random.txt` (约435K pairs)
- ✅ `data/MTI/MTI_pair_random_split.txt` (带split列)
- ✅ `configs/data/miRNA_MTI.yaml`
- ✅ `configs/experiment/MTI_*.yaml` (3个文件)

### Step 2: 运行完整Pipeline（5-6天）

```bash
cd /home/yjq/workspace/rebuttal/PAIRFormer/experiments/EXP8/scripts
./run_all.sh
```

**执行流程**:
1. Phase 1: 数据准备 (2-3h)
2. Phase 2: Cache构建 (14-19h)
3. Phase 3: 三阶段训练 (48-66h)
4. Phase 4: 评估 (2-3h)

---

## ⚙️ 关键参数

### 硬件配置
- **CPU**: 26核 (num_workers=22)
- **GPU**: A100×2
- **内存**: 建议≥128GB
- **磁盘**: 需要200GB可用空间

### 数据配置
- **负类生成**: 随机配对 (seed=1234)
- **数据分割**: 70/15/15 (seed=2020)
- **n=0处理**: 直接丢弃 (~19,937对)

### 训练配置
- **Seeds**: 2020 (主), 2025, 2026 (可选)
- **Budget K**: 64
- **Epochs**: Stage 1/2/3 各100 epochs
- **Batch size**: Stage 1/2=256, Stage 3=32

---

## 📊 预期结果

### 数据规模
- 原始pairs: 237,893
- 丢弃n=0后: ~217,956
- 加上负类: ~435,912
- Test集: ~65,387 pairs

### 性能指标（预期）
- CTS-level PR-AUC: 0.85-0.95
- Pair-level PR-AUC: 0.90-0.98
- F1@0.5: 0.85-0.95

### Rebuttal价值
- 数据集规模: 217× miRAWtest
- 完全独立: 与miRAW无重叠
- 不同负类策略: 合成负类 vs 实验验证负类

---

## 🔍 监控命令

### 查看日志
```bash
# 数据准备
tail -f experiments/EXP8/logs/step1_generate_negatives.log

# Cache构建
tail -f experiments/EXP8/logs/build_window_cache_seed2020.log

# 训练
tail -f experiments/EXP8/logs/train_stage1_seed2020.log
tail -f experiments/EXP8/logs/train_stage3_seed2020.log
```

### GPU监控
```bash
watch -n 1 nvidia-smi
```

### 磁盘空间
```bash
du -sh cache/MTI/
df -h
```

---

## ⚠️ 常见问题

### Q1: 数据准备失败
**A**: 检查 `data/MTI/MTI_mRNALevel.csv` 是否存在，验证Python依赖

### Q2: Cache构建时间过长
**A**: 降低 `batch_size=4096` 或 `num_workers=16`

### Q3: GPU OOM
**A**: 降低 `run.batch_size=16` (Stage 3)

### Q4: 配置文件未找到
**A**: 重新运行 `step3_generate_configs.py`

---

## ✅ Checklist

### 准备阶段
- [ ] 验证原始数据存在
- [ ] 检查Python环境 (pandas, numpy, sklearn, torch, hydra)
- [ ] 检查磁盘空间 (≥200GB)
- [ ] 检查GPU可用 (A100×2)

### 执行阶段
- [ ] 运行 `quick_start.sh`
- [ ] 检查生成的数据文件
- [ ] 检查生成的配置文件
- [ ] 运行 `run_all.sh`

### 监控阶段
- [ ] 定期检查日志
- [ ] 监控GPU使用率
- [ ] 监控磁盘空间
- [ ] 记录关键时间点

### 完成阶段
- [ ] 验证评估结果
- [ ] 复制结果到 `experiments/EXP8/results/`
- [ ] 生成实验报告
- [ ] 更新 `rebuttal_log.md`

---

## 📞 支持

如有问题，请参考：
1. `README.md` - 详细说明
2. `EXP8_MTI_Execution_Plan_Updated.md` - 执行计划
3. 日志文件 - `experiments/EXP8/logs/`

---

**生成工具**: Claude Code Agent
**最后更新**: 2026-03-29
**预计完成**: 2026-04-04 (6天后)
