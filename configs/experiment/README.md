# Experiment Configurations

本目录包含PAIR-Former的所有实验配置文件。配置使用Hydra框架管理。

## 目录结构

```
configs/experiment/
├── README.md                    # 本文件
├── *.yaml                       # 核心实验配置（27个）
├── variants/                    # 实验变体配置（10个）
└── archive/                     # 归档的历史配置（40个）
```

---

## 核心实验配置（27个）

这些配置对应论文中的主要实验，必须保留以确保可复现性。

### miRAW主实验（9个）

#### Stage 1-2训练
- **miRAW_TargetNet_Optimized_baseline.yaml** - Stage 1: 训练TargetNet_Optimized（CTS encoder）
- **miRAW_TargetNet_Optimized.yaml** - Stage 1变体
- **CheapCTSNet.yaml** - Stage 2: 知识蒸馏训练CheapCTSNet
- **CheapCTSNet_noRelDistill.yaml** - Stage 2消融：无关系蒸馏

#### Stage 3训练（Table 1主结果）
- **miRAW_EM_Pipeline.yaml** - 主实验，K=64，F1=0.840±0.022
- **miRAW_pair_maxpool_cache.yaml** - Max pooling baseline

#### Budget分析（Fig 2）
- **miRAW_EM_Pipeline_K1.yaml** - K=1实验

#### 消融实验
- **miRAW_EM_Pipeline_k1ratio05.yaml** - K1 ratio=0.5消融
- **miRAW_EM_Pipeline_splitB.yaml** - Split sensitivity B
- **miRAW_EM_Pipeline_splitC.yaml** - Split sensitivity C

### deepTargetPro外部验证（4个）

#### Table 2实验
- **deepTargetPro_EM_Pipeline.yaml** - Transfer实验，F1=83.9±3.9%
- **deepTargetPro_EM_Pipeline_ext150.yaml** - Full训练150 epochs，F1=83.2±3.2%
- **deepTargetPro_TargetNet_Optimized.yaml** - Stage 1训练

#### Budget分析
- **deepTargetPro_EM_Pipeline_K1.yaml** - K=1实验

### MTI大规模实验（14个）

#### Stage 3训练（Fig 2a）
- **MTI_train_selected_inst.yaml** - 主训练配置，420K pairs
- **MTI_EM_K512.yaml** - K=512实验，F1=0.7925
- **MTI_EM_K1024.yaml** - K=1024实验

#### Stage 1-2 Shard训练
- **MTI_TargetNet_Optimized_shard.yaml** - Stage 1 shard训练
- **MTI_CheapCTSNet_shard.yaml** - Stage 2 shard训练

#### Shard数据处理工具
- **MTI_build_window_shards.yaml** - 构建window shards
- **MTI_build_teacher_shards.yaml** - 构建teacher shards
- **MTI_compact_window_shards.yaml** - 压缩window shards
- **MTI_relabel_window_shards.yaml** - 重标注window shards
- **MTI_eval_window_shard.yaml** - 评估window shard
- **MTI_build_selected_inst.yaml** - 构建selected instance cache

#### 架构消融
- **MTI_PairCNNAggregator.yaml** - CNN aggregator baseline
- **MTI_PairGNNAggregator.yaml** - GNN aggregator baseline

---

## 实验变体配置（variants/，10个）

这些配置是核心实验的变体或补充实验，可能在论文附录或rebuttal中使用。

### miRAW变体
- **miRAW_8020_balanced_K64.yaml** - 8:2 split平衡版本
- **miRAW_8020_baltest_K64.yaml** - 8:2 split平衡测试集
- **miRAW_8020_split_K64.yaml** - 8:2 split标准版本
- **miRAW_pair_agg_set_transformer.yaml** - Set Transformer aggregator变体

### deepTargetPro变体
- **deepTargetPro_8020_split_K64.yaml** - 8:2 split实验

### MTI变体
- **MTI_EM_Pipeline.yaml** - 早期版本（参考）
- **MTI_train_selected_inst_v2.yaml** - 训练配置v2
- **MTI_eval_selected_inst.yaml** - 评估配置
- **MTI_PairGNNMoEAggregator.yaml** - GNN MoE aggregator实验

### Rebuttal实验
- **rebuttal_eval_no_pos_overlap.yaml** - 无位置重叠评估

---

## 归档配置（archive/，40个）

这些配置是历史实验或开发过程中的临时配置，已不再使用但保留以供参考。

### 归档类别
- miRAW实验性split配置（clean/half/8020变体）
- miRAW baseline变体（TargetNet/Transformer）
- deepTargetPro实验性split配置
- MTI smoke test和调试配置
- MTI scalable实验配置
- DeepMirTar配置（未在论文中使用）
- 早期实验配置

---

## 使用方法

### 训练示例

```bash
# Stage 1: 训练CTS encoder
python -m src.launch.train experiment=miRAW_TargetNet_Optimized_baseline

# Stage 2: 蒸馏cheap encoder
python -m src.launch.train experiment=CheapCTSNet

# Stage 3: 训练PAIR-Former
python -m src.launch.train_em experiment=miRAW_EM_Pipeline

# MTI大规模实验
python -m src.launch.train_pair_selected_inst experiment=MTI_train_selected_inst
```

### 评估示例

```bash
# 评估Stage 1-2模型
python -m src.launch.eval experiment=miRAW_TargetNet_Optimized_baseline

# 评估Stage 3模型
python -m src.launch.eval_em experiment=miRAW_EM_Pipeline
```

### 配置覆盖

任何配置值都可以从命令行覆盖：

```bash
python -m src.launch.train_em \
    experiment=miRAW_EM_Pipeline \
    run.num_epochs=50 \
    seed=2025 \
    run.batch_size=128
```

---

## 配置文件结构

每个实验配置使用 `@package _global_` 指令，定义完整的实验设置：

```yaml
# @package _global_

experiment_name: miRAW_EM_Pipeline
seed: 2020
device: cuda

data:
  name: mirna_miRAW
  path: ...

model:
  arch: PairSetTransformerAggregator
  ...

trainer_em:
  lr_agg: 3e-4
  ...

em:
  cheap_ckpt_path: ...
  instance_ckpt_path: ...
```

---

## 论文实验对应关系

### Table 1: miRAW 10-fold CV
- 主结果：`miRAW_EM_Pipeline.yaml`
- Max pooling baseline：`miRAW_pair_maxpool_cache.yaml`
- TargetNet baseline：使用官方checkpoint评估
- Mimosa baseline：使用官方checkpoint评估

### Table 2: deepTargetPro外部验证
- Transfer：`deepTargetPro_EM_Pipeline.yaml`
- Full：`deepTargetPro_EM_Pipeline_ext150.yaml`

### Figure 2a: MTI性能 vs Budget K
- K=64：`MTI_train_selected_inst.yaml`
- K=512：`MTI_EM_K512.yaml`
- K=1024：`MTI_EM_K1024.yaml`

### Figure 2b: Pool-size robustness
- 使用`MTI_train_selected_inst.yaml`，动态调整可见pool size

### Figure 3: Runtime分析
- 使用`src/launch/bench_compute_vs_k.py`和inference脚本

### Figure 4: Budget分析
- miRAW：`miRAW_EM_Pipeline_K1.yaml`等
- deepTargetPro：`deepTargetPro_EM_Pipeline_K1.yaml`等

### 消融实验
- K1 ratio：`miRAW_EM_Pipeline_k1ratio05.yaml`
- Split sensitivity：`miRAW_EM_Pipeline_splitB.yaml`, `splitC.yaml`
- Distillation：`CheapCTSNet_noRelDistill.yaml`
- Aggregator：`MTI_PairCNNAggregator.yaml`, `MTI_PairGNNAggregator.yaml`

---

## 维护指南

### 添加新配置
1. 确定配置类型（核心/变体/归档）
2. 放置在相应目录
3. 更新本README
4. 如果是论文实验，更新"论文实验对应关系"部分

### 归档旧配置
1. 确认配置不再使用
2. 使用`git mv`移动到archive/
3. 更新本README

### 配置命名规范
- 格式：`{dataset}_{model/task}_{variant}.yaml`
- 示例：`miRAW_EM_Pipeline_K1.yaml`
- 数据集：miRAW, deepTargetPro, MTI, DeepMirTar
- 任务：EM_Pipeline, TargetNet_Optimized, CheapCTSNet等
- 变体：K1, splitB, k1ratio05等（可选）

---

## 相关文档
- 主README：`../../README.md`
- CLAUDE.md：`../../CLAUDE.md`
- 配置分析：`../../config_analysis.md`
