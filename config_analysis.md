# 实验配置文件分类分析

## 总览
- **总数**: 77个配置文件
- **数据集**: miRAW (31), deepTargetPro (16), MTI (27), DeepMirTar (3)

---

## 论文核心实验配置（必须保留 - 约20个）

### miRAW主实验（Table 1）
1. **miRAW_EM_Pipeline.yaml** ✅ - 主实验，K=64
2. **miRAW_TargetNet_Optimized_baseline.yaml** ✅ - Stage 1训练
3. **CheapCTSNet.yaml** ✅ - Stage 2蒸馏
4. **miRAW_pair_maxpool_cache.yaml** ✅ - Max pooling baseline

### deepTargetPro外部验证（Table 2）
5. **deepTargetPro_EM_Pipeline.yaml** ✅ - Transfer实验
6. **deepTargetPro_EM_Pipeline_ext150.yaml** ✅ - Full训练（150 epochs）
7. **deepTargetPro_TargetNet_Optimized.yaml** ✅ - Stage 1

### MTI大规模实验（Fig 2a）
8. **MTI_train_selected_inst.yaml** ✅ - 主训练配置
9. **MTI_TargetNet_Optimized_shard.yaml** ✅ - Stage 1 shard训练
10. **MTI_CheapCTSNet_shard.yaml** ✅ - Stage 2 shard训练
11. **MTI_EM_K512.yaml** ✅ - K=512实验
12. **MTI_EM_K1024.yaml** ✅ - K=1024实验（如果用到）

### MTI Shard工具配置（必须保留）
13. **MTI_build_window_shards.yaml** ✅
14. **MTI_build_teacher_shards.yaml** ✅
15. **MTI_compact_window_shards.yaml** ✅
16. **MTI_relabel_window_shards.yaml** ✅
17. **MTI_eval_window_shard.yaml** ✅

### Budget分析（Fig 2）
18. **miRAW_EM_Pipeline_K1.yaml** ✅ - K=1实验
19. **deepTargetPro_EM_Pipeline_K1.yaml** ✅ - K=1实验

### 消融实验
20. **miRAW_EM_Pipeline_k1ratio05.yaml** ✅ - K1 ratio ablation
21. **miRAW_EM_Pipeline_splitB.yaml** ✅ - Split sensitivity
22. **miRAW_EM_Pipeline_splitC.yaml** ✅ - Split sensitivity
23. **CheapCTSNet_noRelDistill.yaml** ✅ - Distillation ablation

### 架构消融
24. **MTI_PairCNNAggregator.yaml** ✅ - CNN aggregator baseline
25. **MTI_PairGNNAggregator.yaml** ✅ - GNN aggregator baseline

---

## 实验性/开发配置（可能归档 - 约40个）

### miRAW实验性配置
- miRAW_8020_*.yaml (7个) - 8:2 split实验
- miRAW_clean_split_*.yaml (2个) - Clean split实验
- miRAW_half_split_*.yaml (2个) - Half split实验
- miRAW_mixed_8020_K64.yaml
- miRAW_targetnet_*.yaml (3个) - TargetNet style实验
- miRAW_TargetNet.yaml, miRAW_TargetNet_baseline.yaml
- miRAW_TargetNet_Optimized.yaml, miRAW_TargetNet_Optimized_v1.yaml
- miRAW_Transformer*.yaml (2个)
- miRAW_pair_agg_*.yaml (2个)

### deepTargetPro实验性配置
- deepTargetPro_8020_split_*.yaml (2个)
- deepTargetPro_clean_split_*.yaml (2个)
- deepTargetPro_half_split_*.yaml (2个)
- deepTargetPro_pair_agg_baseline.yaml

### MTI实验性配置
- MTI_EM_Pipeline.yaml - 可能是早期版本
- MTI_EM_Scalable_*.yaml (3个) - Scalable实验
- MTI_PairGNNMoEAggregator.yaml - MoE实验
- MTI_TargetNet_Optimized.yaml - 非shard版本
- MTI_CheapCTSNet.yaml - 非shard版本
- MTI_build_selected_inst*.yaml (2个)
- MTI_eval_selected_*.yaml (3个)
- MTI_train_selected_*.yaml (多个变体)
- MTI_scalable_smoke.yaml

### DeepMirTar配置
- DeepMirTar_*.yaml (3个) - 可能未在论文中使用

### 其他
- experiment_mirna.yaml, experiment_sirna.yaml
- rebuttal_eval_no_pos_overlap.yaml

---

## 分类决策

### 第一优先级：核心论文实验（25个）
**保留在 configs/experiment/**

#### miRAW (8个)
- miRAW_EM_Pipeline.yaml
- miRAW_EM_Pipeline_K1.yaml
- miRAW_EM_Pipeline_k1ratio05.yaml
- miRAW_EM_Pipeline_splitB.yaml
- miRAW_EM_Pipeline_splitC.yaml
- miRAW_TargetNet_Optimized_baseline.yaml
- miRAW_pair_maxpool_cache.yaml
- CheapCTSNet.yaml
- CheapCTSNet_noRelDistill.yaml

#### deepTargetPro (4个)
- deepTargetPro_EM_Pipeline.yaml
- deepTargetPro_EM_Pipeline_ext150.yaml
- deepTargetPro_EM_Pipeline_K1.yaml
- deepTargetPro_TargetNet_Optimized.yaml

#### MTI (13个)
- MTI_train_selected_inst.yaml
- MTI_TargetNet_Optimized_shard.yaml
- MTI_CheapCTSNet_shard.yaml
- MTI_EM_K512.yaml
- MTI_EM_K1024.yaml
- MTI_PairCNNAggregator.yaml
- MTI_PairGNNAggregator.yaml
- MTI_build_window_shards.yaml
- MTI_build_teacher_shards.yaml
- MTI_compact_window_shards.yaml
- MTI_relabel_window_shards.yaml
- MTI_eval_window_shard.yaml
- MTI_build_selected_inst.yaml (如果用于cache构建)

### 第二优先级：可能有用的实验（10个）
**保留在 configs/experiment/variants/**

- miRAW_8020_split_K64.yaml - 可能用于split sensitivity
- deepTargetPro_8020_split_K64.yaml
- MTI_EM_Pipeline.yaml - 可能是早期版本，保留参考
- MTI_train_selected_inst_v2.yaml
- MTI_eval_selected_inst.yaml
- rebuttal_eval_no_pos_overlap.yaml - Rebuttal实验
- miRAW_pair_agg_set_transformer.yaml
- miRAW_TargetNet_Optimized.yaml
- deepTargetPro_pair_agg_baseline.yaml
- MTI_PairGNNMoEAggregator.yaml

### 第三优先级：归档（42个）
**移动到 configs/experiment/archive/**

所有其他实验性配置，包括：
- 所有8020/clean/half split变体（除了上述保留的）
- TargetNet baseline变体
- Transformer baseline
- DeepMirTar配置
- Smoke test配置
- 早期实验配置

---

## 执行计划

### Step 1: 创建目录结构
```bash
mkdir -p configs/experiment/archive
mkdir -p configs/experiment/variants
```

### Step 2: 移动核心配置（保持原位）
无需操作，25个核心配置保留在 configs/experiment/

### Step 3: 移动变体配置
```bash
git mv configs/experiment/miRAW_8020_split_K64.yaml configs/experiment/variants/
git mv configs/experiment/deepTargetPro_8020_split_K64.yaml configs/experiment/variants/
# ... 其他9个
```

### Step 4: 归档过时配置
```bash
git mv configs/experiment/miRAW_8020_*.yaml configs/experiment/archive/
git mv configs/experiment/miRAW_clean_split_*.yaml configs/experiment/archive/
# ... 其他40个
```

### Step 5: 创建README
创建 configs/experiment/README.md 说明配置组织

---

## 验证清单

- [ ] 核心配置（25个）在 configs/experiment/
- [ ] 变体配置（10个）在 configs/experiment/variants/
- [ ] 归档配置（42个）在 configs/experiment/archive/
- [ ] 创建 configs/experiment/README.md
- [ ] 更新 CLAUDE.md 中的配置路径
- [ ] 验证核心实验配置可加载
