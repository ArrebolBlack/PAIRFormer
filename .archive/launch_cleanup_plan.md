# Launch文件精简方案

## 当前状态：29个文件

### 保留的核心文件（7个）

#### 训练入口（3个）
1. **train.py** → 保留（Stage 1-2训练）
2. **train_em.py** → 保留（Stage 3训练）
3. **train_pair_selected_inst.py** → 保留（Pair-level训练）

#### 评估入口（2个）
4. **eval.py** → 保留（Stage 1-2评估）
5. **eval_em.py** → 保留（Stage 3评估）

#### 性能分析（2个）
6. **bench_compute_vs_k.py** → 保留（Runtime分析）
7. **bench_utils.py** → 保留（Benchmark工具）

### 需要删除的文件（22个）

#### 推理脚本（3个）- 可以合并到inference.py
- inference_BR-MIL_online.py → 删除
- inference_Naive_online.py → 删除
- inference_targetnet_like_online.py → 删除

#### Cache构建（10个）- 可以合并到build_cache.py
- build_cache_ddp.py → 删除
- build_cache_window.py → 删除
- build_cache_window_with_statistics_and_plot.py → 删除
- build_cheap_cache.py → 删除
- build_selected_inst_cache.py → 删除
- build_selected_pair_cache.py → 删除
- build_selected_pair_cache_parallel.py → 删除
- build_selection_cache.py → 删除
- build_teacher_shards.py → 删除
- build_window_shards.py → 删除

#### Shard工具（6个）- MTI专用，移到scripts/mti/
- compact_window_shards.py → 移到scripts/mti/
- relabel_window_shards.py → 移到scripts/mti/
- merge_inst_shards.py → 移到scripts/mti/
- train_cheapcts_shard.py → 移到scripts/mti/
- train_targetnet_shard.py → 移到scripts/mti/
- eval_window_shard.py → 移到scripts/mti/

#### 其他（3个）
- dryrun_dynamic_pair_batch.py → 删除（调试用）
- pair_selected_runtime.py → 删除（性能测试，已有bench）
- eval_pair_selected.py → 合并到eval.py
- train_pair_selected_raw.py → 删除（冗余）

## 新的scripts/结构

```
scripts/
├── train_stage1.py          # Stage 1训练（封装train.py）
├── train_stage2.py          # Stage 2训练（封装train.py）
├── train_stage3.py          # Stage 3训练（封装train_em.py）
├── evaluate.py              # 统一评估入口
├── inference.py             # 统一推理入口
├── build_cache.py           # 统一cache构建
├── benchmark.py             # 性能测试
└── mti/                     # MTI专用工具
    ├── train_shard.py
    ├── build_shards.py
    └── merge_shards.py
```

## 执行步骤

1. 创建scripts/mti/目录
2. 移动MTI shard工具
3. 删除冗余文件
4. 创建新的统一入口脚本
