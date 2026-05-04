# Launch入口点分类分析

## 核心入口点（保留 - 7个）

### 训练入口
1. **train.py** (35K) - Stage 1-2训练（CTS-level）
   - 用途：训练TargetNet/TargetNet_Optimized和CheapCTSNet
   - 论文相关：Table 1基线，Stage 1-2
   - 状态：✅ 保留

2. **train_em.py** (52K) - Stage 3训练（Pair-level with EM）
   - 用途：训练PAIR-Former主模型（PairSetTransformerAggregator）
   - 论文相关：Table 1主结果，Table 2，Fig 2-4
   - 状态：✅ 保留

3. **train_pair_selected_inst.py** (16K) - Pair-level训练（简化版）
   - 用途：训练pair-level模型（不使用EM cache）
   - 论文相关：可能用于消融实验
   - 状态：✅ 保留

### 评估入口
4. **eval.py** (13K) - Stage 1-2评估
   - 用途：评估CTS-level模型
   - 论文相关：Stage 1-2 baseline评估
   - 状态：✅ 保留

5. **eval_em.py** (40K) - Stage 3评估
   - 用途：评估PAIR-Former模型
   - 论文相关：所有主要实验评估
   - 状态：✅ 保留

6. **eval_pair_selected.py** (11K) - Pair-level评估（简化版）
   - 用途：评估pair-level模型
   - 论文相关：可能用于消融实验
   - 状态：✅ 保留

### 性能分析
7. **bench_compute_vs_k.py** (30K) - Runtime benchmark
   - 用途：测量不同K值下的计算开销
   - 论文相关：Fig 3 runtime analysis
   - 状态：✅ 保留

## 推理入口（保留 - 3个）

8. **inference_BR-MIL_online.py** (48K) - BR-MIL在线推理
   - 用途：在线推理性能测试
   - 论文相关：Fig 3 runtime analysis
   - 状态：✅ 保留

9. **inference_Naive_online.py** (44K) - Naive baseline在线推理
   - 用途：对比baseline性能
   - 论文相关：Fig 3 runtime analysis
   - 状态：✅ 保留

10. **inference_targetnet_like_online.py** (31K) - TargetNet-like在线推理
    - 用途：对比TargetNet性能
    - 论文相关：Fig 3 runtime analysis
    - 状态：✅ 保留

## Cache构建工具（部分保留 - 4个核心）

### EM Pipeline Cache（保留）
11. **build_cheap_cache.py** (17K) - 构建cheap cache
    - 用途：预构建CheapCTSNet embeddings
    - 论文相关：Stage 3训练依赖
    - 状态：✅ 保留

12. **build_selection_cache.py** (12K) - 构建selection cache
    - 用途：预构建STSelector输出
    - 论文相关：Stage 3训练依赖
    - 状态：✅ 保留

13. **build_selected_inst_cache.py** (8.1K) - 构建instance cache
    - 用途：预构建TargetNet_Optimized embeddings
    - 论文相关：Stage 3训练依赖
    - 状态：✅ 保留

14. **build_cache_ddp.py** (13K) - DDP cache构建
    - 用途：多GPU并行构建cache
    - 论文相关：MTI大规模实验
    - 状态：✅ 保留

### Pair Cache（可能归档）
15. **build_selected_pair_cache.py** (4.3K) - 构建pair cache
    - 用途：预构建pair-level cache
    - 论文相关：不确定
    - 状态：⚠️ 需审查

16. **build_selected_pair_cache_parallel.py** (5.0K) - 并行构建pair cache
    - 用途：并行版本
    - 论文相关：不确定
    - 状态：⚠️ 需审查

### Window Shard工具（可能归档）
17. **build_window_shards.py** (2.3K) - 构建window shards
    - 用途：分片处理window数据
    - 论文相关：MTI数据预处理？
    - 状态：⚠️ 需审查

18. **build_teacher_shards.py** (3.1K) - 构建teacher shards
    - 用途：分片处理teacher模型输出
    - 论文相关：Stage 2蒸馏？
    - 状态：⚠️ 需审查

19. **compact_window_shards.py** (5.9K) - 压缩window shards
    - 用途：压缩shard文件
    - 论文相关：数据预处理
    - 状态：⚠️ 需审查

20. **relabel_window_shards.py** (3.8K) - 重标注window shards
    - 用途：使用teacher模型重标注
    - 论文相关：Stage 2蒸馏
    - 状态：⚠️ 需审查

21. **merge_inst_shards.py** (12K) - 合并instance shards
    - 用途：合并分片的instance cache
    - 论文相关：MTI数据处理
    - 状态：⚠️ 需审查

22. **build_cache_window.py** (2.9K) - 构建window cache
    - 用途：构建window-level cache
    - 论文相关：不确定
    - 状态：⚠️ 需审查

23. **build_cache_window_with_statistics_and_plot.py** (12K) - 构建window cache并统计
    - 用途：构建cache并生成统计图
    - 论文相关：可能用于Fig 2的n分布统计
    - 状态：⚠️ 需审查

## Shard训练工具（可能归档）

24. **train_cheapcts_shard.py** (16K) - Shard训练cheap encoder
    - 用途：在shard数据上训练CheapCTSNet
    - 论文相关：MTI Stage 2？
    - 状态：⚠️ 需审查

25. **train_targetnet_shard.py** (15K) - Shard训练TargetNet
    - 用途：在shard数据上训练TargetNet
    - 论文相关：MTI Stage 1？
    - 状态：⚠️ 需审查

26. **train_pair_selected_raw.py** (12K) - 训练pair模型（raw版本）
    - 用途：训练pair-level模型（另一个版本）
    - 论文相关：不确定
    - 状态：⚠️ 需审查

27. **eval_window_shard.py** (3.7K) - 评估window shard
    - 用途：评估shard数据上的模型
    - 论文相关：不确定
    - 状态：⚠️ 需审查

## 调试/测试工具（可能归档）

28. **dryrun_dynamic_pair_batch.py** (16K) - Dry run pair batch
    - 用途：测试动态batch构建
    - 论文相关：开发调试
    - 状态：⚠️ 可能归档

29. **pair_selected_runtime.py** (6.7K) - Pair runtime测试
    - 用途：测试pair-level runtime
    - 论文相关：性能分析
    - 状态：⚠️ 需审查

30. **bench_utils.py** (19K) - Benchmark工具函数
    - 用途：benchmark辅助函数
    - 论文相关：Fig 3支持代码
    - 状态：✅ 保留（作为bench_compute_vs_k.py的依赖）

## 分类总结

### 必须保留（14个）
- 核心训练/评估：train.py, train_em.py, train_pair_selected_inst.py, eval.py, eval_em.py, eval_pair_selected.py
- 性能分析：bench_compute_vs_k.py, bench_utils.py
- 在线推理：inference_BR-MIL_online.py, inference_Naive_online.py, inference_targetnet_like_online.py
- EM cache：build_cheap_cache.py, build_selection_cache.py, build_selected_inst_cache.py

### 需要审查（16个）
- DDP cache：build_cache_ddp.py（MTI实验可能需要）
- Pair cache：build_selected_pair_cache.py, build_selected_pair_cache_parallel.py
- Shard工具：build_window_shards.py, build_teacher_shards.py, compact_window_shards.py, relabel_window_shards.py, merge_inst_shards.py
- Window cache：build_cache_window.py, build_cache_window_with_statistics_and_plot.py
- Shard训练：train_cheapcts_shard.py, train_targetnet_shard.py, train_pair_selected_raw.py, eval_window_shard.py
- 测试工具：dryrun_dynamic_pair_batch.py, pair_selected_runtime.py

## 下一步行动

1. 检查MTI实验是否使用shard工具
2. 检查Fig 2的n分布统计是否使用build_cache_window_with_statistics_and_plot.py
3. 确认DDP cache构建是否在论文实验中
4. 决定是否归档shard训练工具
5. 创建src/launch/archive/目录
6. 移动确认可归档的文件
