# MTI Stage 1/2 Scalable Redesign

目标：为 MTI 这种 `48万 pair / 10亿级 CTS window` 的规模，重构 Stage 1 和 Stage 2，使其同时满足：

1. 不再依赖旧的全量 `.pt` block cache。
2. 预处理和训练能稳定利用多核 CPU 与单/双 GPU。
3. 存储格式适合 VepFS 这类“大文件顺序传输友好、小文件高频 IO 差”的平台。
4. 为 Stage 3 提供更强的 instance encoder 和 cheap selector。
5. 不破坏旧 pipeline；全部通过新增文件落地。


## 1. 现状问题

当前 Stage 1/2 仍然使用旧链路：

`pair txt -> 全量 CTS .pt block cache -> ChunkedCTSDataset -> DataLoader -> TargetNet / CheapCTSNet`

主要问题：

1. `src/data/cache.py` 会先生成大量 `.pt` block，小文件很多，不适合 VepFS。
2. `src/data/dataset.py` 训练期间频繁 `torch.load`，依赖随机小文件读取。
3. `src/data/collate.py` 之前在 CPU 上提前把 `uint8` 转成 `float32`，增加 CPU 内存带宽压力。
4. Stage 1 的 window-level 标签来自 pair label，正 pair 中所有 CTS 都被标正，监督噪声很大。
5. Stage 2 的 cheap distill 质量高度依赖 Stage 1 teacher，因此上游噪声会被继续传递。


## 2. 监督方案建议

### 2.1 不推荐

#### 方案 A：`MTI_siteLevel` 全正类训练

问题：

1. 没有负类，无法直接学到可靠分类边界。
2. 更适合作为自监督/对比学习预训练，而不是二分类 teacher。

#### 方案 B：`MTI_mRNALevel` 扫描后正 pair 全窗口标正

问题：

1. 正 bag 中绝大多数窗口并不是真实 functional site。
2. 会把 pair-level 标签噪声直接灌进 instance encoder。
3. 这是当前 Stage 3 上限被压住的核心来源之一。


### 2.2 推荐

#### 方案 C：Pseudo-site 弱监督训练

定义：

1. 负 pair 的所有 CTS window 标负。
2. 正 pair 不再全标正。
3. 只给每个正 pair 的少量高置信窗口正标签，其余窗口设为 `ignore`。

高置信窗口来源：

1. 启发式初始化：
   - `ESA` 高分
   - `cheap/topk` 排名前 `r`
   - 可选加入位置先验
2. 迭代自训练：
   - 用当前 instance/cheap 模型更新 pseudo positive 集合
   - 下一轮重训 instance

这是最推荐的 Stage 1 重训路径。


### 2.3 可选增强

#### 方案 D：Ranking / Contrastive instance training

定义：

1. 正 pair 中 top windows 与负 pair windows 做 ranking loss。
2. 不要求所有正 pair windows 都是正样本。

优点：

1. 比 hard label 更抗噪声。
2. 适合与方案 C 结合。


## 3. 新数据流

### 3.1 Stage 1

`raw pair table -> parallel CTS generator -> pseudo-site shard cache -> train TargetNet`

### 3.2 Stage 2

`pseudo-site shard cache -> train CheapCTSNet distill from TargetNet`

### 3.3 Stage 3

继续使用新的 scalable Stage 3：

`raw pair -> selected_raw -> selected_inst -> pair training`

但 cheap/instance checkpoint 来自新的 Stage 1/2。


## 4. 新 cache 设计

旧格式：

1. 大量 `.pt` block
2. 每块 `torch.save`
3. 训练靠 `torch.load`

新格式：

每个 split 使用少量大 shard，推荐 1GB 到 4GB 一个 shard。

### 4.1 `window_shard_raw`

字段：

1. `X.u8.mmap`
   - shape: `[N, C, L]`
   - dtype: `uint8`
2. `esa.f16.mmap`
   - shape: `[N]`
   - dtype: `float16`
3. `pos.f16.mmap`
   - shape: `[N]`
   - dtype: `float16`
4. `label.i8.mmap`
   - shape: `[N]`
   - 值：
     - `1` = positive
     - `0` = negative
     - `-1` = ignore/unlabeled
5. `pair_id.i32.mmap`
   - shape: `[N]`
6. `meta.json`
   - `num_samples`
   - `num_pairs`
   - `split`
   - `source_file`
   - `label_policy`
   - `esa_min_score`
   - `shard_id`

这个 cache 同时供 Stage 1 和 Stage 2 使用。


### 4.2 `teacher_logit_shard`（可选）

如果 Stage 2 不想每次重新跑 teacher：

1. `teacher_logit.f16.mmap`
2. `teacher_emb.f16.mmap`
3. `uid.i64.mmap`
4. `meta.json`

注意：

1. 这不是必须的第一步。
2. 只有当 Stage 2 teacher forward 成为瓶颈时才建议落盘。


## 5. 并行进程模型

### 5.1 CTS 生成阶段

使用 producer-consumer 架构。

#### CPU workers

职责：

1. 读取 pair 行
2. 滑窗扫描
3. 计算 ESA / pos / encode_RNA
4. 生成候选窗口
5. 根据 label policy 产生：
   - positive
   - negative
   - ignore

输出：

1. shard-ready 的 numpy / torch buffer

#### Writer process

职责：

1. 聚合 worker 输出
2. 顺序写入大 mmap shard
3. 定期 flush

这样可以：

1. 用满所有 vCPU
2. 避免大量小文件写入
3. 让 VepFS 承担顺序大文件传输


### 5.2 Stage 1/2 训练阶段

训练不再依赖 `ChunkedCTSDataset`。

改用：

1. `WindowShardDataset`
2. 或 `IterableWindowShardDataset`

要求：

1. worker 按 shard 顺序预取
2. 同一个 batch 尽量来自连续样本区域
3. 保持 `X` 为 `uint8`，到 GPU 后再转 `float`


## 6. 推荐新增文件

### 6.1 数据结构

1. `src/data/window_shard_cache.py`
   - shard writer / reader
2. `src/data/window_shard_dataset.py`
   - Stage 1/2 新 dataset
3. `src/data/window_shard_collate.py`
   - 保持 `uint8`

### 6.2 预处理

1. `src/precompute/window_shard_builder.py`
   - 单机并行 builder
2. `src/precompute/pseudo_site_labeler.py`
   - 生成 pseudo positive / ignore / negative
3. `src/precompute/window_shard_builder_parallel.py`
   - producer-consumer 版本

### 6.3 入口

1. `src/launch/build_window_shards.py`
2. `src/launch/train_targetnet_shard.py`
3. `src/launch/train_cheapcts_shard.py`
4. `src/launch/build_teacher_shards.py`（可选）

### 6.4 配置

1. `configs/experiment/MTI_TargetNet_Optimized_shard.yaml`
2. `configs/experiment/MTI_CheapCTSNet_shard.yaml`
3. `configs/experiment/MTI_build_window_shards.yaml`


## 7. 训练目标重定义

### 7.1 Stage 1

推荐支持 3 种模式：

1. `all_positive`
   - 仅做兼容 baseline
2. `pseudo_topr`
   - 正 pair 只给 top-r 窗口正标签
   - 其余 ignore
3. `ranking`
   - 正负窗口排序式训练

默认建议：

`pseudo_topr`

参数：

1. `pseudo_r`
2. `pseudo_score_source`
   - `esa`
   - `cheap_logit`
   - `hybrid`


### 7.2 Stage 2

cheap model 仍然蒸馏 teacher，但训练数据应与 Stage 1 一致：

1. 只对 `label != -1` 的窗口做 supervised / KD
2. ignore 样本不计 supervised loss


## 8. 性能优化优先级

### 优先级 1

去掉旧 `.pt block cache`，换成大 shard。

### 优先级 2

CTS 生成改成多进程 producer + 单 writer。

### 优先级 3

保持 `uint8` 到 GPU 后再 cast。

### 优先级 4

如果 teacher forward 成本过高，再加 `teacher_shard`。


## 9. 建议实施顺序

### Phase 1

先实现：

1. `window_shard_raw`
2. `build_window_shards.py`
3. `WindowShardDataset`
4. `train_targetnet_shard.py`

监督先支持：

1. `all_positive`
2. `pseudo_topr`

### Phase 2

再实现：

1. `train_cheapcts_shard.py`
2. `ignore` 样本的 distill 支持

### Phase 3

再做：

1. `ranking` 训练
2. teacher shard
3. EM 式 pseudo label 刷新


## 10. 预期收益

### 训练效果

如果不修监督，只修吞吐：

1. 训练会更快
2. 但 Stage 3 上限大概率仍卡在当前区间

如果同时修监督和吞吐：

1. instance encoder 的信息质量会明显改善
2. cheap selector 排序质量会改善
3. Stage 3 的 pair classifier 才有机会继续抬升

### 工程收益

1. VepFS 更友好
2. 小文件数量大幅减少
3. CPU/GPU 利用率更高
4. Stage 1/2 和 Stage 3 的 scalable 思路统一

