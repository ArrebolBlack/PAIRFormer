# 新服务器部署指南

本指南说明如何在新服务器上部署 PAIRFormer 代码，并构建缓存以进行训练。

## 准备工作

### 1. 源服务器文件

确保以下文件在源服务器的以下路径：
```
/data/MTI/MTI_pair_random_split.txt          (785MB - 原始数据)
checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/checkpoints/best.pt
checkpoints/MTI_TargetNet_Optimized_shard_v2_relabel_top4/best.pt
checkpoints/MTI_A_TargetNet_Optimized_shard_v3_compact_r4/checkpoints/best.pt
```

### 2. 源服务器访问

确保你可以从当前服务器访问源服务器：
```bash
# 测试连接
ping <SOURCE_SERVER_IP>
scp <SOURCE_USER>@<SOURCE_SERVER_IP>:<SOURCE_REPO_PATH>/data/MTI/MTI_pair_random_split.txt /tmp/test.txt
```

### 3. 目标服务器环境

- 操作系统：Linux
- Python：3.10+
- PyTorch：2.4.1+
- CUDA：可用（GPU）
- 存储空间：至少 500GB（用于缓存）

---

## 部署步骤

### 步骤 1：复制代码仓库

在目标服务器上执行：

```bash
# 1. 克隆或复制代码仓库
cd /vepfs-mlp2/mlp-public/haoce/yjq/
git pull  # 或使用 rsync 从源服务器同步

# 2. 验证文件存在
ls -la data/MTI/MTI_pair_random_split.txt
ls -la configs/experiment/MTI_EM_Scalable_selected_raw_parallel.yaml
```

如果使用 rsync 从源服务器同步：
```bash
rsync -avz --progress \
    --exclude=".git" \
    --exclude="cache" \
    --exclude="checkpoints" \
    --exclude="outputs" \
    --exclude="wandb" \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    <SOURCE_USER>@<SOURCE_SERVER_IP>:<SOURCE_REPO_PATH>/ \
    .
```

### 步骤 2：复制数据文件

#### 方法 A：使用 scp（单个文件）

```bash
# 复制原始数据文件
scp <SOURCE_USER>@<SOURCE_SERVER_IP>:<SOURCE_REPO_PATH>/data/MTI/MTI_pair_random_split.txt data/MTI/

# 复制 checkpoint 文件
scp <SOURCE_USER>@<SOURCE_SERVER_IP>:<SOURCE_REPO_PATH>/checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/checkpoints/best.pt checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/
scp <SOURCE_USER>@<SOURCE_SERVER_IP>:<SOURCE_REPO_PATH>/checkpoints/MTI_TargetNet_Optimized_shard_v2_relabel_top4/checkpoints/best.pt checkpoints/MTI_TargetNet_Optimized_shard_v2_relabel_top4/
```

#### 方法 B：使用 rsync（推荐）

```bash
# 只同步 data/ 目录（排除其他）
rsync -avz --progress \
    <SOURCE_USER>@<SOURCE_SERVER_IP>:<SOURCE_REPO_PATH>/data/MTI/ \
    data/MTI/

# 同步 checkpoint 文件
rsync -avz --progress \
    <SOURCE_USER>@<SOURCE_SERVER_IP>:<SOURCE_REPO_PATH>/checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/ \
    checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/

rsync -avz --progress \
    <SOURCE_USER>@<SOURCE_SERVER_IP>:<SOURCE_REPO_PATH>/checkpoints/MTI_TargetNet_Optimized_shard_v2_relabel_top4/ \
    checkpoints/MTI_TargetNet_Optimized_shard_v2_relabel_top4/
```

#### 方法 C：使用提供的脚本

```bash
bash scripts/setup_new_server.sh <SOURCE_SERVER_IP> <SOURCE_USER> <SOURCE_REPO_PATH>
```

这个脚本会自动：
1. 从源服务器复制代码仓库
2. 从源服务器复制数据文件
3. 从源服务器复制 checkpoint 文件
4. 创建必要的目录结构

### 步骤 3：验证文件完整性

```bash
# 检查数据文件
ls -lh data/MTI/MTI_pair_random_split.txt

# 检查 checkpoint 文件
ls -lh checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/checkpoints/best.pt
ls -lh checkpoints/MTI_TargetNet_Optimized_shard_v2_relabel_top4/checkpoints/best.pt

# 检查实验配置
ls -la configs/experiment/MTI_EM_Scalable_selected_raw_parallel.yaml
ls -la configs/experiment/MTI_build_selected_inst.yaml
```

---

## 构建缓存

### 步骤 1：构建 selected_raw 缓存

**耗时**：约 2 小时（3 个 split）

```bash
# 单卡模式
CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_pair_cache_parallel \
    experiment=MTI_EM_Scalable_selected_raw_parallel \
    scalable.cache_root=cache/mti_full_topk_retrain_r4_v3relbl \
    cheap_ckpt_path=checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/checkpoints/best.pt \
    run.split=train run.kmax=64 scalable.num_pairs_hint=480000
```

**监控 GPU 使用**：
```bash
# 终端 1：监控
watch -n 1 nvidia-smi

# 终端 2：查看 GPU 内存
nvidia-smi
```

### 步骤 2：构建 selected_inst 缓存

**耗时**：约 6 分钟（3 个 split）

```bash
# 单卡模式（或修改为 DDP 多卡）
CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_inst_cache \
    experiment=MTI_build_selected_inst \
    scalable.cache_root=cache/mti_full_topk_retrain_r4_v3relbl \
    instance_ckpt_path=checkpoints/MTI_TargetNet_Optimized_shard_v2_relabel_top4/checkpoints/best.pt \
    run.split=train run.batch_size=4096 \
    instance_model.num_channels=[64,64,128,128] \
    instance_model.num_blocks=[3,3,3] \
    instance_model.multi_scale=true \
    instance_model.se_type=cbam \
    instance_model.use_bn=true \
    instance_model.dropout=0.1
```

**预期输出**：
```
每个 split 的缓存文件：
cache/mti_full_topk_retrain_r4_v3relbl/
├── selection/
│   ├── meta.json
│   ├── sel_uids.i32.mmap
│   └── sel_len.i16.mmap
└── selected_raw/
    ├── X.u8.mmap
    ├── esa.f16.mmap
    ├── pos.f16.mmap
    ├── label.f32.mmap
    ├── cheap_logit.f16.mmap
    └── meta.json
```

### 步骤 3：验证缓存构建

```bash
# 检查 meta.json 文件
find cache/mti_full_topk_retrain_r4_v3relbl/selection -name meta.json
find cache/mti_full_topk_retrain_r4_v3relbl/selected_raw -name meta.json

# 查看文件大小
du -sh cache/mti_full_topk_retrain_r4_v3relbl/
```

---

## 训练

### 单卡训练（推荐用于调试）

```bash
python -m src.launch.train_pair_selected_inst experiment=MTI_train_selected_inst
```

### 多卡训练（推荐用于生产）

```bash
# 4 GPU 训练
bash scripts/run_ddp_train_pair_selected.sh 4

# 8 GPU 训练
bash scripts/run_ddp_train_pair_selected.sh 8
```

**训练时间估计**：
- 单卡：约 5-6 小时
- 4 卡：约 1.5-2 小时
- 8 卡：约 45 分钟 - 1 小时

---

## 故障排除

### 问题 1：找不到 checkpoint 文件

```bash
# 检查路径
ls -lh checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/checkpoints/best.pt

# 检查软链接
readlink -f checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/checkpoints/best.pt
```

### 问题 2：数据文件不存在

```bash
# 检查数据目录
ls -la data/MTI/

# 如果需要重新生成随机 split
python -m src.data.generate_random_split
```

### 问题 3：GPU 内存不足

```bash
# 降低 batch size
python -m src.launch.train_pair_selected_inst \
    experiment=MTI_train_selected_inst \
    run.batch_size=16  # 从 32 降低到 16
```

### 问题 4：缓存构建失败

```bash
# 检查 cheap model 是否正确加载
python -c "import torch; m=torch.load('checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/checkpoints/best.pt', map_location='cpu'); print(m.keys())"

# 手动构建一个 split 验证
CUDA_VISIBLE_DEVICES=0 python -m src.launch.build_selected_pair_cache_parallel \
    experiment=MTI_EM_Scalable_selected_raw_parallel \
    scalable.cache_root=cache/test_run \
    run.split=train run.kmax=16 scalable.num_pairs_hint=1000 \
    cheap_ckpt_path=checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/checkpoints/best.pt
```

---

## 性能优化建议

### 1. 数据加载优化

- 使用 `num_workers=8-16` 并行数据加载
- 启用 `persistent_workers=True`
- 使用 SSD 存储缓存（如果可用）

### 2. GPU 利用优化

- 根据可用显存调整 `batch_size`
- 启用混合精度训练（`run.use_amp: true`）
- 使用 `nccl` 后端（NVIDIA GPU）

### 3. 缓存优化

- 将缓存放在 SSD 上（而非 HDD）
- 使用合适的 chunk size 平衡 I/O 和内存

---

## 配置调整建议

### 根据可用 GPU 调整

| GPU 数量 | batch_size (train_pair) | batch_size (build_inst) | num_workers |
|---------|------------------------|---------------------|-----------|
| 1       | 32                         | 2048                           | 4         |
| 2       | 32                         | 4096                           | 8         |
| 4       | 32                         | 4096                           | 16        |
| 8       | 64                         | 4096                           | 16        |

### 根据显存调整

| 显存（GB） | batch_size (train_pair) | batch_size (build_inst) |
|-----------|------------------------|---------------------|
| 16       | 64                         | 2048                           |
| 24       | 32                         | 4096                           |
| 32       | 16                         | 2048                           |
| 40       | 16                         | 1024                           |

---

## 监控和日志

### 实时监控

```bash
# 终端 1：GPU 使用
watch -n 1 nvidia-smi

# 终端 2：进程监控
top -p $(pgrep python | awk '{print $1}')
```

### 查看训练日志

```bash
# 训练日志
tail -f logs/MTI_train_selected_inst/<timestamp>/train.log

# 验证日志
tail -f logs/MTI_train_selected_inst/<timestamp>/val.log
```

---

## 清理和重置

### 清理旧缓存

```bash
# 删除旧缓存
rm -rf cache/mti_full_topk_retrain_r4_v3relbl/
rm -rf cache/mti_full_topk_retrain_r4_v3relbl/selection/
rm -rf cache/mti_full_topk_retrain_r4_v3relbl/selected_raw/
```

### 重置实验

```bash
# 清理旧输出
rm -rf outputs/MTI_train_selected_inst/

# 删除旧 checkpoint
rm -rf checkpoints/MTI_train_selected_inst/

# 重新开始训练
python -m src.launch.train_pair_selected_inst experiment=MTI_train_selected_inst
```

---

## 快速命令参考

### 完整流程（从零到训练）

```bash
# 1. 复制文件（如果还没复制）
bash scripts/setup_new_server.sh <SOURCE_SERVER_IP> <SOURCE_USER> <SOURCE_REPO_PATH>

# 2. 构建缓存（约 2 小时）
bash scripts/run_cache_build_new_server.sh

# 3. 训练（约 5 小时）
bash scripts/run_ddp_train_pair_selected.sh 8
```

### 仅训练（假设文件已就绪）

```bash
# 单卡调试
python -m src.launch.train_pair_selected_inst experiment=MTI_train_selected_inst

# 8卡生产
bash scripts/run_ddp_train_pair_selected.sh 8
```

---

## 预期时间线

| 阶段 | 耗时 | 累计耗时 |
|------|------|---------|
| 复制文件 | 10-30 分钟 | 10-30 分钟 |
| 构建缓存 | 2 小时 | 2 小时 30 分钟 |
| 训练（8 卡） | 1 小时 | 4 小时 |
| **总计** | **~4-5 小时** | **从源代码到开始训练** |

---

## 联系和反馈

如果遇到问题：

1. 检查源服务器上的文件路径是否正确
2. 确认网络连接稳定（用于 rsync/scp）
3. 验证 GPU 驱动版本兼容性
4. 查看完整的错误堆栈信息

```bash
# 生成环境报告
echo "=== 环境报告 ===" > env_report.txt
echo "GPU 信息:" >> env_report.txt
nvidia-smi >> env_report.txt
echo "CUDA 信息:" >> env_report.txt
nvcc --version >> env_report.txt
echo "Python 信息:" >> env_report.txt
python --version >> env_report.txt
echo "文件系统信息:" >> env_report.txt
df -h >> env_report.txt

cat env_report.txt
```
