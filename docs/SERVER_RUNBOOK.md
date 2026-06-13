# 火山云 服务器运行手册（PAIRFormer · refactor/2026-06）

> 目标：让你在火山引擎 veMLP 上**顺畅地 (a) 在开发机 (1×A100) debug 所有实验，
> (b) 把可吃 8 卡的重训练提交为自定义任务，硬件效率（CPU/GPU）拉满**。
> 平台细节依据 `D:\2026中关村访学\火山云机器学习平台开发指南`（01/02/09 篇）。
> 实验清单见仓库 `EXPERIMENTS.md`（E1–E18）。本手册给出每条的 debug 命令 + 8 卡提交方式。

---

## 0. 你的平台真实值（直接抄，来自指南 09 篇）

| 项 | 值 |
|---|---|
| 镜像 | `cr-mlp-cn-beijing.cr.volces.com/public/yjq:latest` |
| 资源队列 ID | `q-20260318205115-lvfng`（queue010）；Priority ≤ 4 |
| vePFS ID / 挂载 | `vepfs-cnbj2c98dea54433` → `/vepfs-mlp2/queue010/20252203765` |
| TOS 个人目录 | `tos://tos-mlp-zgci/yinjiaqi/` |
| 开发机规格 | `ml.pni2.3xlarge`（1×A100-80G / 14 核 / 245G） |
| 8 卡规格 | 单机 8×A100-80G（NVLink，**无需 hpc/RDMA**），UI 下拉确认（应为 `ml.pni2.28xlarge` 一类） |
| 区域 | cn-beijing-b；CUDA 12.2 / 驱动 535 |
| torchrun 注入变量 | `$MLP_WORKER_NUM`(--nnodes) `$MLP_WORKER_GPU`(--nproc_per_node) `$MLP_ROLE_INDEX`(--node_rank) `$MLP_WORKER_0_HOST/_PORT`(--master_addr/port) |

约定本手册用：`$VEP=/vepfs-mlp2/queue010/20252203765`；代码放 `$VEP/PAIRFormer`，
conda 环境 `$VEP/envs/pairformer`，缓存/输出/ckpt 全部写 `$VEP/...`（**不要写云盘、不要直接读 TOS fuse**）。

---

## 1. 一次性环境准备（开发机内）

```bash
VEP=/vepfs-mlp2/queue010/20252203765
pip config set global.index-url https://mirrors.ivolces.com/pypi/simple/   # 内网源

# 代码：clone 到 vePFS，切到重构分支
cd $VEP && git clone <repo-url> PAIRFormer && cd PAIRFormer && git checkout refactor/2026-06
git submodule update --init   # external/Mimosa, external/TargetNet（E1 baseline 才需要）

# conda 环境建在 vePFS（开发机和任务容器共用）
conda create -p $VEP/envs/pairformer python=3.10 -y && conda activate $VEP/envs/pairformer
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.in        # 已修 hydra==2.5、补 eval 依赖（见 README）

# 数据：从 TOS 预热到 vePFS（内网 tosutil，多一个 i）
tosutil config -e=tos-cn-beijing.ivolces.com -re=cn-beijing -i=<AK> -k=<SK>
tosutil cp -r -j 32 -p 16 -u tos://tos-mlp-zgci/yinjiaqi/data/ $VEP/PAIRFormer/data/
# checkpoints 同理（cheap/instance/BR-MIL ckpt，见 EXPERIMENTS.md §0）
```

冒烟自检（确认环境通）：`python tools/refactor_verify/dump_configs.py --repo . --out /tmp/c && python tools/refactor_verify/cfgdiff.py --dir tools/refactor_verify/golden_configs /tmp/c`（应 77/77 0 differ），再 `python tests/test_pair_aggregators_golden.py`。

---

## 2. 硬件效率开关（"拉满" 清单）—— 本次重构已实装

> 这些是**改变数值的性能优化**，默认关（保证等价基线）；真实训练时**全部打开**。

| 开关 | 怎么开 | 作用 | 注意 |
|---|---|---|---|
| **FlashAttention (SDPA)** | 环境变量 `PF_EFFICIENT_ATTN=1` | Set/Pair Transformer 注意力走 `scaled_dot_product_attention` → A100 上 Flash/mem-efficient 内核，**K=512/1024 时显著加速+省显存** | 数值与手写路径不同（flash reduction）；本次重构已实装为 opt-in |
| **TF32 + cudnn.benchmark** | 环境变量 `PF_DETERMINISTIC=0` | A100 matmul/conv 用 TF32，大幅提速 | 数值略变；仅真实训练用 |
| **AMP（混合精度）** | 配置 `trainer_em.use_amp=true`（pair: `trainer_pair_selected.use_amp=true`；window: `train.amp=true`） | fp16 autocast，省显存+提速 | 已支持；如遇不稳可改 bf16（小改 autocast dtype） |
| **DataLoader 打满 CPU** | `run.num_workers=<≈核数>`（开发机 14；8 卡机按规格核数/进程数分配，建议每进程 6–8）、`run.pin_memory=true`、`run.persistent_workers=true` | CPU 预取喂满 GPU | 数据须在 vePFS（非 TOS fuse） |
| **加大 batch** | `run.batch_size=<更大>`（A100-80G 余量大；K=64 可 256→512） | 提升 GPU 利用率 | 多卡时是 per-GPU batch，全局= per-GPU×卡数；LR 按需线性缩放 |
| **DDP 单机 8 卡** | `torchrun --nproc_per_node=8`（NVLink，无需 RDMA） | 近线性加速 | `find_unused_parameters` 未开（聚合器全参数参与，安全且快） |

**统一的"满效率"环境变量块**（每次真实训练都带上）：
```bash
export PF_EFFICIENT_ATTN=1      # FlashAttention / SDPA（默认不设=手写注意力，等价基线用）
export PF_DETERMINISTIC=0       # TF32 + cudnn.benchmark（快；默认不设=确定性，等价基线用）
export OMP_NUM_THREADS=8        # 每进程 CPU 线程数（≈核数/每机进程数），防线程争抢
export NCCL_DEBUG=WARN          # 首次可设 INFO 确认走 NVLink
```
> 进一步可选：`torch.compile`（本次未默认开，有 `_orig_mod.` ckpt 前缀坑）——若要用，单独提需求我加成默认关开关。

---

## 3. 实验分类：开发机 debug vs 8 卡提交

| ID | 实验 | 类型 | 跑法 |
|---|---|---|---|
| E1 | miRAW 10-fold 主结果 | **8 卡**（10 fold 可并行/串行） | 提交（§5）；baseline TargetNet/Mimosa 是独立脚本，开发机 debug |
| E2 | deepTargetPro 外部验证 | **8 卡** | 提交 |
| E3 | 大规模 MTI K∈{64..512} | **8 卡（重点）** | 提交，每 K 一个任务或一任务内循环 |
| E4 | K-sweep retrain/truncate | **8 卡** | 提交（按 K 矩阵） |
| E5 | robustness vs n（eval-only） | **开发机** | 单卡 eval，用 K=512 ckpt |
| E6 | 运行时/显存 bench | **开发机** | 单卡 `bench_compute_vs_k` |
| E7 | 聚合器家族消融 | **8 卡** | 提交（3 个 arch） |
| E8 | SetTransformer 容量消融 | **8 卡** | 提交 |
| E9 | CTS 编码器缩放 | 开发机/8 卡 | Stage-1 训练，按规模定 |
| E10 | 跨域 CAMELYON16/Musk2 | 开发机（多为小数据/脚本） | `scripts/rebuttal/exp6_*` |
| E11 | n-分布统计 | **开发机** | `scripts/rebuttal/n_distribution/*`（纯分析） |
| E12 | 小预算选择器消融 | 开发机/8 卡 | `scripts/rebuttal/selector_ablation/*` |
| E13 | K=1 基线 | 开发机/8 卡 | `run.kmax=1` |
| E14 | 无正样本重叠 eval | **开发机** | `eval_em experiment=rebuttal_eval_no_pos_overlap` |
| E15 | 划分敏感性 | **8 卡**（多划分×种子） | `scripts/rebuttal/split_sensitivity/*` |
| E16 | cheap recall@K | **开发机** | `scripts/rebuttal/cheap_ranking_v2/*`（分析） |
| E17 | rel-distill 消融 | 开发机 | `CheapCTSNet` vs `_noRelDistill` |
| E18 | k1_ratio 消融 | 开发机/8 卡 | `scripts/rebuttal/k1_ratio_ablation/*` |

---

## 4. 开发机 debug（单卡，1×A100）

进入开发机、`conda activate $VEP/envs/pairformer && cd $VEP/PAIRFormer`，带满效率开关：
```bash
export PF_EFFICIENT_ATTN=1 PF_DETERMINISTIC=0
# 例：E3 MTI K=64 单卡 debug（小步先确认跑通，再提交 8 卡）
python -m src.launch.train_em experiment=MTI_train_selected_inst run.kmax=64 \
  trainer_em.use_amp=true run.num_workers=12 run.batch_size=64 \
  logging.wandb.enabled=false hydra.run.dir=$VEP/runs/dbg_mti_k64

# 例：E6 bench（单卡，已就位 bench_fig3 数据）
python -m src.launch.bench_compute_vs_k experiment=miRAW_EM_Pipeline \
  run.bench_pair_ids_path=paper/figures/bench_fig3/pair_ids_test_1024.json

# 例：E14 无正样本重叠 eval（单卡）
python scripts/rebuttal/create_no_pos_overlap_test.py
python -m src.launch.eval_em experiment=rebuttal_eval_no_pos_overlap run.checkpoint=checkpoints/BR-MIL/checkpoints/best.pt
```
开发机只跑小步/eval 确认无误（GPU 8h 利用率<20% 会被自动关机）；重训练务必提交任务。

---

## 5. 提交 8 卡自定义任务

### 5.1 入口脚本（幂等续训，放 `scripts/task_entry.sh`）

把它做成幂等：每次启动自动从 `last.pt` 续训（配合 Preemptible+Retry）。
```bash
#!/usr/bin/env bash
set -euo pipefail
cd /vepfs-mlp2/queue010/20252203765/PAIRFormer
source activate /vepfs-mlp2/queue010/20252203765/envs/pairformer
export PF_EFFICIENT_ATTN=1 PF_DETERMINISTIC=0 OMP_NUM_THREADS=8 NCCL_DEBUG=WARN
EXP=${EXP:-MTI_train_selected_inst}; KMAX=${KMAX:-512}
RUNDIR=/vepfs-mlp2/queue010/20252203765/runs/$EXP-k$KMAX
CKPT=$RUNDIR/checkpoints/last.pt
RESUME=""; [ -f "$CKPT" ] && RESUME="run.resume=true run.checkpoint=$CKPT"   # 幂等续训
torchrun --nnodes=$MLP_WORKER_NUM --nproc_per_node=$MLP_WORKER_GPU \
  --node_rank=$MLP_ROLE_INDEX --master_addr=$MLP_WORKER_0_HOST --master_port=$MLP_WORKER_0_PORT \
  -m src.launch.train_em experiment=$EXP run.kmax=$KMAX \
  trainer_em.use_amp=true run.num_workers=8 run.batch_size=128 \
  logging.wandb.enabled=false hydra.run.dir=$RUNDIR $RESUME
```
> 说明：`run.batch_size` 是**每卡** batch；8 卡全局 = 8×。LR 可按需缩放（见 docs/DDP_TRAINING.md）。
> 单机 8 卡走 NVLink，不需要 hpc/RDMA 镜像。

### 5.2 任务 YAML（控制台 GetJob 导出格式，来自指南 09；首次 CLI 提交若报字段名错按别名换）

`task-mti-k512.yaml`：
```yaml
Name: pairformer-mti-k512
ImageSpec: {Type: Custom, Url: cr-mlp-cn-beijing.cr.volces.com/public/yjq:latest}
EntrypointPath: "bash scripts/task_entry.sh"
Framework: PyTorchDDP
ResourceQueueId: q-20260318205115-lvfng
Priority: 4
ActiveDeadlineSeconds: 864000
Envs:
  - {Name: EXP, Value: MTI_train_selected_inst, IsPrivate: false}
  - {Name: KMAX, Value: "512", IsPrivate: false}
TaskRoleSpecs:
  - {RoleName: worker, RoleReplicas: 1, ResourceSpecId: ml.pni2.28xlarge}   # 单机 8 卡（UI 确认规格名）
Storages:
  - {Type: Vepfs, VepfsId: vepfs-cnbj2c98dea54433, MountPath: /vepfs-mlp2/queue010/20252203765, SubPath: queue010/20252203765, ReadOnly: false}
Preemptible: true
CacheType: Cloudfs
RetryOptions:
  EnableRetry: true
  MaxRetryTimes: 5
  IntervalSeconds: 120
  EnableReserveResourceOnRetry: true
  PolicySets: [InstanceReclaimed, Failed, MachineFailure, VecclHangDetectionFailed, LogDetectionFailed, PythonDetectionFailed]
```
提交：`volc configure`（首次填 AK/SK + region cn-beijing）→ `volc ml_task submit -c task-mti-k512.yaml`。
监控：`volc ml_task logs --id <id>`；控制台看 GPU/NVLink 利用率曲线确认"拉满"。

### 5.3 批量（E3/E4 K-sweep / E7 聚合器 / E15 划分）
复制上面的 YAML，改 `Name` + `Envs`（如 `KMAX=64/128/256`，或 `EXP=MTI_PairCNNAggregator` 等），各提交一个任务（小任务有插队优势，可并行排队）。每 K 一个任务最简单。

---

## 6. 验证 DDP 修复（必跑一次）
本次重构修了 PairSelected 的多卡崩溃，请先在 8 卡机跑：
```bash
bash scripts/verify_ddp_pair_selected.sh 8     # 机制自测 + 单卡 vs DDP + ckpt 互通
```
EM 路径（train_em）的多卡早已可用。

---

## 7. 收尾
- ckpt 归档：`tosutil cp -r -j 32 -p 16 -u $VEP/runs/<run>/ tos://tos-mlp-zgci/yinjiaqi/ckpts/<run>/`
- 跑完把关键指标回填 `EXPERIMENTS.md`（把 UNKNOWN 项的实际命令/资源/结果补全）。
- 开发机长期不用就删（云盘只放系统，删无损失）。
