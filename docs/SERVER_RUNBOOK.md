# 火山云 服务器运行手册（PAIRFormer · refactor/2026-06）

> 目标：进开发机就能**照抄运行**——(a) 1×A100 debug 全部实验，(b) 把吃 8 卡的重训练提交为自定义任务，CPU/GPU 拉满。
> 平台细节据 `D:\2026中关村访学\火山云机器学习平台开发指南`（01/02/09 篇）；实验清单见 `EXPERIMENTS.md`（E1–E18）。
> 本手册每条命令都已对仓库实际的 launch 入口 / experiment 配置 / override 键 / 脚本路径核对过（refactor/2026-06）。
> **本轮检查改动**：补全前置 ckpt/数据清单、缓存效率配方、修正 AMP 键名、E1–E18 全部映射到仓库现成脚本。

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

约定：`$VEP=/vepfs-mlp2/queue010/20252203765`；代码 `$VEP/PAIRFormer`、conda `$VEP/envs/pairformer`、
缓存/输出/ckpt 全写 `$VEP/...`（**不要写云盘、不要直读 TOS fuse**）。

---

## 1. 一次性环境准备（开发机内）

```bash
VEP=/vepfs-mlp2/queue010/20252203765
pip config set global.index-url https://mirrors.ivolces.com/pypi/simple/

# 代码（含 LFS：ckpt + 数据切分都是 LFS 对象，务必 lfs pull）
cd $VEP && git clone <repo-url> PAIRFormer && cd PAIRFormer && git checkout refactor/2026-06
git lfs install && git lfs pull            # 拉 checkpoints/* 与 data/* 真实文件（否则是指针）
git submodule update --init                # external/Mimosa, external/TargetNet（仅 E1 baseline 需要）

# conda（建在 vePFS，开发机和任务容器共用同一份）
conda create -p $VEP/envs/pairformer python=3.10 -y && conda activate $VEP/envs/pairformer
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.in             # 已修 hydra 错包、补 eval 依赖（见 README）
```

### 1.1 前置产物自检（不齐就先补，否则 EM pipeline 起不来）

EM 训练（train_em）启动时会**自动现建缓存（仅 rank0）**，但它依赖两个**预训练 ckpt** 和原始数据：

```bash
# (1) EM 必需的 cheap / instance ckpt（em_core.yaml:225/229 写死引用，缺则从随机权重起，结果不对）
ls checkpoints/CheapCTSNet/checkpoints/last.pt                    # cheap 模型
ls checkpoints/miRAW_TargetNet_Optimized_dp-0.1/checkpoints/last.pt  # instance 模型
ls checkpoints/BR-MIL/checkpoints/best.pt                         # miRAW eval（E13/E14）用
# (2) 数据（仓库已带；确认非 LFS 指针 = 文件 >1KB）
ls -la data/miRAW_Train_Validation.txt data/MTI/MTI_pair_random_split.txt
ls data/rebuttal/miRAW_fold1_split/ data/rebuttal/miRAW_8020_balanced/   # 各 plan 脚本要的切分
# (3) 配置等价闸门（确认环境/分支没坏）
python tools/refactor_verify/dump_configs.py --repo . --out /tmp/c
python tools/refactor_verify/cfgdiff.py --dir tools/refactor_verify/golden_configs /tmp/c   # 应 77/77 0 differ
python tests/test_pair_aggregators_golden.py
```

> 缺 MTI 大规模专属数据（`data/MTI/MTI_CTS_train_val.txt`、`data/siRNA_raw_data/*`）或 MTI 训练好的 ckpt
> （`MTI_v3_*`，由 E3 产出）时，对应实验要先在服务器训出来——见 §3 状态列。

---

## 2. 硬件效率开关（"拉满"清单）—— 本次重构已实装

> 这些多为**改数值的性能优化**，默认关（保等价基线）；真实训练时按下表全开。

| 开关 | 怎么开 | 作用 | 注意 |
|---|---|---|---|
| **FlashAttention (SDPA)** | 环境变量 `PF_EFFICIENT_ATTN=1` | Set/Pair Transformer 走 `scaled_dot_product_attention`→A100 Flash 内核，**K=512/1024 显著加速+省显存** | 数值与手写路径不同；已实装 opt-in |
| **TF32 + cudnn.benchmark** | 环境变量 `PF_DETERMINISTIC=0` | A100 matmul/conv 用 TF32 | 数值略变；仅真实训练用 |
| **AMP（EM/实例训练）** | `trainer_em.use_amp=true`（默认 false） | fp16 autocast，省显存+提速 | 键已核对 `em_core.yaml`；pair 路径用 `trainer_pair_selected.use_amp=true` |
| **AMP（建缓存阶段）** | `em.cheap_cache.amp`（默认 **true**）、`em.instance_cache.use_amp`（默认 **true**） | 建 cheap/instance 缓存时混精 | 默认已开，无需动；不稳时设 false |
| **DataLoader 打满 CPU** | `run.num_workers=<≈核数>`（开发机 12、8 卡机每进程 6–8）+ `run.pin_memory=true` + `run.persistent_workers=true` | CPU 预取喂满 GPU | 数据须在 vePFS（非 TOS fuse） |
| **加大 batch** | `run.batch_size=<更大>`（A100-80G；K=64 可 256→512） | 提升 GPU 利用率 | 多卡为**每卡** batch，全局=×卡数；LR 按需线性缩放 |
| **DDP 单机 8 卡** | `torchrun --nproc_per_node=8`（NVLink） | 近线性加速 | `find_unused_parameters` 未开（全参数参与，安全且快） |
| **ESA 比对加速** | **自动**（本轮 lru 记忆化，逐位等价） | 建缓存时跳过重复 miRNA/区段比对 | 无需开关；MTI 全 UTR 扫描路径受益最大，详见 `docs/PERF_AUDIT.md` |

**统一"满效率"环境变量块**（真实训练每次都带）：
```bash
export PF_EFFICIENT_ATTN=1      # FlashAttention/SDPA（默认不设=手写注意力=等价基线）
export PF_DETERMINISTIC=0       # TF32+cudnn.benchmark（默认不设=确定性=等价基线）
export OMP_NUM_THREADS=8        # 每进程 CPU 线程（≈核数/每机进程数），防争抢
export NCCL_DEBUG=WARN          # 首次可设 INFO 确认走 NVLink
```
> 可选 `torch.compile`：本轮未默认开（有 `_orig_mod.` ckpt 前缀坑），见 PERF_AUDIT §G3。

---

## 2.5 缓存策略（效率与续训的关键）

train_em 的缓存（cheap→selection→instance memmap）**默认随训练 rank0 现建**（`src/launch/train_em.py:490-536`）。三种用法，按场景选：

| 场景 | 配置 | 说明 |
|---|---|---|
| **首跑/debug** | 默认即可（`em.bootstrap.enabled=true`） | 自动建缓存再训；小步确认跑通 |
| **8 卡正式跑（推荐）** | 先建一次缓存 → 再训时 `em.bootstrap.enabled=false run.force_overwrite_bootstrap=false run.train_instance_mode=cached run.val_instance_mode=cached` | 避免 8 卡任务把 GPU 时间耗在串行建缓存上；缓存放 vePFS 复用 |
| **极致吞吐（大 K）** | 追加 `em_cache_root=/dev/shm/cache_kXXX` | 缓存放内存盘（RAM），消除 IO 瓶颈（参考 `scripts/auto_resume_k1024.sh` 的 K1024 配方）|

**幂等续训**：train_em 认 `run.resume=true run.checkpoint=<last.pt>`（`train_em.py:856`）。配合任务的 Preemptible+Retry，
入口脚本检测到 `last.pt` 就自动续（见 §5.1）。`scripts/auto_resume_k1024.sh` 是现成的"每 30 分钟自动续训"模板
（⚠️ **内含别人环境的写死路径/`DDP_BACKEND=gloo`/`LD_PRELOAD`，照抄前务必改成你的 `$VEP` 路径**）。

---

## 3. 实验总表 E1–E18：类型 + 现成脚本 + 状态

> "脚本"列优先用仓库已有的（已核对存在）；没有现成脚本的给 §4/§5 的命令。

| ID | 实验 | 类型 | 现成脚本 / 入口（已核对存在） |
|---|---|---|---|
| E1 | miRAW 10-fold 主结果 | **8 卡** | `scripts/rebuttal/tuning/plan_D_10fold_final.sh`；baseline `scripts/rebuttal/baselines/eval_{targetnet,mimosa,maxpool_10fold}.py`（开发机）|
| E2 | deepTargetPro 外部验证 | **8 卡** | `scripts/rebuttal/tuning/plan_D_deepTargetPro_10fold.sh` / `plan_deepTargetPro_full_pipeline.sh`；transfer `scripts/run_seed{2025,2026}_ext150.sh` |
| E3 | 大规模 MTI K∈{64..1024} | **8 卡（重点）** | `bash scripts/run_ddp_train_em.sh 8 MTI_EM_K512`；K1024 用 `scripts/auto_resume_k1024.sh`（改路径） |
| E4 | K-sweep retrain/truncate | **8 卡** | `scripts/rebuttal/tuning/plan_A_k_sweep.sh`；truncate `scripts/rebuttal/k_sensitivity/run_k_sensitivity.sh` |
| E5 | robustness vs n（eval-only） | **开发机** | `scripts/run_robustness_vs_n.sh 0 512 <MTI_K512_ckpt>`（需 MTI 训好的 ckpt）|
| E6 | 运行时/显存 bench | **开发机** | `python -m src.launch.bench_compute_vs_k`（§4）|
| E7 | 聚合器家族 CNN/GNN/SAB | **8 卡** | `run_ddp_train_em.sh 8 MTI_PairCNNAggregator` / `MTI_PairGNNAggregator` / `MTI_train_selected_inst model.block_type=sab` |
| E8 | SetTransformer 容量 | **8 卡** | `plan_B1_model_8xA100.sh`（容量）+ `plan_C1_training_8xA100.sh`（bs×warmup）；2 卡/5090 用 `_B2/_B3 _C2/_C3` |
| E9 | CTS 编码器缩放 | 开发机/8 卡 | `src.launch.train experiment=MTI_TargetNet_Optimized`（§4，改 channels/blocks）|
| E10 | 跨域 CAMELYON16/Musk2 | **开发机** | `scripts/rebuttal/exp6_camelyon16_v4.py`；`scripts/rebuttal/exp6-musk2/run_sweep.sh` |
| E11 | n-分布统计 | **开发机** | `scripts/rebuttal/n_distribution/compute_n_distribution.py` → `plot_n_distribution.py` |
| E12 | 小预算选择器消融 | 开发机/8 卡 | `scripts/rebuttal/selector_ablation/generate_configs.py` → `run_priority_experiments.sh` |
| E13 | K=1 基线 | 开发机 | `run.kmax=1`（§4）；`scripts/rebuttal/eval_k1_online.py` |
| E14 | 无正样本重叠 eval | **开发机** | `scripts/rebuttal/create_no_pos_overlap_test.py` → `eval_em experiment=rebuttal_eval_no_pos_overlap` |
| E15 | 划分敏感性 | **8 卡** | `scripts/rebuttal/split_sensitivity/create_split_data.py` → `run_all.sh` |
| E16 | cheap recall@K | **开发机** | `scripts/rebuttal/cheap_ranking_v2/cheap_ranking_v2.py` |
| E17 | rel-distill 消融 | 开发机 | `train experiment=CheapCTSNet` vs `=CheapCTSNet_noRelDistill`；`scripts/rebuttal/rel_distill_ablation/run_all.sh` |
| E18 | k1_ratio 消融 | 开发机/8 卡 | `scripts/rebuttal/k1_ratio_ablation/run_all.sh` |

---

## 4. 开发机 debug（单卡 1×A100，照抄）

```bash
conda activate $VEP/envs/pairformer && cd $VEP/PAIRFormer
export PF_EFFICIENT_ATTN=1 PF_DETERMINISTIC=0     # debug 也可拉满；要对等价基线时不设这两个

# —— E3 MTI 单卡小步确认（再提 8 卡）——
python -m src.launch.train_em experiment=MTI_train_selected_inst run.kmax=64 \
  trainer_em.use_amp=true run.num_workers=12 run.batch_size=64 \
  run.num_epochs=1 logging.wandb.enabled=false hydra.run.dir=$VEP/runs/dbg_mti_k64

# —— E6 bench（数据已就位 paper/figures/bench_fig3/）——
python -m src.launch.bench_compute_vs_k experiment=miRAW_EM_Pipeline \
  run.bench_pair_ids_path=paper/figures/bench_fig3/pair_ids_test_1024.json

# —— E13 K=1 基线 ——
python -m src.launch.train_em experiment=miRAW_EM_Pipeline run.kmax=1 \
  trainer_em.use_amp=true logging.wandb.enabled=false

# —— E14 无正样本重叠 eval ——
python scripts/rebuttal/create_no_pos_overlap_test.py
python -m src.launch.eval_em experiment=rebuttal_eval_no_pos_overlap \
  run.checkpoint=checkpoints/BR-MIL/checkpoints/best.pt

# —— E5 robustness vs n（eval-only，需 MTI K512 ckpt）——
bash scripts/run_robustness_vs_n.sh 0 512 checkpoints/<你的_MTI_K512>/best.pt

# —— E9 Stage-1 CTS 编码器缩放（注意入口是 train，不是 train_em）——
python -m src.launch.train experiment=MTI_TargetNet_Optimized \
  instance_model.num_channels=[64,64,128,128] instance_model.num_blocks=[3,3,3,3]

# —— E10/E11/E16/E17/E18：直接跑对应脚本（见 §3 表）——
python scripts/rebuttal/exp6_camelyon16_v4.py
python scripts/rebuttal/n_distribution/compute_n_distribution.py
bash   scripts/rebuttal/rel_distill_ablation/run_all.sh
```
> 开发机只跑小步/eval（GPU 8h 利用率<20% 会被自动关机）；重训练务必提交任务（§5）。

---

## 5. 提交 8 卡自定义任务

### 5.1 入口脚本 `scripts/task_entry.sh`（幂等续训，已随仓库）

```bash
#!/usr/bin/env bash
set -euo pipefail
VEP=/vepfs-mlp2/queue010/20252203765
cd $VEP/PAIRFormer && source activate $VEP/envs/pairformer
export PF_EFFICIENT_ATTN=1 PF_DETERMINISTIC=0 OMP_NUM_THREADS=8 NCCL_DEBUG=WARN
EXP=${EXP:-MTI_EM_K512}; KMAX=${KMAX:-512}
RUNDIR=$VEP/runs/$EXP-k$KMAX; CKPT=$RUNDIR/checkpoints/last.pt
RESUME=""; [ -f "$CKPT" ] && RESUME="run.resume=true run.checkpoint=$CKPT"   # 幂等续训
torchrun --nnodes=$MLP_WORKER_NUM --nproc_per_node=$MLP_WORKER_GPU \
  --node_rank=$MLP_ROLE_INDEX --master_addr=$MLP_WORKER_0_HOST --master_port=$MLP_WORKER_0_PORT \
  -m src.launch.train_em experiment=$EXP run.kmax=$KMAX \
  trainer_em.use_amp=true run.num_workers=8 run.batch_size=128 \
  run.train_instance_mode=cached run.val_instance_mode=cached \
  logging.wandb.enabled=false hydra.run.dir=$RUNDIR $RESUME
```
> `run.batch_size` 是**每卡**；8 卡全局=×8。首次会自动建缓存（rank0）；想避免可先在开发机建好缓存放 `$VEP`，
> 再追加 `em.bootstrap.enabled=false`（见 §2.5）。简单交互式多卡也可直接 `bash scripts/run_ddp_train_em.sh 8 <EXP> <overrides...>`。

### 5.2 任务 YAML `task-mti-k512.yaml`（控制台 GetJob 格式，来自指南 09）

```yaml
Name: pairformer-mti-k512
ImageSpec: {Type: Custom, Url: cr-mlp-cn-beijing.cr.volces.com/public/yjq:latest}
EntrypointPath: "bash scripts/task_entry.sh"
Framework: PyTorchDDP
ResourceQueueId: q-20260318205115-lvfng
Priority: 4
ActiveDeadlineSeconds: 864000
Envs:
  - {Name: EXP, Value: MTI_EM_K512, IsPrivate: false}
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
监控：`volc ml_task logs --id <id>`；控制台看 GPU/NVLink 利用率确认"拉满"。

### 5.3 批量（E3 K-sweep / E7 聚合器 / E8 容量 / E15 划分）
复制 YAML 改 `Name`+`Envs`：
- **E3 K-sweep**：`KMAX=64/128/256/512` 各一个任务（小任务有插队优势，可并行排队）。
- **E7**：`EXP=MTI_PairCNNAggregator` / `MTI_PairGNNAggregator` / `MTI_train_selected_inst`（后者加 `model.block_type=sab`）。
- **E8 容量**：入口换成 `EntrypointPath: "bash scripts/rebuttal/tuning/plan_B1_model_8xA100.sh"`（脚本内部已 8 卡并行分发）。
- **E1/E2 10-fold**：`EntrypointPath: "bash scripts/rebuttal/tuning/plan_D_10fold_final.sh"`（/ deepTargetPro 版）。

---

## 6. 验证 DDP 修复（8 卡机必跑一次）
本轮修了 PairSelected 多卡崩溃：
```bash
bash scripts/verify_ddp_pair_selected.sh 8     # 机制自测 + 单卡 vs DDP + ckpt 互通
```
EM 路径（train_em）多卡早已可用。

---

## 7. 收尾
- ckpt 归档：`tosutil cp -r -j 32 -p 16 -u $VEP/runs/<run>/ tos://tos-mlp-zgci/yinjiaqi/ckpts/<run>/`
- 跑完把指标回填 `EXPERIMENTS.md`（补全 UNKNOWN 项的实际命令/资源/结果）。
- 开发机长期不用就删（云盘只放系统，删无损失）。

---

## 附：本手册引用的入口/脚本/键（均已核对存在）
- launch 入口：`src.launch.{train_em, train, eval_em, eval_pair_selected, bench_compute_vs_k}`
- override 键：`run.{kmax,batch_size,num_workers,pin_memory,persistent_workers,num_epochs,resume,checkpoint,force_overwrite_bootstrap,train_instance_mode,val_instance_mode,test_instance_mode}`、
  `trainer_em.{use_amp,num_epochs,monitor,greater_is_better}`、`em.{cheap_cache.amp,instance_cache.use_amp,bootstrap.enabled,selection_cache.candidate_pool_size}`、`em_cache_root`、`paths.cache_root`、`model.{d_model,n_layers,n_heads,dim_ff,block_type}`
- ckpt（EM bootstrap 依赖）：`checkpoints/CheapCTSNet/checkpoints/last.pt`、`checkpoints/miRAW_TargetNet_Optimized_dp-0.1/checkpoints/last.pt`、`checkpoints/BR-MIL/checkpoints/best.pt`
