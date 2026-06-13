# 火山云任务提交（PAIRFormer）

真实平台值见 `docs/SERVER_RUNBOOK.md` §0 / 《开发指南 09》。本目录是可直接 `volc ml_task submit -c` 的成品 YAML。

## 一次性环境（开发机，按你 2026-06-14 的实测状态）

`PAIRFormer_exp8_final` **不是 git 仓库** → 代码用**全新 code-only 克隆**，数据/ckpt **软链** exp8（都在 vePFS，零下载）：

```bash
VEP=/vepfs-mlp2/queue010/20252203765 ; EX=$VEP/PAIRFormer_exp8_final
# 1) 代码（只取代码，不取 LFS blob；走代理，体量小）
GIT_LFS_SKIP_SMUDGE=1 git clone -b refactor/2026-06 <repo-url> $VEP/PAIRFormer
cd $VEP/PAIRFormer
# 2) 数据 / ckpt 复用 exp8（不下载）。先删掉克隆出来的 LFS 指针 data/ckpt，再软链 exp8 的：
rm -rf data checkpoints && ln -s $EX/data data && ln -s $EX/checkpoints checkpoints
# 3) env 用 myenv（已含 torch2.5.1 + Bio/hydra/sklearn/pandas）
source $VEP/miniconda3/etc/profile.d/conda.sh && conda activate myenv
# 4) 自检：配置等价闸门 + src 解析
python tools/refactor_verify/dump_configs.py --repo . --out /tmp/c && \
python tools/refactor_verify/cfgdiff.py --dir tools/refactor_verify/golden_configs /tmp/c   # 应 77/77 0 differ
```
> ⚠️ myenv 是 **torch 2.5.1**（本地等价基线是 2.4.1）→ 服务器结果不会与本地 golden 逐位一致，属正常（服务器跑的是真实指标，不是等价证明）。

## 单卡 dry-run（强烈建议，先确认 reuse 路径跑通再提 8 卡）

```bash
export PYTHONPATH=$PWD PF_EFFICIENT_ATTN=1 PF_DETERMINISTIC=0
python -m src.launch.train_pair_selected_inst experiment=MTI_train_selected_inst \
  run.kmax=512 run.num_epochs=1 \
  scalable.cache_root=$EX/cache_mti_full_st05_retrain_r4_v3relbl \
  instance_ckpt_path=checkpoints/MTI_TargetNet_Optimized_shard_v2_relabel_top4/best.pt
```
跑通（能读缓存、前向反向、出 test 指标）即可提任务。

## 提交双任务（你的"复用 vs 重建 + 对比"）

```bash
volc configure                                   # 首次：AK/SK（AccessKey.txt）+ region cn-beijing
volc ml_task submit -c deploy/volc/task-mti-reuse.yaml    # 任务A：复用 exp8 ST05 缓存（最快）
volc ml_task submit -c deploy/volc/task-mti-build.yaml    # 任务B：重建缓存再训练（对照）
volc ml_task logs --id <id>                      # 或开发机 tail -f $VEP/task_logs/<task>_mti_w0.log
```
跑完比两个 `$VEP/runs/mti_selected_{reuse,build}_k512` 下的 test 指标 → 一致即证明缓存可复用。

> CLI 报字段名错 → 按《开发指南 09》"两套 schema"把 `ImageSpec/ResourceQueueId/EntrypointPath/...`
> 换成 `ImageUrl/ResourceQueueID/Entrypoint/...` 别名；**值不变**（这也是 07 清单最后一项待验证，提一次即清）。

## 其它实验
- dense EM（miRAW/deepTargetPro/MTI_EM_K512）：用 `scripts/task_entry.sh`（会自动建 em_cache）。
- E1/E2 10-fold、E8 容量：入口换成对应 `scripts/rebuttal/tuning/plan_*_8xA100.sh`。
- E5 robustness / E4 K-sweep：exp8 里已有现成 ckpt 和缓存（`MTI_v3_SWA_K512`、`cache_robustness_k64_n*`、`cache_mti_full_topk_..._k*`）→ 多为开发机 eval，见 `docs/SERVER_RUNBOOK.md` §3。
