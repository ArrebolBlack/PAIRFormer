# EXPERIMENTS — PAIRFormer 关键实验清单

> 论文（NeurIPS 2026 修订版）核心实验 ↔ 仓库配置/脚本的可复制运行清单。
> 来源：`tools/refactor_verify` 勘察 + 论文/rebuttal 材料映射。**草稿**：标 UNKNOWN 的项与
> 资源估计待最终核对（阶段4 总验收时定稿）。命令默认在仓库根目录、激活项目环境后运行。

## 0. 前置

- **环境**：torch 2.4.1 + hydra-core 1.3.2 等（见 `requirements.in`，已修 `hydra==2.5` 错误包并补 eval 依赖）。
  Windows 本机安装见 `baseline/run/BASELINE.md`。
- **数据**：`data/` 下 miRAW_*/MTI/deepTargetPro/DeepMirTar/rebuttal 已随仓库（LFS）。
  缺 `data/MTI/MTI_CTS_train_val.txt`、`data/siRNA_raw_data/*`（服务器专属）。
- **首跑缓存**：EM/CTS 路径首跑会从 raw txt 现建缓存（Biopython ESA，较慢）；scalable 路径需先
  build cache（见各实验）。
- **外部 baseline 子模块**（E1 的 TargetNet/Mimosa）：`git submodule update --init`（external/ 当前未初始化）。
- **关键 checkpoint**（docs/reports/EXPERIMENT_REPORT.md App A）：MTI K=512 `checkpoints/MTI_v3_K512_expG/best.pt`；
  SWA `checkpoints/MTI_v3_SWA_K512/best.pt`；instance `checkpoints/MTI_v3_xlarge_resume/best.pt`；
  cheap `checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/best.pt`；miRAW BR-MIL `checkpoints/BR-MIL/checkpoints/best.pt`（本机已存在）。
- **多卡启动约定**：`torchrun --nproc_per_node=N -m src.launch.<entry> experiment=<NAME> ...`，或
  `bash scripts/run_ddp_train_em.sh N <NAME> <overrides>`。per-GPU batch_size = 配置值（全局随 N 增大），
  LR 缩放见 `docs/DDP_TRAINING.md`。checkpoint 单↔多卡互通（不存 `module.` 前缀）。
  PairSelected 路径多卡见 `scripts/verify_ddp_pair_selected.sh`（已修复，refactor/2026-06）。
- **硬件**：本机单卡 RTX3060 仅够小烟雾；真实/多卡跑在服务器（2×A100 火山云 / 8×A100 提交 / 5090）。

## 1. 主结果 / 主图主表

| ID | 实验 | 论文位置 | 配置/脚本 | 单卡命令 | 多卡命令 | 预期指标 | 资源 | 状态 |
|---|---|---|---|---|---|---|---|---|
| E1 | miRAW 10-fold 主结果（vs TargetNet/Mimosa/MaxPool） | Table 1 §6.2 | `miRAW_EM_Pipeline`(K=64)；baseline `scripts/rebuttal/baselines/eval_{targetnet,mimosa,maxpool_10fold}.py` | `python -m src.launch.train_em experiment=miRAW_EM_Pipeline run.kmax=64` | `bash scripts/run_ddp_train_em.sh 2 miRAW_EM_Pipeline run.kmax=64` | PAIR-Former PR-AUC 0.869±0.031, F1 0.840±0.022；TN 0.779；Mimosa 0.788；MaxPool 0.798 | 2×A100 数小时/fold | 待跑 |
| E2 | deepTargetPro 外部验证（transfer+full） | Table 2 §6.2 | `deepTargetPro_EM_Pipeline`/`_ext150`；transfer `scripts/run_seed{2025,2026}_ext150.sh` | `python -m src.launch.train_em experiment=deepTargetPro_EM_Pipeline` | `bash scripts/run_ddp_train_em.sh 2 deepTargetPro_EM_Pipeline` | transfer F1 83.9±3.9%, full 83.2±3.2%；>TEC-miTarget 79.11% | 2×A100 | 待跑 |
| E3 | 大规模 MTI（K∈{64,128,256,512}） | §6.2 Fig3(a) | `MTI_train_selected_inst`(+`MTI_EM_K512/K1024`) | `python -m src.launch.train_em experiment=MTI_train_selected_inst run.kmax=512` | `bash scripts/run_ddp_train_em.sh 8 MTI_train_selected_inst run.kmax=512` | K=512 F1 0.792, PR-AUC 0.874；K=64 F1 0.771 | 8×A100 ~30min/epoch | 待跑 |
| E4 | K-sweep retrain@K vs truncate@Kmax | Fig3(a) §6.3 | `MTI_EM_K512`(truncate, st_selector_prefix)+per-K；`scripts/rebuttal/k_sensitivity/run_k_sensitivity.sh` | retrain: `... run.kmax=$K`；truncate: 用 prefix selector 在更小 kmax 评 K512 ckpt | `bash scripts/run_ddp_train_em.sh 8 MTI_train_selected_inst run.kmax=$K` | 单调饱和，retrain≥truncate；F1 ~0.74(K1)→0.792(K512) | 8×A100 | 待跑 |
| E5 | Robustness vs 可见池 n（固定 K*=64） | Fig3(b) §6.3 | `MTI_train_selected_inst` + `em.selection_cache.candidate_pool_size`；`EXP_PLAN_1_Robustness_2xA100.md` | eval-only：`eval_pair_selected ... candidate_pool_size={64..2048} run.kmax=512`（用 K512 ckpt） | 同上 torchrun | n=64 F1~0.76 → n≥512 饱和 0.7914/PR-AUC 0.8729 | 2×A100（eval） | 待跑 |
| E6 | 运行时/显存（online inference vs K） | Fig4 §6.3 | 入口 `src.launch.bench_compute_vs_k`；输出 `paper/figures/bench_fig3/*.csv`（已存在） | `python -m src.launch.bench_compute_vs_k experiment=miRAW_EM_Pipeline run.bench_pair_ids_path=paper/figures/bench_fig3/pair_ids_test_1024.json` | —（单卡基准） | K=64 BRMIL≈TargetNet-like 延迟/吞吐；CPU gather 主导 | 单卡 | 数据已在 paper/figures/bench_fig3/ |

## 2. 消融（Table 3 §6.4）

| ID | 实验 | 论文位置 | 配置 | 命令（单卡；多卡同换 torchrun） | 预期(Val F1) | 状态 |
|---|---|---|---|---|---|---|
| E7 | 聚合器家族 CNN/GNN/SAB | Table 3 | `MTI_PairCNNAggregator`/`MTI_PairGNNAggregator`/`MTI_train_selected_inst block_type=sab` | `python -m src.launch.train_em experiment=MTI_PairCNNAggregator` 等 | CNN 0.685, GNN 0.760, SAB 0.7715；ISAB 0.642 | 待跑 |
| E8 | SetTransformer 容量(d_model,L,H) | Table 3 | `MTI_train_selected_inst` + `model.{d_model,n_layers,n_heads,block_type}` | `... model.d_model=1024 model.n_layers=4 model.n_heads=16 block_type=sab` | Small 0.727, Sweet(d1024/L4) 0.735 | 待跑 |
| E9 | CTS 编码器缩放(Std/X-Large/XX-Large) | §6.4 文字 | UNKNOWN 单文件；近似 `MTI_TargetNet_Optimized` + num_channels/num_blocks | `python -m src.launch.train experiment=MTI_TargetNet_Optimized instance_model.num_channels=[64,64,128,128] num_blocks=[3,3,3,3]` | CTS-level 饱和 ~0.685；pair-level 0.642→0.735 | 待跑/UNKNOWN |

## 3. 审稿补充 / rebuttal（多为独立脚本；is_reviewer_supplementary）

| ID | 实验 | 触发 | 配置/脚本 | 命令 | 预期 | 状态 |
|---|---|---|---|---|---|---|
| E10 | 跨域 CAMELYON16 + Musk2 | Table4 §6.5 (F4i7) | `scripts/rebuttal/exp6_camelyon16_v4.py`,`exp6-musk2/*` | `python scripts/rebuttal/exp6_camelyon16_v4.py` | CAM16 AUC0.980/F10.966；Musk2 +5.5pp F1 | 待跑 |
| E11 | n-分布统计（Fig2+附录） | Fig2/AppC.1 (F4i7) | `scripts/rebuttal/n_distribution/*` | `python .../compute_n_distribution.py` | median n 640~993；92-95% 对>K64 | 待跑 |
| E12 | 小预算选择器消融 S0/S1/S2 (K8/16) | (ohuZ/KXKP) | `scripts/rebuttal/selector_ablation/generate_configs.py` | `generate_configs.py → run_priority_experiments.sh` | 差异<0.005 PR-AUC | 待跑 |
| E13 | K=1 基线（隔离 SetTransformer） | (KXKP) | `*_EM_Pipeline run.kmax=1`；`scripts/rebuttal/eval_k1_online.py` | `python -m src.launch.train_em experiment=miRAW_EM_Pipeline run.kmax=1` | miRAW K=1 F1~0.925 vs K64 0.950 | 待跑 |
| E14 | 无正样本重叠评估（泄漏审计） | (ohuZ-Q1) | `rebuttal_eval_no_pos_overlap` | `create_no_pos_overlap_test.py → eval_em experiment=rebuttal_eval_no_pos_overlap` | PR-AUC 0.9815→0.9809 | 待跑 |
| E15 | 划分敏感性 | (KXKP-Q4) | `scripts/rebuttal/split_sensitivity/*` | `create_split_data.py → run_all.sh` | 跨划分 PR-AUC 稳定 | 待跑 |
| E16 | cheap recall@K（STSelector vs TopK） | (KXKP-Q3) | `scripts/rebuttal/cheap_ranking{,_v2}/*` | `cheap_ranking_v2.py` | recall@K，STSelector≥TopK | 待跑 |
| E17 | 关系蒸馏 L_rel 消融 | (KXKP) | `CheapCTSNet` vs `CheapCTSNet_noRelDistill`（已去重为 base+2键覆盖） | `python -m src.launch.train experiment=CheapCTSNet` / `=CheapCTSNet_noRelDistill` | L_rel 影响(小) | 待跑 |
| E18 | k1_ratio 消融 | (ohuZ/KXKP) | `scripts/rebuttal/k1_ratio_ablation/run_all.sh`（覆盖 `em.selector_module.cfg.k1_ratio`） | `bash .../run_all.sh` | 高K近平、小K多样性有益 | 待跑 |

## 4. 说明 / 待定稿（阶段4 收口）

- **数值口径**：Table 1 用**新 10-fold balanced**（PR-AUC 0.869）；rebuttal CSV 仍是旧 half-split
  饱和值（~0.99）。复现主表用新口径。
- **UNKNOWN 配置**：E6（有入口无 yaml）、E9（Stage-1 缩放无单一命名文件）、E10/E11/E15/E16/E18
  （独立 python 脚本）。阶段4 落实精确命令 + 资源估计 + 现存 ckpt 路径。
- **配置去重不影响实验**：本次重构把 miRAW/deepTargetPro/MTI 等 experiment 配置改为继承 `_base/*`，
  但合成结果与重构前**逐一 0 差异**（`tools/refactor_verify`），故上述 `experiment=<NAME>` 命令行为不变。
- 详细映射见 `D:\2026中关村访学\PAIRFormer重构\stage1\STAGE1_EXPERIMENT_MAP.md`（审稿线索 ohuZ/U4C9/KXKP/F4i7）。
