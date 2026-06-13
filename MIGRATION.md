# MIGRATION — refactor/2026-06

记录本次重构的结构变化与等价性保证。底线：**行为等价**（见
`tools/refactor_verify/`）。本文件随分层执行持续更新。

## 等价性方法（配置层）

- 重构前在 main(HEAD=aee69ce) 上用 `tools/refactor_verify/dump_configs.py` 把全部 77 个
  实验配置合成快照（`golden_configs/`）。
- 每次改配置后重新合成并 `cfgdiff.py --dir golden_configs <new>`，要求 **0 differ**
  （语义等价：摊平 dotted-key 比较，list 整体比较）。
- 因此"配置去重"对喂给代码的最终配置**零改变** → 行为 bit-identical。

## 配置层（L1）—— 进行中

### 已完成：EM-pipeline 家族（F1/F3/F4），cfgdiff = 77/77 0 differ
新增 `configs/experiment/_base/`：
- `em_core.yaml` —— EM 流水线共享体（原 miRAW_EM_Pipeline 主体的字节拷贝，**减去**
  benchmark 参数块）。
- `em_bench.yaml` —— 12 个 `run.bench_*` 参数（仅 `src.launch.bench_compute_vs_k`/Fig4
  读取）。从 em_core 抽出，因为部分 EM 配置（split_sensitivity、K-scaling）原本就**不含**
  这些键；只有原本含它们的配置才 include 此文件。
- `mti_em_core.yaml` —— `em_core` + MTI 专属覆盖（数据列/路径、大 instance 编码器、ckpt、
  f1 监控、MTI wandb），不含 bench；是 MTI_EM_Pipeline/K512/K1024 的共享父。
  （注意：Hydra 的 defaults 路径相对**该文件所在目录**解析，故 mti_em_core 引用同目录
  兄弟 `em_core` 而非 `_base/em_core`。）

旧配置改为薄覆盖（`defaults: [_base/<core>, (_base/em_bench), _self_]` + 仅差异键）：
miRAW/deepTargetPro/MTI_EM_Pipeline、MTI_EM_K512/K1024、miRAW_EM_Pipeline_splitB/splitC/
k1ratio05。每个的覆盖键集由 `cfgdiff golden_configs/<base> golden_configs/<member>` 精确抽取。
F2 rebuttal 子配置（已是薄覆盖范式）随其父被抽取自动受益，合成不变。

### 已完成：F5 pair_agg / F7 window-TNopt / F9 cheap（commit L1.3）+ 死配置归档（commit L1.4）
- `_base/pair_agg_core`（把 `data.pair.cache_root` 抽出，让 DeepMirTar 可不含它）、
  `_base/window_tnopt_core`、`_base/cheap_core`。8 个配置改薄覆盖。
- `miRAW_pair_agg_set_transformer` 保持独立（架构本就不同，非复制）。
- MTI_TargetNet_Optimized 的 `task.threshold` 由标量 0.5 → dict（含 sweep），覆盖时整节点替换（OmegaConf 语义），已验证等价。
- 死配置归档见 `configs/_legacy/README.md`。
- "按数据集缺键"模式（bench、DeepMirTar cache_root）统一用"基类不含该键、含的成员各自添加"处理，保 0-diff。

### 已完成：F8 评估 + 聚合器消融去重（commit L1.5）
- **聚合器消融**（真复制子族）：`_base/mti_agg_ablation_core.yaml`（共享 selected-inst body）+
  MTI_PairGNNAggregator/PairCNNAggregator/PairGNNMoEAggregator 三个薄覆盖（仅 model 块 + name）。
- **其余 MTI-scalable 配置经 cfgdiff 评估为"运维上彼此不同，非复制粘贴"，刻意保持独立**：
  - `MTI_build_selected_inst` 相对 train **删 57 键**（建缓存、无训练）；`MTI_eval_selected_inst`
    按 run.mode + 代码路径不同（删 17 加 5）；selected_raw / window-shard / shard 变体各是
    不同 run.mode 的独立操作。强行继承只会降低可读性而几乎不消除真实重复 → 不并入。
  - 单例 `experiment_sirna`、`DeepMirTar_Transformer` 与 F6 链根 `miRAW_TargetNet_baseline`
    保持原样。
- `experiment_name` 在所有合成结果里都是 `${experiment.name}`（由 config.yaml 的 `_self_`
  最后覆盖），故薄覆盖成员只需设 `experiment.name`。

## L1 配置层 —— 完成
6 个复制家族去重（EM / pair-agg / window-TNopt / cheap / 聚合器消融）+ 死配置归档，
**全程 cfgdiff --dir golden = 77/77 0 differ**。配置去重目标（消除复制粘贴家族）达成。

## 已知 bug / 历史值（保留以保等价，规范化属后续 bugfix 层）

- **`em.policy.instance_mode` 的中文全角引号**（D6）：`em_core` 原样保留原 miRAW 主配置里的
  `"“hybrid”"`（值含全角引号，≠ 干净的 `hybrid`）。代码 `src/em/update_policy.py` 用
  `== "hybrid"` 精确判断，故全角引号会**禁用 epoch-force-online 路径**。但 miRAW 策略参数
  `instance_update_every_epochs=0`，该路径的触发条件 `every_epochs>0` 本就不满足 →
  **全角引号与干净值在此参数下行为一致，未影响论文主结果**。原本为干净 `hybrid` 的配置
  （MTI/deepTargetPro/split*/k1ratio05）显式覆盖回 `hybrid`。
  → 规范化（统一为 `hybrid`）留待 bugfix 层，并以 golden 训练对比确认零数值变化。

## 代码层 + 依赖（进行中）

### ✅ L2a checkpoint 工具（commit e89bcb0）
`src/utils/checkpoint.py: clean_state_dict_keys(sd, prefixes)` 统一 7 处 per-key 前缀剥离
（各传原始 prefix 元组保行为，golden 单测 4/4 验证）。`eval_pair_selected.py` 的整字典公共前缀
剥离器**刻意不动**（语义不同；统一它非等价）。死代码 `ddp_entry.py` → `_legacy/`。

### ✅ L4 模型/损失/死代码（commit 96dc5a3 / af3f904 / a4b45e6 / 55ba258 / fbbe2e3）
- 聚合器 golden 回归（`tests/test_pair_aggregators_golden.py`，6 个 bit-identical）。
- `PairGNNMoEAggregator` 近克隆 → 薄子类（num_experts 移入基类）。
- `BasePairAggregator`：Set/CNN/GNN 共享 `_normalize_mask`/掩码预处理/分类头（`_build_head`
  在原构造点调用以保权重初始化 RNG 顺序；CNN 传 GELU）。
- **loss 统一**：`Trainer._compute_loss` binary 分支委托给 `trainer/loss.BinaryClassificationLoss`，
  删重复的 `Trainer._binary_focal_loss`（`tests/test_loss_equivalence.py` 证 24/24 配置逐位等价；
  miRAW 基线 bit-identical）。
- 死/破损文件归档：`update_model_name.sh` / `test_config_and_checkpoint.py` /
  `test_checkpoint_update.yaml` → `_legacy/`；空文件 `train` 删除。

### ✅ L6 依赖修正（部分）
`requirements.in`：删除误写的 `hydra==2.5`（错误包，需 MSVC 编译）；补 eval 实际需要但原 Linux
freeze 缺失的 `scikit-learn/pandas/matplotlib/seaborn`（与 numpy 1.24.4/py3.10 兼容，不影响训练数值）。
本机 Windows 安装：cu121 index 装 torch 2.4.1 → 按 pin 装纯 python 依赖（见 baseline/run/BASELINE.md）。

## 待办（后续层）
- **L2b** DDP launcher 样板收口到 `src/utils/ddp.py`；**L3** 修 PairSelected 崩溃
  （launcher 传 local_rank 给无此参数的 trainer）+ 加 DDP 梯度同步。EM/pair 路径本机缺 ckpt
  难复跑 → import/结构校验 + 服务器 DDP=1/多卡 + Workflow 对抗核验。
- **L4 续**：`evaluate_test_abc_once` 收编三处 A/B/C 手抄；trainer.py 注释死代码。
- **L5** 数值优化开关（AMP/compile，默认关）；**L6 续** requirements lock 重生 + README 安装更新 +
  EXPERIMENTS.md（阶段4）。每层等价闸门见 refactor_plan.md。
