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

### 待完成：F8 MTI-scalable 家族（~15-18 个配置）
共享 `MTI_train_selected_inst` 那套 ~133 行 selected-inst/scalable body（task=pair_train_selected_inst，
用 trainer_pair_selected + scalable + token_provider），分支为：
- 聚合器消融（PairCNN/GNN/MoE）：仅 `model` 块（arch 相关键，含缺键）+ name 不同；
- build/train/eval：`run.mode` + trainer 块有无；
- window-shard / selected_raw / shard 变体：更偏一次性运维配置。
计划：抽 `_base/mti_scalable_inst_core`（不含 model 块，避免聚合器缺键问题），各成员补 model+mode。
这是 L1 剩余的独立子任务（每个 130+ 键，缺键子坑多），单独一轮做。

## 已知 bug / 历史值（保留以保等价，规范化属后续 bugfix 层）

- **`em.policy.instance_mode` 的中文全角引号**（D6）：`em_core` 原样保留原 miRAW 主配置里的
  `"“hybrid”"`（值含全角引号，≠ 干净的 `hybrid`）。代码 `src/em/update_policy.py` 用
  `== "hybrid"` 精确判断，故全角引号会**禁用 epoch-force-online 路径**。但 miRAW 策略参数
  `instance_update_every_epochs=0`，该路径的触发条件 `every_epochs>0` 本就不满足 →
  **全角引号与干净值在此参数下行为一致，未影响论文主结果**。原本为干净 `hybrid` 的配置
  （MTI/deepTargetPro/split*/k1ratio05）显式覆盖回 `hybrid`。
  → 规范化（统一为 `hybrid`）留待 bugfix 层，并以 golden 训练对比确认零数值变化。

## 待办（后续层）
- DDP 统一 / checkpoint 工具（L2）、PairSelected 崩溃修复（L3）、质量去重（L4）、
  数值优化开关（L5）、依赖/文档（L6）。每层各自的等价闸门见 refactor_plan.md。
