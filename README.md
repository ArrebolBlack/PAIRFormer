Project Architecture Overview

通用深度学习模板（CNN / Transformer / DiT 均可挂接），基于 Hydra + OmegaConf。

核心设计：
所有东西都通过配置解耦，入口脚本只负责「组装」，真正的逻辑藏在 data / model / trainer / evaluator 这些模块里。

1. 顶层 Mental Model

一行命令背后发生了什么？

python -m src.launch.train \
  experiment=experiment_mirna \
  model=targetnet_default \
  data=mirna_miraw


大致流程：

Hydra 读取 configs/config.yaml 和所有 defaults group，合成一个大 cfg

cfg.data → DataConfig → build_dataset_and_loader → DataLoader

cfg.model → build_model → nn.Module

cfg.train + cfg.task → Trainer

cfg.run + cfg.paths → 目录结构（checkpoints / eval / cache）

cfg.logging → wandb / log 行为

Trainer.train_one_epoch / validate_one_epoch，最后 evaluate_with_trainer 做完整评估

2. Config 依赖关系图

配置 group：

data

model

train

task

run

eval

logging

paths

入口脚本：

src/launch/train.py

src/launch/eval.py

2.1 总体依赖图（文本版）
                    +----------------------+
                    |     configs/*.yaml   |
                    |  (Hydra + OmegaConf) |
                    +----------+-----------+
                               |
                               v
                        DictConfig cfg
                               |
    +---------------------------+-----------------------------------+
    |                           |                                   |
    v                           v                                   v
 cfg.data                 cfg.model                           cfg.paths
   |                         |                                   |
   v                         v                                   |
DataConfig           ModelConfig(可选)/DictConfig                |
   |                         |                                   |
   |                         v                                   |
   |                 models.registry.build_model                 |
   |                                                             |
   v                                                             v
build_dataset_and_loader                                 output_root / cache_root / logs_root
   |                                                             |
   v                                                             |
DataLoader(train/val/test)                                      cfg.run
   |                                                             |
   +---------------------------+---------------------------------+
                               v
                            cfg.train   +  cfg.task   +   cfg.logging
                               |                |              |
                               v                v              |
                          Trainer(model, train_cfg, task_cfg, logger/wandb)
                               |
                               v
        -----------------------------------------------------
        |                  训练 & 验证 & 评测               |
        |  - train_one_epoch                              |
        |  - validate_one_epoch (调用 compute_metrics)     |
        |  - evaluator.evaluate_with_trainer              |
        -----------------------------------------------------
                               |
                               v
                    checkpoints / eval 结果 / wandb

2.2 各 group 谁在用？

cfg.data

在 train.py / eval.py 中：

DataConfig.from_omegaconf(cfg.data)

build_dataset_and_loader(data_cfg, ...)

DataConfig 决定：

原始 txt 路径（path.train / path.val / path.test*）

是否带 ESA 特征等

cfg.model

在 train.py / eval.py 中：

model_name = cfg.model.get("arch", cfg.model.get("name"))

build_model(model_name, cfg.model, data_cfg=data_cfg)

模型内部可以拿 cfg.model 或 ModelConfig 里的 params 做结构超参（通道数、层数、dropout 等）

cfg.train

只在 Trainer 里落地：

优化器：optimizer, lr, weight_decay, momentum

调度器：scheduler, scheduler_factor, scheduler_patience, scheduler_t_max, scheduler_step_size, scheduler_gamma

loss：loss_type (bce / mse / custom)

训练技巧：amp, grad_clip, ema.enabled, ema.decay

监控指标：monitor, greater_is_better

cfg.task

在 Trainer / compute_metrics / evaluator 中使用：

problem_type: "binary_classification" / "regression"

from_logits: 是否需要在 metrics 内部自动加 sigmoid

threshold: 固定阈值或者（扩展后）更复杂的 threshold 配置

compute_metrics(y_true, y_pred_raw, task_cfg) 内部根据 problem_type + threshold 计算 F1 / AUC 等

cfg.run

在 train.py 中：

num_epochs

batch_size, num_workers, pin_memory

cache_path（结合 paths.cache_root → 决定 cache 位置）

resume / checkpoint

ckpt_subdir / eval_subdir（决定 run 内部输出目录）

在 eval.py 中：

batch_size, num_workers, pin_memory

cache_path

checkpoint（必须指定）

eval_subdir（评估输出根目录）

cfg.eval

在 evaluator.evaluate_with_trainer 中使用：

是否做 threshold sweep：do_threshold_sweep, sweep_num_thresholds

输出内容：save_metrics_json, save_report_txt, save_threshold_csv, save_curves_png

文件名控制：roc_curve_file, pr_curve_file

在 eval.py 中：

use_val_best_threshold, best_threshold_path：

若启用，会加载 best_threshold.json，并重写一份 task_eval_cfg.threshold，用固定阈值在 test 上跑。

cfg.logging

在 train.py / eval.py 的 setup_wandb 中使用：

logging.wandb.enabled, project, entity, mode, group, tags

在 evaluator 中：

logging.eval.use_wandb

logging.eval.wandb_prefix（例如 "eval"，写入 summary 时加前缀）

cfg.paths

统一定义路径命名空间：

output_root: 实验输出根（通常配合 Hydra hydra.run.dir 用）

cache_root: 数据 cache 根，通常相对于项目 root

logs_root: 日志根（可选，给以后 logger 用）

cfg.run.cache_path: ${paths.cache_root}：复用 paths 的定义

💡 推荐实践（最后落地版）：
在 configs/config.yaml 中使用 hydra.run.dir: ${paths.output_root}/${now:%Y-%m-%d_%H-%M-%S}，
再在 run.default.yaml 中配置 ckpt_subdir: "checkpoints", eval_subdir: "eval"，
这样每个 run 的结构清晰统一。

3. 一次标准训练流程（Train）
3.1 命令行例子

最小版（使用默认 experiment / data / model）：

python -m src.launch.train


指定实验模板 + 模型：

python -m src.launch.train \
  experiment=experiment_mirna \
  model=targetnet_default


常见 override 示例：

# 改学习率
python -m src.launch.train train.lr=3e-4

# 改 batch size 和 epoch 数
python -m src.launch.train run.batch_size=2048 run.num_epochs=50

# 指定 cache 目录（例如 SSD）
python -m src.launch.train run.cache_path=/ssd/cache_mirna

# 开启/关闭 AMP
python -m src.launch.train train.amp=false

# 打开 WandB
python -m src.launch.train logging.wandb.enabled=true logging.wandb.project=mirna_project

3.2 Train 脚本内部流程（简要）

src/launch/train.py：

@hydra.main(...) 读取 config

set_seeds(cfg.seed)，设置随机种子（在 src/utils.set_seeds 统一实现）

解析 device：CPU / CUDA

根据 cfg.run.ckpt_subdir / cfg.run.eval_subdir + Path.cwd() 创建

run_dir/checkpoints/

run_dir/eval/

setup_wandb(cfg)：若开启，则 wandb.init(config=cfg)

data_cfg = DataConfig.from_omegaconf(cfg.data)

使用 cfg.run.cache_path（结合 paths.cache_root）构造 cache 路径，并传给 build_dataset_and_loader

build_dataset_and_loader 构建：

train_loader（split_idx="train"）

val_loader（split_idx="val"）

val_set_labels = get_set_labels(data_cfg, "val")：读取 set-level 标签

model = build_model(model_name, cfg.model, data_cfg=data_cfg)

trainer = Trainer(model, task_cfg=cfg.task, train_cfg=cfg.train, device=device)

如 cfg.run.resume / cfg.run.checkpoint 打开，则 trainer.load_checkpoint(...)

for epoch in range(trainer.state.epoch, cfg.run.num_epochs):

train_metrics = trainer.train_one_epoch(train_loader)

val_metrics = trainer.validate_one_epoch(val_loader, set_labels=val_set_labels, aggregate_sets=True, use_ema=True)

保存 last.pt / best.pt

若 wandb 打开，记录 train/loss, val/loss, val/f1, val/roc_auc, val/pr_auc 等

训练结束后：

用 evaluate_with_trainer(...) 在 val 上做一次完整评估（含 threshold sweep / ROC / PR 曲线等）

输出到 run_dir/eval/val

若 wandb 打开，写入 wandb.run.summary["val/xxx"]

4. 一次标准评估流程（Eval）
4.1 模式 1：固定阈值（来自 cfg.task.threshold）

最简单版本：用当前 config 的 threshold 直接评估 test 集。

python -m src.launch.eval \
  run.checkpoint=/path/to/best.pt \
  data=mirna_miraw


特点：

cfg.task.threshold 决定使用的阈值（例如 0.5）

若 cfg.eval.do_threshold_sweep=true，则会在当前 split 上扫一遍（可选）

4.2 模式 2：复用训练阶段的 best_threshold

假设你在训练时，val 评估阶段生成了：

<run_dir>/eval/val/best_threshold.json
{
  "best_threshold": 0.7345,
  "monitor": "f1",
  ...
}


评估脚本中：

cfg.eval.use_val_best_threshold=true

cfg.eval.best_threshold_path=/abs/path/to/best_threshold.json
（若不指定，默认尝试 run_dir/eval/val/best_threshold.json）

命令：

python -m src.launch.eval \
  run.checkpoint=/path/to/best.pt \
  eval.use_val_best_threshold=true \
  eval.best_threshold_path=/abs/run_xxx/eval/val/best_threshold.json


内部行为：

读入 best_threshold

克隆一份 task_eval_cfg = OmegaConf.create(cfg.task)

改写：

task_eval_cfg.threshold.value = best_threshold

task_eval_cfg.threshold.fixed = true

task_eval_cfg.threshold.sweep = false（如果你后续扩展了这两个字段）

用这份 task_eval_cfg 对所有 data_cfg.path.keys()（通常 ["test"] 或 ["test0", "test1", ...]）进行评估

输出结果至：

run_dir/eval/<split_idx>/metrics.json

run_dir/eval/<split_idx>/report.txt

run_dir/eval/<split_idx>/roc_curve.png / pr_curve.png

若 wandb 打开，summary 中会有：

<split_idx>/f1, <split_idx>/roc_auc, <split_idx>/pr_auc

eval/best_threshold

5. 配置自检：一键打印全部配置

你已经有脚本：

python ./scripts/print_all_configs.py


它会打印当前组合后的：

data

model

train

task

run

eval

logging

paths

用于快速 sanity check，比如你刚刚的输出：

==== data ====
name: mirna_miraw
...

==== model ====
name: targetnet_default
arch: TargetNet
...

...


推荐习惯：

每次大改 config 结构后，先跑一遍 print_all_configs.py 确认没有 keyError、没有 typo

把典型的实验组合（比如 experiment_mirna）写在 README 的「实验清单」里，未来要复现时就照着点菜。



1. 单次训练：最基本的命令

使用默认 config.yaml + 默认 experiment：

python -m src.launch.train


指定一个实验 preset（比如 configs/experiment/mirna_baseline.yaml）：

python -m src.launch.train experiment=mirna_baseline


常见的覆写方式（命令行优先级最高）：

python -m src.launch.train \
  experiment=mirna_baseline \
  train.lr=3e-4 \
  train.weight_decay=0.0 \
  run.num_epochs=50 \
  logging.wandb.enabled=true \
  logging.wandb.group=mirna_baseline_v2 \
  logging.wandb.tags="[mirna,baseline,v2]"


说明：

experiment=mirna_baseline
→ 用 configs/experiment/mirna_baseline.yaml 覆盖默认配置

train.*
→ 覆盖训练超参

run.*
→ 控制 epoch / batch_size / checkpoint 等

logging.wandb.*
→ 控制 WandB 项目、分组和标签（不用改 yaml 文件）

2. 评估脚本：加载 checkpoint 做完整评测

给定一个已经训练好的 checkpoint（例如：outputs/exp1/checkpoints/best.pt），做 test 评测：

python -m src.launch.eval \
  experiment=mirna_baseline \
  run.checkpoint="outputs/exp1/checkpoints/best.pt"


如果想在 eval 阶段也用 WandB 记录指标：

python -m src.launch.eval \
  experiment=mirna_baseline \
  run.checkpoint="outputs/exp1/checkpoints/best.pt" \
  logging.wandb.enabled=true \
  logging.wandb.group=mirna_eval_v1 \
  logging.wandb.tags="[mirna,eval]"


说明：

run.checkpoint 必须显式指定

eval 脚本内部会遍历 data.path 下所有 split（例如 test, test0, test1），分别评估

评估结果会写到 ${run.eval_subdir}（默认 outputs/eval 之类）

3. 多任务 / 多超参搜索：Hydra multirun (-m)

Hydra 原生支持多组参数一键跑，典型用法是加 -m（multirun）。

3.1 对一个实验做超参网格搜索

例如：在 baseline 上做 (lr, batch_size) 笛卡尔积搜索：

python -m src.launch.train -m \
  experiment=mirna_baseline \
  train.lr=1e-3,3e-4,1e-4 \
  run.batch_size=512,1024


Hydra 会自动展开成 3 × 2 = 6 个 run，每个 run 有自己独立的 hydra.run.dir，例如：

multirun/2025-11-27/00-00-00/0

multirun/2025-11-27/00-00-00/1

…

每个目录下都有：

该次 run 的配置快照：.hydra/config.yaml

你的输出：outputs/checkpoints、outputs/eval 等

你可以进一步指定 multi-run 的输出根目录：

python -m src.launch.train -m \
  hydra.sweep.dir="multirun/miRNA_lr_bs_sweep" \
  experiment=mirna_baseline \
  train.lr=1e-3,3e-4,1e-4 \
  run.batch_size=512,1024

3.2 多个 experiment 一次性跑完

比如你有两个 preset：

experiment=mirna_baseline

experiment=sirna_baseline

可以这样一次跑两个：

python -m src.launch.train -m \
  experiment=mirna_baseline,sirna_baseline \
  train.lr=1e-3 \
  run.num_epochs=30


Hydra 会生成两条 run，分别对应不同 experiment 的配置。

3.3 配合 WandB 的分组策略（推荐实践）

常见模式是：

每一条 multirun 的实验，WandB 用同一个 group；

每个子 run 自动带自己的 hydra.job.num（你可以自己加到 tags/name 里）。

示例：

python -m src.launch.train -m \
  experiment=mirna_baseline \
  logging.wandb.enabled=true \
  logging.wandb.project=targetnet-refactor \
  logging.wandb.entity=myuser \
  logging.wandb.group="mirna_lr_bs_sweep" \
  train.lr=1e-3,3e-4,1e-4 \
  run.batch_size=512,1024


然后在 WandB 面板中按 group = mirna_lr_bs_sweep 过滤，就能看到这一组超参搜索的所有 track。

如果你之后想把 Hydra 的 hydra.job.num 拼进 run name，可以在代码里用：

job_num = cfg.hydra.job.num  # int
run_name = f"{cfg.experiment_name}_job{job_num}"