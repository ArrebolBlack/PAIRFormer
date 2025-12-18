# src/launch/eval.py
"""
eval.py
======================
本文件是「纯评估入口脚本」，负责在 **给定 checkpoint + 数据配置** 的前提下，
对一个或多个数据集 split（通常是 test*）执行完整评测，并输出：

- metrics.json
- report.txt
- 各类曲线（ROC / PR / confusion matrix / prob distribution / threshold sweep）
- 如启用 WandB，则把最终 test metrics 写入 wandb.run.summary

典型用法
--------
1. 只用固定阈值（来自 cfg.task.threshold.value）评测：
   $ python -m src.launch.eval run.mode=eval run.checkpoint=/path/to/best.pt

2. 使用训练阶段在 val 上选出的 best_threshold：
   假设 train run 生成了 `runs/xxx/eval/val/best_threshold.json`：
   $ python -m src.launch.eval \
       run.mode=eval \
       run.checkpoint=/path/to/best.pt \
       eval.use_val_best_threshold=true \
       eval.best_threshold_path=/absolute/path/to/that/best_threshold.json

   脚本会：
     - 读取 best_threshold
     - 克隆一份 cfg.task，修改 threshold.fixed=True / sweep=False / value=best_threshold
     - 用该 task 配置在 test 集上评测，保证对不同模型完全公平。

设计要点
--------
- 评测逻辑完全委托给 src.evaluator.evaluator.evaluate_with_trainer；
- eval.py 只负责：
    * 从 Hydra 配置构造 DataConfig / Model / Trainer
    * 读取 checkpoint
    * 决定 threshold 策略（是否读 best_threshold.json）
    * 遍历所有 data_cfg.path.keys()（如 "test0", "test1"），对每个 split 分别评估。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple
import numbers
import numpy as np
import torch

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd  

from src.config.data_config import DataConfig
from src.data.builder import (
    build_dataset_and_loader,
    get_set_labels,
    build_pair_level_dataset_and_loader,
)
from src.trainer.trainer import Trainer
from src.models.registry import build_model
from src.evaluator.evaluator import evaluate_with_trainer
from src.utils import set_seeds


from src.config.arch_space import ARCH_SPACE

def apply_arch_variant(cfg):
    v = cfg.model.get("arch_variant", None)
    if v is not None:
        arch = ARCH_SPACE[v]
        cfg.model.num_channels = arch["num_channels"]
        cfg.model.num_blocks   = arch["num_blocks"]
        cfg.model.multi_scale  = arch["multi_scale"]

def iter_scalar_metrics(metrics: Dict[str, Any]) -> Iterable[Tuple[str, float]]:
    """
    只迭代 metrics 里的“真·标量”指标：
    - 跳过 list / dict / np.ndarray（例如 confusion_matrix）
    - numpy 标量 -> Python 标量
    """
    for k, v in metrics.items():
        if isinstance(v, (list, dict, np.ndarray)):
            continue
        if isinstance(v, (np.generic,)):
            v = v.item()
        try:
            v_float = float(v)
        except (TypeError, ValueError):
            continue
        yield k, v_float


def to_serializable(obj: Any) -> Any:
    """
    把 all_metrics 里可能存在的 numpy 类型转换成 JSON 可序列化对象。
    """
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, (np.generic,)):
        return obj.item()
    return obj




def setup_wandb(cfg: DictConfig):
    """
    评估阶段的 WandB 初始化（若启用），与 train.py 略同。
    """
    try:
        import wandb  # type: ignore
    except ImportError:
        return None

    if "logging" not in cfg or "wandb" not in cfg.logging:
        return None

    wandb_cfg = cfg.logging.wandb
    enabled = bool(wandb_cfg.get("enabled", False))
    if not enabled:
        return None

    project = wandb_cfg.get("project", "default_project")
    entity = wandb_cfg.get("entity", None)
    mode = wandb_cfg.get("mode", "online")
    group = wandb_cfg.get("group", None)
    tags = wandb_cfg.get("tags", None)

    # 评估 run name 可以加个后缀，避免和训练混淆
    base_name = cfg.get("experiment_name", "eval_run")
    run_name = f"{base_name}_eval"

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        group=group,
        tags=tags,
        mode=mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    return run


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """
    评估主入口：
    - 假设 cfg.run.mode=="eval"（可以在 configs/run/*.yaml 中定义）
    - 需要 cfg.run.checkpoint 指定要加载的权重
    """
    apply_arch_variant(cfg)
    seed = int(cfg.get("seed", 2020))
    set_seeds(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.get("device", "cuda") != "cpu" else "cpu"
    )

    run_dir = Path.cwd()

    # 评估阶段的输出顶层目录：
    # 1) 优先用 cfg.run.eval_dir（通常是 ${hydra:run.dir}/eval）
    # 2) 若没有 eval_dir，则退回 eval_subdir 的写法
    eval_root_cfg = cfg.run.get("eval_dir", cfg.run.get("eval_subdir", "eval"))
    eval_root = Path(eval_root_cfg)
    if not eval_root.is_absolute():
        eval_root = run_dir / eval_root
    eval_root.mkdir(parents=True, exist_ok=True)


    # WandB
    wandb_run = setup_wandb(cfg)

    # DataConfig
    data_cfg = DataConfig.from_omegaconf(cfg.data)
    orig_cwd = Path(get_original_cwd())
    if "paths" in cfg:
        default_cache = cfg.paths.get("cache_root", "cache")
    else:
        default_cache = "cache"

    cache_root_cfg = cfg.run.get("cache_path", default_cache)
    cache_root = Path(cache_root_cfg)
    if not cache_root.is_absolute():
        cache_root = orig_cwd / cache_root
    cache_root = str(cache_root)
    batch_size = int(cfg.run.get("batch_size", 1024))
    num_workers = int(cfg.run.get("num_workers", 4))
    pin_memory = bool(cfg.run.get("pin_memory", True))

    # 构建模型 & Trainer
    model_name = cfg.model.get("arch", cfg.model.get("name"))
    model = build_model(model_name, cfg.model, data_cfg=data_cfg)

    trainer = Trainer(
        model=model,
        task_cfg=cfg.task,
        train_cfg=cfg.train,
        run_cfg=cfg.run,
        device=device,
        logger=wandb_run,
    )

    # checkpoint 必须指定
    ckpt_path = cfg.run.get("checkpoint", None)
    if ckpt_path is None:
        raise ValueError("[Eval] cfg.run.checkpoint 不能为空。请指定要评估的模型权重路径。")

    ckpt_path = str(Path(ckpt_path))
    trainer.load_checkpoint(ckpt_path, map_location=device)
    print(f"[Eval] Loaded checkpoint from {ckpt_path}")

    # 是否使用训练时 val 的 best_threshold
    use_val_best = bool(cfg.eval.get("use_val_best_threshold", False))
    best_thr_path = cfg.eval.get("best_threshold_path", None)
    best_threshold = None

    if use_val_best:
        if best_thr_path is None:
            # 默认尝试从 eval/val/best_threshold.json 读取（相对于当前 run_dir）
            candidate = eval_root / "val" / "best_threshold.json"
            if candidate.exists():
                best_thr_path = str(candidate)

        if best_thr_path is not None and Path(best_thr_path).exists():
            with open(best_thr_path, "r") as f:
                obj = json.load(f)
            best_threshold = float(obj.get("threshold"))
            print(f"[Eval] Loaded best_threshold={best_threshold:.4f} from {best_thr_path}")

        else:
            print("[Eval] use_val_best_threshold=True 但找不到 best_threshold_path，"
                  "将退回到 cfg.task.threshold 配置。")
            best_threshold = None

    # 若需要用 best_threshold 固定评估 test，则克隆一份 task_cfg
    if best_threshold is not None:
        task_eval_cfg = OmegaConf.create(OmegaConf.to_container(cfg.task, resolve=True))
        # 这里 threshold 就是一个 float，直接覆盖
        task_eval_cfg.threshold = float(best_threshold)
    else:
        task_eval_cfg = cfg.task

    # ==== 根据 experiment.task 判断评估模式（window-level / pair-level） ====
    experiment_cfg = cfg.get("experiment", None)
    task_mode = getattr(experiment_cfg, "task", None) if experiment_cfg is not None else None
    pair_cfg = cfg.data.get("pair", None)

    # 开始对所有 split 评估（通常 data_cfg.path.keys() 类似 ["test0", "test1", ...]）
    all_metrics = {}

    # 可以只对名字以 "test" 开头的 split 评估，也可以全部评估
    # 这里简单起见全部评估
    eval_splits = list(data_cfg.path.keys())
    for split_idx in eval_splits:

        print(f"\n[Eval] Processing split: {split_idx}")

        if task_mode == "pair_level_train":
            # ===================== Pair-level 分支 =====================
            if pair_cfg is None:
                raise ValueError(
                    "[Eval] experiment.task='pair_level_train' 但 cfg.data.pair 缺失。"
                )

            # 使用 pair-level builder 构建 Dataset + DataLoader
            ds, loader = build_pair_level_dataset_and_loader(
                pair_cfg=pair_cfg,
                split=str(split_idx),    
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle=False,
                drop_last=False,
            )
            set_labels = None
            aggregate_sets = False

        else:
            # ===================== Window-level 分支（原逻辑） =====================
            ds, loader = build_dataset_and_loader(
                data_cfg=data_cfg,
                split_idx=split_idx,
                cache_data_path=cache_root,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle=False,
                drop_last=False,
            )
            # window-level：需要 set-level label，并在 evaluator 里做 window→set 聚合
            set_labels = get_set_labels(data_cfg, split_idx)
            aggregate_sets = True

        split_eval_dir = eval_root / split_idx
        split_eval_dir.mkdir(parents=True, exist_ok=True)

        result = evaluate_with_trainer(
            trainer=trainer,
            loader=loader,
            task_cfg=task_eval_cfg,
            logging_cfg=cfg.logging,
            output_dir=str(split_eval_dir),
            set_labels=set_labels,
            aggregate_sets=True,
            tag=split_idx,
            do_threshold_sweep=cfg.eval.do_threshold_sweep,
            sweep_num_thresholds=cfg.eval.sweep_num_thresholds,
            reduction=cfg.run.get("test_reduction", "max"),
            softmax_temp=cfg.run.get("test_softmax_temp", 1.0),
            topk=cfg.run.get("test_topk", 3),
        )

        metrics = result.get("metrics", {})
        all_metrics[split_idx] = metrics

        best_thr = result.get("best_threshold", None)

        print(f"[Eval] Metrics for {split_idx}:")
        # 只打印标量指标
        for k, v in iter_scalar_metrics(metrics):
            print(f"  {k}: {v:.4f}")

        # 混淆矩阵等复杂对象单独打印
        if "confusion_matrix" in metrics:
            print("  confusion_matrix:")
            print(np.array(metrics["confusion_matrix"]))

        if best_thr is not None:
            print(f"  (split {split_idx}) best_threshold = {float(best_thr):.4f}")


    # 把所有 split 的 metrics 汇总写入一个总表（先做 JSON 可序列化转换）
    summary_path = eval_root / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(to_serializable(all_metrics), f, indent=2)
    print(f"\n[Eval] Saved metrics summary to {summary_path}")

    # WandB summary（只写标量）
    if wandb_run is not None:
        import wandb  # type: ignore

        for split_idx, metrics in all_metrics.items():
            for k, v in iter_scalar_metrics(metrics):
                wandb_run.summary[f"{split_idx}/{k}"] = v

        # 如果你想把各 split 的 best_threshold 也记进去，可以在上面循环里：
        #   if "best_threshold" in metrics: wandb_run.summary[f"{split_idx}/best_threshold"] = float(metrics["best_threshold"])
        # 目前 best_threshold 已经保存在各 split 对应的 eval 目录下，有需要再加。

        wandb.finish()



if __name__ == "__main__":
    main()
