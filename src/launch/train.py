# src/launch/train.py
"""
train.py
======================
本文件是「训练入口脚本」，负责把 **Hydra 配置 → Data / Model / Trainer / Evaluator / WandB**
串起来，形成一个完整、现代化的训练 pipeline。

主要职责
--------
1. 读取 Hydra 配置（configs/ 下的 config.yaml + 各 group）
   - cfg.data     : 数据集配置（DataConfig.from_omegaconf）
   - cfg.model    : 模型结构配置（给 models.registry.build_model 用）
   - cfg.train    : 训练超参配置（喂给 Trainer，控制 optimizer / scheduler / loss 等）
   - cfg.task     : 任务配置（problem_type + threshold 扫描策略等）
   - cfg.run      : 运行策略（num_epochs / batch_size / cache_path / 模式等）
   - cfg.eval     : 验证与测试阶段评估策略
   - cfg.logging  : 日志与 WandB 配置
   - cfg.experiment（可选）: 实验级别的描述信息

2. 构建数据：
   - 使用 DataConfig + src.data.builder.build_dataset_and_loader
   - 获得 (train_dataset, train_loader) / (val_dataset, val_loader)
   - 读取 set-level 标签（set_labels），用于 window→set 聚合

3. 构建模型与 Trainer：
   - models.registry.build_model(name, cfg.model, data_cfg)
   - Trainer(model, task_cfg=cfg.task, train_cfg=cfg.train, device=device)

4. 训练 loop：
   - for epoch in range(cfg.run.num_epochs):
       * trainer.train_one_epoch(train_loader)
       * trainer.validate_one_epoch(val_loader, set_labels=val_set_labels)
       * 打印 / 记录指标（stdout + WandB）
       * 保存最新 checkpoint & 若有提升则保存 best checkpoint

5. 训练结束后的评估（可选）：
   - 使用 evaluator.evaluate_with_trainer 在 **val 集** 上做一次完整评估：
       * threshold sweep（若 cfg.task.threshold.sweep=True）
       * 保存 best_threshold.json / metrics.json / 各类曲线
   - 若需要，也可以在此脚本中顺手对 test 集做评估（示例中留了接口位置）。

6. WandB 集成（可选）：
   - 若 cfg.logging.wandb.enabled=True 且已安装 wandb：
       * 在开头 init
       * 每个 epoch log train/val 的 loss / F1 / AUC 等
       * evaluator 在测试阶段可把最终 metrics 写入 wandb.run.summary

注意
----
- 本文件不直接关心具体的模型结构、数据组织细节，全部通过：
    * DataConfig + build_dataset_and_loader
    * models.registry.build_model
    * Trainer
    * evaluator.evaluate_with_trainer
  进行解耦。
- 这里的路径均依赖 Hydra 的工作目录（run.dir），默认是每次运行一个新的输出目录。
  所有输出（checkpoint / eval 结果）都放在当前运行目录的子目录下，便于长期维护。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List
import numbers 

import torch

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd

# 项目内模块
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

import time 


def apply_arch_variant(cfg):
    v = cfg.model.get("arch_variant", None)
    if v is not None:
        arch = ARCH_SPACE[v]
        cfg.model.num_channels = arch["num_channels"]
        cfg.model.num_blocks   = arch["num_blocks"]
        cfg.model.multi_scale  = arch["multi_scale"]

# ---------------------- #
# 辅助函数
# ---------------------- #


import numpy as np
from typing import Dict, Any, Iterable, Tuple

def iter_scalar_metrics(metrics: Dict[str, Any]) -> Iterable[Tuple[str, float]]:
    """
    只迭代 metrics 里“可以当成标量”的项：
    - 跳过 list / dict / np.ndarray 等复杂对象
    - numpy 标量会转成 Python float/int
    - 不能 float(...) 的一律丢弃
    """
    for k, v in metrics.items():
        # 跳过明显不是标量的
        if isinstance(v, (list, dict, np.ndarray)):
            continue

        # numpy 标量 -> Python 标量
        if isinstance(v, (np.generic,)):
            v = v.item()

        try:
            v_float = float(v)
        except (TypeError, ValueError):
            continue

        yield k, v_float




def setup_wandb(cfg: DictConfig):
    """
    根据 cfg.logging.wandb 初始化 WandB 运行（若启用且安装了 wandb）。

    返回
    ----
    run : wandb.run or None
        若未启用或未安装 wandb，则返回 None。
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

    # run name 尽量用 experiment_name，否则 Hydra 默认名
    run_name = cfg.get("experiment_name", None)

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


# Distill 

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

def load_model_state(model: nn.Module, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            k = k[len("model."):]
        if k.startswith("net."):
            k = k[len("net."):]
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[Teacher] Warning: missing keys: {len(missing)}")
    if unexpected:
        print(f"[Teacher] Warning: unexpected keys: {len(unexpected)}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def build_teacher_from_cfg(cfg: DictConfig, data_cfg: DataConfig, device: torch.device) -> nn.Module:
    # 需要在 cfg.run 里提供：
    #   - distill_teacher_arch: e.g. "TargetNet_Optimized"
    #   - distill_teacher_ckpt: 路径
    #   - distill_teacher_model: DictConfig（teacher 的 model 超参）
    teacher_arch = str(cfg.run.distill_teacher_arch)
    teacher_ckpt = str(cfg.run.distill_teacher_ckpt)

    if not teacher_ckpt or not os.path.exists(teacher_ckpt):
        raise FileNotFoundError(f"[Teacher] ckpt not found: {teacher_ckpt}")

    teacher_model_cfg = cfg.run.distill_teacher_model
    teacher = build_model(teacher_arch, teacher_model_cfg, data_cfg=data_cfg).to(device)
    load_model_state(teacher, teacher_ckpt, device)
    return teacher


# ---------------------- #
# 主入口
# ---------------------- #

@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """
    Hydra 主入口：
    - 由 configs/config.yaml 提供 defaults（data/model/train/task/run/eval/logging 等）
    - 命令行可以通过 override 改动任意字段
    """
    apply_arch_variant(cfg)
    # ---- 基本环境准备 ----
    seed = int(cfg.get("seed", 2020))
    set_seeds(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.get("device", "cuda") != "cpu" else "cpu"
    )

    # 当前 run 目录（Hydra 会自动为每次运行创建独立目录）
    run_dir = Path.cwd()

    # 优先用 run.ckpt_dir / run.eval_dir（已经在 config 里展开了 ${hydra:run.dir}/...）
    # 为兼容旧配置，若缺失则退回 ckpt_subdir / eval_subdir 的相对写法
    ckpt_dir_cfg = cfg.run.get("ckpt_dir", cfg.run.get("ckpt_subdir", "checkpoints"))
    eval_dir_cfg = cfg.run.get("eval_dir", cfg.run.get("eval_subdir", "eval"))

    ckpt_dir = Path(ckpt_dir_cfg)
    eval_dir = Path(eval_dir_cfg)

    # 如果用户写的是相对路径（例如直接写 "checkpoints"），就挂在本次 run 目录下
    if not ckpt_dir.is_absolute():
        ckpt_dir = run_dir / ckpt_dir
    if not eval_dir.is_absolute():
        eval_dir = run_dir / eval_dir

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ---- WandB 初始化（如果配置启用） ----
    wandb_run = setup_wandb(cfg)

    # ---- 构建 DataConfig ----
    data_cfg = DataConfig.from_omegaconf(cfg.data)

    # cache 路径：可以从 cfg.run.cache_path 或 cfg.get("cache_root") 里取
    # 原始项目根目录（Hydra 启动前的 cwd）
    orig_cwd = Path(get_original_cwd())

    # 优先用 run.cache_path；如果没配，则退回 paths.cache_root；再没就 "cache"
    if "paths" in cfg:
        default_cache = cfg.paths.get("cache_root", "cache")
    else:
        default_cache = "cache"

    cache_root_cfg = cfg.run.get("cache_path", default_cache)
    cache_root = Path(cache_root_cfg)

    # 若是相对路径，则相对于 original_cwd
    if not cache_root.is_absolute():
        cache_root = orig_cwd / cache_root

    cache_root = str(cache_root)

    # ---- 构建 train/val Dataset + DataLoader ----
    batch_size = int(cfg.run.get("batch_size", 1024))
    num_workers = int(cfg.run.get("num_workers", 4))
    pin_memory = bool(cfg.run.get("pin_memory", True))

    # 安全读取 experiment.task（老实验没有 experiment 字段）
    experiment_cfg = cfg.get("experiment", None)
    task_mode = getattr(experiment_cfg, "task", None) if experiment_cfg is not None else None
    
    # 用于效率实验
    def build_loader_for_split(split_name: str):
        """
        split_name: "val" or "test" (或你的其它 split 名)
        返回 (dataset, loader, set_labels, aggregate_sets_flag)
        """
        if task_mode == "pair_level_train":
            ds, ld = build_pair_level_dataset_and_loader(
                pair_cfg=cfg.data.pair,
                split=split_name,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle=False,
                drop_last=False,
            )
            return ds, ld, None, False
        else:
            ds, ld = build_dataset_and_loader(
                data_cfg=data_cfg,
                split_idx=split_name,
                cache_data_path=cache_root,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle=False,
                drop_last=False,
            )
            # window-level：一定是 set-level 聚合
            set_labels_local = get_set_labels(data_cfg, split_name)
            return ds, ld, set_labels_local, True


    if task_mode == "pair_level_train":
        # ===================== Pair-level 分支 =====================
        pair_cfg = cfg.data.pair

        train_ds, train_loader = build_pair_level_dataset_and_loader(
            pair_cfg=pair_cfg,
            split="train",
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
            drop_last=False,
        )

        val_ds, val_loader = build_pair_level_dataset_and_loader(
            pair_cfg=pair_cfg,
            split="val",
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
            drop_last=False,
        )


        # pair-level: 每个 sample 就是一个 pair，不需要额外 set 聚合
        val_set_labels = None
        aggregate_sets = False

    else:
        # ===================== 原有 window-level 分支 =====================
        train_ds, train_loader = build_dataset_and_loader(
            data_cfg=data_cfg,
            split_idx="train",
            cache_data_path=cache_root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
            drop_last=False,
        )
        val_ds, val_loader = build_dataset_and_loader(
            data_cfg=data_cfg,
            split_idx="val",
            cache_data_path=cache_root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
            drop_last=False,
        )


        val_set_labels = get_set_labels(data_cfg, "val")
        aggregate_sets = True



    # ---- 构建模型 ----
    # model_name 通常用 cfg.model.arch 或 cfg.model.name 作为 registry key
    model_name = cfg.model.get("arch", cfg.model.get("name"))
    model = build_model(model_name, cfg.model, data_cfg=data_cfg)

    # ---- 构建 teacher（可选）----
    teacher_model = None
    if bool(cfg.run.get("distill_enabled", False)):
        teacher_model = build_teacher_from_cfg(cfg, data_cfg=data_cfg, device=device)


    # ---- 构建 Trainer ----
    trainer = Trainer(
        model=model,
        task_cfg=cfg.task,
        train_cfg=cfg.train,
        run_cfg=cfg.run,
        device=device,
        logger=wandb_run,
        teacher_model=teacher_model,   # <<<<<< Distill新增
    )

    # 若需要从 checkpoint 恢复
    if cfg.run.get("resume", False) or cfg.run.get("checkpoint") is not None:
        ckpt_path = cfg.run.get("checkpoint", str(ckpt_dir / "best.pt"))
        if ckpt_path is not None and os.path.exists(ckpt_path):
            trainer.load_checkpoint(ckpt_path, map_location=device)
            print(f"[Train] Resumed from checkpoint: {ckpt_path}")
        else:
            print(f"[Train] No checkpoint found at {ckpt_path}, start from scratch.")

    # ---- 训练循环 ----
    num_epochs = int(cfg.run.get("num_epochs", 30))

    for epoch in range(trainer.state.epoch, num_epochs):
        trainer.state.epoch = epoch

        # 1) Train
        train_metrics = trainer.train_one_epoch(train_loader)
        train_loss = train_metrics["loss"]
        print(f"[Epoch {epoch+1}/{num_epochs}] Train loss = {train_loss:.4f}")

        # 2) Validation（window→set 聚合 + 指标计算）
        prev_best = trainer.state.best_metric
        val_metrics = trainer.validate_one_epoch(
            loader=val_loader,
            set_labels=val_set_labels,
            aggregate_sets=aggregate_sets,
            use_ema=True,
        )
        val_loss = val_metrics["loss"]
        val_pr_auc = val_metrics["pr_auc"]
        val_roc_auc = val_metrics["roc_auc"]
        monitor_name = trainer.monitor
        monitor_value = val_metrics.get(monitor_name, val_loss)

        val_f1 = float(val_metrics.get("f1", float("nan")))

        print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"Val loss = {val_loss:.4f} | F1 = {val_f1:.4f} | pr_auc = {val_pr_auc:.4f} | roc_auc = {val_roc_auc:.4f} | {monitor_name} = {monitor_value:.4f}"
        )

        # 3) 保存 latest checkpoint
        latest_ckpt_path = ckpt_dir / "last.pt"
        trainer.save_checkpoint(str(latest_ckpt_path))

        # 4) 若当前 epoch 提升了 best_metric，则保存 best checkpoint
        if trainer.state.best_metric != prev_best:
            best_ckpt_path = ckpt_dir / "best.pt"
            trainer.save_checkpoint(str(best_ckpt_path))
            print(
                f"[Epoch {epoch+1}] Improved {monitor_name}: "
                f"{prev_best:.6f} → {trainer.state.best_metric:.6f}. Saved best checkpoint."
            )

        # 5) WandB 日志
        if wandb_run is not None:
            # 为保险起见，lazy import
            import wandb  # type: ignore

            log_dict = {
                "epoch": epoch + 1,
                "train/loss": float(train_loss),
                "val/loss": float(val_loss),
            }
            # 常见指标，如果存在就 log
            for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
                if key in val_metrics:
                    log_dict[f"val/{key}"] = float(val_metrics[key])

            wandb.log(log_dict, step=trainer.state.global_step)

    # ---- 训练结束：在 Val 上做完整评估（含 threshold sweep / 图像等） ----
    # 这里使用 evaluator.evaluate_with_trainer，输出到 eval/val 子目录
    val_eval_dir = eval_dir / "val"
    val_eval_dir.mkdir(parents=True, exist_ok=True)

    val_eval_result = evaluate_with_trainer(
        trainer=trainer,
        loader=val_loader,
        task_cfg=cfg.task,
        logging_cfg=cfg.logging,
        output_dir=str(val_eval_dir),
        set_labels=val_set_labels,
        aggregate_sets=aggregate_sets,
        tag="val",
        do_threshold_sweep=cfg.eval.do_threshold_sweep,
        sweep_num_thresholds=cfg.eval.sweep_num_thresholds,
        reduction=cfg.run.get("eval_reduction", "max"),
        softmax_temp=cfg.run.get("eval_softmax_temp", 1.0),
        topk=cfg.run.get("eval_topk", 3),
    )
    # 约定：evaluate_with_trainer 返回 dict，至少包含：
    #   - "metrics": Dict[str,float]
    #   - "best_threshold": Optional[float] （若启用 sweep）
    best_threshold = val_eval_result.get("best_threshold", None)
    metrics = val_eval_result.get("metrics", {})

    print("\n[Train] Final val metrics:")
    for k, v in val_eval_result.get("metrics", {}).items():
        # numpy 标量、普通 float/int 都按标量处理
        if isinstance(v, numbers.Number):
            print(f"  {k}: {float(v):.4f}")
        else:
            # 对于 list / confusion_matrix 之类，直接打印原始对象
            print(f"  {k}: {v}")

    if best_threshold is not None:
        print(f"[Train] Best threshold on val = {best_threshold:.4f}")
    else:
        print("[Train] No best_threshold from evaluator (maybe sweep disabled).")

    # ---- 终端打印：只对标量用 {:.4f}，复杂对象原样打印 ----
    print("\n[Train] Final val metrics (scalar only):")
    for k, v in iter_scalar_metrics(metrics):
        print(f"  {k}: {v:.4f}")

    # 如果想看混淆矩阵之类，可以单独打印
    if "confusion_matrix" in metrics:
        print("  confusion_matrix:")
        print(np.array(metrics["confusion_matrix"]))

    if best_threshold is not None:
        print(f"[Train] Best threshold on val = {float(best_threshold):.4f}")
    else:
        print("[Train] No best_threshold from evaluator (maybe sweep disabled).")


    # ---- 把最终 val 标量 metrics 写入 WandB summary（若启用） ----
    if wandb_run is not None:
        import wandb  # type: ignore

        for k, v in iter_scalar_metrics(metrics):
            wandb_run.summary[f"val/{k}"] = v

        if best_threshold is not None:
            try:
                wandb_run.summary["val/best_threshold"] = float(best_threshold)
            except (TypeError, ValueError):
                pass


    # ============================================================
    # Efficiency: inference throughput benchmark (pairs/s)
    # - 默认在 val 上测；可通过 cfg.run.bench_inference_split 指定 "test"
    # - 结果用于填 Table: Inference (pairs/s)
    # ============================================================
    if cfg.run.get("bench_inference", False):
        bench_split = str(cfg.run.get("bench_inference_split", "val"))
        warmup_batches = int(cfg.run.get("bench_infer_warmup_batches", 10))
        max_batches = cfg.run.get("bench_infer_max_batches", None)
        max_batches = int(max_batches) if max_batches is not None else None

        print(f"\n[Efficiency] Benchmarking inference throughput on split='{bench_split}' ...")
        _, bench_loader, _, _ = build_loader_for_split(bench_split)

        # 需要 Trainer 实现 benchmark_inference()
        infer_stats = trainer.benchmark_inference(
            bench_loader,
            use_ema=True,
            warmup_batches=warmup_batches,
            max_batches=max_batches,
        )

        # 统一输出
        print(
            f"[Efficiency] split='{bench_split}': "
            f"pairs/s={infer_stats['infer_pairs_per_s']:.2f}, "
            f"peak_vram_gb={infer_stats['infer_peak_vram_gb']:.3f}, "
            f"elapsed_s={infer_stats['infer_elapsed_s']:.3f}, "
            f"total_pairs={int(infer_stats['infer_total_pairs'])}"
        )

        # 写入 wandb：同时 log + summary（便于直接抄表）
        if wandb_run is not None:
            import wandb  # type: ignore
            log_dict = {
                f"eff/{bench_split}/infer_pairs_per_s": float(infer_stats["infer_pairs_per_s"]),
                f"eff/{bench_split}/infer_peak_vram_gb": float(infer_stats["infer_peak_vram_gb"]),
                f"eff/{bench_split}/infer_elapsed_s": float(infer_stats["infer_elapsed_s"]),
                f"eff/{bench_split}/infer_total_pairs": float(infer_stats["infer_total_pairs"]),
            }
            wandb.log(log_dict, step=trainer.state.global_step)

            for k, v in log_dict.items():
                wandb_run.summary[k] = float(v)

                
    # ============================================================
    # 额外：可选地在训练结束后，直接对 test 集做评估
    # - 可以选择用 last 权重（当前 trainer 状态）
    # - 也可以选择加载 best.pt 后再评估
    # ============================================================
    if cfg.run.get("eval_test_after_train", False):
        print("\n[Train] eval_test_after_train=True, start evaluating on test set...")

        # flags: 是否对 last / best 进行测试
        eval_with_last = cfg.run.get("eval_test_with_last", True)
        eval_with_best = cfg.run.get("eval_test_with_best", False)

        # best ckpt 路径（可在 yaml 中覆盖；默认 ckpt_dir/best.pt）
        best_ckpt_path_cfg = cfg.run.get("best_ckpt_path", None)
        default_best_ckpt_path = str(ckpt_dir / "best.pt")
        best_ckpt_path = best_ckpt_path_cfg or default_best_ckpt_path

        test_splits = cfg.run.get("test_splits", ["test"])

        # 安全读取 experiment.task（老实验没有 experiment 字段）
        experiment_cfg = cfg.get("experiment", None)
        task_mode = getattr(experiment_cfg, "task", None) if experiment_cfg is not None else None
        pair_cfg = cfg.data.get("pair", None)

        def run_test_eval_for_current_trainer(tag_prefix: str) -> None:
            """
            使用当前 trainer 的权重，对所有 test_splits 做一次完整评估。
            tag_prefix:
                - "last" : 当前 trainer 状态（训练结束时的权重）
                - "best" : 加载 best.pt 后的状态
            目录结构：
                eval/test/<split>/<tag_prefix>/thr0_5
                eval/test/<split>/<tag_prefix>/val_best
                eval/test/<split>/<tag_prefix>/sweep
            WandB summary key 前缀：
                test/<split>/<tag_prefix>_thr0.5/...
                test/<split>/<tag_prefix>_valbest/...
                test/<split>/<tag_prefix>_sweep/...
            """
            for split_idx in test_splits:
                print(f"[Train][{tag_prefix}] Building test loader for split='{split_idx}'")

                # ===================== 按 task 分支构建 test dataset =====================
                if task_mode == "pair_level_train":
                    if pair_cfg is None:
                        raise ValueError("[Train] experiment.task='pair_level_train' but cfg.data.pair is missing.")

                    test_ds, test_loader = build_pair_level_dataset_and_loader(
                        pair_cfg=pair_cfg,
                        split=split_idx,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        shuffle=False,
                        drop_last=False,
                    )

                    # pair-level：每个 sample 就是一个 pair，不需要 set-level label
                    test_set_labels = None

                else:
                    # 原有 window-level 模式
                    test_ds, test_loader = build_dataset_and_loader(
                        data_cfg=data_cfg,
                        split_idx=split_idx,
                        cache_data_path=cache_root,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        shuffle=False,
                        drop_last=False,
                    )

                    if aggregate_sets:
                        test_set_labels = get_set_labels(data_cfg, split_idx)
                    else:
                        test_set_labels = None

                # 为不同 ckpt 的 test 结果区分子目录：
                # eval/test/<split>/<tag_prefix>/...
                test_root = eval_dir / "test" / str(split_idx) / tag_prefix
                test_root.mkdir(parents=True, exist_ok=True)

                # ---------- (A) 阈值 0.5 的报告 ----------
                from copy import deepcopy
                task_fixed = OmegaConf.create(OmegaConf.to_container(cfg.task, resolve=True))
                # 这里假设 task.threshold 是一个 float（当前实现就是）
                task_fixed.threshold = 0.5

                out_dir_fixed = test_root / "thr0_5"
                out_dir_fixed.mkdir(parents=True, exist_ok=True)

                print(f"[Train][Test {split_idx}][{tag_prefix}] Eval with fixed threshold = 0.5")
                res_fixed = evaluate_with_trainer(
                    trainer=trainer,
                    loader=test_loader,
                    task_cfg=task_fixed,
                    logging_cfg=cfg.logging,
                    output_dir=str(out_dir_fixed),
                    set_labels=test_set_labels,
                    aggregate_sets=aggregate_sets,
                    tag=f"{split_idx}_{tag_prefix}_thr0.5",
                    do_threshold_sweep=False,               # ✅ 只看 0.5，不扫
                    sweep_num_thresholds=cfg.eval.sweep_num_thresholds,
                    reduction=cfg.run.get("test_reduction", "max"),
                    softmax_temp=cfg.run.get("test_softmax_temp", 1.0),
                    topk=cfg.run.get("test_topk", 3),
                )

                # ---------- (B) 使用 val 上 best_threshold 的报告（如果存在） ----------
                if best_threshold is not None:
                    task_valbest = OmegaConf.create(OmegaConf.to_container(cfg.task, resolve=True))
                    task_valbest.threshold = float(best_threshold)

                    out_dir_valbest = test_root / "val_best"
                    out_dir_valbest.mkdir(parents=True, exist_ok=True)

                    print(
                        f"[Train][Test {split_idx}][{tag_prefix}] "
                        f"Eval with val best_threshold = {float(best_threshold):.4f}"
                    )
                    res_valbest = evaluate_with_trainer(
                        trainer=trainer,
                        loader=test_loader,
                        task_cfg=task_valbest,
                        logging_cfg=cfg.logging,
                        output_dir=str(out_dir_valbest),
                        set_labels=test_set_labels,
                        aggregate_sets=aggregate_sets,
                        tag=f"{split_idx}_{tag_prefix}_valbest",
                        do_threshold_sweep=False,           # ✅ 固定 val best 阈值
                        sweep_num_thresholds=cfg.eval.sweep_num_thresholds,
                        reduction=cfg.run.get("test_reduction", "max"),
                        softmax_temp=cfg.run.get("test_softmax_temp", 1.0),
                        topk=cfg.run.get("test_topk", 3),
                    )
                else:
                    print(f"[Train][Test {split_idx}][{tag_prefix}] Skip val-best eval because best_threshold is None.")

                # ---------- (C) 在 test 上做 sweep，找 test F1 最大阈值 ----------
                task_sweep = OmegaConf.create(OmegaConf.to_container(cfg.task, resolve=True))
                # task_sweep.threshold 用不用都行，这里保留原值
                out_dir_sweep = test_root / "sweep"
                out_dir_sweep.mkdir(parents=True, exist_ok=True)

                print(f"[Train][Test {split_idx}][{tag_prefix}] Eval with threshold sweep on test")
                res_sweep = evaluate_with_trainer(
                    trainer=trainer,
                    loader=test_loader,
                    task_cfg=task_sweep,
                    logging_cfg=cfg.logging,
                    output_dir=str(out_dir_sweep),
                    set_labels=test_set_labels,
                    aggregate_sets=aggregate_sets,
                    tag=f"{split_idx}_{tag_prefix}_sweep",
                    do_threshold_sweep=True,               # ✅ 在 test 上扫阈值
                    sweep_num_thresholds=cfg.eval.sweep_num_thresholds,
                    reduction=cfg.run.get("test_reduction", "max"),
                    softmax_temp=cfg.run.get("test_softmax_temp", 1.0),
                    topk=cfg.run.get("test_topk", 3),
                )

                best_thr_test = res_sweep.get("best_threshold", None)

                # ---- (A) Fixed threshold = 0.5 ----
                metrics_fixed = res_fixed.get("metrics", {})

                print(f"\n[Test {split_idx}][{tag_prefix}] Fixed threshold=0.5 metrics:")
                for k, v in iter_scalar_metrics(metrics_fixed):
                    print(f"  {k}: {v:.4f}")

                cm_fixed = metrics_fixed.get("confusion_matrix", None)
                if cm_fixed is not None:
                    print("  confusion_matrix:")
                    print(np.array(cm_fixed))

                # ---- (B) 使用 val best_threshold ----
                metrics_valbest = res_valbest.get("metrics", {}) if res_valbest is not None else None
                if metrics_valbest is not None:
                    print(
                        f"\n[Test {split_idx}][{tag_prefix}] "
                        f"Using val best_threshold={float(best_threshold):.4f} metrics:"
                    )
                    for k, v in iter_scalar_metrics(metrics_valbest):
                        print(f"  {k}: {v:.4f}")

                    cm_valbest = metrics_valbest.get("confusion_matrix", None)
                    if cm_valbest is not None:
                        print("  confusion_matrix:")
                        print(np.array(cm_valbest))

                # ---- (C) Sweep：优先使用 best_threshold 下的指标 ----
                metrics_sweep = res_sweep.get("metrics_at_best", res_sweep.get("metrics", {}))

                if best_thr_test is not None and "metrics_at_best" in res_sweep:
                    print(
                        f"\n[Test {split_idx}][{tag_prefix}] "
                        f"Sweep on test metrics (best threshold={float(best_thr_test):.4f}):"
                    )
                else:
                    print(
                        f"\n[Test {split_idx}][{tag_prefix}] "
                        f"Sweep on test metrics (base threshold={float(task_sweep.threshold):.4f}):"
                    )

                for k, v in iter_scalar_metrics(metrics_sweep):
                    print(f"  {k}: {v:.4f}")

                cm_sweep = metrics_sweep.get("confusion_matrix", None)
                if cm_sweep is not None:
                    print("  confusion_matrix:")
                    print(np.array(cm_sweep))

                if best_thr_test is not None:
                    print(
                        f"[Test {split_idx}][{tag_prefix}] "
                        f"Best threshold on test (from sweep) = {float(best_thr_test):.4f}"
                    )
                else:
                    print(
                        f"[Test {split_idx}][{tag_prefix}] "
                        "No best_threshold from sweep (unexpected if do_threshold_sweep=True)."
                    )

                # ---- 如启用 WandB，把 test 的标量 metrics 也写入 summary ----
                if wandb_run is not None:
                    prefix = f"test/{split_idx}/{tag_prefix}"
                    # # 固定 0.5
                    # for k, v in iter_scalar_metrics(metrics_fixed):
                    #     wandb_run.summary[f"{prefix}_thr0.5/{k}"] = v
                    # val best
                    # if metrics_valbest is not None:
                    #     for k, v in iter_scalar_metrics(metrics_valbest):
                    #         wandb_run.summary[f"{prefix}_valbest/{k}"] = v
                    # sweep：记录 best-threshold 对应的 metrics
                    for k, v in iter_scalar_metrics(metrics_sweep):
                        wandb_run.summary[f"{prefix}_sweep/{k}"] = v
                    if best_thr_test is not None:
                        try:
                            wandb_run.summary[f"{prefix}_sweep/best_threshold"] = float(best_thr_test)
                        except (TypeError, ValueError):
                            pass

        # 先用 last 权重（当前 trainer 状态）做一次测试
        if eval_with_last:
            print("\n[Train] Evaluating on test set with LAST checkpoint (current trainer state)...")
            run_test_eval_for_current_trainer(tag_prefix="last")

        # 再可选地加载 best.ckpt，再测一遍
        if eval_with_best:
            if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
                print(f"\n[Train] Loading BEST checkpoint from: {best_ckpt_path}")
                trainer.load_checkpoint(best_ckpt_path, map_location=device)
                print("[Train] Evaluating on test set with BEST checkpoint...")
                run_test_eval_for_current_trainer(tag_prefix="best")
            else:
                print(
                    f"\n[Train] Skipped eval_test_with_best=True because "
                    f"best checkpoint not found at: {best_ckpt_path}"
                )


    # ---- 最后再 finish WandB ----
    if wandb_run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()
