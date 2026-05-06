# src/launch/train.py
"""
train.py - Training Entry Point
================================
Orchestrates Hydra config → Data / Model / Trainer / Evaluator / WandB pipeline.

Main responsibilities:
1. Load Hydra config (data, model, train, task, run, eval, logging)
2. Build datasets and dataloaders (train/val/test)
3. Build model via registry and Trainer
4. Training loop: train_one_epoch → validate_one_epoch → checkpoint saving
5. Post-training evaluation with threshold sweep (optional)
6. WandB logging integration (optional)

All outputs (checkpoints, eval results) are saved to Hydra's run directory.
Decoupled via DataConfig, build_model, Trainer, and evaluate_with_trainer.
"""

from __future__ import annotations

import numbers
import os
import time
from pathlib import Path
from typing import List

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from src.config.arch_space import ARCH_SPACE
from src.config.data_config import DataConfig
from src.data.builder import (
    build_dataset_and_loader,
    build_pair_level_dataset_and_loader,
    get_set_labels,
)
from src.evaluator.evaluator import evaluate_with_trainer
from src.models.registry import build_model
from src.trainer.trainer import Trainer
from src.utils import set_seeds


def apply_arch_variant(cfg):
    v = cfg.model.get("arch_variant", None)
    if v is not None:
        arch = ARCH_SPACE[v]
        cfg.model.num_channels = arch["num_channels"]
        cfg.model.num_blocks = arch["num_blocks"]
        cfg.model.multi_scale = arch["multi_scale"]


from typing import Any, Dict, Iterable, Tuple

import numpy as np


def iter_scalar_metrics(metrics: Dict[str, Any]) -> Iterable[Tuple[str, float]]:
    """
    Iterate only scalar metrics from dict:
    - Skip list / dict / np.ndarray
    - Convert numpy scalars to Python float/int
    - Discard non-floatable values
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


def setup_wandb(cfg: DictConfig):
    """
    Initialize WandB run if enabled and installed.

    Returns
    -------
    run : wandb.run or None
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


from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


def load_model_state(model: nn.Module, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            k = k[len("model.") :]
        if k.startswith("net."):
            k = k[len("net.") :]
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[Teacher] Warning: missing keys: {len(missing)}")
    if unexpected:
        print(f"[Teacher] Warning: unexpected keys: {len(unexpected)}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def build_teacher_from_cfg(
    cfg: DictConfig, data_cfg: DataConfig, device: torch.device
) -> nn.Module:
    """
    Build teacher model from config for knowledge distillation.
    Requires cfg.run.distill_teacher_arch, distill_teacher_ckpt, distill_teacher_model.
    """
    teacher_arch = str(cfg.run.distill_teacher_arch)
    teacher_ckpt = str(cfg.run.distill_teacher_ckpt)

    if not teacher_ckpt or not os.path.exists(teacher_ckpt):
        raise FileNotFoundError(f"[Teacher] ckpt not found: {teacher_ckpt}")

    teacher_model_cfg = cfg.run.distill_teacher_model
    teacher = build_model(teacher_arch, teacher_model_cfg, data_cfg=data_cfg).to(device)
    load_model_state(teacher, teacher_ckpt, device)
    return teacher


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """
    Hydra main entry point.
    Loads config from configs/config.yaml with all groups (data/model/train/task/run/eval/logging).
    Command-line overrides supported.
    """
    apply_arch_variant(cfg)
    seed = int(cfg.get("seed", 2020))
    set_seeds(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.get("device", "cuda") != "cpu" else "cpu"
    )

    run_dir = Path.cwd()

    ckpt_dir_cfg = cfg.run.get("ckpt_dir", cfg.run.get("ckpt_subdir", "checkpoints"))
    eval_dir_cfg = cfg.run.get("eval_dir", cfg.run.get("eval_subdir", "eval"))

    ckpt_dir = Path(ckpt_dir_cfg)
    eval_dir = Path(eval_dir_cfg)

    if not ckpt_dir.is_absolute():
        ckpt_dir = run_dir / ckpt_dir
    if not eval_dir.is_absolute():
        eval_dir = run_dir / eval_dir

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = setup_wandb(cfg)

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

    experiment_cfg = cfg.get("experiment", None)
    task_mode = getattr(experiment_cfg, "task", None) if experiment_cfg is not None else None

    def build_train_val_loaders():
        """Build train and val loaders based on task mode."""
        if task_mode == "pair_level_train":
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
            return train_ds, train_loader, val_ds, val_loader, None, False
        else:
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
            return train_ds, train_loader, val_ds, val_loader, val_set_labels, True

    train_ds, train_loader, val_ds, val_loader, val_set_labels, aggregate_sets = build_train_val_loaders()

    model_name = cfg.model.get("arch", cfg.model.get("name"))
    model = build_model(model_name, cfg.model, data_cfg=data_cfg)

    teacher_model = None
    if bool(cfg.run.get("distill_enabled", False)):
        teacher_model = build_teacher_from_cfg(cfg, data_cfg=data_cfg, device=device)

    trainer = Trainer(
        model=model,
        task_cfg=cfg.task,
        train_cfg=cfg.train,
        run_cfg=cfg.run,
        device=device,
        logger=wandb_run,
        teacher_model=teacher_model,
    )

    if cfg.run.get("resume", False) or cfg.run.get("checkpoint") is not None:
        ckpt_path = cfg.run.get("checkpoint", str(ckpt_dir / "best.pt"))
        if ckpt_path is not None and os.path.exists(ckpt_path):
            trainer.load_checkpoint(ckpt_path, map_location=device)
            print(f"[Train] Resumed from checkpoint: {ckpt_path}")
        else:
            print(f"[Train] No checkpoint found at {ckpt_path}, start from scratch.")

    num_epochs = int(cfg.run.get("num_epochs", 30))

    for epoch in range(trainer.state.epoch, num_epochs):
        trainer.state.epoch = epoch

        train_metrics = trainer.train_one_epoch(train_loader)
        train_loss = train_metrics["loss"]
        print(f"[Epoch {epoch+1}/{num_epochs}] Train loss = {train_loss:.4f}")

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

        latest_ckpt_path = ckpt_dir / "last.pt"
        trainer.save_checkpoint(str(latest_ckpt_path))

        if trainer.state.best_metric != prev_best:
            best_ckpt_path = ckpt_dir / "best.pt"
            trainer.save_checkpoint(str(best_ckpt_path))
            print(
                f"[Epoch {epoch+1}] Improved {monitor_name}: "
                f"{prev_best:.6f} → {trainer.state.best_metric:.6f}. Saved best checkpoint."
            )

        if wandb_run is not None:
            import wandb  # type: ignore

            log_dict = {
                "epoch": epoch + 1,
                "train/loss": float(train_loss),
                "val/loss": float(val_loss),
            }
            for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
                if key in val_metrics:
                    log_dict[f"val/{key}"] = float(val_metrics[key])

            wandb.log(log_dict, step=trainer.state.global_step)

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
    best_threshold = val_eval_result.get("best_threshold", None)
    metrics = val_eval_result.get("metrics", {})

    print("\n[Train] Final val metrics:")
    for k, v in val_eval_result.get("metrics", {}).items():
        if isinstance(v, numbers.Number):
            print(f"  {k}: {float(v):.4f}")
        else:
            print(f"  {k}: {v}")

    if best_threshold is not None:
        print(f"[Train] Best threshold on val = {best_threshold:.4f}")
    else:
        print("[Train] No best_threshold from evaluator (maybe sweep disabled).")

    print("\n[Train] Final val metrics (scalar only):")
    for k, v in iter_scalar_metrics(metrics):
        print(f"  {k}: {v:.4f}")

    if "confusion_matrix" in metrics:
        print("  confusion_matrix:")
        print(np.array(metrics["confusion_matrix"]))

    if best_threshold is not None:
        print(f"[Train] Best threshold on val = {float(best_threshold):.4f}")
    else:
        print("[Train] No best_threshold from evaluator (maybe sweep disabled).")

    if wandb_run is not None:
        import wandb  # type: ignore

        for k, v in iter_scalar_metrics(metrics):
            wandb_run.summary[f"val/{k}"] = v

        if best_threshold is not None:
            try:
                wandb_run.summary["val/best_threshold"] = float(best_threshold)
            except (TypeError, ValueError):
                pass

    if cfg.run.get("bench_inference", False):
        bench_split = str(cfg.run.get("bench_inference_split", "val"))
        warmup_batches = int(cfg.run.get("bench_infer_warmup_batches", 10))
        max_batches = cfg.run.get("bench_infer_max_batches", None)
        max_batches = int(max_batches) if max_batches is not None else None

        print(f"\n[Efficiency] Benchmarking inference throughput on split='{bench_split}' ...")

        # Build loader for benchmark split
        if task_mode == "pair_level_train":
            _, bench_loader, _, _ = (
                *build_pair_level_dataset_and_loader(
                    pair_cfg=cfg.data.pair,
                    split=bench_split,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    shuffle=False,
                    drop_last=False,
                ),
                None,
                False,
            )
        else:
            _, bench_loader, _, _ = (
                *build_dataset_and_loader(
                    data_cfg=data_cfg,
                    split_idx=bench_split,
                    cache_data_path=cache_root,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    shuffle=False,
                    drop_last=False,
                ),
                get_set_labels(data_cfg, bench_split),
                True,
            )

        infer_stats = trainer.benchmark_inference(
            bench_loader,
            use_ema=True,
            warmup_batches=warmup_batches,
            max_batches=max_batches,
        )

        print(
            f"[Efficiency] split='{bench_split}': "
            f"pairs/s={infer_stats['infer_pairs_per_s']:.2f}, "
            f"peak_vram_gb={infer_stats['infer_peak_vram_gb']:.3f}, "
            f"elapsed_s={infer_stats['infer_elapsed_s']:.3f}, "
            f"total_pairs={int(infer_stats['infer_total_pairs'])}"
        )

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

    if cfg.run.get("eval_test_after_train", False):
        print("\n[Train] eval_test_after_train=True, start evaluating on test set...")

        eval_with_last = cfg.run.get("eval_test_with_last", True)
        eval_with_best = cfg.run.get("eval_test_with_best", False)

        best_ckpt_path_cfg = cfg.run.get("best_ckpt_path", None)
        default_best_ckpt_path = str(ckpt_dir / "best.pt")
        best_ckpt_path = best_ckpt_path_cfg or default_best_ckpt_path

        test_splits = cfg.run.get("test_splits", ["test"])

        experiment_cfg = cfg.get("experiment", None)
        task_mode = getattr(experiment_cfg, "task", None) if experiment_cfg is not None else None
        pair_cfg = cfg.data.get("pair", None)

        def run_test_eval_for_current_trainer(tag_prefix: str) -> None:
            for split_idx in test_splits:
                print(f"[Train][{tag_prefix}] Building test loader for split='{split_idx}'")

                if task_mode == "pair_level_train":
                    if pair_cfg is None:
                        raise ValueError(
                            "[Train] experiment.task='pair_level_train' but cfg.data.pair is missing."
                        )

                    test_ds, test_loader = build_pair_level_dataset_and_loader(
                        pair_cfg=pair_cfg,
                        split=split_idx,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        shuffle=False,
                        drop_last=False,
                    )

                    test_set_labels = None

                else:
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

                test_root = eval_dir / "test" / str(split_idx) / tag_prefix
                test_root.mkdir(parents=True, exist_ok=True)

                from copy import deepcopy

                task_fixed = OmegaConf.create(OmegaConf.to_container(cfg.task, resolve=True))
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
                    do_threshold_sweep=False,
                    sweep_num_thresholds=cfg.eval.sweep_num_thresholds,
                    reduction=cfg.run.get("test_reduction", "max"),
                    softmax_temp=cfg.run.get("test_softmax_temp", 1.0),
                    topk=cfg.run.get("test_topk", 3),
                )

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
                        do_threshold_sweep=False,
                        sweep_num_thresholds=cfg.eval.sweep_num_thresholds,
                        reduction=cfg.run.get("test_reduction", "max"),
                        softmax_temp=cfg.run.get("test_softmax_temp", 1.0),
                        topk=cfg.run.get("test_topk", 3),
                    )
                else:
                    print(
                        f"[Train][Test {split_idx}][{tag_prefix}] Skip val-best eval because best_threshold is None."
                    )

                task_sweep = OmegaConf.create(OmegaConf.to_container(cfg.task, resolve=True))
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
                    do_threshold_sweep=True,
                    sweep_num_thresholds=cfg.eval.sweep_num_thresholds,
                    reduction=cfg.run.get("test_reduction", "max"),
                    softmax_temp=cfg.run.get("test_softmax_temp", 1.0),
                    topk=cfg.run.get("test_topk", 3),
                )

                best_thr_test = res_sweep.get("best_threshold", None)

                metrics_fixed = res_fixed.get("metrics", {})

                print(f"\n[Test {split_idx}][{tag_prefix}] Fixed threshold=0.5 metrics:")
                for k, v in iter_scalar_metrics(metrics_fixed):
                    print(f"  {k}: {v:.4f}")

                cm_fixed = metrics_fixed.get("confusion_matrix", None)
                if cm_fixed is not None:
                    print("  confusion_matrix:")
                    print(np.array(cm_fixed))

                metrics_valbest = (
                    res_valbest.get("metrics", {}) if res_valbest is not None else None
                )
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

                if wandb_run is not None:
                    prefix = f"test/{split_idx}/{tag_prefix}"
                    for k, v in iter_scalar_metrics(metrics_fixed):
                        wandb_run.summary[f"{prefix}_thr0.5/{k}"] = v
                    if metrics_valbest is not None:
                        for k, v in iter_scalar_metrics(metrics_valbest):
                            wandb_run.summary[f"{prefix}_valbest/{k}"] = v
                    for k, v in iter_scalar_metrics(metrics_sweep):
                        wandb_run.summary[f"{prefix}_sweep/{k}"] = v
                    if best_thr_test is not None:
                        try:
                            wandb_run.summary[f"{prefix}_sweep/best_threshold"] = float(
                                best_thr_test
                            )
                        except (TypeError, ValueError):
                            pass

        if eval_with_last:
            print(
                "\n[Train] Evaluating on test set with LAST checkpoint (current trainer state)..."
            )
            run_test_eval_for_current_trainer(tag_prefix="last")

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

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
