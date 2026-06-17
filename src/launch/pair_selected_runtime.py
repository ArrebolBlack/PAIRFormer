from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import copy
import functools
import json
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from src.data.selected_pair_collate import selected_pair_collate
from src.data.selected_pair_dataset import SelectedPairDataset
from src.evaluator.metrics import compute_metrics, find_best_threshold_by_f1, sweep_thresholds
from src.launch.train import iter_scalar_metrics


def build_selected_loader(
    *,
    cache_root: str,
    split: str,
    cache_type: str,
    batch_size: int,
    num_workers: int,
    max_pairs: Optional[int] = None,
    shuffle: bool = False,
    truncate_k: Optional[int] = None,
):
    ds = SelectedPairDataset(
        cache_root,
        split=split,
        cache_type=cache_type,
        max_pairs=max_pairs,
    )

    # 必须用模块级 partial，不能用局部闭包：spawn 启动的 DataLoader worker 要 pickle collate_fn，
    # 局部闭包 build_selected_loader.<locals>._collate 不可 pickle → 训后 test 评估会 spawn 失败崩溃。
    collate = functools.partial(selected_pair_collate, truncate_k=truncate_k)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=collate,
        drop_last=False,
    )
    return ds, loader


def log_epoch_to_wandb(
    *,
    wandb_run: Any,
    epoch: int,
    global_step: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
) -> None:
    if wandb_run is None:
        return
    import wandb  # type: ignore

    log_dict = {"epoch": int(epoch) + 1}
    for k, v in iter_scalar_metrics(train_metrics):
        if k == "optimizer_steps":
            continue
        log_dict[f"train/{k}"] = v
    for k, v in iter_scalar_metrics(val_metrics):
        log_dict[f"val/{k}"] = v
    wandb.log(log_dict, step=int(global_step))


def summarize_metrics_to_wandb(*, wandb_run: Any, prefix: str, metrics: Dict[str, float]) -> None:
    if wandb_run is None:
        return
    for k, v in iter_scalar_metrics(metrics):
        wandb_run.summary[f"{prefix}/{k}"] = v


def run_selected_eval(
    *,
    trainer,
    cache_root: str,
    split: str,
    cache_type: str,
    batch_size: int,
    num_workers: int,
    max_pairs: Optional[int] = None,
) -> Dict[str, float]:
    _, loader = build_selected_loader(
        cache_root=cache_root,
        split=split,
        cache_type=cache_type,
        batch_size=batch_size,
        num_workers=num_workers,
        max_pairs=max_pairs,
        shuffle=False,
    )
    return trainer.validate_one_epoch(loader)


def run_selected_predict(
    *,
    trainer,
    cache_root: str,
    split: str,
    cache_type: str,
    batch_size: int,
    num_workers: int,
    max_pairs: Optional[int] = None,
):
    _, loader = build_selected_loader(
        cache_root=cache_root,
        split=split,
        cache_type=cache_type,
        batch_size=batch_size,
        num_workers=num_workers,
        max_pairs=max_pairs,
        shuffle=False,
    )
    return trainer.predict(loader)


def clone_task_with_threshold(task_cfg: Any, threshold: float):
    if task_cfg is None:
        return {"problem_type": "binary_classification", "from_logits": True, "threshold": float(threshold)}
    try:
        cloned = OmegaConf.create(OmegaConf.to_container(task_cfg, resolve=True))
        cloned.threshold = float(threshold)
        return cloned
    except Exception:
        cloned = copy.deepcopy(task_cfg)
        if isinstance(cloned, dict):
            cloned["threshold"] = float(threshold)
        else:
            setattr(cloned, "threshold", float(threshold))
        return cloned


def evaluate_prediction_arrays(
    *,
    y_true: np.ndarray,
    logits: np.ndarray,
    task_cfg: Any,
    fixed_threshold: float = 0.5,
    enable_fixed: bool = True,
    enable_val_best: bool = False,
    val_best_threshold: Optional[float] = None,
    enable_sweep: bool = False,
    sweep_num_thresholds: int = 101,
) -> Dict[str, Dict[str, float] | float]:
    out: Dict[str, Dict[str, float] | float] = {}

    if enable_fixed:
        metrics_fixed = compute_metrics(
            y_true,
            logits,
            clone_task_with_threshold(task_cfg, fixed_threshold),
            return_details=False,
        )
        out["thr0_5"] = metrics_fixed

    if enable_val_best and (val_best_threshold is not None):
        metrics_valbest = compute_metrics(
            y_true,
            logits,
            clone_task_with_threshold(task_cfg, float(val_best_threshold)),
            return_details=False,
        )
        out["val_best"] = metrics_valbest
        out["val_best_threshold"] = float(val_best_threshold)

    if enable_sweep:
        details = compute_metrics(
            y_true,
            logits,
            clone_task_with_threshold(task_cfg, fixed_threshold),
            return_details=True,
        )
        y_prob = np.asarray(details["y_prob"]).reshape(-1)
        thresholds = np.linspace(0.0, 1.0, int(sweep_num_thresholds))
        sweep_res = sweep_thresholds(y_true, y_prob, thresholds=thresholds)
        best_thr_info = find_best_threshold_by_f1(sweep_res)
        metrics_sweep = compute_metrics(
            y_true,
            logits,
            clone_task_with_threshold(task_cfg, float(best_thr_info["threshold"])),
            return_details=False,
        )
        out["sweep"] = metrics_sweep
        out["sweep_best_threshold"] = float(best_thr_info["threshold"])

    return out


def write_eval_strategy_outputs(output_root: Path, metrics_by_strategy: Dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for k, v in metrics_by_strategy.items():
        if isinstance(v, dict):
            with open(output_root / f"{k}.json", "w") as f:
                json.dump({kk: float(vv) for kk, vv in v.items() if isinstance(vv, (int, float))}, f, indent=2, sort_keys=True)
        elif isinstance(v, (int, float)):
            with open(output_root / f"{k}.txt", "w") as f:
                f.write(f"{float(v):.8f}\n")


def print_scalar_metrics(title: str, metrics: Dict[str, Any]) -> None:
    print(title)
    for k, v in iter_scalar_metrics(metrics):
        print(f"  {k}: {v:.4f}")


def default_best_checkpoint(ckpt_dir: Path) -> Path:
    return ckpt_dir / "best.pt"


def resolve_pair_selected_ckpt_dir(run_dir: Path, cfg: Any) -> Path:
    ckpt_subdir = str(cfg.run.get("ckpt_subdir", "checkpoints"))
    if ckpt_subdir == "checkpoints":
        exp_name = str(cfg.run.get("checkpoint_experiment_name", cfg.get("experiment_name", "pair_selected")))
        return run_dir / "checkpoints" / exp_name
    return run_dir / ckpt_subdir
