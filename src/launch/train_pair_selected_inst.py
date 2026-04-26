# src/launch/train_pair_selected_inst.py
from __future__ import annotations

import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from pathlib import Path
import os

import hydra
import torch
import torch.nn as nn
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from src.config.data_config import DataConfig
from src.em.cheap_runner import load_ckpt_into_model
from src.launch.pair_selected_runtime import (
    build_selected_loader,
    default_best_checkpoint,
    evaluate_prediction_arrays,
    log_epoch_to_wandb,
    print_scalar_metrics,
    resolve_pair_selected_ckpt_dir,
    run_selected_predict,
    summarize_metrics_to_wandb,
)
from src.launch.train import setup_wandb
from src.models.registry import build_model
from src.trainer.trainer_pair_selected import PairSelectedTrainer, PairSelectedTrainerConfig
from src.utils import set_seeds

# DDP imports
from src.utils.ddp import setup_ddp, cleanup_ddp, is_ddp, is_rank0, get_rank, get_world_size, print_on_rank0, barrier
from src.utils.ddp_sampler import get_ddp_sampler, set_epoch_for_sampler, disable_persistent_workers_if_ddp


def _apply_sync_batchnorm(model: torch.nn.Module) -> torch.nn.Module:
    """Convert BatchNorm1d layers to SyncBatchNorm for DDP training."""
    return nn.SyncBatchNorm.convert_sync_batchnorm(model)


def _resolve_ckpt(path_str: str | None, orig_cwd: Path | None) -> Path | None:
    if path_str is None:
        return None
    p = Path(os.path.expandvars(os.path.expanduser(str(path_str))))
    if not p.is_absolute():
        p = (orig_cwd if orig_cwd is not None else Path(get_original_cwd())) / p
    return p


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # ---- DDP setup ----
    # Must call setup_ddp() unconditionally — is_ddp() checks dist.is_initialized()
    # which is False before setup_ddp() is called (chicken-and-egg).
    rank, local_rank, world_size = setup_ddp()
    if world_size > 1:
        print_on_rank0(f"[train_pair_selected_inst] DDP mode: rank={rank} local_rank={local_rank} world_size={world_size}")
        device = torch.device(f"cuda:{local_rank}")
    else:
        print("[train_pair_selected_inst] Single GPU mode")
        device_str = str(cfg.get("device", "cuda")) if torch.cuda.is_available() else "cpu"
        if device_str == "cuda":
            device_str = "cuda:0"
        device = torch.device(device_str)

    seed = int(cfg.get("seed", 2020))
    set_seeds(seed + rank)  # offset seed per rank for DDP

    run_dir = Path.cwd()
    orig_cwd = Path(get_original_cwd())
    ckpt_dir = resolve_pair_selected_ckpt_dir(orig_cwd, cfg)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- wandb (rank 0 only) ----
    wandb_run = setup_wandb(cfg)

    split_train = str(cfg.run.get("split", "train"))
    split_val = str(cfg.run.get("val_split", "val"))
    cache_root = str(cfg.scalable.cache_root)
    max_train_pairs = cfg.scalable.get("max_train_pairs", None)
    max_val_pairs = cfg.scalable.get("max_val_pairs", None)

    batch_size = int(cfg.run.get("batch_size", 16))
    num_workers = int(cfg.run.get("num_workers", 4))

    # In DDP mode, disable persistent_workers to avoid stale workers
    persistent_workers = bool(cfg.run.get("persistent_workers", True))
    if is_ddp():
        persistent_workers = disable_persistent_workers_if_ddp(persistent_workers, num_workers)

    # ---- Build datasets ----
    train_ds, _ = build_selected_loader(
        cache_root=cache_root,
        split=split_train,
        cache_type="selected_inst",
        batch_size=batch_size,
        num_workers=num_workers,
        max_pairs=(None if max_train_pairs is None else int(max_train_pairs)),
        shuffle=True,
    )
    val_ds, _ = build_selected_loader(
        cache_root=cache_root,
        split=split_val,
        cache_type="selected_inst",
        batch_size=batch_size,
        num_workers=num_workers,
        max_pairs=(None if max_val_pairs is None else int(max_val_pairs)),
        shuffle=False,
    )

    # Build DataLoaders with proper DDP support
    train_sampler = get_ddp_sampler(train_ds, shuffle=True)
    val_sampler = get_ddp_sampler(val_ds, shuffle=False)

    use_persistent = persistent_workers and (num_workers > 0)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # only shuffle if no sampler
        num_workers=num_workers,
        collate_fn=train_ds.collate_fn,
        pin_memory=True,
        persistent_workers=use_persistent,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_ds.collate_fn,
        pin_memory=True,
        persistent_workers=use_persistent,
        drop_last=False,
    )

    print_on_rank0(
        f"[train_pair_selected_inst] train_split={split_train} val_split={split_val} "
        f"train_pairs={len(train_ds)} val_pairs={len(val_ds)} batch_size={batch_size}"
    )

    # ---- Build models ----
    data_cfg = DataConfig.from_omegaconf(cfg.data)
    agg_cfg = cfg.model
    agg_model = build_model(str(agg_cfg.get("arch", agg_cfg.get("name"))), agg_cfg, data_cfg=data_cfg).to(device)

    # Apply SyncBatchNorm for DDP
    if is_ddp():
        agg_model = _apply_sync_batchnorm(agg_model)

    instance_model = None
    if bool(cfg.run.get("load_instance_into_ckpt", False)):
        inst_cfg = cfg.instance_model
        instance_model = build_model(str(inst_cfg.get("arch", inst_cfg.get("name"))), inst_cfg, data_cfg=data_cfg).to(device)

        if is_ddp():
            instance_model = _apply_sync_batchnorm(instance_model)

        inst_ckpt = _resolve_ckpt(cfg.get("instance_ckpt_path", None), orig_cwd=run_dir)
        if inst_ckpt is not None and inst_ckpt.exists():
            load_ckpt_into_model(instance_model, inst_ckpt, device=device, use_ema_shadow=False)

    # ---- Build trainer ----
    tr_node = cfg.trainer_pair_selected
    betas = tr_node.get("betas", [0.9, 0.999])
    tr_cfg = PairSelectedTrainerConfig(
        num_epochs=int(tr_node.get("num_epochs", cfg.run.num_epochs)),
        log_every=int(tr_node.get("log_every", 50)),
        grad_accum_steps=int(tr_node.get("grad_accum_steps", 1)),
        clip_grad_norm=float(tr_node.get("clip_grad_norm", 0.0)),
        use_amp=bool(tr_node.get("use_amp", False)),
        optimizer=str(tr_node.get("optimizer", "adamw")),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(tr_node.get("eps", 1e-8)),
        amsgrad=bool(tr_node.get("amsgrad", False)),
        lr_agg=float(tr_node.get("lr_agg", 3e-4)),
        wd_agg=float(tr_node.get("wd_agg", 1e-2)),
        lr_inst=float(tr_node.get("lr_inst", 1e-5)),
        wd_inst=float(tr_node.get("wd_inst", 0.0)),
        scheduler_agg=str(tr_node.get("scheduler_agg", "cosine")),
        scheduler_inst=str(tr_node.get("scheduler_inst", "cosine")),
        scheduler_t_max=int(tr_node.get("scheduler_t_max", cfg.run.num_epochs)),
        scheduler_step_size=int(tr_node.get("scheduler_step_size", 10)),
        scheduler_gamma=float(tr_node.get("scheduler_gamma", 0.1)),
        scheduler_factor=float(tr_node.get("scheduler_factor", 0.2)),
        scheduler_patience=int(tr_node.get("scheduler_patience", 5)),
        warmup_steps=int(tr_node.get("warmup_steps", 0)),
        loss_type=str(tr_node.get("loss_type", "bce")),
        focal_alpha=float(tr_node.get("focal_alpha", 0.25)),
        focal_gamma=float(tr_node.get("focal_gamma", 2.0)),
        focal_lambda=float(tr_node.get("focal_lambda", 1.0)),
        label_smoothing=bool(tr_node.get("label_smoothing", False)),
        smooth_neg=float(tr_node.get("smooth_neg", 0.0)),
        smooth_pos=float(tr_node.get("smooth_pos", 1.0)),
        bce_lambda=float(tr_node.get("bce_lambda", 1.0)),
        pos_weight=float(tr_node.get("pos_weight", 1.0)),
        monitor=str(tr_node.get("monitor", "loss")),
        greater_is_better=bool(tr_node.get("greater_is_better", False)),
        token_dropout_rate=float(tr_node.get("token_dropout_rate", 0.0)),
        swa_enabled=bool(tr_node.get("swa_enabled", False)),
        swa_start_epoch=int(tr_node.get("swa_start_epoch", 70)),
        swa_lr=float(tr_node.get("swa_lr", 1e-5)),
    )

    assemble = cfg.token_provider.assemble
    trainer = PairSelectedTrainer(
        cfg=tr_cfg,
        device=device,
        agg_model=agg_model,
        instance_model=instance_model,
        task_cfg=cfg.task,
        use_inst_emb=bool(assemble.get("use_inst_emb", True)),
        use_inst_logit=bool(assemble.get("use_inst_logit", True)),
        use_esa=bool(assemble.get("use_esa", True)),
        use_pos=bool(assemble.get("use_pos", True)),
        train_instance_model=False,
        local_rank=local_rank,
    )

    # ---- Resume ----
    resumed = False
    if bool(cfg.run.get("resume", False)) or (cfg.run.get("checkpoint", None) is not None):
        ckpt_path_cfg = cfg.run.get("checkpoint", None)
        ckpt_path = _resolve_ckpt(ckpt_path_cfg, orig_cwd=run_dir) if ckpt_path_cfg is not None else (ckpt_dir / "best.pt")
        if ckpt_path is not None and ckpt_path.exists():
            trainer.load_checkpoint(str(ckpt_path), map_location=device)
            print_on_rank0(f"[train_pair_selected_inst] Resumed from checkpoint: {ckpt_path}")
            resumed = True
        else:
            print_on_rank0(f"[train_pair_selected_inst] No checkpoint found at {ckpt_path}, start from scratch.")

    start_epoch = int(trainer.epoch) + 1 if resumed else int(trainer.epoch)

    # ---- Training loop ----
    for epoch in range(start_epoch, int(tr_cfg.num_epochs)):
        trainer.epoch = epoch

        # Set sampler epoch for proper shuffling in DDP
        set_epoch_for_sampler(train_loader, epoch)
        set_epoch_for_sampler(val_loader, epoch)

        train_metrics = trainer.train_one_epoch(train_loader)
        val_metrics = trainer.validate_one_epoch(val_loader)
        if float(train_metrics.get("optimizer_steps", 0.0)) > 0:
            trainer.step_schedulers(val_metrics)
        improved = trainer.update_best(val_metrics)

        print_on_rank0(
            f"[Epoch {epoch+1}/{tr_cfg.num_epochs}] "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_f1={float(val_metrics.get('f1', float('nan'))):.4f} "
            f"val_pr_auc={float(val_metrics.get('pr_auc', float('nan'))):.4f}"
        )
        log_epoch_to_wandb(
            wandb_run=wandb_run,
            epoch=epoch,
            global_step=trainer.global_step,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )
        summarize_metrics_to_wandb(wandb_run=wandb_run, prefix="val", metrics=val_metrics)

        trainer.save_checkpoint(str(ckpt_dir / "last.pt"))
        if improved:
            trainer.save_checkpoint(str(ckpt_dir / "best.pt"))

    # ---- Post-training evaluation ----
    if bool(cfg.run.get("eval_test_after_train", False)):
        eval_with_last = bool(cfg.run.get("eval_test_with_last", True))
        eval_with_best = bool(cfg.run.get("eval_test_with_best", False))
        eval_fixed = bool(cfg.run.get("eval_fixed_threshold", True))
        eval_valbest = bool(cfg.run.get("eval_with_val_best_threshold", False))
        eval_sweep = bool(cfg.run.get("eval_with_threshold_sweep", False))
        sweep_num_thresholds = int(cfg.run.get("sweep_num_thresholds", 101))
        test_splits = cfg.run.get("test_splits", ["test"])
        if isinstance(test_splits, str):
            test_splits = [test_splits]
        best_ckpt_path = _resolve_ckpt(cfg.run.get("best_ckpt_path", str(default_best_checkpoint(ckpt_dir))), orig_cwd=run_dir)

        def run_test_eval_for_current_trainer(tag_prefix: str) -> None:
            val_best_threshold = None
            if eval_valbest:
                pred_val = run_selected_predict(
                    trainer=trainer,
                    cache_root=cache_root,
                    split=split_val,
                    cache_type="selected_inst",
                    batch_size=batch_size,
                    num_workers=num_workers,
                    max_pairs=None,
                )
                val_sweep = evaluate_prediction_arrays(
                    y_true=pred_val["labels"],
                    logits=pred_val["logits"],
                    task_cfg=cfg.task,
                    fixed_threshold=float(cfg.task.get("threshold", 0.5)),
                    enable_fixed=False,
                    enable_val_best=False,
                    val_best_threshold=None,
                    enable_sweep=True,
                    sweep_num_thresholds=sweep_num_thresholds,
                )
                val_best_threshold = float(val_sweep["sweep_best_threshold"])
                print_on_rank0(f"[train_pair_selected_inst][{tag_prefix}] val_best_threshold={val_best_threshold:.6f}")

            for sp in [str(x) for x in list(test_splits)]:
                print_on_rank0(f"[train_pair_selected_inst][{tag_prefix}] Evaluating split='{sp}'")
                pred_test = run_selected_predict(
                    trainer=trainer,
                    cache_root=cache_root,
                    split=sp,
                    cache_type="selected_inst",
                    batch_size=batch_size,
                    num_workers=num_workers,
                    max_pairs=None,
                )
                metrics_by_strategy = evaluate_prediction_arrays(
                    y_true=pred_test["labels"],
                    logits=pred_test["logits"],
                    task_cfg=cfg.task,
                    fixed_threshold=float(cfg.task.get("threshold", 0.5)),
                    enable_fixed=eval_fixed,
                    enable_val_best=eval_valbest,
                    val_best_threshold=val_best_threshold,
                    enable_sweep=eval_sweep,
                    sweep_num_thresholds=sweep_num_thresholds,
                )
                for name in ("thr0_5", "val_best", "sweep"):
                    if name in metrics_by_strategy:
                        print_scalar_metrics(
                            f"[train_pair_selected_inst][{tag_prefix}][{sp}][{name}] metrics:",
                            metrics_by_strategy[name],
                        )
                        summarize_metrics_to_wandb(
                            wandb_run=wandb_run,
                            prefix=f"test/{sp}/{tag_prefix}/{name}",
                            metrics=metrics_by_strategy[name],
                        )

        if eval_with_last:
            run_test_eval_for_current_trainer("last")
        if eval_with_best and best_ckpt_path is not None and best_ckpt_path.exists():
            trainer.load_checkpoint(str(best_ckpt_path), map_location=device)
            print_on_rank0(f"[train_pair_selected_inst] Loaded BEST checkpoint for test eval: {best_ckpt_path}")
            run_test_eval_for_current_trainer("best")

    # ---- Cleanup ----
    if wandb_run is not None:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    if is_ddp():
        cleanup_ddp()


if __name__ == "__main__":
    main()
