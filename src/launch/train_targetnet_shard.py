from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.data_config import DataConfig
from src.data.sampler import ChunkAwareBatchSampler
from src.data.stream_pair_dataset import StreamPairDataset
from src.data.window_shard_collate import window_shard_collate
from src.data.window_shard_dataset import WindowShardDataset
from src.evaluator.metrics import compute_metrics
from src.launch.train import setup_wandb
from src.models.extractors import get_embedding_and_logit
from src.models.registry import build_model
from src.trainer.loss import BinaryClassificationLoss
from src.utils import set_seeds


def _pair_labels(data_cfg: DataConfig, split: str) -> List[float]:
    return [float(rec.label) for rec in StreamPairDataset(data_cfg, split).iter_records()]


def _resolve_ckpt_dir(run_dir: Path, cfg: DictConfig) -> Path:
    subdir = str(cfg.run.get("ckpt_subdir", "checkpoints"))
    exp_name = str(cfg.get("experiment_name", "targetnet_shard"))
    return run_dir / subdir / exp_name


def _forward_logits(model, x: torch.Tensor) -> torch.Tensor:
    _, logits = get_embedding_and_logit(model, x)
    return logits.view(-1)


def _make_loader(
    root: str, split: str, batch_size: int, num_workers: int, max_samples=None, shuffle=False
):
    ds = WindowShardDataset(root, split=split, include_ignore=False, max_samples=max_samples)
    use_chunk_aware = True
    if use_chunk_aware:
        batch_sampler = ChunkAwareBatchSampler(
            ds.cum_sizes,
            batch_size=batch_size,
            drop_last=False,
            shuffle=shuffle,
        )
        ld = DataLoader(
            ds,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            collate_fn=window_shard_collate,
        )
    else:
        ld = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            collate_fn=window_shard_collate,
            drop_last=False,
        )
    return ds, ld


def _aggregate_pair_logits(
    pair_np: np.ndarray, logits_np: np.ndarray, num_pairs: int
) -> np.ndarray:
    """Vectorized pair-level logit aggregation via max-pooling."""
    pair_logits = np.full((num_pairs,), -1e9, dtype=np.float32)
    valid = (pair_np >= 0) & (pair_np < num_pairs)
    np.maximum.at(pair_logits, pair_np[valid], logits_np[valid])
    return pair_logits


def _build_lr_scheduler(optimizer, train_cfg, num_epochs):
    """Build LR scheduler with optional linear warmup + cosine annealing."""
    sched_name = str(train_cfg.get("scheduler", "none")).lower()
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
    base_lr = float(train_cfg.get("lr", 5e-3))

    def warmup_fn(epoch):
        if warmup_epochs <= 0:
            return 1.0
        return min(1.0, epoch / warmup_epochs)

    main_scheduler = None

    if sched_name == "cosine":
        t_max = int(train_cfg.get("scheduler_t_max", num_epochs))
        eta_min = float(train_cfg.get("scheduler_eta_min", 1e-6))
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        )
    elif sched_name == "step":
        step_size = int(train_cfg.get("scheduler_step_size", 10))
        gamma = float(train_cfg.get("scheduler_gamma", 0.1))
        main_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_name == "plateau":
        factor = float(train_cfg.get("scheduler_factor", 0.2))
        patience = int(train_cfg.get("scheduler_patience", 5))
        greater_is_better = bool(train_cfg.get("greater_is_better", True))
        mode = "max" if greater_is_better else "min"
        main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience
        )

    if warmup_epochs > 0 and main_scheduler is not None:
        warmup_sched = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)
        return optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, main_scheduler],
            milestones=[warmup_epochs],
        )
    elif warmup_epochs > 0:
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)
    elif main_scheduler is not None:
        return main_scheduler
    return None


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_seeds(int(cfg.get("seed", 2020)))
    device = torch.device(
        "cuda" if torch.cuda.is_available() and str(cfg.get("device", "cuda")) != "cpu" else "cpu"
    )
    run_dir = Path.cwd()
    ckpt_dir = _resolve_ckpt_dir(run_dir, cfg)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = setup_wandb(cfg)

    data_cfg = DataConfig.from_omegaconf(cfg.data)
    batch_size = int(cfg.run.get("batch_size", 256))
    num_workers = int(cfg.run.get("num_workers", 4))
    cache_root = str(cfg.scalable.cache_root)
    max_train = cfg.scalable.get("max_train_samples", None)
    max_val = cfg.scalable.get("max_val_samples", None)

    train_ds, train_loader = _make_loader(
        cache_root, "train", batch_size, num_workers, max_train, shuffle=True
    )
    val_ds, val_loader = _make_loader(
        cache_root, "val", batch_size, num_workers, max_val, shuffle=False
    )
    val_pair_labels = _pair_labels(data_cfg, "val")
    print(
        f"[train_targetnet_shard] train_samples={len(train_ds)} val_samples={len(val_ds)} batch_size={batch_size}"
    )

    model = build_model(
        str(cfg.model.get("arch", cfg.model.get("name"))), cfg.model, data_cfg=data_cfg
    ).to(device)
    train_cfg = cfg.train
    criterion = BinaryClassificationLoss(train_cfg=train_cfg)
    lr = float(train_cfg.get("lr", 5e-3))
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    amp_enabled = bool(train_cfg.get("amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    monitor = str(train_cfg.get("monitor", "f1"))
    greater_is_better = bool(train_cfg.get("greater_is_better", True))
    best_metric = -1e9 if greater_is_better else 1e9
    resume = bool(cfg.run.get("resume", False))
    ckpt_path = Path(str(cfg.run.get("checkpoint", ckpt_dir / "best.pt")))
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=False)
        best_metric = float(ckpt.get("best_metric", best_metric))
        print(f"[train_targetnet_shard] Resumed from {ckpt_path}")

    num_epochs = int(cfg.run.get("num_epochs", 30))
    val_interval = int(cfg.run.get("val_interval", 1))
    scheduler = _build_lr_scheduler(optimizer, train_cfg, num_epochs)
    if scheduler is not None:
        print(
            f"[train_targetnet_shard] lr={lr}, scheduler={train_cfg.get('scheduler')}, warmup={train_cfg.get('warmup_epochs', 0)}"
        )
    print(f"[train_targetnet_shard] val_interval={val_interval}")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_seen = 0
        for batch in tqdm(train_loader, desc=f"train_targetnet_shard epoch {epoch}"):
            x = batch["inputs"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True).view(-1)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True).view(-1)
            x = x.float()
            if valid_mask.sum().item() == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                logits = _forward_logits(model, x)
                loss = criterion.compute(logits[valid_mask], y[valid_mask])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bs = int(valid_mask.sum().item())
            train_loss += float(loss.item()) * bs
            train_seen += bs

        # Step scheduler (per-epoch)
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                pass  # stepped after val
            else:
                scheduler.step()

        # Always save last.pt
        torch.save(
            {"state_dict": model.state_dict(), "epoch": epoch, "best_metric": best_metric},
            ckpt_dir / "last.pt",
        )

        do_val = (
            (val_interval <= 1)
            or ((epoch + 1) % val_interval == 0)
            or (epoch == 0)
            or (epoch == num_epochs - 1)
        )
        if not do_val:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[Epoch {epoch+1}/{num_epochs}] "
                f"train_loss={train_loss / max(1, train_seen):.4f} lr={current_lr:.6f} (val skipped)"
            )
            if wandb_run is not None:
                import wandb  # type: ignore

                wandb.log({"epoch": epoch + 1, "train/loss": train_loss / max(1, train_seen)})
            continue

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_seen = 0
        all_logits: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        all_pair: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"val_targetnet_shard epoch {epoch}"):
                x = batch["inputs"].to(device, non_blocking=True).float()
                y = batch["labels"].to(device, non_blocking=True).view(-1)
                pair = batch["set_idx"].to(device, non_blocking=True).view(-1)
                valid_mask = batch["valid_mask"].to(device, non_blocking=True).view(-1)
                if valid_mask.sum().item() == 0:
                    continue
                logits = _forward_logits(model, x)
                loss = criterion.compute(logits[valid_mask], y[valid_mask])
                bs = int(valid_mask.sum().item())
                val_loss += float(loss.item()) * bs
                val_seen += bs
                all_logits.append(logits[valid_mask].detach().cpu())
                all_labels.append(y[valid_mask].detach().cpu())
                all_pair.append(pair[valid_mask].detach().cpu())

        metrics: Dict[str, float] = {"loss": val_loss / max(1, val_seen)}
        if all_logits:
            logits_np = torch.cat(all_logits).numpy()
            labels_np = torch.cat(all_labels).numpy()
            pair_np = torch.cat(all_pair).numpy().astype("int64")
            pair_logits = _aggregate_pair_logits(pair_np, logits_np, len(val_pair_labels))
            metrics.update(
                compute_metrics(
                    y_true=np.asarray(val_pair_labels, dtype=np.float32),
                    y_pred_raw=pair_logits,
                    task_cfg=cfg.task,
                )
            )

        current = float(metrics.get(monitor, metrics["loss"]))
        improved = (current > best_metric) if greater_is_better else (current < best_metric)
        if improved:
            best_metric = current
            torch.save(
                {"state_dict": model.state_dict(), "epoch": epoch, "best_metric": best_metric},
                ckpt_dir / "best.pt",
            )

        # Step plateau scheduler with val metric
        if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(current)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"train_loss={train_loss / max(1, train_seen):.4f} "
            f"val_loss={metrics['loss']:.4f} val_f1={float(metrics.get('f1', float('nan'))):.4f} "
            f"val_pr_auc={float(metrics.get('pr_auc', float('nan'))):.4f} lr={current_lr:.6f}"
        )
        if wandb_run is not None:
            import wandb  # type: ignore

            log_dict = {"epoch": epoch + 1, "train/loss": train_loss / max(1, train_seen)}
            for k, v in metrics.items():
                try:
                    log_dict[f"val/{k}"] = float(v)
                except Exception:
                    pass
            wandb.log(log_dict)

    if bool(cfg.run.get("eval_test_after_train", False)):
        test_ds, test_loader = _make_loader(
            cache_root, "test", batch_size, num_workers, None, shuffle=False
        )
        test_pair_labels = _pair_labels(data_cfg, "test")
        model.eval()
        all_logits: List[torch.Tensor] = []
        all_pair: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="test_targetnet_shard"):
                x = batch["inputs"].to(device, non_blocking=True).float()
                pair = batch["set_idx"].to(device, non_blocking=True).view(-1)
                valid_mask = batch["valid_mask"].to(device, non_blocking=True).view(-1)
                if valid_mask.sum().item() == 0:
                    continue
                logits = _forward_logits(model, x)
                all_logits.append(logits[valid_mask].detach().cpu())
                all_pair.append(pair[valid_mask].detach().cpu())
        if all_logits:
            logits_np = torch.cat(all_logits).numpy()
            pair_np = torch.cat(all_pair).numpy().astype("int64")
            pair_logits = _aggregate_pair_logits(pair_np, logits_np, len(test_pair_labels))
            test_metrics = compute_metrics(
                y_true=np.asarray(test_pair_labels, dtype=np.float32),
                y_pred_raw=pair_logits,
                task_cfg=cfg.task,
            )
            print(
                f"[test_targetnet_shard] f1={float(test_metrics.get('f1', float('nan'))):.4f} "
                f"pr_auc={float(test_metrics.get('pr_auc', float('nan'))):.4f}"
            )
            if wandb_run is not None:
                import wandb  # type: ignore

                wandb.log(
                    {
                        f"test/{k}": float(v)
                        for k, v in test_metrics.items()
                        if isinstance(v, (int, float))
                    }
                )

    if wandb_run is not None:
        try:
            import wandb  # type: ignore

            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
