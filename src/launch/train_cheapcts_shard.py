from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.data_config import DataConfig
from src.data.sampler import ChunkAwareBatchSampler
from src.data.stream_pair_dataset import StreamPairDataset
from src.data.window_shard_collate import window_shard_collate
from src.data.window_shard_dataset import WindowShardDataset
from src.distill.teacher_extractor import TeacherRunner
from src.evaluator.metrics import compute_metrics
from src.launch.train import load_model_state, setup_wandb
from src.models.registry import build_model
from src.trainer.loss import BinaryClassificationLoss
from src.utils import set_seeds


def _pair_labels(data_cfg: DataConfig, split: str) -> List[float]:
    return [float(rec.label) for rec in StreamPairDataset(data_cfg, split).iter_records()]


def _resolve_ckpt_dir(run_dir: Path, cfg: DictConfig) -> Path:
    subdir = str(cfg.run.get("ckpt_subdir", "checkpoints"))
    exp_name = str(cfg.get("experiment_name", "cheapcts_shard"))
    return run_dir / subdir / exp_name


def _make_loader(
    root: str, split: str, batch_size: int, num_workers: int, max_samples=None, shuffle=False
):
    ds = WindowShardDataset(
        root,
        split=split,
        include_ignore=True,
        max_samples=max_samples,
        teacher_root=(root if bool(max_samples is None) and False else None),
    )
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
    return ds, ld


def _distill_bce(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    T = max(float(temperature), 1e-6)
    teacher_prob = torch.sigmoid(teacher_logits / T).detach()
    return F.binary_cross_entropy_with_logits(student_logits / T, teacher_prob) * (T * T)


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

    teacher_root = (
        str(cfg.run.get("teacher_cache_root", cache_root))
        if bool(cfg.run.get("use_teacher_shard", False))
        else None
    )
    train_ds = WindowShardDataset(
        cache_root,
        split="train",
        include_ignore=True,
        max_samples=max_train,
        teacher_root=teacher_root,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=ChunkAwareBatchSampler(
            train_ds.cum_sizes, batch_size=batch_size, drop_last=False, shuffle=True
        ),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=window_shard_collate,
    )
    val_ds = WindowShardDataset(
        cache_root, split="val", include_ignore=True, max_samples=max_val, teacher_root=teacher_root
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=ChunkAwareBatchSampler(
            val_ds.cum_sizes, batch_size=batch_size, drop_last=False, shuffle=False
        ),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=window_shard_collate,
    )
    val_pair_labels = _pair_labels(data_cfg, "val")
    print(
        f"[train_cheapcts_shard] train_samples={len(train_ds)} val_samples={len(val_ds)} batch_size={batch_size}"
    )

    student = build_model(
        str(cfg.model.get("arch", cfg.model.get("name"))), cfg.model, data_cfg=data_cfg
    ).to(device)
    teacher_runner = None
    if not bool(cfg.run.get("use_teacher_shard", False)):
        teacher = build_model(
            str(cfg.run.distill_teacher_arch), cfg.run.distill_teacher_model, data_cfg=data_cfg
        ).to(device)
        load_model_state(teacher, str(cfg.run.distill_teacher_ckpt), device)
        teacher_runner = TeacherRunner(
            teacher=teacher,
            device=device,
            amp_enabled=bool(
                cfg.run.get("distill_teacher_amp", True) and cfg.train.get("amp", False)
            ),
            need_feat=bool(cfg.run.get("distill_need_feat", True)),
        )

    student_proj: nn.Module | None = None
    if float(cfg.run.get("distill_beta_feat", 0.0)) > 0:
        student_proj = nn.Linear(
            int(cfg.run.get("distill_student_emb_dim", 64)),
            int(cfg.run.get("distill_teacher_feat_dim", 384)),
        ).to(device)

    criterion = BinaryClassificationLoss(train_cfg=cfg.train)
    params = list(student.parameters()) + (
        [] if student_proj is None else list(student_proj.parameters())
    )
    optimizer = optim.AdamW(
        params,
        lr=float(cfg.train.get("lr", 6e-3)),
        weight_decay=float(cfg.train.get("weight_decay", 0.0)),
    )
    amp_enabled = bool(cfg.train.get("amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    monitor = str(cfg.train.get("monitor", "f1"))
    greater_is_better = bool(cfg.train.get("greater_is_better", True))
    best_metric = -1e9 if greater_is_better else 1e9

    alpha_start = float(cfg.run.get("distill_alpha_start", 0.8))
    alpha_end = float(cfg.run.get("distill_alpha_end", 0.5))
    beta_kd = float(cfg.run.get("distill_beta_kd", 1.0))
    beta_feat = float(cfg.run.get("distill_beta_feat", 0.1))
    temperature = float(cfg.run.get("distill_T", 2.0))
    num_epochs = int(cfg.run.get("num_epochs", 30))
    resume = bool(cfg.run.get("resume", False))
    ckpt_path = Path(str(cfg.run.get("checkpoint", ckpt_dir / "best.pt")))
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        if "state_dict" in ckpt:
            student.load_state_dict(ckpt["state_dict"], strict=False)
        if student_proj is not None and "student_proj" in ckpt:
            student_proj.load_state_dict(ckpt["student_proj"], strict=False)
        best_metric = float(ckpt.get("best_metric", best_metric))
        print(f"[train_cheapcts_shard] Resumed from {ckpt_path}")

    for epoch in range(num_epochs):
        student.train()
        if student_proj is not None:
            student_proj.train()
        train_loss = 0.0
        train_seen = 0
        alpha = alpha_start + (alpha_end - alpha_start) * (epoch / max(1, num_epochs - 1))

        for batch in tqdm(train_loader, desc=f"train_cheapcts_shard epoch {epoch}"):
            x = batch["inputs"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True).view(-1)
            esa = batch["esa_scores"].to(device, non_blocking=True).view(-1)
            pos = batch["pos"].to(device, non_blocking=True).view(-1)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True).view(-1)
            x = x.float()
            if valid_mask.sum().item() == 0:
                continue

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                if bool(cfg.run.get("use_teacher_shard", False)):
                    teacher_logits = (
                        batch["teacher_logit"].to(device, non_blocking=True).view(-1)[valid_mask]
                    )
                    teacher_feat = None
                    if "teacher_feat" in batch:
                        teacher_feat = batch["teacher_feat"].to(device, non_blocking=True)[
                            valid_mask
                        ]
                else:
                    assert teacher_runner is not None
                    teacher_feat, teacher_logits = teacher_runner(x[valid_mask])

            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                student_emb, student_logits, student_emb_raw = student(
                    x,
                    esa_scores=esa,
                    pos=pos,
                    return_normalized_emb=True,
                    return_emb_raw=True,
                )
                sup = criterion.compute(student_logits[valid_mask], y[valid_mask])
                kd = _distill_bce(student_logits[valid_mask], teacher_logits, temperature)
                feat = torch.zeros((), device=device)
                if student_proj is not None and teacher_feat is not None:
                    feat = F.mse_loss(
                        student_proj(student_emb_raw[valid_mask]), teacher_feat.detach()
                    )
                loss = (1.0 - alpha) * sup + alpha * (beta_kd * kd + beta_feat * feat)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bs = int(valid_mask.sum().item())
            train_loss += float(loss.item()) * bs
            train_seen += bs

        student.eval()
        if student_proj is not None:
            student_proj.eval()
        val_loss = 0.0
        val_seen = 0
        all_logits: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        all_pair: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"val_cheapcts_shard epoch {epoch}"):
                x = batch["inputs"].to(device, non_blocking=True).float()
                y = batch["labels"].to(device, non_blocking=True).view(-1)
                esa = batch["esa_scores"].to(device, non_blocking=True).view(-1)
                pos = batch["pos"].to(device, non_blocking=True).view(-1)
                pair = batch["set_idx"].to(device, non_blocking=True).view(-1)
                valid_mask = batch["valid_mask"].to(device, non_blocking=True).view(-1)
                if valid_mask.sum().item() == 0:
                    continue
                _, logits = student(x, esa_scores=esa, pos=pos)
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
            pair_np = torch.cat(all_pair).numpy().astype("int64")
            pair_logits = np.full((len(val_pair_labels),), -1e9, dtype=np.float32)
            for pid, lg in zip(pair_np.tolist(), logits_np.tolist()):
                if 0 <= int(pid) < len(val_pair_labels):
                    pair_logits[int(pid)] = max(float(pair_logits[int(pid)]), float(lg))
            metrics.update(
                compute_metrics(
                    y_true=np.asarray(val_pair_labels, dtype=np.float32),
                    y_pred_raw=pair_logits,
                    task_cfg=cfg.task,
                )
            )

        current = float(metrics.get(monitor, metrics["loss"]))
        improved = (current > best_metric) if greater_is_better else (current < best_metric)
        state = {
            "state_dict": student.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
        }
        if student_proj is not None:
            state["student_proj"] = student_proj.state_dict()
        if improved:
            best_metric = current
            state["best_metric"] = best_metric
            torch.save(state, ckpt_dir / "best.pt")
        torch.save(state, ckpt_dir / "last.pt")

        print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"train_loss={train_loss / max(1, train_seen):.4f} "
            f"val_loss={metrics['loss']:.4f} val_f1={float(metrics.get('f1', float('nan'))):.4f} "
            f"val_pr_auc={float(metrics.get('pr_auc', float('nan'))):.4f}"
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
        test_ds = WindowShardDataset(
            cache_root, split="test", include_ignore=True, teacher_root=teacher_root
        )
        test_loader = DataLoader(
            test_ds,
            batch_sampler=ChunkAwareBatchSampler(
                test_ds.cum_sizes, batch_size=batch_size, drop_last=False, shuffle=False
            ),
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            collate_fn=window_shard_collate,
        )
        test_pair_labels = _pair_labels(data_cfg, "test")
        student.eval()
        all_logits = []
        all_pair = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="test_cheapcts_shard"):
                x = batch["inputs"].to(device, non_blocking=True).float()
                esa = batch["esa_scores"].to(device, non_blocking=True).view(-1)
                pos = batch["pos"].to(device, non_blocking=True).view(-1)
                pair = batch["set_idx"].to(device, non_blocking=True).view(-1)
                valid_mask = batch["valid_mask"].to(device, non_blocking=True).view(-1)
                if valid_mask.sum().item() == 0:
                    continue
                _, logits = student(x, esa_scores=esa, pos=pos)
                all_logits.append(logits[valid_mask].detach().cpu())
                all_pair.append(pair[valid_mask].detach().cpu())
        if all_logits:
            logits_np = torch.cat(all_logits).numpy()
            pair_np = torch.cat(all_pair).numpy().astype("int64")
            pair_logits = np.full((len(test_pair_labels),), -1e9, dtype=np.float32)
            for pid, lg in zip(pair_np.tolist(), logits_np.tolist()):
                if 0 <= int(pid) < len(test_pair_labels):
                    pair_logits[int(pid)] = max(float(pair_logits[int(pid)]), float(lg))
            test_metrics = compute_metrics(
                y_true=np.asarray(test_pair_labels, dtype=np.float32),
                y_pred_raw=pair_logits,
                task_cfg=cfg.task,
            )
            print(
                f"[test_cheapcts_shard] f1={float(test_metrics.get('f1', float('nan'))):.4f} "
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
