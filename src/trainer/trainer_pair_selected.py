from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from torch.optim.swa_utils import AveragedModel
import torch.nn as nn

from src.evaluator.metrics import compute_metrics
from src.models.extractors import get_embedding_and_logit
from src.trainer.loss import BinaryClassificationLoss

# DDP imports
from src.utils.ddp import (
    is_ddp, is_rank0, get_rank, get_world_size,
    all_reduce_dict, save_on_rank0, print_on_rank0,
)


@dataclass
class PairSelectedTrainerConfig:
    num_epochs: int = 10
    log_every: int = 50
    grad_accum_steps: int = 1
    clip_grad_norm: float = 0.0
    use_amp: bool = False

    optimizer: str = "adamw"
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False

    lr_agg: float = 3e-4
    wd_agg: float = 1e-2
    lr_inst: float = 1e-5
    wd_inst: float = 0.0

    scheduler_agg: str = "cosine"
    scheduler_inst: str = "cosine"
    scheduler_t_max: int = 10
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    scheduler_factor: float = 0.2
    scheduler_patience: int = 5
    warmup_steps: int = 0

    token_dropout_rate: float = 0.0

    loss_type: str = "bce"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    focal_lambda: float = 1.0
    label_smoothing: bool = False
    smooth_neg: float = 0.0
    smooth_pos: float = 1.0
    bce_lambda: float = 1.0
    pos_weight: float = 1.0

    monitor: str = "loss"
    greater_is_better: bool = False

    # SWA (Stochastic Weight Averaging)
    swa_enabled: bool = False
    swa_start_epoch: int = 70
    swa_lr: float = 1e-5


class PairSelectedTrainer:
    def __init__(
        self,
        *,
        cfg: PairSelectedTrainerConfig,
        device: torch.device,
        agg_model: torch.nn.Module,
        instance_model: Optional[torch.nn.Module],
        task_cfg: Optional[Any] = None,
        use_inst_emb: bool = True,
        use_inst_logit: bool = True,
        use_esa: bool = True,
        use_pos: bool = True,
        train_instance_model: bool = True,
        local_rank: int = 0,
    ):
        self.cfg = cfg
        self.device = device
        self.local_rank = local_rank

        # Wrap agg_model with DDP if applicable
        if is_ddp():
            self.agg_model = agg_model.to(device)
            ddp_kwargs = {}
            if device.type == "cuda":
                ddp_kwargs["device_ids"] = [local_rank]
                ddp_kwargs["output_device"] = local_rank
            self.agg_model_ddp = nn.parallel.DistributedDataParallel(
                self.agg_model, **ddp_kwargs,
            )
        else:
            self.agg_model = agg_model.to(device)
            self.agg_model_ddp = self.agg_model

        self.instance_model = instance_model.to(device) if instance_model is not None else None
        self.task_cfg = task_cfg

        self.use_inst_emb = bool(use_inst_emb)
        self.use_inst_logit = bool(use_inst_logit)
        self.use_esa = bool(use_esa)
        self.use_pos = bool(use_pos)
        self.train_instance_model = bool(train_instance_model and self.instance_model is not None)

        self.crit = BinaryClassificationLoss(train_cfg=cfg)
        self.opt_agg = optim.AdamW(
            self.agg_model.parameters(),
            lr=cfg.lr_agg,
            weight_decay=cfg.wd_agg,
            betas=cfg.betas,
            eps=cfg.eps,
            amsgrad=cfg.amsgrad,
        )
        self.opt_inst = None
        if self.train_instance_model:
            self.opt_inst = optim.AdamW(
                self.instance_model.parameters(),
                lr=cfg.lr_inst,
                weight_decay=cfg.wd_inst,
                betas=cfg.betas,
                eps=cfg.eps,
                amsgrad=cfg.amsgrad,
            )

        self.sched_agg = self._build_scheduler(self.opt_agg, cfg.scheduler_agg)
        self.sched_inst = self._build_scheduler(self.opt_inst, cfg.scheduler_inst) if self.opt_inst is not None else None
        self.amp_enabled = bool(cfg.use_amp and device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        # warmup state
        self.warmup_steps: int = max(0, int(getattr(cfg, "warmup_steps", 0)))
        self._warmup_done: bool = (self.warmup_steps == 0)
        self._base_lrs_agg: list = [pg["lr"] for pg in self.opt_agg.param_groups]

        self.token_dropout_rate: float = float(getattr(cfg, "token_dropout_rate", 0.0))

        # SWA
        self.swa_enabled: bool = bool(getattr(cfg, "swa_enabled", False))
        self.swa_start_epoch: int = int(getattr(cfg, "swa_start_epoch", 70))
        self.swa_lr: float = float(getattr(cfg, "swa_lr", 1e-5))
        self.swa_model: Optional[AveragedModel] = None
        self.swa_n_updates: int = 0
        if self.swa_enabled:
            self.swa_model = AveragedModel(self.agg_model)
            self.swa_model.to(device)

        self.best_metric = -1e9 if cfg.greater_is_better else 1e9
        self.epoch = 0
        self.global_step = 0

    def _build_scheduler(self, optimizer: optim.Optimizer, name: str):
        name = str(name).lower()
        if name == "none":
            return None
        if name == "plateau":
            mode = "max" if bool(self.cfg.greater_is_better) else "min"
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=mode,
                factor=float(self.cfg.scheduler_factor),
                patience=int(self.cfg.scheduler_patience),
            )
        if name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.cfg.scheduler_t_max))
        if name == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(self.cfg.scheduler_step_size),
                gamma=float(self.cfg.scheduler_gamma),
            )
        return None

    def _build_tokens(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        pair_id = batch["pair_id"].to(self.device, non_blocking=True)
        y_pair = batch["y_pair"].to(self.device, non_blocking=True)
        mask = batch["mask"].to(self.device, non_blocking=True)

        esa = batch["esa"].to(self.device, non_blocking=True)
        pos = batch["pos"].to(self.device, non_blocking=True)
        X = batch.get("X", None)
        inst_emb_cached = batch.get("inst_emb", None)
        inst_logit_cached = batch.get("inst_logit", None)

        if X is not None:
            X = X.to(self.device, non_blocking=True)
            B, K, C, L = X.shape
        else:
            B, K = esa.shape

        feat_valid = None
        logit_valid = None
        need_online = (
            (self.use_inst_emb and inst_emb_cached is None)
            or (self.use_inst_logit and inst_logit_cached is None)
        )

        if need_online:
            if self.instance_model is None:
                raise RuntimeError("Instance model is required for selected_raw or missing selected_inst features.")
            if X is None:
                raise RuntimeError("Batch does not contain raw windows X, but online instance forward is required.")

            X_flat = X.view(B * K, C, L).float()
            mask_flat = mask.view(B * K)
            valid = torch.nonzero(mask_flat, as_tuple=False).view(-1)

            if valid.numel() == 0:
                return {"pair_id": pair_id, "y_pair": y_pair, "mask": mask, "tokens": None}

            X_valid = X_flat.index_select(0, valid)
            esa_valid = esa.view(B * K).index_select(0, valid).float()
            pos_valid = pos.view(B * K).index_select(0, valid).float()

            with torch.amp.autocast(device_type="cuda", enabled=self.amp_enabled):
                feat_valid, logit_valid = get_embedding_and_logit(
                    self.instance_model,
                    X_valid,
                    esa_scores=esa_valid,
                    pos=pos_valid,
                )

        feats = []
        if self.use_inst_emb:
            if inst_emb_cached is not None:
                feats.append(inst_emb_cached.to(self.device, non_blocking=True))
            else:
                emb_flat = torch.zeros((B * K, feat_valid.shape[1]), device=self.device, dtype=feat_valid.dtype)
                emb_flat.index_copy_(0, valid, feat_valid)
                feats.append(emb_flat.view(B, K, -1))

        if self.use_inst_logit:
            if inst_logit_cached is not None:
                feats.append(inst_logit_cached.to(self.device, non_blocking=True).unsqueeze(-1))
            else:
                log_flat = torch.zeros((B * K,), device=self.device, dtype=logit_valid.dtype)
                log_flat.index_copy_(0, valid, logit_valid)
                feats.append(log_flat.view(B, K).unsqueeze(-1))

        if self.use_esa:
            feat_dtype = feats[0].dtype if feats else torch.float32
            feats.append(esa.unsqueeze(-1).to(device=self.device, dtype=feat_dtype))
        if self.use_pos:
            feat_dtype = feats[0].dtype if feats else torch.float32
            feats.append(pos.unsqueeze(-1).to(device=self.device, dtype=feat_dtype))

        if not feats:
            raise RuntimeError("At least one token feature must be enabled.")

        tokens = torch.cat(feats, dim=-1)
        tokens = tokens * mask.unsqueeze(-1).to(tokens.dtype)
        return {"pair_id": pair_id, "y_pair": y_pair, "mask": mask, "tokens": tokens}

    def train_one_epoch(self, loader: Any) -> Dict[str, float]:
        self.agg_model.train()
        if self.instance_model is not None:
            self.instance_model.train(mode=self.train_instance_model)

        total_loss = 0.0
        total_seen = 0
        grad_accum = max(1, int(self.cfg.grad_accum_steps))
        optimizer_steps = 0

        self.opt_agg.zero_grad(set_to_none=True)
        if self.opt_inst is not None:
            self.opt_inst.zero_grad(set_to_none=True)

        pbar = tqdm(loader, desc=f"PairSelected train epoch {self.epoch}")
        for it, batch in enumerate(pbar):
            out = self._build_tokens(batch)
            tokens = out["tokens"]
            y = out["y_pair"]
            mask = out["mask"]
            if tokens is None:
                self.global_step += 1
                continue

            with torch.amp.autocast(device_type="cuda", enabled=self.amp_enabled):
                # token dropout augmentation: randomly mask a fraction of tokens
                if self.token_dropout_rate > 0.0:
                    _B, _K = mask.shape
                    keep_prob = 1.0 - self.token_dropout_rate
                    rand = torch.rand(_B, _K, device=tokens.device)
                    token_keep_mask = (rand < keep_prob).float().unsqueeze(-1)
                    # at least keep 1 token per sample
                    for i in range(_B):
                        if token_keep_mask[i].sum() == 0:
                            idx = torch.randint(0, tokens.shape[1], (1,), device=tokens.device)
                            token_keep_mask[i, idx] = 1.0
                    tokens = tokens * token_keep_mask
                    # update attn_mask accordingly
                    mask = mask & (token_keep_mask.squeeze(-1) > 0)

                logits = self.agg_model_ddp(tokens, attn_mask=mask).view(-1)
                loss = self.crit.compute(logits, y.view(-1).float())
                loss_to_backward = loss / float(grad_accum)

            self.scaler.scale(loss_to_backward).backward()
            self.global_step += 1

            bs = int(y.shape[0])
            total_loss += float(loss.item()) * bs
            total_seen += bs

            is_last = (it == len(loader) - 1) if hasattr(loader, "__len__") else False
            if ((it + 1) % grad_accum == 0) or is_last:
                if float(self.cfg.clip_grad_norm) > 0:
                    self.scaler.unscale_(self.opt_agg)
                    if self.opt_inst is not None:
                        self.scaler.unscale_(self.opt_inst)
                    torch.nn.utils.clip_grad_norm_(self.agg_model.parameters(), float(self.cfg.clip_grad_norm))
                    if self.instance_model is not None and self.train_instance_model:
                        torch.nn.utils.clip_grad_norm_(self.instance_model.parameters(), float(self.cfg.clip_grad_norm))
                self.scaler.step(self.opt_agg)
                if self.opt_inst is not None:
                    self.scaler.step(self.opt_inst)
                self.scaler.update()
                self.opt_agg.zero_grad(set_to_none=True)
                if self.opt_inst is not None:
                    self.opt_inst.zero_grad(set_to_none=True)
                optimizer_steps += 1

                # linear warmup: override LR during warmup phase
                if not self._warmup_done:
                    if optimizer_steps < self.warmup_steps:
                        progress = optimizer_steps / max(1, self.warmup_steps)
                        for pg, base_lr in zip(self.opt_agg.param_groups, self._base_lrs_agg):
                            pg["lr"] = base_lr * progress
                    else:
                        self._warmup_done = True

                # SWA: constant LR + update averaged model
                if self.swa_enabled and self.epoch >= self.swa_start_epoch:
                    for pg in self.opt_agg.param_groups:
                        pg["lr"] = self.swa_lr
                    if self.swa_model is not None:
                        self.swa_model.update_parameters(self.agg_model)
                        self.swa_n_updates += 1

            if self.cfg.log_every > 0 and self.global_step % int(self.cfg.log_every) == 0:
                pbar.set_postfix({"loss": f"{total_loss / max(1, total_seen):.4f}"})

        return {
            "loss": total_loss / max(1, total_seen),
            "lr_agg": float(self.opt_agg.param_groups[0]["lr"]),
            "lr_inst": float(self.opt_inst.param_groups[0]["lr"]) if self.opt_inst is not None else 0.0,
            "optimizer_steps": float(optimizer_steps),
        }

    @torch.no_grad()
    def validate_one_epoch(self, loader: Any) -> Dict[str, float]:
        # Use SWA model for validation if SWA is active and has been updated
        use_swa = (
            self.swa_enabled
            and self.swa_model is not None
            and self.epoch >= self.swa_start_epoch
            and self.swa_n_updates > 0
        )
        # In DDP, use the DDP wrapper's module for SWA
        if use_swa:
            # AveragedModel wraps the original model, so use it directly
            eval_model = self.swa_model
        else:
            # Use DDP wrapper for evaluation (it's safe in eval mode)
            eval_model = self.agg_model_ddp
        eval_model.eval()
        if self.instance_model is not None:
            self.instance_model.eval()

        total_loss = 0.0
        total_seen = 0
        all_logits = []
        all_labels = []

        tag = "SWA" if use_swa else "val"
        for batch in tqdm(loader, desc=f"PairSelected {tag} epoch {self.epoch}"):
            out = self._build_tokens(batch)
            tokens = out["tokens"]
            y = out["y_pair"]
            mask = out["mask"]
            if tokens is None:
                continue

            logits = eval_model(tokens, attn_mask=mask).view(-1)
            loss = self.crit.compute(logits, y.view(-1).float())

            bs = int(y.shape[0])
            total_loss += float(loss.item()) * bs
            total_seen += bs
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.detach().cpu())

        metrics: Dict[str, float] = {"loss": total_loss / max(1, total_seen)}
        if all_labels:
            # Gather all logits/labels from all ranks
            if is_ddp():
                from src.utils.ddp import gather_tensors
                all_logits = gather_tensors(torch.cat(all_logits))
                all_labels = gather_tensors(torch.cat(all_labels))
                # Only process on rank 0
                if is_rank0():
                    logits_np = all_logits.numpy()
                    labels_np = all_labels.numpy()
                    cm = compute_metrics(labels_np, logits_np, self.task_cfg)
                    for k, v in cm.items():
                        try:
                            metrics[k] = float(v)
                        except Exception:
                            pass
            else:
                logits_np = torch.cat(all_logits).numpy()
                labels_np = torch.cat(all_labels).numpy()
                cm = compute_metrics(labels_np, logits_np, self.task_cfg)
                for k, v in cm.items():
                    try:
                        metrics[k] = float(v)
                    except Exception:
                        pass

        return metrics

    @torch.no_grad()
    def predict(self, loader: Any) -> Dict[str, Any]:
        # Use DDP wrapper for inference
        self.agg_model_ddp.eval()
        if self.instance_model is not None:
            self.instance_model.eval()

        all_logits = []
        all_labels = []
        all_pair_id = []

        for batch in tqdm(loader, desc=f"PairSelected predict epoch {self.epoch}"):
            out = self._build_tokens(batch)
            tokens = out["tokens"]
            y = out["y_pair"]
            mask = out["mask"]
            pair_id = out["pair_id"]
            if tokens is None:
                continue

            logits = self.agg_model_ddp(tokens, attn_mask=mask).view(-1)
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.detach().cpu())
            all_pair_id.append(pair_id.detach().cpu())

        return {
            "logits": torch.cat(all_logits).numpy() if all_logits else np.zeros((0,), dtype=np.float32),
            "labels": torch.cat(all_labels).numpy() if all_labels else np.zeros((0,), dtype=np.float32),
            "pair_id": torch.cat(all_pair_id).numpy() if all_pair_id else np.zeros((0,), dtype=np.int64),
        }

    def step_schedulers(self, metrics: Dict[str, float]) -> None:
        monitor_val = float(metrics.get(self.cfg.monitor, metrics.get("loss", 0.0)))
        for sched in [self.sched_agg, self.sched_inst]:
            if sched is None:
                continue
            if isinstance(sched, optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(monitor_val)
            else:
                sched.step()

    def update_best(self, metrics: Dict[str, float]) -> bool:
        v = float(metrics.get(self.cfg.monitor, metrics.get("loss", 0.0)))
        old = self.best_metric
        if self.cfg.greater_is_better:
            if v > self.best_metric:
                self.best_metric = v
        else:
            if v < self.best_metric:
                self.best_metric = v
        return self.best_metric != old

    def save_checkpoint(self, path: str) -> None:
        # Only save on rank 0
        save_on_rank0(
            self._save_checkpoint_impl,
            path,
        )

    def _save_checkpoint_impl(self, path: str) -> None:
        # Save raw model (not DDP wrapper)
        agg_state_dict = self.agg_model.state_dict()
        inst_state_dict = self.instance_model.state_dict() if self.instance_model is not None else None
        opt_agg = self.opt_agg.state_dict()
        opt_inst = self.opt_inst.state_dict() if self.opt_inst is not None else None
        sched_agg = None if self.sched_agg is None else self.sched_agg.state_dict()
        sched_inst = None if self.sched_inst is None else self.sched_inst.state_dict()
        scaler = None if not self.amp_enabled else self.scaler.state_dict()

        ckpt = {
            "agg_state_dict": agg_state_dict,
            "inst_state_dict": inst_state_dict,
            "opt_agg": opt_agg,
            "opt_inst": opt_inst,
            "sched_agg": sched_agg,
            "sched_inst": sched_inst,
            "scaler": scaler,
            "state": {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "best_metric": self.best_metric,
            },
            "cfg": self.cfg.__dict__,
        }
        if self.swa_model is not None:
            ckpt["swa_state_dict"] = self.swa_model.state_dict()
            ckpt["swa_n_updates"] = self.swa_n_updates
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str, *, map_location: Optional[torch.device] = None) -> None:
        ckpt = torch.load(path, map_location=("cpu" if map_location is None else map_location))

        # Helper to strip DDP wrapper prefix
        def strip_ddp_prefix(k: str) -> str:
            for prefix in ("module.",):
                if k.startswith(prefix):
                    return k[len(prefix):]
            return k

        # Process agg_state_dict
        agg_sd = ckpt.get("agg_state_dict", None)
        if isinstance(agg_sd, dict):
            agg_sd_cleaned = {strip_ddp_prefix(k): v for k, v in agg_sd.items()}
            self.agg_model.load_state_dict(agg_sd_cleaned, strict=False)

        # Process inst_state_dict
        inst_sd = ckpt.get("inst_state_dict", None)
        if self.instance_model is not None and isinstance(inst_sd, dict):
            inst_sd_cleaned = {strip_ddp_prefix(k): v for k, v in inst_sd.items()}
            self.instance_model.load_state_dict(inst_sd_cleaned, strict=False)

        # Process optimizer states
        if ckpt.get("opt_agg", None) is not None:
            self.opt_agg.load_state_dict(ckpt["opt_agg"])
        if self.opt_inst is not None and ckpt.get("opt_inst", None) is not None:
            self.opt_inst.load_state_dict(ckpt["opt_inst"])

        # Process scheduler states
        if self.sched_agg is not None and ckpt.get("sched_agg", None) is not None:
            self.sched_agg.load_state_dict(ckpt["sched_agg"])
        if self.sched_inst is not None and ckpt.get("sched_inst", None) is not None:
            self.sched_inst.load_state_dict(ckpt["sched_inst"])

        # Process scaler
        if self.amp_enabled and ckpt.get("scaler", None) is not None:
            self.scaler.load_state_dict(ckpt["scaler"])

        # Restore SWA state
        if self.swa_model is not None and ckpt.get("swa_state_dict", None) is not None:
            self.swa_model.load_state_dict(ckpt["swa_state_dict"])
        self.swa_n_updates = int(ckpt.get("swa_n_updates", 0))

        # Restore training state
        st = ckpt.get("state", None)
        if not isinstance(st, dict):
            st = {
                "epoch": int(ckpt.get("epoch", 0)),
                "global_step": int(ckpt.get("global_step", 0)),
                "best_metric": float(ckpt.get("best_metric", self.best_metric)),
            }
        self.epoch = int(st.get("epoch", 0))
        self.global_step = int(st.get("global_step", 0))
        self.best_metric = float(st.get("best_metric", self.best_metric))
