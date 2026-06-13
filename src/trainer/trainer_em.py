# src/trainer/trainer_em.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from src.em.token_provider import TokenProvider
from src.em.controller import EMPipelineController

from omegaconf import DictConfig
from src.evaluator.metrics import compute_metrics

import time
from src.utils.efficiency import EffMeter

from src.trainer.loss import BinaryClassificationLoss

# DDP imports
from src.utils.ddp import is_ddp, is_rank0, get_rank, get_world_size, all_reduce_dict, save_on_rank0, print_on_rank0
import torch.distributed as dist

# ----------------------- #
# Training state
# ----------------------- #

@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float = 1e9  # default: monitor loss (lower is better)


# ----------------------- #
# EMA
# ----------------------- #

class EMAHelper:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
                continue
            self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.detach()

    def swap_parameters(self, model: nn.Module):
        return _EMASwapContext(self, model)


class _EMASwapContext:
    def __init__(self, ema: EMAHelper, model: nn.Module):
        self.ema = ema
        self.model = model

    def __enter__(self):
        self.ema.backup = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.ema.shadow:
                self.ema.backup[name] = param.data.clone()
                param.data.copy_(self.ema.shadow[name])

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.ema.backup:
                param.data.copy_(self.ema.backup[name])
        self.ema.backup = {}


class _nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


# ----------------------- #
# TrainerEMConfig
# ----------------------- #

@dataclass
class TrainerEMConfig:
    # loop control
    num_epochs: int = 10
    log_every: int = 50
    grad_accum_steps: int = 1
    clip_grad_norm: float = 0.0
    use_amp: bool = False

    # optimizer
    optimizer: str = "adamw"
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False

    lr_agg: float = 3e-4
    wd_agg: float = 1e-2
    lr_inst: float = 1e-5
    wd_inst: float = 0.0

    # scheduler
    scheduler_agg: str = "cosine"
    scheduler_inst: str = "cosine"
    scheduler_t_max: int = 10
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    scheduler_factor: float = 0.2
    scheduler_patience: int = 5

    # loss
    loss_type: str = "bce"
    esa_weighting: bool = False

    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    focal_lambda: float = 1.0

    label_smoothing: bool = False
    smooth_neg: float = 0.0
    smooth_pos: float = 1.0

    bce_lambda: float = 1.0
    pos_weight: float = 1.0
    bce_pos_weight: float = 1.0

    esa_scale: float = 10.0
    esa_lambda_pos: float = 1.0
    esa_lambda_neg: float = 0.5

    # monitor/best
    monitor: str = "loss"
    greater_is_better: bool = False

    # EMA
    ema_enabled: bool = False
    ema_decay: float = 0.999


# ----------------------- #
# TrainerEM
# ----------------------- #

class TrainerEM:
    """
    EM Trainer:
      - token_provider handles per-step cache/online decisions
      - controller handles E-step cache refresh
      - DDP: agg_model wrapped with DistributedDataParallel
      - DDP: instance_model NOT wrapped, gradients manually synchronized
    """

    def __init__(
        self,
        *,
        cfg: TrainerEMConfig,
        device: torch.device,
        agg_model: nn.Module,
        instance_model: nn.Module,
        token_provider: TokenProvider,
        controller: Optional[EMPipelineController],
        train_loader: Any,
        val_loader: Optional[Any] = None,
        logger: Optional[Any] = None,
        task_cfg: Optional[DictConfig] = None,
        token_provider_val: Optional[TokenProvider] = None,
        local_rank: int = 0,
    ):
        self.cfg = cfg
        self.device = device
        self.local_rank = local_rank

        self.agg_model = agg_model.to(device)

        # Wrap agg_model with DDP if applicable
        if is_ddp():
            ddp_kwargs = {}
            if device.type == "cuda":
                ddp_kwargs["device_ids"] = [local_rank]
                ddp_kwargs["output_device"] = local_rank
            self.agg_model_ddp = nn.parallel.DistributedDataParallel(
                self.agg_model, **ddp_kwargs,
            )
        else:
            self.agg_model_ddp = self.agg_model

        self.instance_model = instance_model.to(device)

        self.token_provider = token_provider
        self.controller = controller
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.logger = logger
        self.task_cfg = task_cfg

        self.token_provider_val = token_provider_val

        # epoch-level override for train token plan
        self._train_plan_override: Optional[Dict[str, bool]] = None

        # loss
        self.crit = BinaryClassificationLoss(train_cfg=cfg)

        # Two optimizers
        self.opt_agg = optim.AdamW(
            self.agg_model.parameters(), lr=cfg.lr_agg,
            weight_decay=cfg.wd_agg, betas=self.cfg.betas,
            eps=self.cfg.eps, amsgrad=self.cfg.amsgrad,
        )
        self.opt_inst = optim.AdamW(
            self.instance_model.parameters(), lr=cfg.lr_inst,
            weight_decay=cfg.wd_inst, betas=self.cfg.betas,
            eps=self.cfg.eps, amsgrad=self.cfg.amsgrad,
        )

        # Two schedulers
        self.sched_agg = self._build_scheduler(self.opt_agg, cfg.scheduler_agg)
        self.sched_inst = self._build_scheduler(self.opt_inst, cfg.scheduler_inst)

        # AMP
        amp_enabled = bool(cfg.use_amp and device.type == "cuda")
        self.amp_enabled = amp_enabled
        self.scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

        # state
        self.state = TrainState(
            epoch=0,
            global_step=0,
            best_metric=(-1e9 if cfg.greater_is_better else 1e9),
        )

        # EMA (only for agg)
        self.ema: Optional[EMAHelper] = None
        if bool(cfg.ema_enabled):
            self.ema = EMAHelper(self.agg_model, decay=float(cfg.ema_decay))

        # compatibility aliases
        self.model = self.agg_model
        self.monitor = str(self.cfg.monitor)
        self.greater_is_better = bool(self.cfg.greater_is_better)

    def set_train_plan_override(self, plan: Optional[Dict[str, bool]]) -> None:
        """Force train-time token plan for the whole epoch."""
        self._train_plan_override = (None if plan is None else dict(plan))

    def _compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.crit.set_sample_weight(None)
        return self.crit.compute(logits.view(-1), y.view(-1).float())

    def set_train_loader(self, loader: Any) -> None:
        self.train_loader = loader

    # ----------------------- #
    # logging
    # ----------------------- #

    def _log_metrics(self, metrics: Dict[str, float], stage: str, step: Optional[int] = None) -> None:
        if self.logger is None:
            return
        if step is None:
            step = self.state.global_step

        out = {}
        for k, v in metrics.items():
            try:
                out[f"{stage}/{k}"] = float(v)
            except Exception:
                continue

        if not out:
            return

        if hasattr(self.logger, "log_metrics"):
            try:
                self.logger.log_metrics(out, step=step)
                return
            except TypeError:
                pass

        if hasattr(self.logger, "log"):
            try:
                self.logger.log(out, step=step)
            except TypeError:
                self.logger.log(out)

    # ----------------------- #
    # scheduler builder
    # ----------------------- #

    def _build_scheduler(self, optimizer: optim.Optimizer, name: str):
        name = str(name).lower()
        if name == "none":
            return None
        if name == "plateau":
            mode = "max" if bool(self.cfg.greater_is_better) else "min"
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=mode,
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

    # ----------------------- #
    # token build
    # ----------------------- #

    def _build_tokens_train(self, batch_cpu: Dict[str, Any], *, epoch: int, global_step: int) -> Dict[str, Any]:
        plan = self._train_plan_override
        if plan is None:
            plan = self.token_provider.policy.step_plan(epoch, global_step)
        plan = dict(plan)

        train_inst = bool(plan.get("train_instance", False)) and (not bool(plan.get("use_instance_cache", True)))
        self.instance_model.train(train_inst)

        return self.token_provider.build_tokens(
            batch_cpu, epoch=epoch, global_step=global_step, plan=plan,
        )

    @torch.inference_mode()
    def _build_tokens_eval(
        self,
        batch_cpu: Dict[str, Any],
        *,
        epoch: int,
        use_instance_cache: Optional[bool] = None,
    ) -> Dict[str, Any]:
        tp = self.token_provider_val or self.token_provider
        self.instance_model.eval()

        plan = {"train_instance": False, "train_cheap": False}
        if use_instance_cache is not None:
            plan["use_instance_cache"] = bool(use_instance_cache)
        return tp.build_tokens(batch_cpu, epoch=epoch, global_step=self.state.global_step, plan=plan)

    def set_instance_trainable(self, trainable: bool) -> None:
        self.instance_model.requires_grad_(bool(trainable))

    # ----------------------- #
    # fit
    # ----------------------- #

    def fit(self) -> None:
        for epoch in range(int(self.cfg.num_epochs)):
            self.state.epoch = epoch
            self.token_provider.on_epoch_begin(epoch)

            if self.controller is not None:
                new_loader = self.controller.maybe_refresh_and_rebuild(epoch=epoch)
                if new_loader is not None:
                    self.set_train_loader(new_loader)

            train_metrics = self.train_one_epoch(self.train_loader)
            if self.val_loader is not None:
                val_metrics = self.validate_one_epoch(self.val_loader)
                self._step_schedulers(val_metrics)
                self._update_best(val_metrics)
            else:
                self._step_schedulers({"loss": train_metrics.get("loss", 0.0)})

    # ----------------------- #
    # train / val core loops
    # ----------------------- #

    def train_one_epoch(self, loader: Any) -> Dict[str, float]:
        self.agg_model.train()

        grad_accum = max(1, int(self.cfg.grad_accum_steps))
        clip = float(self.cfg.clip_grad_norm)

        total_loss = 0.0
        total_seen = 0

        n_steps = 0
        n_used_cache = 0
        n_train_inst = 0
        n_skipped = 0

        agg_accum_has_grad = False
        inst_accum_has_grad = False

        self.opt_agg.zero_grad(set_to_none=True)
        self.opt_inst.zero_grad(set_to_none=True)

        pbar = tqdm(loader, desc=f"TrainEM epoch {self.state.epoch}", disable=not is_rank0())

        for it, batch_cpu in enumerate(pbar):
            if it % grad_accum == 0:
                self.opt_agg.zero_grad(set_to_none=True)
                self.opt_inst.zero_grad(set_to_none=True)
                agg_accum_has_grad = False
                inst_accum_has_grad = False

            out = self._build_tokens_train(batch_cpu, epoch=self.state.epoch, global_step=self.state.global_step)

            tokens = out["tokens"]
            mask = out["mask"]
            y = out["y_pair"]
            train_inst = bool(out["train_instance"])
            used_cache = bool(out["used_cache"])

            n_steps += 1
            n_used_cache += int(used_cache)
            n_train_inst += int(train_inst)

            if tokens is None:
                n_skipped += 1
                self.state.global_step += 1
                if self.cfg.log_every > 0 and (self.state.global_step % self.cfg.log_every == 0):
                    pbar.set_postfix({"loss": "skip", "cache": used_cache, "inst": int(train_inst)})
                continue

            with torch.amp.autocast(device_type="cuda", enabled=self.amp_enabled):
                # Use DDP-wrapped model for forward pass (handles gradient sync)
                logits = self.agg_model_ddp(tokens, attn_mask=mask)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                logits = logits.view(-1)
                loss = self._compute_loss(logits, y.view(-1))
                loss_to_backward = loss / float(grad_accum)

            self.scaler.scale(loss_to_backward).backward()
            agg_accum_has_grad = True

            if train_inst:
                inst_accum_has_grad = True

            bs = int(y.shape[0])
            total_loss += float(loss.detach().item()) * bs
            total_seen += bs

            self.state.global_step += 1

            is_last = (it == (len(loader) - 1)) if hasattr(loader, "__len__") else False
            do_step = ((it + 1) % grad_accum == 0) or is_last

            if do_step and agg_accum_has_grad:
                if clip > 0:
                    self.scaler.unscale_(self.opt_agg)
                    torch.nn.utils.clip_grad_norm_(self.agg_model.parameters(), clip)
                    if inst_accum_has_grad:
                        self.scaler.unscale_(self.opt_inst)
                        torch.nn.utils.clip_grad_norm_(self.instance_model.parameters(), clip)

                # DDP: manually sync instance model gradients before optimizer step
                if is_ddp() and inst_accum_has_grad:
                    for p in self.instance_model.parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                            p.grad.div_(get_world_size())

                self.scaler.step(self.opt_agg)
                if inst_accum_has_grad:
                    self.scaler.step(self.opt_inst)
                self.scaler.update()

                # EMA (only for agg)
                if self.ema is not None:
                    self.ema.update(self.agg_model)

            # log
            if self.cfg.log_every > 0 and (self.state.global_step % self.cfg.log_every == 0):
                avg = total_loss / max(1, total_seen)
                pbar.set_postfix(
                    {
                        "loss": f"{avg:.4f}",
                        "cache%": f"{(n_used_cache/max(1,n_steps))*100:.1f}",
                        "inst%": f"{(n_train_inst/max(1,n_steps))*100:.1f}",
                        "skip%": f"{(n_skipped/max(1,n_steps))*100:.1f}",
                    }
                )

        avg_loss = total_loss / max(1, total_seen)
        metrics = {
            "loss": avg_loss,
            "lr_agg": float(self.opt_agg.param_groups[0].get("lr", 0.0)),
            "lr_inst": float(self.opt_inst.param_groups[0].get("lr", 0.0)),
            "used_cache_pct": 100.0 * (n_used_cache / max(1, n_steps)),
            "train_inst_pct": 100.0 * (n_train_inst / max(1, n_steps)),
            "skipped_pct": 100.0 * (n_skipped / max(1, n_steps)),
        }
        self._log_metrics(metrics, stage="train", step=self.state.global_step)
        return metrics

    @torch.no_grad()
    def validate_one_epoch(
        self,
        loader: Any,
        *,
        use_ema: bool = True,
        use_instance_cache: Optional[bool] = None,
    ) -> Dict[str, float]:
        ctx = self.ema.swap_parameters(self.agg_model) if (use_ema and self.ema is not None) else _nullcontext()
        with ctx:
            self.agg_model.eval()
            self.instance_model.eval()

            n_steps = 0
            n_used_cache = 0

            total_loss, total_seen = 0.0, 0
            all_logits, all_labels = [], []

            for batch_cpu in tqdm(loader, desc=f"Valid epoch {self.state.epoch}", disable=not is_rank0()):
                out = self._build_tokens_eval(
                    batch_cpu,
                    epoch=self.state.epoch,
                    use_instance_cache=use_instance_cache,
                )

                n_steps += 1
                n_used_cache += int(bool(out.get("used_cache", False)))

                tokens = out.get("tokens", None)
                y = out.get("y_pair", None)
                mask = out.get("mask", None)

                if tokens is None or y is None:
                    continue

                # Use agg_model_ddp for forward (handles SyncBatchNorm correctly)
                logits = self.agg_model_ddp(tokens, attn_mask=mask)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                logits = logits.view(-1)

                y = y.view(-1).float()
                loss = self._compute_loss(logits, y)

                bs = int(y.numel())
                total_loss += float(loss.item()) * bs
                total_seen += bs

                all_logits.append(logits.detach().cpu())
                all_labels.append(y.detach().cpu())

            avg_loss = total_loss / max(1, total_seen)
            metrics: Dict[str, float] = {"loss": float(avg_loss)}

            # DDP: gather logits/labels from all ranks and compute metrics on rank 0
            if len(all_labels) > 0:
                local_logits = torch.cat(all_logits)
                local_labels = torch.cat(all_labels)

                if is_ddp():
                    from src.utils.ddp import gather_tensors
                    gathered_logits = gather_tensors(local_logits)
                    gathered_labels = gather_tensors(local_labels)
                    # Concatenate all ranks
                    all_rank_logits = torch.cat(gathered_logits)
                    all_rank_labels = torch.cat(gathered_labels)

                    if is_rank0():
                        cm = compute_metrics(
                            y_true=all_rank_labels.numpy(),
                            y_pred_raw=all_rank_logits.numpy(),
                            task_cfg=self.task_cfg,
                        )
                        for k, v in cm.items():
                            try:
                                metrics[k] = float(v)
                            except Exception:
                                pass

                    # Broadcast metrics from rank 0 to all ranks
                    # Use broadcast_object_list to handle variable-length dicts
                    metrics_payload = [metrics]
                    dist.broadcast_object_list(metrics_payload, src=0)
                    metrics = metrics_payload[0]
                else:
                    logits_np = local_logits.numpy()
                    labels_np = local_labels.numpy()
                    cm = compute_metrics(y_true=labels_np, y_pred_raw=logits_np, task_cfg=self.task_cfg)
                    for k, v in cm.items():
                        try:
                            metrics[k] = float(v)
                        except Exception:
                            pass

            metrics["lr_agg"] = float(self.opt_agg.param_groups[0].get("lr", 0.0))
            metrics["lr_inst"] = float(self.opt_inst.param_groups[0].get("lr", 0.0))
            metrics["used_cache_pct"] = 100.0 * (n_used_cache / max(1, n_steps))

            self._log_metrics(metrics, stage="val", step=self.state.global_step)
            return metrics

    # ----------------------- #
    # scheduler step / best metric
    # ----------------------- #

    def _step_schedulers(self, metrics: Dict[str, float]) -> None:
        monitor_val = float(metrics.get(self.cfg.monitor, metrics.get("loss", 0.0)))
        for sched in [self.sched_agg, self.sched_inst]:
            if sched is None:
                continue
            if isinstance(sched, optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(monitor_val)
            else:
                sched.step()

    def _update_best(self, metrics: Dict[str, float]) -> None:
        v = metrics.get(self.cfg.monitor, None)
        if v is None:
            return
        v = float(v)
        if bool(self.cfg.greater_is_better):
            if v > self.state.best_metric:
                self.state.best_metric = v
        else:
            if v < self.state.best_metric:
                self.state.best_metric = v

    def step_schedulers(self, metrics: Dict[str, float]) -> None:
        self._step_schedulers(metrics)

    def update_best(self, metrics: Dict[str, float]) -> bool:
        prev = self.state.best_metric
        self._update_best(metrics)
        return self.state.best_metric != prev

    # ----------------------- #
    # save/load checkpoint
    # ----------------------- #

    def save_checkpoint(self, path: str) -> None:
        """Save checkpoint only on rank 0."""
        save_on_rank0(self._save_checkpoint_impl, path)

    def _save_checkpoint_impl(self, path: str) -> None:
        ckpt = {
            "agg_state_dict": self.agg_model.state_dict(),
            "inst_state_dict": self.instance_model.state_dict(),
            "opt_agg": self.opt_agg.state_dict(),
            "opt_inst": self.opt_inst.state_dict(),
            "sched_agg": None if self.sched_agg is None else self.sched_agg.state_dict(),
            "sched_inst": None if self.sched_inst is None else self.sched_inst.state_dict(),
            "scaler": None if not self.amp_enabled else self.scaler.state_dict(),
            "state": {
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "best_metric": self.state.best_metric,
            },
            "ema_shadow": None if self.ema is None else {k: v.cpu() for k, v in self.ema.shadow.items()},
            "cfg": self.cfg.__dict__,
        }
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str, *, map_location: Optional[torch.device] = None) -> None:
        ckpt = torch.load(path, map_location=("cpu" if map_location is None else map_location))

        # Strip DDP prefix if present. Consolidated into clean_state_dict_keys
        # (behavior unchanged; verified by tests/test_checkpoint_utils.py).
        from src.utils.checkpoint import clean_state_dict_keys

        agg_sd = ckpt.get("agg_state_dict", ckpt)
        if isinstance(agg_sd, dict):
            agg_sd_cleaned = clean_state_dict_keys(agg_sd, ("module.",))
            self.agg_model.load_state_dict(agg_sd_cleaned, strict=False)

        inst_sd = ckpt.get("inst_state_dict", None)
        if isinstance(inst_sd, dict):
            inst_sd_cleaned = clean_state_dict_keys(inst_sd, ("module.",))
            self.instance_model.load_state_dict(inst_sd_cleaned, strict=False)

        if "opt_agg" in ckpt and ckpt["opt_agg"] is not None:
            self.opt_agg.load_state_dict(ckpt["opt_agg"])
        if "opt_inst" in ckpt and ckpt["opt_inst"] is not None:
            self.opt_inst.load_state_dict(ckpt["opt_inst"])

        if self.sched_agg is not None and ckpt.get("sched_agg", None) is not None:
            self.sched_agg.load_state_dict(ckpt["sched_agg"])
        if self.sched_inst is not None and ckpt.get("sched_inst", None) is not None:
            self.sched_inst.load_state_dict(ckpt["sched_inst"])

        if self.amp_enabled and ckpt.get("scaler", None) is not None:
            self.scaler.load_state_dict(ckpt["scaler"])

        st = ckpt.get("state", {})
        self.state.epoch = int(st.get("epoch", 0))
        self.state.global_step = int(st.get("global_step", 0))
        self.state.best_metric = float(st.get("best_metric", self.state.best_metric))

        ema_shadow = ckpt.get("ema_shadow", None)
        if self.ema is not None and isinstance(ema_shadow, dict):
            for k, v in ema_shadow.items():
                self.ema.shadow[k] = v.to(self.device)

    @torch.no_grad()
    def predict(self, loader: Any, use_ema: bool = True, use_instance_cache: Optional[bool] = None) -> Dict[str, Any]:
        ctx = self.ema.swap_parameters(self.agg_model) if (use_ema and self.ema is not None) else _nullcontext()
        with ctx:
            self.agg_model.eval()
            self.instance_model.eval()

            all_logits, all_labels = [], []
            all_pair_id = []

            for batch_cpu in tqdm(loader, desc="Predict", disable=not is_rank0()):
                out = self._build_tokens_eval(batch_cpu, epoch=self.state.epoch, use_instance_cache=use_instance_cache)

                tokens = out.get("tokens", None)
                y = out.get("y_pair", None)
                mask = out.get("mask", None)
                pid = out.get("pair_id", None)

                if tokens is None or y is None:
                    continue

                logits = self.agg_model_ddp(tokens, attn_mask=mask)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                logits = logits.view(-1)

                y = y.view(-1).float()

                all_logits.append(logits.detach().cpu())
                all_labels.append(y.detach().cpu())

                if pid is not None and torch.is_tensor(pid):
                    all_pair_id.append(pid.detach().cpu())

            out_dict: Dict[str, Any] = {
                "logits": torch.cat(all_logits).numpy() if len(all_logits) > 0 else np.zeros((0,), dtype=np.float32),
                "labels": torch.cat(all_labels).numpy() if len(all_labels) > 0 else np.zeros((0,), dtype=np.float32),
            }
            if len(all_pair_id) > 0:
                out_dict["pair_id"] = torch.cat(all_pair_id).numpy()
            return out_dict

    @torch.inference_mode()
    def benchmark_inference(
        self,
        loader: Any,
        *,
        use_ema: bool = True,
        warmup_batches: int = 10,
        max_batches: Optional[int] = None,
        plan_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """End-to-end inference throughput benchmark."""
        device_is_cuda = (self.device.type == "cuda")
        if device_is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

        meter = EffMeter(device=self.device, enabled=True)
        meter.reset_epoch()

        ctx = self.ema.swap_parameters(self.agg_model) if (use_ema and self.ema is not None) else _nullcontext()
        with ctx:
            self.agg_model.eval()
            self.instance_model.eval()

            base_plan = {
                "train_instance": False,
                "use_instance_cache": True,
                "train_cheap": False,
                "use_cheap_cache": True,
            }
            if plan_override is not None:
                base_plan.update(plan_override)

            it = iter(loader)

            n_warm = 0
            while n_warm < int(warmup_batches):
                try:
                    batch_cpu = next(it)
                except StopIteration:
                    break

                out = (self.token_provider_val or self.token_provider).build_tokens(
                    batch_cpu, epoch=int(self.state.epoch),
                    global_step=int(self.state.global_step), plan=base_plan,
                )
                tokens = out.get("tokens", None)
                mask = out.get("mask", None)
                y = out.get("y_pair", None)

                if tokens is None or y is None:
                    continue

                with torch.amp.autocast(device_type="cuda", enabled=self.amp_enabled):
                    _ = self.agg_model(tokens, attn_mask=mask)

                meter.update_peak_cpu()
                n_warm += 1

            if device_is_cuda:
                torch.cuda.synchronize(self.device)

            total_pairs = 0
            nb = 0
            t0 = time.perf_counter()

            while True:
                if max_batches is not None and nb >= int(max_batches):
                    break
                try:
                    batch_cpu = next(it)
                except StopIteration:
                    break

                out = (self.token_provider_val or self.token_provider).build_tokens(
                    batch_cpu, epoch=int(self.state.epoch),
                    global_step=int(self.state.global_step), plan=base_plan,
                )
                tokens = out.get("tokens", None)
                mask = out.get("mask", None)
                y = out.get("y_pair", None)

                if tokens is None or y is None:
                    continue

                with torch.amp.autocast(device_type="cuda", enabled=self.amp_enabled):
                    _ = self.agg_model(tokens, attn_mask=mask)

                total_pairs += int(y.numel())
                nb += 1
                meter.update_peak_cpu()

            if device_is_cuda:
                torch.cuda.synchronize(self.device)

            t1 = time.perf_counter()
            elapsed = max(t1 - t0, 1e-9)

            peak_vram_gb = 0.0
            if device_is_cuda:
                peak_vram_gb = float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 3))

            pairs_per_s = float(total_pairs / elapsed)

            return {
                "infer_pairs_per_s": pairs_per_s,
                "infer_peak_vram_gb": peak_vram_gb,
                "infer_peak_cpu_rss_gb": float(meter.stats.peak_cpu_rss_gb),
                "infer_elapsed_s": float(elapsed),
                "infer_total_pairs": float(total_pairs),
            }
