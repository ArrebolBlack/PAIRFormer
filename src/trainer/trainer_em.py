# src/trainer/trainer_em.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from src.em.token_provider import TokenProvider
from src.em.controller import EMPipelineController

# 可选：如果你希望 val 也输出 F1/AUC 等（与旧 Trainer 对齐）
try:
    from omegaconf import DictConfig  # type: ignore
    from src.evaluator.metrics import compute_metrics  # type: ignore
except Exception:
    DictConfig = Any
    compute_metrics = None


# ----------------------- #
# 训练状态对象（复用旧 Trainer 语义）
# ----------------------- #

@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float = 1e9  # 默认监控 loss：越小越好


# ----------------------- #
# EMA（复用旧 Trainer 语义；默认只对 agg 做 EMA 更合理）
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
# TrainerEMConfig：保持你现有字段，并扩展几个“旧 Trainer”常用开关
# ----------------------- #

@dataclass
class TrainerEMConfig:
    num_epochs: int = 10
    log_every: int = 50
    grad_accum_steps: int = 1
    clip_grad_norm: float = 0.0
    use_amp: bool = False

    # optimizer 超参（默认 AdamW）
    lr_agg: float = 1e-4
    wd_agg: float = 0.0
    lr_inst: float = 1e-5
    wd_inst: float = 0.0

    # scheduler（尽量对齐旧 Trainer 的字符串风格：none / plateau / cosine / step）
    scheduler_agg: str = "none"
    scheduler_inst: str = "none"
    scheduler_t_max: int = 10
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    scheduler_factor: float = 0.2
    scheduler_patience: int = 5

    # monitor（默认监控 val/loss）
    monitor: str = "loss"
    greater_is_better: bool = False  # loss: False

    # EMA（默认只对 agg）
    ema_enabled: bool = False
    ema_decay: float = 0.999


# ----------------------- #
# TrainerEM
# ----------------------- #

class TrainerEM:
    """
    EM Trainer（单卡）：
      - token_provider 负责每 step 是否训练 instance / 是否用 cache
      - controller 负责 E-step + 重建 train_loader（单卡下 barrier 无意义但不影响）
      - 复用旧 Trainer 的稳定结构：TrainState / EMA / AMP / scheduler / logger 风格
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
        task_cfg: Optional[DictConfig] = None,  # 可选：输出 F1/AUC 等
        token_provider_val: Optional[TokenProvider] = None,
    ):
        self.cfg = cfg
        self.device = device

        self.agg_model = agg_model.to(device)
        self.instance_model = instance_model.to(device)

        self.token_provider = token_provider
        self.controller = controller
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.logger = logger
        self.task_cfg = task_cfg

        self.token_provider_val = token_provider_val

        # loss（pair-level 二分类）
        self.crit = nn.BCEWithLogitsLoss()

        # 两套 optimizer
        self.opt_agg = optim.AdamW(self.agg_model.parameters(), lr=cfg.lr_agg, weight_decay=cfg.wd_agg)
        self.opt_inst = optim.AdamW(self.instance_model.parameters(), lr=cfg.lr_inst, weight_decay=cfg.wd_inst)

        # 两套 scheduler（可选）
        self.sched_agg = self._build_scheduler(self.opt_agg, cfg.scheduler_agg)
        self.sched_inst = self._build_scheduler(self.opt_inst, cfg.scheduler_inst)

        # AMP scaler（复用旧 Trainer 风格）
        amp_enabled = bool(cfg.use_amp and device.type == "cuda")
        self.amp_enabled = amp_enabled
        self.scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

        # state
        self.state = TrainState(epoch=0, global_step=0, best_metric=( -1e9 if cfg.greater_is_better else 1e9 ))

        # EMA（默认只对 agg）
        self.ema: Optional[EMAHelper] = None
        if bool(cfg.ema_enabled):
            self.ema = EMAHelper(self.agg_model, decay=float(cfg.ema_decay))


        # ---- compatibility for old evaluator ----
        self.model = self.agg_model                 # old evaluator may use trainer.model
        self.monitor = str(self.cfg.monitor)        # old train.py prints trainer.monitor
        self.greater_is_better = bool(self.cfg.greater_is_better)


    def set_train_loader(self, loader: Any) -> None:
        self.train_loader = loader

    # ----------------------- #
    # logging（复用旧 Trainer 的“尽量兼容 WandB/TB”的策略）
    # ----------------------- #

    def _log_metrics(self, metrics: Dict[str, float], stage: str, step: Optional[int] = None) -> None:
        if self.logger is None:
            return
        if step is None:
            step = self.state.global_step

        # 过滤成 float
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
    # scheduler builder（对齐旧 Trainer 的字符串风格）
    # ----------------------- #

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
        # 未知：默认不用
        return None

    # ----------------------- #
    # token build
    # ----------------------- #
        
    def _build_tokens_train(self, batch_cpu: Dict[str, Any], *, epoch: int, global_step: int) -> Dict[str, Any]:
        plan = self.token_provider.policy.step_plan(epoch, global_step)   # dict

        # 关键：只有在“online 且 train_instance=True”时才把 instance_model 置为 train
        train_inst = bool(plan.get("train_instance", False)) and (not bool(plan.get("use_instance_cache", True)))
        self.instance_model.train(train_inst)

        # 统一入口：由 TokenProvider 决定 cached/online/hybrid，并返回 used_cache/train_instance 等标记
        return self.token_provider.build_tokens(
            batch_cpu,
            epoch=epoch,
            global_step=global_step,
            plan=plan,
        )

    @torch.inference_mode()
    def _build_tokens_eval(self, batch_cpu: Dict[str, Any], *, epoch: int) -> Dict[str, Any]:
        tp = self.token_provider_val or self.token_provider
        self.instance_model.eval()
        # val 永远不训练 instance，且尽量用 cache（缺 cache 才由 TokenProvider 决策 cache_missing 策略）
        plan = {
            "train_instance": False,
            "use_instance_cache": True,
            "train_cheap": False,
            "use_cheap_cache": True,
        }
        return tp.build_tokens(batch_cpu, epoch=epoch, global_step=self.state.global_step, plan=plan)

    # ----------------------- #
    # fit
    # ----------------------- #

    def fit(self) -> None:
        for epoch in range(int(self.cfg.num_epochs)):
            self.state.epoch = epoch
            self.token_provider.on_epoch_begin(epoch)

            # E-step（单卡：controller 内部 barrier 无影响）
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
                # 无 val：对非-plateau scheduler，每 epoch 走一步
                self._step_schedulers({"loss": train_metrics.get("loss", 0.0)})

    # ----------------------- #
    # train / val 核心循环（旧 Trainer 风格 + 双 optimizer + grad accumulation）
    # ----------------------- #

    def train_one_epoch(self, loader: Any) -> Dict[str, float]:
        self.agg_model.train()
        # instance_model 的 train/eval 由每 step 的 policy 决定

        grad_accum = max(1, int(self.cfg.grad_accum_steps))
        clip = float(self.cfg.clip_grad_norm)

        total_loss = 0.0
        total_seen = 0

        # 统计：cache/inst 比例，便于你 debug policy
        n_steps = 0
        n_used_cache = 0
        n_train_inst = 0
        n_skipped = 0

        # accumulation window flags（避免“空 step 的 AdamW weight decay”）
        agg_accum_has_grad = False
        inst_accum_has_grad = False

        self.opt_agg.zero_grad(set_to_none=True)
        self.opt_inst.zero_grad(set_to_none=True)

        pbar = tqdm(loader, desc=f"TrainEM epoch {self.state.epoch}")

        for it, batch_cpu in enumerate(pbar):
            # 进入新的 accumulation window 时清梯度
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
                # 全 padding：不做 backward，不做 step（但 global_step 仍前进，保持 policy 时间轴一致）
                n_skipped += 1
                self.state.global_step += 1

                # 若这是 window 的最后一步，可能会触发 step：这里直接让 window 结束时因 has_grad=False 而跳过
                if self.cfg.log_every > 0 and (self.state.global_step % self.cfg.log_every == 0):
                    pbar.set_postfix({"loss": "skip", "cache": used_cache, "inst": int(train_inst)})
                continue

            with torch.amp.autocast(device_type="cuda", enabled=self.amp_enabled):
                logits = self.agg_model(tokens, attn_mask=mask)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                logits = logits.view(-1)
                loss = self.crit(logits, y.view(-1))

                # grad accumulation 标准缩放
                loss_to_backward = loss / float(grad_accum)

            self.scaler.scale(loss_to_backward).backward()
            agg_accum_has_grad = True
            if train_inst:
                inst_accum_has_grad = True

            # 统计 loss（用未缩放的 loss，更符合直觉）
            bs = int(y.shape[0])
            total_loss += float(loss.detach().item()) * bs
            total_seen += bs

            self.state.global_step += 1

            # 到 accumulation window 末尾 or 最后一个 batch：尝试 step
            is_last = (it == (len(loader) - 1)) if hasattr(loader, "__len__") else False
            do_step = ((it + 1) % grad_accum == 0) or is_last

            if do_step and agg_accum_has_grad:
                if clip > 0:
                    # agg
                    self.scaler.unscale_(self.opt_agg)
                    torch.nn.utils.clip_grad_norm_(self.agg_model.parameters(), clip)
                    # inst（只在本窗口真的训练过 inst 时才裁剪）
                    if inst_accum_has_grad:
                        self.scaler.unscale_(self.opt_inst)
                        torch.nn.utils.clip_grad_norm_(self.instance_model.parameters(), clip)

                # step（关键：inst 只有在窗口内训练过才 step，避免 AdamW 纯 weight decay 漂移）
                self.scaler.step(self.opt_agg)
                if inst_accum_has_grad:
                    self.scaler.step(self.opt_inst)
                self.scaler.update()

                # EMA（只对 agg）
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

    @torch.inference_mode()
    def validate_one_epoch(self, loader: Any, *, use_ema: bool = True) -> Dict[str, float]:
        # val：agg eval + inst eval，且强制 train_inst=False 构 token
        ctx = self.ema.swap_parameters(self.agg_model) if (use_ema and self.ema is not None) else _nullcontext()
        with ctx:
            self.agg_model.eval()
            self.instance_model.eval()

            total_loss = 0.0
            total_seen = 0

            all_logits = []
            all_labels = []

            for batch_cpu in tqdm(loader, desc=f"ValEM epoch {self.state.epoch}"):
                out = self._build_tokens_eval(batch_cpu, epoch=self.state.epoch)

                tokens = out["tokens"]
                mask = out["mask"]
                y = out["y_pair"]

                if tokens is None:
                    continue

                logits = self.agg_model(tokens, attn_mask=mask)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                logits = logits.view(-1)

                loss = self.crit(logits, y.view(-1))

                bs = int(y.shape[0])
                total_loss += float(loss.detach().item()) * bs
                total_seen += bs

                all_logits.append(logits.detach().cpu())
                all_labels.append(y.view(-1).detach().cpu())

            avg_loss = total_loss / max(1, total_seen)

            metrics: Dict[str, float] = {
                "loss": avg_loss,
                "lr_agg": float(self.opt_agg.param_groups[0].get("lr", 0.0)),
                "lr_inst": float(self.opt_inst.param_groups[0].get("lr", 0.0)),
            }

            # 若提供 task_cfg 且 compute_metrics 可用，则输出与旧 Trainer 一致的指标
            if (self.task_cfg is not None) and (compute_metrics is not None) and len(all_logits) > 0:
                logits_np = torch.cat(all_logits).numpy()
                labels_np = torch.cat(all_labels).numpy()
                m = compute_metrics(y_true=labels_np, y_pred_raw=logits_np, task_cfg=self.task_cfg)
                for k, v in m.items():
                    try:
                        metrics[k] = float(v)
                    except Exception:
                        pass

            self._log_metrics(metrics, stage="val", step=self.state.global_step)
            return metrics

    # ----------------------- #
    # scheduler step / best metric（对齐旧 Trainer）
    # ----------------------- #

    def _step_schedulers(self, metrics: Dict[str, float]) -> None:
        # plateau：用 monitor；其他：每 epoch step
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
    # save/load ckpt（对齐旧 Trainer）
    # ----------------------- #
                
    def save_checkpoint(self, path: str) -> None:
        ckpt = {
            "agg_state_dict": self.agg_model.state_dict(),
            "inst_state_dict": self.instance_model.state_dict(),
            "opt_agg": self.opt_agg.state_dict(),
            "opt_inst": self.opt_inst.state_dict(),
            "sched_agg": None if self.sched_agg is None else self.sched_agg.state_dict(),
            "sched_inst": None if self.sched_inst is None else self.sched_inst.state_dict(),
            "scaler": None if not self.amp_enabled else self.scaler.state_dict(),
            "state": {"epoch": self.state.epoch, "global_step": self.state.global_step, "best_metric": self.state.best_metric},
            "ema_shadow": None if self.ema is None else {k: v.cpu() for k, v in self.ema.shadow.items()},
            "cfg": self.cfg.__dict__,
        }
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str, *, map_location: Optional[torch.device] = None) -> None:
        ckpt = torch.load(path, map_location=("cpu" if map_location is None else map_location))
        self.agg_model.load_state_dict(ckpt["agg_state_dict"], strict=False)
        self.instance_model.load_state_dict(ckpt["inst_state_dict"], strict=False)

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

    import numpy as np
    import time

    @torch.inference_mode()
    def predict(
        self,
        loader: Any,
        *,
        set_labels: Optional[Any] = None,
        aggregate_sets: bool = False,
        use_ema: bool = True,
        reduction: str = "max",
        softmax_temp: float = 1.0,
        topk: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Compatibility API for src.evaluator.evaluator.evaluate_with_trainer.

        Returns
        -------
        y_true : np.ndarray, shape [N] (or [num_sets] if aggregate_sets=True)
        y_pred_raw : np.ndarray, shape [N] (logits)
        set_idx : Optional[np.ndarray], shape [N] (only meaningful for window->set aggregation)
        """
        ctx = self.ema.swap_parameters(self.agg_model) if (use_ema and self.ema is not None) else _nullcontext()
        with ctx:
            self.agg_model.eval()
            self.instance_model.eval()

            all_logits: list[torch.Tensor] = []
            all_labels: list[torch.Tensor] = []
            all_set_idx: list[torch.Tensor] = []

            for batch_cpu in tqdm(loader, desc=f"PredictEM epoch {self.state.epoch}"):
                out = self._build_tokens_eval(batch_cpu, epoch=self.state.epoch)

                tokens = out.get("tokens", None)
                mask = out.get("mask", None)
                y = out.get("y_pair", None)

                if tokens is None or y is None:
                    continue

                logits = self.agg_model(tokens, attn_mask=mask)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                logits = logits.view(-1)

                all_logits.append(logits.detach().cpu())
                all_labels.append(y.view(-1).detach().cpu())

                # optional: some evaluators want set_idx; for pair-level it can be None
                # try to find set_idx / pair_id, but don't require it
                sid = out.get("set_idx", None)
                if sid is None:
                    sid = out.get("pair_id", None)
                if sid is not None:
                    sid = torch.as_tensor(sid).view(-1).detach().cpu()
                    all_set_idx.append(sid)

            if len(all_logits) == 0:
                y_true = np.zeros((0,), dtype=np.float32)
                y_pred = np.zeros((0,), dtype=np.float32)
                return y_true, y_pred, None

            y_pred_raw = torch.cat(all_logits).numpy()
            y_true = torch.cat(all_labels).numpy()

            # window->set aggregation (mostly for old window-level tasks)
            if aggregate_sets:
                if set_labels is None:
                    raise ValueError("aggregate_sets=True requires set_labels.")
                if len(all_set_idx) == 0:
                    raise ValueError("aggregate_sets=True requires set_idx in batch/out.")

                set_idx = torch.cat(all_set_idx).numpy()

                # lazy import to avoid heavy import / circular
                from src.trainer.trainer import aggregate_by_set_idx  # reuse old implementation

                y_true_set, y_pred_set = aggregate_by_set_idx(
                    y_true_window=y_true,
                    y_pred_window=y_pred_raw,
                    set_idx_window=set_idx,
                    set_labels=set_labels,
                    reduction=str(reduction),
                    softmax_tau=float(softmax_temp),
                    topk=int(topk),
                )
                return y_true_set, y_pred_set, None

            return y_true, y_pred_raw, None



    @torch.inference_mode()
    def benchmark_inference(
        self,
        loader: Any,
        *,
        use_ema: bool = True,
        warmup_batches: int = 10,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Match the keys used in old train.py:
        infer_pairs_per_s, infer_peak_vram_gb, infer_elapsed_s, infer_total_pairs
        """
        device_is_cuda = (self.device.type == "cuda")
        if device_is_cuda:
            torch.cuda.reset_peak_memory_stats()

        ctx = self.ema.swap_parameters(self.agg_model) if (use_ema and self.ema is not None) else _nullcontext()
        with ctx:
            self.agg_model.eval()
            self.instance_model.eval()

            # warmup
            n = 0
            for batch_cpu in loader:
                out = self._build_tokens_eval(batch_cpu, epoch=self.state.epoch)
                tokens = out.get("tokens", None)
                mask = out.get("mask", None)
                y = out.get("y_pair", None)
                if tokens is None or y is None:
                    continue
                _ = self.agg_model(tokens, attn_mask=mask)
                n += 1
                if n >= int(warmup_batches):
                    break

            if device_is_cuda:
                torch.cuda.synchronize()

            # timed
            total_pairs = 0
            t0 = time.time()
            nb = 0

            for batch_cpu in loader:
                if (max_batches is not None) and (nb >= int(max_batches)):
                    break

                out = self._build_tokens_eval(batch_cpu, epoch=self.state.epoch)
                tokens = out.get("tokens", None)
                mask = out.get("mask", None)
                y = out.get("y_pair", None)
                if tokens is None or y is None:
                    continue

                _ = self.agg_model(tokens, attn_mask=mask)

                bs = int(y.shape[0])
                total_pairs += bs
                nb += 1

            if device_is_cuda:
                torch.cuda.synchronize()
            elapsed = time.time() - t0

            peak_gb = 0.0
            if device_is_cuda:
                peak_gb = float(torch.cuda.max_memory_allocated() / 1e9)

            pairs_per_s = float(total_pairs / max(elapsed, 1e-9))
            return {
                "infer_pairs_per_s": pairs_per_s,
                "infer_peak_vram_gb": peak_gb,
                "infer_elapsed_s": float(elapsed),
                "infer_total_pairs": float(total_pairs),
            }
