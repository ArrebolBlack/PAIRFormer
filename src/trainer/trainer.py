# src/trainer/trainer.py
"""
trainer.py
======================
通用 Trainer 模板（现代化版本）

设计目标
--------
1. 通用性：
   - 支持二分类 / 回归（problem_type）
   - 支持多种优化器 / 调度器 / 损失函数（由上层 config 覆盖）
   - 兼容 tuple batch (x, y, set_idx) 和 dict batch（便于后续接 Transformer / DiT）

2. 可配置性：
   - train_cfg.optimizer: "adamw" / "adam" / "sgd" / "rmsprop" / ...
   - train_cfg.scheduler: "none" / "plateau" / "cosine" / "step"
   - train_cfg.loss_type: "bce" / "mse" / "custom"
   - train_cfg.weight_decay: float
   - train_cfg.grad_clip: float or None
   - train_cfg.amp: bool
   - train_cfg.ema.enabled: bool
   - train_cfg.ema.decay: float (e.g. 0.999)
   - train_cfg.monitor: str (例如 "loss" / "f1" / "roc_auc")
   - train_cfg.greater_is_better: bool
        * 对 loss 一般为 False（越小越好）
        * 对 F1/AUC 一般为 True（越大越好）

3. 保留你项目的特性：
   - 支持 window-level → set-level 聚合（max pooling），与旧 Logger.aggregate 一致；
   - validate 时返回统一 metrics（loss + F1/AUC/PR-AUC 等），用于对比旧模型；
   - 支持输出 logits + labels + set_idx（predict 接口），方便下游 sweep threshold / 画图。

4. 可维护性：
   - 训练 loop 保持 HuggingFace 风格：
        * train_one_epoch
        * validate_one_epoch
        * predict
   - 默认不与 argparse / Hydra / WandB 强耦合：
        * 通过 DictConfig 接收配置（天然支持 Hydra）
        * 通过可选 logger 抽象支持 WandB / TensorBoard / 自定义 Logger
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from omegaconf import DictConfig

from src.evaluator.metrics import compute_metrics  # 你的统一评测函数


# ----------------------- #
# 训练状态对象
# ----------------------- #

@dataclass
class TrainState:
    """
    TrainState
    ==========
    保存训练过程中的全局状态（可序列化到 checkpoint）：

    字段
    ----
    epoch : int
        当前已完成的 epoch 数（从 0 开始）。
    global_step : int
        到目前为止执行过多少个 optimization step。
    best_metric : float
        当前为止最好的验证指标（例如 F1 / AUC 或 loss）。
        * 当 greater_is_better=True 时，初值为一个很小的数（-1e9）；
        * 当 greater_is_better=False 时，初值为一个很大的数（+1e9）。
    """
    epoch: int = 0
    global_step: int = 0
    best_metric: float = -1e9


# ----------------------- #
# EMA 辅助类
# ----------------------- #

class EMAHelper:
    """
    EMAHelper
    =========
    管理模型参数的指数滑动平均 (Exponential Moving Average)。

    用法示例
    --------
    ema = EMAHelper(model, decay=0.999)

    # 每个 optimization step 后调用：
    ema.update(model)

    # 验证 / 测试时：
    with ema.swap_parameters(model):
        ... 用 EMA 参数跑 forward ...
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}

        # 初始化 shadow 参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        在每一步优化之后调用：更新 EMA 参数。
        """
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                # 新出现的参数（极少见），直接初始化
                self.shadow[name] = param.detach().clone()
                continue
            new_avg = self.decay * self.shadow[name] + (1.0 - self.decay) * param.detach()
            self.shadow[name] = new_avg

    def swap_parameters(self, model: nn.Module):
        """
        上下文管理器：临时用 EMA 参数替换模型参数，退出时恢复原参数。

        用法：
            with ema.swap_parameters(model):
                ... eval / predict ...
        """
        return _EMASwapContext(self, model)


class _EMASwapContext:
    """
    _EMASwapContext
    ===============
    内部使用的 context manager，用于在 with 块内暂时替换参数为 EMA 版本。
    """

    def __init__(self, ema: EMAHelper, model: nn.Module):
        self.ema = ema
        self.model = model

    def __enter__(self):
        self.ema.backup = {}
        # 备份原始参数并替换为 shadow
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.ema.shadow:
                self.ema.backup[name] = param.data.clone()
                param.data.copy_(self.ema.shadow[name])

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复原始参数
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.ema.backup:
                param.data.copy_(self.ema.backup[name])
        self.ema.backup = {}


# ----------------------- #
# 参数分组：选择性 weight decay
# ----------------------- #

def build_weight_decay_param_groups(
    model: nn.Module,
    weight_decay: float,
    selective: bool = True,
) -> List[Dict[str, Any]]:
    """
    build_weight_decay_param_groups
    ===============================
    构建带 / 不带 weight decay 的参数组。

    设计原则
    --------
    - selective=True（推荐）：
        * 对卷积 / 线性层的 weight 使用 weight decay；
        * 对 bias、BatchNorm / LayerNorm / 其他 norm 参数，不使用 weight decay。
      这是现在 CNN / Transformer 的典型配置。
    - selective=False：
        * 所有参数共用一个 weight_decay（等价于 optimizer(model.parameters(), weight_decay=...)）。

    如果你想沿用原始的 get_params_and_initialize 逻辑：
        - 可以在构建 model 后，先调用那段函数得到 (params_with_decay, params_without_decay)
        - 然后用本函数替换掉里边的组装逻辑。
    """
    if not selective:
        return [{"params": list(model.parameters()), "weight_decay": weight_decay}]

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 典型规则：
        # - norm 层参数 & bias 不做 weight decay
        if (
            name.endswith(".bias")
            or "bn" in name.lower()
            or "batch_norm" in name.lower()
            or "layernorm" in name.lower()
            or "ln" in name.lower()
            or "norm" in name.lower()
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


# ----------------------- #
# Set-level 聚合工具
# ----------------------- #


def aggregate_by_set_idx(
    y_true_window: np.ndarray,
    y_pred_window: np.ndarray,
    set_idx_window: np.ndarray,
    set_labels: List[float],
    reduction: str = "max",
    *,
    softmax_tau: float = 1.0,
    topk: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 window-level 的预测聚合到 set-level。

    reduction 支持:
      - "max"        : max pooling（旧逻辑）
      - "mean"       : mean pooling
      - "softmax"    : softmax 权重加权平均（温度 softmax_tau）
      - "topk_mean"  : 对每个 set 取 top-k window 做平均
      - "sum"         
      
    返回:
      y_true_set: (num_sets,)
      y_pred_set: (num_sets,)
    """
    num_sets = len(set_labels)
    y_pred_window = np.asarray(y_pred_window).reshape(-1)
    set_idx_window = np.asarray(set_idx_window).reshape(-1).astype(np.int64)

    if num_sets == 0:
        # 没有任何 set，直接返回空，避免后面再炸
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    y_pred_set = np.zeros(num_sets, dtype=np.float32)

    for s in range(num_sets):
        mask = (set_idx_window == s)
        if not np.any(mask):
            # 这个 set 在当前 split 下没有任何 window → 预测值置 0
            y_pred_set[s] = 0.0
            continue

        vals = y_pred_window[mask]  # 该 set 下所有 windows 的 logit

        if reduction == "max":
            y_pred_set[s] = float(np.max(vals))

        elif reduction == "mean":
            y_pred_set[s] = float(np.mean(vals))

        elif reduction == "sum":
            y_pred_set[s] = float(np.sum(vals))

        elif reduction == "softmax":
            # softmax 池化：\sum softmax(vals / tau) * vals
            v = vals.astype(np.float64)
            v_scaled = v / max(softmax_tau, 1e-6)
            v_scaled -= v_scaled.max()  # 数值稳定
            w = np.exp(v_scaled)
            w_sum = w.sum()
            if w_sum <= 0:
                y_pred_set[s] = float(v.mean())
            else:
                w /= w_sum
                y_pred_set[s] = float((w * v).sum())

        elif reduction == "topk_mean":
            k = min(topk, len(vals))
            # 只取最大的 k 个 window 做平均
            if k <= 0:
                y_pred_set[s] = float(np.mean(vals))
            else:
                # np.partition 比 full sort 快
                idx = np.argpartition(vals, -k)[-k:]
                top_vals = vals[idx]
                y_pred_set[s] = float(np.mean(top_vals))

        else:
            raise ValueError(f"Unsupported reduction: {reduction}")

    y_true_set = np.array(set_labels, dtype=np.float32)
    return y_true_set, y_pred_set



# ----------------------- #
# 空 context manager
# ----------------------- #

class _nullcontext:
    """
    _nullcontext
    ============
    简易空上下文管理器，使得：

        with _nullcontext():
            ...

    等价于直接执行代码块。
    """
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


# ----------------------- #
# 主 Trainer
# ----------------------- #

class Trainer:
    """
    Trainer
    =======
    通用训练器，提供：

    - train_one_epoch(loader)         : 单个 epoch 的训练
    - validate_one_epoch(loader, ...) : 单个 epoch 的验证（可选择 window / set 级聚合）
    - predict(loader, ...)            : 前向推理，输出 logits / labels / set_idx
    - save_checkpoint(path)           : 保存模型 + 优化器 + 训练状态
    - load_checkpoint(path)           : 加载模型 + 优化器 + 训练状态

    关键特性
    --------
    1. 支持 AMP 混合精度 (train_cfg.amp)
    2. 支持梯度裁剪 (train_cfg.grad_clip)
    3. 支持 EMA 参数平滑 (train_cfg.ema.enabled / decay)
    4. 优化器 / 调度器 / 损失函数均可由上层 config 决定
    5. 内置 set-level 聚合逻辑，与旧版 Logger.aggregate 行为一致
    6. 支持监控指标切换：
        - train_cfg.monitor: "loss" / "f1" / "roc_auc" ...
        - train_cfg.greater_is_better: False (loss) / True (F1/AUC)
    7. 轻量 logger 抽象：
        - __init__(..., logger=None)
        - set_logger(logger)
        - 内部在 epoch 级别自动调用 logger.log_metrics(...)
          以便统一接 WandB / TensorBoard / 自定义实验管理。
    """

    def __init__(
        self,
        model: nn.Module,
        task_cfg: DictConfig,
        train_cfg: DictConfig,
        run_cfg: DictConfig,
        device: torch.device,
        custom_loss_fn: Optional[nn.Module] = None,
        use_selective_weight_decay: bool = True,
        logger: Optional[Any] = None,
    ):
        """
        参数
        ----
        model      : nn.Module
            任意 PyTorch 模型（TargetNet / Transformer / DiT）。
        task_cfg   : DictConfig
            任务配置，至少包含 problem_type；
            同时可以包含 threshold，用于 compute_metrics 中的固定阈值评估。
        train_cfg  : DictConfig
            训练配置，包含 lr / weight_decay / optimizer / scheduler / amp / grad_clip / monitor 等。
        device     : torch.device
        custom_loss_fn : Optional[nn.Module]
            当 loss_type == "custom" 时使用。
        use_selective_weight_decay : bool
            是否启用“选择性 weight decay”（bias/norm 不 decay）。
        logger     : Optional[Any]
            可选的 logger 对象，用于记录训练 / 验证指标。
            约定接口（任意其一即可）：
                - logger.log_metrics(metrics: Dict[str, float], step: Optional[int], prefix: str)
                - logger.log(metrics: Dict[str, float])
        """
        self.model = model.to(device)
        self.task_cfg = task_cfg
        self.train_cfg = train_cfg
        self.run_cfg = run_cfg
        self.device = device
        self.logger = logger  # 可选 logger，对接 WandB / TB / 自定义系统

        # ---------- 监控指标设置（loss or F1/AUC） ---------- #
        self.monitor: str = getattr(train_cfg, "monitor", "loss")
        if hasattr(train_cfg, "greater_is_better"):
            self.greater_is_better: bool = bool(train_cfg.greater_is_better)
        else:
            # 默认：monitor=="loss" → minimize；其他（F1/AUC）→ maximize
            self.greater_is_better = False if self.monitor == "loss" else True

        # 训练状态（best_metric 初值根据 greater_is_better 决定）
        if self.greater_is_better:
            self.state = TrainState(best_metric=-1e9)
        else:
            self.state = TrainState(best_metric=1e9)

        # ---------- 参数组 & 优化器 ---------- #
        param_groups = build_weight_decay_param_groups(
            self.model,
            weight_decay=float(getattr(train_cfg, "weight_decay", 0.0)),
            selective=use_selective_weight_decay,
        )
        self.optimizer = self._build_optimizer(train_cfg, param_groups)

        # ---------- 调度器 ---------- #
        self.scheduler = self._build_scheduler(train_cfg, self.optimizer)

        # ---------- 损失函数 ---------- #
        self.loss_type = getattr(train_cfg, "loss_type", "bce")
        self.criterion, self.criterion_reg = self._build_loss_fns(
            task_cfg, train_cfg, custom_loss_fn
        )

        # ---------- AMP / grad_clip ---------- #
        self.amp_enabled = bool(getattr(train_cfg, "amp", False))
        self.grad_clip = getattr(train_cfg, "grad_clip", None)
        self.scaler = torch.amp.GradScaler(device="cuda", enabled=self.amp_enabled,)
        # ---------- EMA ---------- #
        ema_cfg = getattr(train_cfg, "ema", None)
        self.ema: Optional[EMAHelper] = None
        if ema_cfg is not None and getattr(ema_cfg, "enabled", False):
            decay = float(getattr(ema_cfg, "decay", 0.999))
            self.ema = EMAHelper(self.model, decay=decay)
        # ---------- ESA ---------- #
        # 用于在 batch / loss 之间传递 ESA 权重
        self._last_esa_scores = None
        self._current_sample_weight = None

    # --------- logger 相关工具 --------- #

    def set_logger(self, logger: Any) -> None:
        """
        set_logger
        ==========
        运行时动态设置 / 替换 logger 对象。

        参数
        ----
        logger : Any
            任意实现了 logger.log_metrics(...) 或 logger.log(...) 的对象。
            示例：
                class WandBLogger:
                    def __init__(self, run): self.run = run
                    def log_metrics(self, metrics, step=None, prefix=""):
                        data = {}
                        for k, v in metrics.items():
                            k2 = f"{prefix}{k}" if prefix else k
                            data[k2] = float(v)
                        self.run.log(data, step=step)
        """
        self.logger = logger

    def _log_metrics(
        self,
        metrics: Dict[str, float],
        stage: str,
        step: Optional[int] = None,
    ) -> None:
        """
        _log_metrics
        ============
        内部统一的 metrics 上报入口。

        行为
        ----
        - 若 self.logger 为 None，则什么都不做；
        - 若 logger 有 .log_metrics(...) 方法，则调用：
            logger.log_metrics(metrics, step=step, prefix=f"{stage}/")
        - 否则若 logger 有 .log(...) 方法，则调用：
            logger.log({f"{stage}/{k}": v for k,v in metrics.items()}, step=step)

        注意
        ----
        - 这个函数只在 epoch 级别调用：
            * train_one_epoch 结束后  → stage="train"
            * validate_one_epoch 结束后 → stage="val"
        - step 默认为 self.state.global_step（即迄今为止的总优化步数）。
        """
        if self.logger is None:
            return

        if step is None:
            step = self.state.global_step

        # 给 metrics 加上 stage 前缀
        prefixed = {f"{stage}/{k}": float(v) for k, v in metrics.items() if isinstance(v, (int, float))}

        if hasattr(self.logger, "log_metrics"):
            try:
                # logger.log_metrics 允许自己处理 prefix，这里直接传原始 metrics
                self.logger.log_metrics(metrics, step=step, prefix=f"{stage}/")
                return
            except TypeError:
                # 如果签名不匹配，就退回到 log 接口
                pass

        if hasattr(self.logger, "log"):
            # 常见的是 WandB 的 run.log(...)
            try:
                self.logger.log(prefixed, step=step)
                return
            except TypeError:
                # 有些 logger.log 不接受 step 参数，就只传数据
                self.logger.log(prefixed)

    # --------- 构建 optimizer / scheduler / loss --------- #

    def _build_optimizer(
        self,
        train_cfg: DictConfig,
        param_groups: List[Dict[str, Any]],
    ) -> optim.Optimizer:
        """
        根据 train_cfg.optimizer 构建优化器。

        支持：
        - "adamw" (默认)
        - "adam"
        - "sgd"
        - "rmsprop"
        以后可以继续扩展（如 Lion）。
        """
        opt_name = getattr(train_cfg, "optimizer", "adamw").lower()
        lr = float(train_cfg.lr)

        if opt_name == "adam":
            return optim.Adam(param_groups, lr=lr)
        elif opt_name == "sgd":
            momentum = float(getattr(train_cfg, "momentum", 0.9))
            return optim.SGD(param_groups, lr=lr, momentum=momentum)
        elif opt_name == "rmsprop":
            return optim.RMSprop(param_groups, lr=lr)
        else:  # 默认 AdamW
            return optim.AdamW(param_groups, lr=lr)

    def _build_scheduler(
        self,
        train_cfg: DictConfig,
        optimizer: optim.Optimizer,
    ):
        """
        根据 train_cfg.scheduler 构建调度器。

        支持：
        - "none"    : 不使用调度器
        - "plateau" : ReduceLROnPlateau (监控某个 metric，可最小化或最大化)
        - "cosine"  : CosineAnnealingLR
        - "step"    : StepLR
        """
        sched_name = getattr(train_cfg, "scheduler", "none").lower()

        if sched_name == "none":
            return None

        if sched_name == "plateau":
            factor = float(getattr(train_cfg, "scheduler_factor", 0.2))
            patience = int(getattr(train_cfg, "scheduler_patience", 5))
            # mode 根据 greater_is_better 决定
            mode = "max" if self.greater_is_better else "min"
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
            )
        elif sched_name == "cosine":
            t_max = int(getattr(train_cfg, "scheduler_t_max", 10))
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        elif sched_name == "step":
            step_size = int(getattr(train_cfg, "scheduler_step_size", 10))
            gamma = float(getattr(train_cfg, "scheduler_gamma", 0.1))
            return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        else:
            # 未知字符串：默认不使用调度器
            return None

    def _build_loss_fns(
        self,
        task_cfg: DictConfig,
        train_cfg: DictConfig,
        custom_loss_fn: Optional[nn.Module],
    ) -> Tuple[Optional[nn.Module], Optional[nn.Module]]:
        """
        构建损失函数。

        - task_cfg.problem_type: "binary_classification" / "regression"
        - train_cfg.loss_type  : "bce" / "mse" / "custom"

        返回
        ----
        (criterion_cls, criterion_reg)
        - 对于 binary_classification:
            * criterion_cls: BCEWithLogitsLoss / custom
            * criterion_reg: None
        - 对于 regression:
            * criterion_cls: None
            * criterion_reg: MSELoss / custom
        """
        problem_type = getattr(task_cfg, "problem_type", "binary_classification")
        loss_type = getattr(train_cfg, "loss_type", "bce").lower()

        if loss_type == "custom":
            if custom_loss_fn is None:
                raise ValueError("loss_type='custom' 但没有传入 custom_loss_fn")
            return custom_loss_fn, custom_loss_fn

        if problem_type == "binary_classification":
            # ✅ 所有 BCE / Focal / BCE+Focal 的具体逻辑都在 _compute_loss 里实现
            # 这里统一返回 (None, None) 作为占位
            if loss_type in ["bce", "focal", "bce_focal"]:
                return None, None
            else:
                raise ValueError(
                    f"Unsupported loss_type={loss_type} for binary_classification "
                    "(支持: 'bce', 'focal', 'bce_focal', 'custom')"
                )


        elif problem_type == "regression":
            if loss_type == "mse":
                return None, nn.MSELoss()
            else:
                raise ValueError(f"Unsupported loss_type={loss_type} for regression")

        else:
            raise ValueError(f"Unknown problem_type: {problem_type}")

    # --------- 内部工具：batch 解包 + loss 计算 --------- #

    def _unpack_batch(self, batch: Any):
        """
        解包 batch，统一输出 (inputs, labels, set_idx)。

        支持两种格式：
        1) 旧格式：tuple/list
           (x, y, set_idx)
        2) 新格式：dict
           {
             "inputs": Tensor,
             "labels": Tensor,
             "set_idx": Tensor (optional)
           }
        """
        if isinstance(batch, dict):
            x = batch["inputs"].to(self.device, non_blocking=True)
            y = batch["labels"].to(self.device, non_blocking=True)
            set_idx = batch.get("set_idx", None)
            if set_idx is not None:
                set_idx = set_idx.to(self.device, non_blocking=True)

            esa = batch.get("esa_scores", None)
            if esa is not None:
                esa = esa.to(self.device, non_blocking=True)
        else:
            # 兼容 (x, y, set_idx) / (x, y, set_idx, esa)
            if len(batch) == 3:
                x, y, set_idx = batch
                esa = None
            else:
                x, y, set_idx, esa = batch
                if esa is not None:
                    esa = esa.to(self.device, non_blocking=True)
            
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            if set_idx is not None:
                set_idx = set_idx.to(self.device, non_blocking=True)

        # ✅ 把当前 batch 的 ESA 缓存下来，后面 train_one_epoch 会用
        self._last_esa_scores = esa

        # 你的数据是 uint8 one-hot，需要转 float
        if x.dtype == torch.uint8:
            x = x.float()

        return x, y.float(), set_idx

    def _is_pair_level_batch(self, batch: Any) -> bool:
        """
        判断当前 batch 是否来自 PairLevelDataset。
        约定：pair-level batch 是 dict 且至少包含键 "tokens"。
        """
        return isinstance(batch, dict) and ("tokens" in batch)

    # def _binary_focal_loss(
    #     self,
    #     logits: torch.Tensor,
    #     labels: torch.Tensor,
    #     sample_weight: torch.Tensor = None,
    # ) -> torch.Tensor:
    #     """
    #     Binary Focal Loss，支持 soft label（用于配合 label smoothing）。

    #     参数
    #     ----
    #     logits : (N,)
    #     labels : (N,)  已经做过 label smoothing 的 target（在 [0,1]）
    #     """
    #     logits = logits.view(-1)
    #     labels = labels.view(-1)

    #     gamma = float(getattr(self.train_cfg, "focal_gamma", 2.0))
    #     alpha = float(getattr(self.train_cfg, "focal_alpha", 0.25))

    #     # 概率 p = sigmoid(logit)
    #     p = torch.sigmoid(logits)

    #     # 对“目标类别”的概率 p_t：
    #     # 对于 hard label: p_t = p (y=1), 1-p (y=0)
    #     # 这里 labels 可能是 soft（smoothing 后），做一个线性插值版本：
    #     p_t = p * labels + (1.0 - p) * (1.0 - labels)

    #     # alpha_t 同理，用 soft label 做插值
    #     alpha_t = alpha * labels + (1.0 - alpha) * (1.0 - labels)

    #     eps = 1e-8
    #     # 基础 CE（用 soft label）
    #     ce = -(
    #         labels * torch.log(p.clamp(min=eps))
    #         + (1.0 - labels) * torch.log((1.0 - p).clamp(min=eps))
    #     )

    #     focal_factor = alpha_t * (1.0 - p_t).pow(gamma)
    #     loss = focal_factor * ce

    #     if sample_weight is not None:
    #         sample_weight = sample_weight.view_as(loss)
    #         loss = loss * sample_weight

    #     return loss.mean()


    def _binary_focal_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weight: torch.Tensor = None,
        pos_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        稳定版 Binary Focal Loss，支持 soft label + sample_weight + pos_weight。

        logits : (N,)
        labels : (N,)  已经做过 label smoothing 的 target（在 [0,1]）
        """
        logits = logits.view(-1)
        labels = labels.view(-1).float()

        gamma = float(getattr(self.train_cfg, "focal_gamma", 2.0))
        alpha = float(getattr(self.train_cfg, "focal_alpha", 0.25))

        # 1) 先用官方 BCEWithLogitsLoss 做“稳定的 CE”（不做 reduction）
        #    这里可以同时考虑 pos_weight
        ce = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=pos_weight,
            reduction="none",
        )  # [N]

        # 2) 计算 p 和 soft-label 意义下的 p_t
        p = torch.sigmoid(logits)              # [N]
        # soft p_t: “对”的概率
        p_t = p * labels + (1.0 - p) * (1.0 - labels)
        p_t = p_t.clamp(min=1e-6, max=1.0 - 1e-6)

        # 3) alpha_t 同样用 soft label 做插值
        alpha_t = alpha * labels + (1.0 - alpha) * (1.0 - labels)

        # 4) focal 因子
        focal_factor = alpha_t * (1.0 - p_t).pow(gamma)   # [N]

        # 5) 组合：focal-weighted CE
        loss = focal_factor * ce                          # [N]

        # 6) 可选：sample_weight（如 ESA 权重）
        if sample_weight is not None:
            sample_weight = sample_weight.view_as(loss)
            loss = loss * sample_weight                   # [N]

        return loss.mean()

    def _compute_esa_sample_weight(self, esa_scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        根据 ESA score 生成 per-window 权重。
        只在 train.train_reduction == "none" 且 train.esa_weighting=True 时使用。

        w_pos = 1 + lambda_pos * norm_esa
        w_neg = 1 + lambda_neg * norm_esa
        """
        esa_flat = esa_scores.view(-1).float()
        labels_flat = labels.view(-1).float()

        scale = float(getattr(self.train_cfg, "esa_scale", 10.0))
        lam_pos = float(getattr(self.train_cfg, "esa_lambda_pos", 0.0))
        lam_neg = float(getattr(self.train_cfg, "esa_lambda_neg", 0.0))

        # 简单归一化到 [0, 1]
        scale = max(scale, 1e-6)
        esa_norm = torch.clamp(esa_flat / scale, 0.0, 1.0)

        w_pos = 1.0 + lam_pos * esa_norm
        w_neg = 1.0 + lam_neg * esa_norm

        sample_weight = torch.where(labels_flat > 0.5, w_pos, w_neg)
        return sample_weight


    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        根据 problem_type + train 配置计算 loss。

        Stage 0.1:
          - class weighting (pos_weight)
          - label smoothing (smooth_pos / smooth_neg)

        Stage 0.2:
          - loss_type: "bce" / "focal" / "bce_focal"
            * "bce"      : 只用 BCE（兼容 Stage 0.1 行为）
            * "focal"    : 只用 Focal loss
            * "bce_focal": BCE + lambda_focal * Focal
        """
        problem_type = getattr(self.task_cfg, "problem_type", "binary_classification")

        if problem_type == "binary_classification":
            # 展平成一维，和以前行为一致
            logits_flat = logits.view(-1)
            labels_flat = labels.view(-1).float()

            # 取出 ESA-based per-sample weight（如果有）
            sample_weight = getattr(self, "_current_sample_weight", None)
            if sample_weight is not None:

                # print("[DEBUG] sample_weight min/max:",
                #     sample_weight.min().item(),
                #     sample_weight.max().item())
                
                # 防止 train 残留到 val，长度对不上时直接忽略
                if sample_weight.numel() != logits_flat.numel():
                    sample_weight = None
                else:
                    sample_weight = sample_weight.view_as(logits_flat)

            # -------- Label smoothing -------- #
            use_smoothing = bool(getattr(self.train_cfg, "label_smoothing", False))
            if use_smoothing:
                smooth_pos = float(getattr(self.train_cfg, "smooth_pos", 0.9))
                smooth_neg = float(getattr(self.train_cfg, "smooth_neg", 0.1))

                labels_smooth = torch.where(
                    labels_flat > 0.5,
                    torch.full_like(labels_flat, smooth_pos),
                    torch.full_like(labels_flat, smooth_neg),
                )
            else:
                labels_smooth = labels_flat

            # -------- Class weighting (pos_weight) -------- #
            pos_weight_val = float(getattr(self.train_cfg, "pos_weight", 1.0))
            if abs(pos_weight_val - 1.0) < 1e-8:
                pos_weight_tensor = None  # 和普通 BCE 完全等价
            else:
                pos_weight_tensor = torch.tensor(
                    pos_weight_val,
                    dtype=logits_flat.dtype,
                    device=logits_flat.device,
                )

            # -------- 根据 loss_type 决定怎么用这些东西 -------- #
                        # -------- 根据 loss_type 决定怎么用这些东西 -------- #
            loss_type = str(getattr(self.train_cfg, "loss_type", "bce")).lower()


            if loss_type == "bce":
                # ✅ 完全保留 Stage 0.1 的行为：BCE + (可选) pos_weight + (可选) smoothing
                loss = F.binary_cross_entropy_with_logits(
                    logits_flat,
                    labels_smooth,
                    weight=sample_weight,
                    pos_weight=pos_weight_tensor,
                )
                return loss

            elif loss_type == "focal":
                return self._binary_focal_loss(
                    logits_flat,
                    labels_smooth,
                    sample_weight=sample_weight,
                    pos_weight=pos_weight_tensor,
                )

            elif loss_type == "bce_focal":
                # 1) BCE 部分：保留原有 pos_weight + sample_weight 行为
                bce_loss = F.binary_cross_entropy_with_logits(
                    logits_flat,
                    labels_smooth,
                    weight=sample_weight,
                    pos_weight=pos_weight_tensor,
                )

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print("[WARN] logits contains NaN/Inf in focal loss!")

                # 2) Focal 部分：用稳定版实现 + 同样的 pos_weight / sample_weight
                focal_loss = self._binary_focal_loss(
                    logits_flat,
                    labels_smooth,
                    sample_weight=sample_weight,
                    pos_weight=pos_weight_tensor,
                )

                lambda_focal = float(getattr(self.train_cfg, "focal_lambda", 1.0))
                lambda_bce = float(getattr(self.train_cfg, "bce_lambda", 1.0))
                            
                # print(f"[DEBUG] bce_loss={bce_loss} focal_loss={focal_loss}")

                return lambda_bce * bce_loss + lambda_focal * focal_loss


            else:
                raise ValueError(f"Unknown loss_type for binary_classification: {loss_type}")

        elif problem_type == "regression":
            assert self.criterion_reg is not None
            return self.criterion_reg(logits.view(-1), labels.view(-1))

        else:
            raise ValueError(f"Unknown problem_type: {problem_type}")



    def _apply_mil_reduction(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        set_idx: torch.Tensor,
        *,
        reduction: str = "max",
        temperature: float = 1.0,
        topk: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对 window-level 的 logits/labels 按 set_idx 做 MIL 聚合，返回 pair-level 的 logits/labels。

        参数
        ----
        logits   : (N,) or (N,1)
        labels   : (N,) or (N,1)  -- 这里假设同一 set 内 label 一样（你的数据就是这样）
        set_idx  : (N,)           -- 当前 batch 内的 pair index，0..num_sets-1
        reduction: str            -- "max" / "mean" / "sum" / "softmax" / "topk_mean"
        temperature: float        -- softmax pooling 的温度
        topk       : int          -- topk_mean 的 k 值

        返回
        ----
        agg_logits: (num_sets,)
        agg_labels: (num_sets,)
        """

        # 统一展平成 1D
        logits = logits.view(-1)
        labels = labels.view(-1)
        set_idx = set_idx.view(-1).long()

        device = logits.device

        # ===== 关键：在 batch 内对 set_idx 做重编号 =====
        # unique_sets: 实际出现在本 batch 的 bag id（排序后）
        # new_set_idx: 每个样本对应的“局部 bag id”，范围 0..num_sets_batch-1
        unique_sets, new_set_idx = torch.unique(set_idx, return_inverse=True)
        num_sets = unique_sets.numel()

        dtype_logits = logits.dtype
        dtype_labels = labels.dtype


        if reduction == "max":
            # -------- max pooling --------
            # 用 logits 的 dtype，避免 AMP 下 dtype 不匹配
            min_val = torch.finfo(dtype_logits).min
            agg_logits = torch.full(
                (num_sets,),
                min_val,
                device=device,
                dtype=dtype_logits,
            )
            if hasattr(torch.Tensor, "scatter_reduce_"):
                agg_logits.scatter_reduce_(0, new_set_idx, logits, reduce="amax", include_self=True)
            else:
                for i in range(num_sets):
                    mask = (new_set_idx == i)
                    if mask.any():
                        agg_logits[i] = logits[mask].max()


        elif reduction == "mean":
            # -------- mean pooling --------
            agg_logits = torch.zeros(num_sets, device=device, dtype=dtype_logits)
            counts = torch.zeros(num_sets, device=device, dtype=dtype_logits)
            agg_logits.scatter_add_(0, new_set_idx, logits)
            counts.scatter_add_(0, new_set_idx, torch.ones_like(logits, dtype=dtype_logits))
            agg_logits = agg_logits / counts.clamp_min(1.0)


        elif reduction == "sum":
            # -------- sum pooling --------
            agg_logits = torch.zeros(num_sets, device=device, dtype=dtype_logits)
            agg_logits.scatter_add_(0, new_set_idx, logits)


        elif reduction == "softmax":
            # -------- softmax pooling (attention MIL) --------
            # 对每个 set 内部做 softmax(temperature * logit)，再加权平均 logit
            agg_logits = torch.zeros(num_sets, device=device, dtype=dtype_logits)
            for i in range(num_sets):
                mask = (new_set_idx == i)
                if not mask.any():
                    continue
                scores = logits[mask]
                w = torch.softmax(scores * temperature, dim=0)  # (n_i,)
                agg_logits[i] = (w * scores).sum()


        elif reduction == "topk_mean":
            # -------- top-k mean pooling --------
            agg_logits = torch.zeros(num_sets, device=device, dtype=dtype_logits)
            k = max(1, topk)
            for i in range(num_sets):
                mask = (new_set_idx == i)
                if not mask.any():
                    continue
                scores = logits[mask]
                k_i = min(k, scores.numel())
                topk_vals, _ = torch.topk(scores, k_i)
                agg_logits[i] = topk_vals.mean()


        else:
            raise ValueError(f"Unsupported MIL reduction: {reduction}")

        # label 聚合：因为每个 set 内 label 一样，直接取第一个/最大都行
        agg_labels = torch.zeros(num_sets, device=device, dtype=dtype_labels)
        if hasattr(torch.Tensor, "scatter_reduce_"):
            agg_labels.scatter_reduce_(0, new_set_idx, labels, reduce="amax", include_self=False)
        else:
            for i in range(num_sets):
                mask = (new_set_idx == i)
                if mask.any():
                    agg_labels[i] = labels[mask][0]


        return agg_logits, agg_labels



    # --------- 训练 --------- #

    def train_one_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """
        单个 epoch 训练。

        返回
        ----
        {"loss": avg_loss, "lr": current_lr}
        """
        self.model.train()
        total_loss = 0.0
        num_samples = 0

        mil_reduction = getattr(self.run_cfg, "train_reduction", "none")
        mil_temperature = float(getattr(self.run_cfg, "train_softmax_temp", 1.0))
        mil_topk = int(getattr(self.run_cfg, "train_topk", 3))


        for batch in tqdm(loader, desc=f"Train epoch {self.state.epoch}"):

            # 1) 先解包出 x, y, set_idx（兼容所有数据格式）
            x, y, set_idx = self._unpack_batch(batch)
            esa_scores = self._last_esa_scores

            # 2) 检查是否为 pair-level 模式
            is_pair_level = self._is_pair_level_batch(batch)
            
            # pair-level 模式不需要 ESA weighting；确保不要误开
            if is_pair_level:
                self._current_sample_weight = None
                # 即便 train.esa_weighting=True，这里也不走 ESA 权重逻辑
            else:
                # 只有在非 pair-level 模式下才允许 ESA weighting
                # ✅ 若打开 ESA weighting，但 batch 里没有 esa_scores，直接报错，避免悄悄变成 no-op
                if bool(getattr(self.train_cfg, "esa_weighting", False)) and esa_scores is None:
                    raise RuntimeError(
                        "train.esa_weighting=True 但 batch 中没有 'esa_scores' 字段。\n"
                        "请检查：Dataset.__getitem__ 是否返回 esa，collate_fn 是否把 esa 汇总进 batch，"
                        "以及 _unpack_batch 中对 'esa_scores' 的处理。"
                    )


            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=self.amp_enabled):

                # 3) 前向：根据 batch 类型决定是否传入 mask / pos
                if is_pair_level:
                    # 从 batch 中取出 mask / pos
                    mask = batch["mask"].to(self.device, non_blocking=True)
                    pos = batch.get("pos", None)
                    if pos is not None:
                        pos = pos.to(self.device, non_blocking=True)
                    logits = self.model(x, attn_mask=mask, pos=pos)  # [B]
                else:
                    logits = self.model(x)  # 旧模型（window-level）
                # logits = self.model(x)   # window-level logits, shape ~ (N, 1) or (N,)

                # 4) MIL 聚合（pair-level 本身就是 bag 级别，通常不需要再聚）
                # ====== MIL聚合(可选) ====== #
                if (not is_pair_level) and (set_idx is not None) and (mil_reduction is not None) and (mil_reduction != "none"):
                    logits_for_loss, labels_for_loss = self._apply_mil_reduction(
                        logits=logits,
                        labels=y,
                        set_idx=set_idx,
                        reduction=mil_reduction,
                        temperature=mil_temperature,
                        topk=mil_topk,
                    )
                    # 当前 Stage 0.x 专注 window-wise，bag-level 暂不加 ESA 权重 
                    esa_for_loss = None
                else:
                    # pair-level 直接用 pair 级 logits / labels
                    logits_for_loss, labels_for_loss = logits, y
                    esa_for_loss = None if is_pair_level else esa_scores

                # 5) ESA-aware per-window weighting（仅 window-level 使用）
                # ====== ESA-aware per-window weighting（Stage 0.3） ====== #
                use_esa_weighting = bool(getattr(self.train_cfg, "esa_weighting", False))
                if (not is_pair_level) and use_esa_weighting and (esa_for_loss is not None):
                    self._current_sample_weight = self._compute_esa_sample_weight(
                        esa_for_loss, labels_for_loss
                    )
                else:
                    self._current_sample_weight = None

                loss = self._compute_loss(logits_for_loss, labels_for_loss)

            self.scaler.scale(loss).backward()

            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # EMA 更新
            if self.ema is not None:
                self.ema.update(self.model)

            bs = logits.shape[0]
            total_loss += loss.item() * bs
            num_samples += bs
            self.state.global_step += 1

        avg_loss = total_loss / max(1, num_samples)
        # 记录当前学习率（使用第一个 param_group）
        current_lr = self.optimizer.param_groups[0].get("lr", 0.0)

        metrics = {"loss": avg_loss, "lr": current_lr}
        # 将 train metrics 上报给 logger
        self._log_metrics(metrics, stage="train", step=self.state.global_step)
        return metrics

    # --------- 验证（含 metrics + set-level 聚合） --------- #

    @torch.no_grad()
    def validate_one_epoch(
        self,
        loader: DataLoader,
        set_labels: Optional[List[float]] = None,
        aggregate_sets: bool = True,
        use_ema: bool = True,
    ) -> Dict[str, float]:
        """
        单个 epoch 验证。

        参数
        ----
        loader        : DataLoader
        set_labels    : Optional[List[float]]
                        - 若为 None，则在 window-level 上计算指标；
                        - 若不为 None 且 aggregate_sets=True，
                          则按 set_idx 做 max pooling 聚合到 set-level。
        aggregate_sets: bool
        use_ema       : bool
                        - 若 True 且 self.ema 存在，则使用 EMA 参数进行评估。

        返回
        ----
        metrics : Dict[str,float]
          例如：
            {
              "loss": xx,
              "f1": xx,
              "roc_auc": xx,
              "pr_auc": xx,
              ...
            }
        """

        mil_reduction = getattr(self.run_cfg, "eval_reduction", "max")
        mil_temperature = float(getattr(self.run_cfg, "train_softmax_temp", 1.0))
        mil_topk = int(getattr(self.run_cfg, "train_topk", 3))

        # ✅ 验证阶段不使用 ESA 加权，防止 train 残留的 sample_weight 干扰
        self._current_sample_weight = None
        self._last_esa_scores = None

        ctx = self.ema.swap_parameters(self.model) if (use_ema and self.ema is not None) else _nullcontext()

        with ctx:
            self.model.eval()

            total_loss = 0.0
            num_samples = 0
            all_logits = []
            all_labels = []
            all_set_idx = []

            for batch in tqdm(loader, desc=f"Valid epoch {self.state.epoch}"):
                x, y, set_idx = self._unpack_batch(batch)
                is_pair_level = self._is_pair_level_batch(batch)


                with torch.amp.autocast(device_type="cuda", enabled=self.amp_enabled):
                    if is_pair_level:
                        mask = batch["mask"].to(self.device, non_blocking=True)
                        pos = batch.get("pos", None)
                        if pos is not None:
                            pos = pos.to(self.device, non_blocking=True)
                        logits = self.model(x, attn_mask=mask, pos=pos)
                    else:
                        logits = self.model(x)

                    # logits = self.model(x)
                    # 验证阶段统一在“模型输出的粒度”上算 loss
                    loss = self._compute_loss(logits, y)

                bs = logits.shape[0]
                total_loss += loss.item() * bs
                num_samples += bs

                all_logits.append(logits.detach().cpu())
                all_labels.append(y.detach().cpu())
                if set_idx is not None:
                    all_set_idx.append(set_idx.detach().cpu())

            avg_loss = total_loss / max(1, num_samples)
            logits = torch.cat(all_logits).numpy()
            labels = torch.cat(all_labels).numpy()

            # aggregate_sets 部分里，只有在 len(all_set_idx) > 0 时才会走 set-level 聚合；
            # 而 pair-level batch 没有 set_idx，因此 all_set_idx 为空，自然就直接用 pair-level 预测做 metrics，
            # 若需要 set-level 聚合
            if aggregate_sets and set_labels is not None and len(all_set_idx) > 0:
                set_idx_full = torch.cat(all_set_idx).numpy()
                y_true, y_pred_raw = aggregate_by_set_idx(
                    y_true_window=labels,
                    y_pred_window=logits,
                    set_idx_window=set_idx_full,
                    set_labels=set_labels,
                    reduction=mil_reduction,
                    softmax_tau=mil_temperature,
                    topk=mil_topk,
                )
            else:
                        
                # print("[validate_one_epoch] Warning: No aggregate")
                # print(f"[validate_one_epoch] Warning: aggregate_sets: {aggregate_sets}, set_labels: {set_labels is not None}, all_set_idx: {len(all_set_idx) > 0}")
                y_true, y_pred_raw = labels, logits

            # 由 compute_metrics 负责：
            # - 把 raw logits 转为概率（sigmoid）；
            # - 使用 task_cfg.threshold 计算 F1 等。
            metrics = compute_metrics(
                y_true=y_true,
                y_pred_raw=y_pred_raw,
                task_cfg=self.task_cfg,
            )
            metrics["loss"] = avg_loss
            # 也可以附带当前 lr，方便定位：val metrics 对应哪一个 lr
            current_lr = self.optimizer.param_groups[0].get("lr", 0.0)
            metrics["lr"] = current_lr

            # ---------- 调度器 step ---------- #
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    # Plateau 一般监控某个 metric（loss / f1 / auc）
                    monitor_value = metrics.get(self.monitor, avg_loss)
                    self.scheduler.step(monitor_value)
                else:
                    # 其他 scheduler（Cosine / Step）直接按 epoch 前进
                    self.scheduler.step()

            # ---------- 更新 best_metric ---------- #
            monitor_value = metrics.get(self.monitor, None)
            if monitor_value is not None:
                if self.greater_is_better:
                    if monitor_value > self.state.best_metric:
                        self.state.best_metric = monitor_value
                else:
                    if monitor_value < self.state.best_metric:
                        self.state.best_metric = monitor_value

            # 将 val metrics 上报给 logger
            self._log_metrics(metrics, stage="val", step=self.state.global_step)
            return metrics

    # --------- 推理 / 导出预测 --------- #

    @torch.no_grad()
    def predict(
        self,
        loader: DataLoader,
        use_ema: bool = True,
    ) -> Dict[str, Any]:
        """
        仅做前向推理（不计算 loss），返回原始 logits / labels / set_idx。

        典型用途：
        - 在 test 集上跑一遍，然后用 evaluator / metrics.py：
            * 从 logits / labels 里扫各种 threshold，找到 F1 最优；
            * 画 ROC / PR 曲线。
        """
        # ✅ 推理阶段同样不使用 ESA sample_weight
        self._current_sample_weight = None
        self._last_esa_scores = None

        ctx = self.ema.swap_parameters(self.model) if (use_ema and self.ema is not None) else _nullcontext()
        with ctx:
            self.model.eval()

            all_logits = []
            all_labels = []
            all_set_idx = []

            for batch in tqdm(loader, desc="Predict"):
                x, y, set_idx = self._unpack_batch(batch)
                is_pair_level = self._is_pair_level_batch(batch)

                if is_pair_level:
                    mask = batch["mask"].to(self.device, non_blocking=True)
                    pos = batch.get("pos", None)
                    if pos is not None:
                        pos = pos.to(self.device, non_blocking=True)
                    logits = self.model(x, attn_mask=mask, pos=pos)
                else:
                    logits = self.model(x)

                # logits = self.model(x)

                all_logits.append(logits.detach().cpu())
                all_labels.append(y.detach().cpu())
                if set_idx is not None:
                    all_set_idx.append(set_idx.detach().cpu())

            logits = torch.cat(all_logits).numpy()
            labels = torch.cat(all_labels).numpy()
            set_idx = None
            if len(all_set_idx) > 0:
                set_idx = torch.cat(all_set_idx).numpy()

            return {
                "logits": logits,
                "labels": labels,
                "set_idx": set_idx,
            }



    # --------- checkpoint 存取 --------- #

    def save_checkpoint(self, path: str):
        """
        保存当前模型 / 优化器 / 训练状态（以及可选 EMA shadow）。

        payload 字段
        -----------
        - "state_dict"        : 模型参数
        - "optimizer"         : 优化器状态
        - "train_state"       : TrainState(epoch, global_step, best_metric)
        - "monitor"           : 当前监控指标名
        - "greater_is_better" : 监控指标是否是“越大越好”
        - "ema_shadow"        : 若开启 EMA，则保存 shadow 参数
        """
        payload: Dict[str, Any] = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_state": {
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "best_metric": self.state.best_metric,
            },
            "monitor": self.monitor,
            "greater_is_better": self.greater_is_better,
        }
        if self.ema is not None:
            payload["ema_shadow"] = self.ema.shadow

        torch.save(payload, path)

    def load_checkpoint(self, path: str, map_location=None):
        """
        从 checkpoint 加载模型 / 优化器 / 训练状态。

        注意
        ----
        - 若 checkpoint 中没有 optimizer / train_state 字段，则只恢复模型参数；
        - 若 checkpoint 中含有 ema_shadow 而当前 Trainer 也有 EMA，
          则会恢复 EMA 的 shadow。
        """
        ckpt = torch.load(path, map_location=map_location)
        self.model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "train_state" in ckpt:
            ts = ckpt["train_state"]
            self.state = TrainState(
                epoch=ts.get("epoch", 0),
                global_step=ts.get("global_step", 0),
                best_metric=ts.get("best_metric", -1e9),
            )
        # 恢复 monitor / greater_is_better（防止跑到一半改配置）
        if "monitor" in ckpt:
            self.monitor = ckpt["monitor"]
        if "greater_is_better" in ckpt:
            self.greater_is_better = ckpt["greater_is_better"]
        if "ema_shadow" in ckpt and self.ema is not None:
            self.ema.shadow = ckpt["ema_shadow"]
