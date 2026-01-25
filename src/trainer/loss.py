# src/trainer/loss.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import torch
import torch.nn.functional as F


@dataclass
class BinaryLossInputs:
    """
    兼容旧 Trainer 的行为：
    - loss_type: "bce" / "focal" / "bce_focal"
    - label_smoothing: bool
    - smooth_pos / smooth_neg
    - pos_weight
    - focal_gamma / focal_alpha / focal_lambda / bce_lambda
    - sample_weight: Optional[Tensor] (外部可传入；旧 Trainer 里来自 ESA weighting)
    """
    train_cfg: Any  # DictConfig / dataclass / 任意有属性的对象


class BinaryClassificationLoss:
    """
    从旧 src/trainer/trainer.py 的 _compute_loss / _binary_focal_loss 迁移而来。
    设计为：不依赖 Trainer 类；只依赖 train_cfg 上的字段；支持可选 sample_weight/pos_weight/smoothing。
    """

    def __init__(self, train_cfg: Any):
        self.train_cfg = train_cfg
        self._current_sample_weight: Optional[torch.Tensor] = None

    def set_sample_weight(self, w: Optional[torch.Tensor]) -> None:
        self._current_sample_weight = w

    def _binary_focal_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        稳定版 Binary Focal Loss，支持 soft label + sample_weight + pos_weight。
        逻辑与旧 Trainer 一致。
        """
        logits = logits.view(-1)
        labels = labels.view(-1).float()

        gamma = float(getattr(self.train_cfg, "focal_gamma", 2.0))
        alpha = float(getattr(self.train_cfg, "focal_alpha", 0.25))

        ce = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=pos_weight,
            reduction="none",
        )  # [N]

        p = torch.sigmoid(logits)  # [N]
        p_t = p * labels + (1.0 - p) * (1.0 - labels)
        p_t = p_t.clamp(min=1e-6, max=1.0 - 1e-6)

        alpha_t = alpha * labels + (1.0 - alpha) * (1.0 - labels)
        focal_factor = alpha_t * (1.0 - p_t).pow(gamma)  # [N]

        loss = focal_factor * ce  # [N]

        if sample_weight is not None:
            sample_weight = sample_weight.view_as(loss)
            loss = loss * sample_weight

        return loss.mean()

    def compute(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        等价于旧 Trainer._compute_loss 的 binary_classification 分支（不含 regression）。
        - logits: [B] or [B,1]
        - labels: [B] or [B,1]
        """
        logits_flat = logits.view(-1)
        labels_flat = labels.view(-1).float()

        # sample_weight（可选；旧 Trainer 里来自 self._current_sample_weight）
        sample_weight = self._current_sample_weight
        if sample_weight is not None:
            if sample_weight.numel() != logits_flat.numel():
                sample_weight = None
            else:
                sample_weight = sample_weight.view_as(logits_flat)

        # label smoothing
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

        # pos_weight
        pos_weight_val = float(getattr(self.train_cfg, "pos_weight", 1.0))
        if abs(pos_weight_val - 1.0) < 1e-8:
            pos_weight_tensor = None
        else:
            pos_weight_tensor = logits_flat.new_tensor(pos_weight_val)

        loss_type = str(getattr(self.train_cfg, "loss_type", "bce")).lower()

        if loss_type == "bce":
            return F.binary_cross_entropy_with_logits(
                logits_flat,
                labels_smooth,
                weight=sample_weight,
                pos_weight=pos_weight_tensor,
            )

        elif loss_type == "focal":
            return self._binary_focal_loss(
                logits_flat,
                labels_smooth,
                sample_weight=sample_weight,
                pos_weight=pos_weight_tensor,
            )

        elif loss_type == "bce_focal":
            bce_loss = F.binary_cross_entropy_with_logits(
                logits_flat,
                labels_smooth,
                weight=sample_weight,
                pos_weight=pos_weight_tensor,
            )

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("[WARN] logits contains NaN/Inf in focal loss!")

            focal_loss = self._binary_focal_loss(
                logits_flat,
                labels_smooth,
                sample_weight=sample_weight,
                pos_weight=pos_weight_tensor,
            )

            lambda_focal = float(getattr(self.train_cfg, "focal_lambda", 1.0))
            lambda_bce = float(getattr(self.train_cfg, "bce_lambda", 1.0))
            return lambda_bce * bce_loss + lambda_focal * focal_loss

        else:
            raise ValueError(f"Unknown loss_type for binary_classification: {loss_type}")
