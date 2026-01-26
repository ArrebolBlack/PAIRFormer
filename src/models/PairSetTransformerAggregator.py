# src/models/PairSetTransformerAggregator.py
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.config.data_config import DataConfig
from src.models.registry import register_model
from src.models.modules.set_transformer import (
    SetTransformerConfig, SAB, ISAB, PMA
)


@register_model("PairSetTransformerAggregator")
class PairSetTransformerAggregator(nn.Module):
    """
    PairSetTransformerAggregator
    ============================
    Stage-2: Set Transformer 方式对多个 CTS token 进行集合聚合（Permutation-invariant），输出 pair-level logit。

    输入:
      x:         [B, L, D_in]
      attn_mask: [B, L] (1 valid, 0 pad) 或 [B,1,1,L] 兼容形式
      pos:       忽略（为了接口一致保留；如果 pos 已拼进 token，则属于 x 的特征一部分）

    输出:
      logits: [B]
    """

    def __init__(self, model_cfg: DictConfig, data_cfg: Optional[DataConfig] = None):
        super().__init__()
        p = model_cfg

        # ---- 1) dims ----
        self.in_dim: int = int(p.get("in_dim"))
        if self.in_dim <= 0:
            raise ValueError("PairSetTransformerAggregator requires cfg.model.in_dim > 0")

        d_model: int = int(p.get("d_model", 256))
        n_heads: int = int(p.get("n_heads", 8))
        d_ff: int = int(p.get("dim_ff", 4 * d_model))
        dropout: float = float(p.get("dropout", 0.1))
        ff_activation: str = str(p.get("ff_activation", "gelu")).lower()

        self.n_layers: int = int(p.get("n_layers", 2))

        # Encoder type: "isab" (default) or "sab"
        self.block_type: str = str(p.get("block_type", "isab")).lower()
        self.num_inducing_points: int = int(p.get("num_inducing_points", 16))  # m in ISAB

        # Decoder seeds: k in PMA; for binary classification usually k=1
        self.num_seeds: int = int(p.get("num_seeds", 1))
        self.use_output_sab: bool = bool(p.get("use_output_sab", self.num_seeds > 1))

        # ---- 2) input proj ----
        self.input_proj = nn.Linear(self.in_dim, d_model)

        # ---- 3) set transformer cfg ----
        stcfg = SetTransformerConfig(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            ff_activation=ff_activation,
        )

        # ---- 4) encoder ----
        layers = []
        for _ in range(self.n_layers):
            if self.block_type == "isab":
                layers.append(ISAB(stcfg, m=self.num_inducing_points))
            elif self.block_type == "sab":
                layers.append(SAB(stcfg))
            else:
                raise ValueError("cfg.model.block_type must be one of ['isab','sab']")
        self.encoder = nn.ModuleList(layers)

        # ---- 5) decoder ----
        self.pma = PMA(stcfg, k=self.num_seeds)
        self.out_sab = SAB(stcfg) if self.use_output_sab and self.num_seeds > 1 else None

        # ---- 6) head ----
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    @staticmethod
    def _normalize_mask(attn_mask: Optional[torch.Tensor], L: int, device: torch.device) -> torch.Tensor:
        """
        Return: mask [B, L] bool, True=valid.
        """
        if attn_mask is None:
            return torch.ones(1, L, device=device, dtype=torch.bool)  # will broadcast later if needed

        if attn_mask.dim() == 2:
            # [B, L] with 1/0
            return attn_mask.to(device=device).bool()

        if attn_mask.dim() == 4:
            # [B,1,1,L] -> [B,L]
            return attn_mask[:, 0, 0, :].to(device=device).bool()

        raise ValueError(f"Unsupported attn_mask shape: {tuple(attn_mask.shape)}")

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, Din = x.shape
        if Din != self.in_dim:
            raise ValueError(f"Expected in_dim={self.in_dim}, but got {Din}")

        # mask: [B, L] True=valid
        mask = self._normalize_mask(attn_mask, L, x.device)
        if mask.size(0) == 1 and B > 1:
            mask = mask.expand(B, -1)

        # 防止“全 padding”导致 attention softmax 全 -inf -> NaN
        empty = (mask.sum(dim=1) == 0)
        if empty.any():
            mask = mask.clone()
            mask[empty, 0] = True

        x = x.to(dtype=self.input_proj.weight.dtype)

        # input proj + zero-out pad tokens (important to avoid pad leaking via residuals)
        h = self.input_proj(x)                      # [B,L,D]
        h = h * mask.unsqueeze(-1).float()

        # encoder (permutation equivariant)
        for layer in self.encoder:
            if isinstance(layer, (SAB, ISAB)):
                h = layer(h, mask=mask)
            else:
                h = layer(h)

            # keep padded positions zero
            h = h * mask.unsqueeze(-1).float()

        # decoder pooling (permutation invariant)
        pooled_k = self.pma(h, mask=mask)  # [B,k,D]
        if self.out_sab is not None:
            pooled_k = self.out_sab(pooled_k, mask=None)  # k tokens are all valid

        if self.num_seeds == 1:
            pooled = pooled_k[:, 0, :]  # [B,D]
        else:
            pooled = pooled_k.mean(dim=1)  # [B,D] (simple, stable baseline)

        pooled = self.norm(pooled)
        logits = self.classifier(pooled).squeeze(-1)  # [B]
        return logits
