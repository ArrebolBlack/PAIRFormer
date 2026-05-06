"""PairCNNAggregator — sorted 1D-CNN baseline for pair-level aggregation.

Sorts tokens by inst_logit (descending), applies dilated Conv1d residual
blocks, then masked global average pooling + classifier.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.config.data_config import DataConfig
from src.models.registry import register_model


@register_model("PairCNNAggregator")
class PairCNNAggregator(nn.Module):
    """Sorted 1D-CNN aggregator.

    Forward signature identical to PairSetTransformerAggregator:
        forward(x [B, K, D_in], attn_mask=[B,K], pos=None) -> [B]
    """

    def __init__(self, model_cfg: DictConfig, data_cfg: Optional[DataConfig] = None):
        super().__init__()
        p = model_cfg

        self.in_dim: int = int(p.get("in_dim"))
        assert self.in_dim > 0

        d_model: int = int(p.get("d_model", 256))
        self.d_model = d_model
        dim_ff: int = int(p.get("dim_ff", d_model * 2))
        self.n_layers: int = int(p.get("n_layers", 4))
        dropout: float = float(p.get("dropout", 0.1))
        ff_activation: str = str(p.get("ff_activation", "gelu")).lower()
        kernel_size: int = int(p.get("kernel_size", 3))

        self.input_proj = nn.Linear(self.in_dim, d_model)

        # Dilated Conv1d residual blocks
        self.conv_blocks = nn.ModuleList()
        for i in range(self.n_layers):
            dilation = 2**i
            padding = dilation * (kernel_size - 1) // 2
            block = nn.ModuleDict(
                {
                    "conv": nn.Conv1d(
                        d_model, d_model, kernel_size, padding=padding, dilation=dilation
                    ),
                    "bn": nn.BatchNorm1d(d_model),
                    "drop": nn.Dropout(dropout),
                }
            )
            self.conv_blocks.append(block)

        self.norm = nn.LayerNorm(d_model)
        act = nn.GELU() if ff_activation == "gelu" else nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            act,
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    @staticmethod
    def _normalize_mask(attn_mask, L, device):
        if attn_mask is None:
            return torch.ones(1, L, device=device, dtype=torch.bool)
        if attn_mask.dim() == 2:
            return attn_mask.to(device=device).bool()
        if attn_mask.dim() == 4:
            return attn_mask[:, 0, 0, :].to(device=device).bool()
        raise ValueError(
            f"[PairCNNAggregator._normalize_mask] "
            f"Unsupported attn_mask shape: {tuple(attn_mask.shape)}"
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, K, Din = x.shape
        assert Din == self.in_dim

        mask = self._normalize_mask(attn_mask, K, x.device)
        if mask.size(0) == 1 and B > 1:
            mask = mask.expand(B, -1)

        # Prevent all-pad samples
        empty = mask.sum(dim=1) == 0
        if empty.any():
            mask = mask.clone()
            mask[empty, 0] = True

        x = x.to(dtype=self.input_proj.weight.dtype)

        # Sort by inst_logit (index -3 in 387-dim: [emb(384), logit(1), esa(1), pos(1)])
        logit_scores = x[:, :, -3]
        sort_scores = logit_scores.masked_fill(~mask, float("-inf"))
        sort_idx = sort_scores.argsort(dim=-1, descending=True)
        x = torch.gather(x, 1, sort_idx.unsqueeze(-1).expand_as(x))
        mask = torch.gather(mask, 1, sort_idx)

        # Input projection
        h = self.input_proj(x)  # [B, K, d_model]
        h = h * mask.unsqueeze(-1).float()

        # Conv1d blocks: [B, K, D] -> [B, D, K] -> conv -> [B, K, D]
        for block in self.conv_blocks:
            residual = h
            h_t = h.transpose(1, 2)  # [B, D, K]
            h_t = block["conv"](h_t)
            h_t = block["bn"](h_t)
            h_t = F.gelu(h_t)
            h_t = block["drop"](h_t)
            h = residual + h_t.transpose(1, 2)
            h = h * mask.unsqueeze(-1).float()

        # Masked global average pooling
        h_sum = (h * mask.unsqueeze(-1).float()).sum(dim=1)  # [B, D]
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1).float()  # [B, 1]
        pooled = h_sum / counts  # [B, D]

        # Classifier
        pooled = self.norm(pooled)
        logits = self.classifier(pooled).squeeze(-1)
        return logits
