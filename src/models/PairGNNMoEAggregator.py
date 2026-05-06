"""PairGNNMoEAggregator — GNN + Soft Mixture-of-Experts aggregator.

Identical to PairGNNAggregator but replaces FFN layers with Soft MoE.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.config.data_config import DataConfig
from src.models.modules.gnn_layers import GATBlock, GlobalAttentionPooling, build_knn_graph
from src.models.registry import register_model


@register_model("PairGNNMoEAggregator")
class PairGNNMoEAggregator(nn.Module):
    """GAT + Soft MoE aggregator for pair-level prediction."""

    def __init__(self, model_cfg: DictConfig, data_cfg: Optional[DataConfig] = None):
        super().__init__()
        p = model_cfg

        self.in_dim: int = int(p.get("in_dim"))
        assert self.in_dim > 0

        d_model: int = int(p.get("d_model", 256))
        dim_ff: int = int(p.get("dim_ff", d_model * 2))
        n_heads: int = int(p.get("n_heads", 4))
        self.n_layers: int = int(p.get("n_layers", 3))
        dropout: float = float(p.get("dropout", 0.1))
        ff_activation: str = str(p.get("ff_activation", "gelu")).lower()
        self.num_neighbors: int = int(p.get("num_neighbors", 8))
        num_experts: int = int(p.get("num_experts", 4))

        self.input_proj = nn.Linear(self.in_dim, d_model)

        self.encoder = nn.ModuleList(
            [
                GATBlock(d_model, n_heads, dim_ff, dropout, ff_activation, num_experts=num_experts)
                for _ in range(self.n_layers)
            ]
        )

        self.pool = GlobalAttentionPooling(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
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
            f"[PairGNNMoEAggregator._normalize_mask] "
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

        empty = mask.sum(dim=1) == 0
        if empty.any():
            mask = mask.clone()
            mask[empty, 0] = True

        x = x.to(dtype=self.input_proj.weight.dtype)

        h = self.input_proj(x)
        h = h * mask.unsqueeze(-1).float()

        h_flat = h.reshape(B * K, -1)
        mask_flat = mask.reshape(B * K)

        edge_index, batch_vec = build_knn_graph(h, mask, self.num_neighbors)

        node_mask = mask_flat
        for layer in self.encoder:
            h_flat = layer(h_flat, edge_index, node_mask)

        pooled = self.pool(h_flat, node_mask, batch_vec, B)

        pooled = self.norm(pooled)
        logits = self.classifier(pooled).squeeze(-1)
        return logits
