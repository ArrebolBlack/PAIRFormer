"""PairGNNAggregator — GAT-based graph neural network aggregator.

Builds a k-NN graph over CTS tokens, applies multi-head graph attention
layers, then global attention pooling + classifier.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.config.data_config import DataConfig
from src.models.base_pair_aggregator import BasePairAggregator
from src.models.registry import register_model
from src.models.modules.gnn_layers import (
    build_knn_graph,
    GATBlock,
    GlobalAttentionPooling,
)


@register_model("PairGNNAggregator")
class PairGNNAggregator(BasePairAggregator):
    """GAT-based aggregator for pair-level prediction.

    Forward signature identical to PairSetTransformerAggregator:
        forward(x [B, K, D_in], attn_mask=[B,K], pos=None) -> [B]
    """

    # None => plain FFN inside each GATBlock (original PairGNNAggregator behavior).
    # Subclasses (PairGNNMoEAggregator) override to enable Soft-MoE FFN.
    DEFAULT_NUM_EXPERTS = None

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
        # num_experts absent => plain FFN (no kwarg passed, == original PairGNN).
        # present => Soft-MoE FFN (== original PairGNNMoEAggregator).
        num_experts = p.get("num_experts", self.DEFAULT_NUM_EXPERTS)
        gat_kwargs = {} if num_experts is None else {"num_experts": int(num_experts)}

        self.input_proj = nn.Linear(self.in_dim, d_model)

        self.encoder = nn.ModuleList([
            GATBlock(d_model, n_heads, dim_ff, dropout, ff_activation, **gat_kwargs)
            for _ in range(self.n_layers)
        ])

        self.pool = GlobalAttentionPooling(d_model)
        # norm + classifier (shared head). Construct LAST to preserve weight-init RNG order.
        self._build_head(d_model, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, K, Din = x.shape
        assert Din == self.in_dim

        mask = self._prep_mask(attn_mask, B, K, x.device)

        x = x.to(dtype=self.input_proj.weight.dtype)

        # Input projection
        h = self.input_proj(x)  # [B, K, D]
        h = h * mask.unsqueeze(-1).float()

        # Flatten to [B*K, D] for graph operations
        h_flat = h.reshape(B * K, -1)
        mask_flat = mask.reshape(B * K)

        # Build k-NN graph
        edge_index, batch_vec = build_knn_graph(h, mask, self.num_neighbors)

        # GAT encoder
        node_mask = mask_flat  # [B*K] bool
        for layer in self.encoder:
            h_flat = layer(h_flat, edge_index, node_mask)

        # Global attention pooling -> [B, D]
        pooled = self.pool(h_flat, node_mask, batch_vec, B)

        # Classifier
        pooled = self.norm(pooled)
        logits = self.classifier(pooled).squeeze(-1)
        return logits
