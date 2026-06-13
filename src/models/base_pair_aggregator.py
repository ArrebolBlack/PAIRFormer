"""BasePairAggregator — shared scaffolding for the Pair*Aggregator family.

The Stage-2 aggregators SetTransformer / CNN / GNN(+MoE) share an identical mask
contract and classifier head. This base consolidates the byte-for-byte duplicated:
  - ``_normalize_mask``: accept attn_mask [B,L] or [B,1,1,L] -> [B,L] bool (True=valid).
  - ``_prep_mask``:      normalize + broadcast to B + all-pad-row guard.
  - ``_build_head``:     LayerNorm + (Linear, activation, Dropout, Linear->1) classifier.

Each subclass keeps its own backbone and forward flow; only this shared scaffolding is
hoisted. IMPORTANT for equivalence: ``_build_head`` constructs ``self.norm`` then
``self.classifier`` in that order, so subclasses must call it at the SAME point in
``__init__`` where they originally constructed those modules — otherwise weight-init RNG
order shifts. Behavior is verified bit-identical by tests/test_pair_aggregators_golden.py.

NOTE: PairTransformerAggregator (CLS-token mask handling) and PairMaxPoolCache (no learned
backbone) intentionally do NOT use this base — their mask/forward handling differs.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


class BasePairAggregator(nn.Module):
    @staticmethod
    def _normalize_mask(attn_mask, L, device):
        """attn_mask [B,L] or [B,1,1,L] (1 valid / 0 pad) -> [B,L] bool. None -> all-valid [1,L]."""
        if attn_mask is None:
            return torch.ones(1, L, device=device, dtype=torch.bool)
        if attn_mask.dim() == 2:
            return attn_mask.to(device=device).bool()
        if attn_mask.dim() == 4:
            return attn_mask[:, 0, 0, :].to(device=device).bool()
        raise ValueError(f"Unsupported attn_mask shape: {tuple(attn_mask.shape)}")

    def _prep_mask(self, attn_mask, B: int, L: int, device) -> torch.Tensor:
        """Normalize -> broadcast to B -> guard all-pad rows (set first token valid)."""
        mask = self._normalize_mask(attn_mask, L, device)
        if mask.size(0) == 1 and B > 1:
            mask = mask.expand(B, -1)
        empty = mask.sum(dim=1) == 0
        if empty.any():
            mask = mask.clone()
            mask[empty, 0] = True
        return mask

    def _build_head(self, d_model: int, dropout: float, activation: Optional[nn.Module] = None) -> None:
        """Build self.norm (LayerNorm) + self.classifier (2-layer MLP). Default activation=ReLU."""
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU() if activation is None else activation,
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
