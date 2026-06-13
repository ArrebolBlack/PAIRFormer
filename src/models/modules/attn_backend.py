"""Opt-in efficient attention backend (refactor/2026-06, L5 numerical-optimization layer).

The hand-written attention in transformer.py / set_transformer.py computes
softmax(QK^T/sqrt(d))V manually. On Ampere+ (A100) `torch.nn.functional.
scaled_dot_product_attention` dispatches to FlashAttention / memory-efficient kernels,
which are much faster and lower-memory — especially for the K=512/1024 Set Transformer
runs on MTI.

This is a NUMERICAL optimization (flash uses a different reduction order, so results are
not bit-identical to the manual path). It is therefore OPT-IN and OFF by default, so the
equivalence baseline + golden tests stay bit-identical:

  - enable per-process via env:  PF_EFFICIENT_ATTN=1
  - or programmatically:         from src.models.modules.attn_backend import set_efficient_attention; set_efficient_attention(True)

When off, the original manual path runs unchanged.
"""
from __future__ import annotations

import os

_TRUTHY = ("1", "true", "yes", "on")
_USE_SDPA: bool = os.environ.get("PF_EFFICIENT_ATTN", "0").strip().lower() in _TRUTHY


def set_efficient_attention(enabled: bool) -> None:
    """Globally enable/disable SDPA (FlashAttention) for the hand-written attention modules."""
    global _USE_SDPA
    _USE_SDPA = bool(enabled)


def efficient_attention_enabled() -> bool:
    return _USE_SDPA
