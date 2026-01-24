# src/data/pair_batch_contract.py
from __future__ import annotations
from typing import Dict, Any, Optional
import torch

MASK_TRUE_IS_VALID = True  # 约定：mask=True 表示 valid，False 表示 pad

def validate_pair_batch(
    batch: Dict[str, Any],
    *,
    name: str = "batch",
    require_tokens: bool = True,
) -> None:
    for k in ("pair_id", "y_pair", "mask"):
        if k not in batch:
            raise KeyError(f"[{name}] missing key: {k}")

    pair_id = batch["pair_id"]
    y_pair = batch["y_pair"]
    mask = batch["mask"]

    if pair_id.dim() != 1:
        raise RuntimeError(f"[{name}] pair_id must be [B], got {tuple(pair_id.shape)}")
    if y_pair.dim() != 1:
        raise RuntimeError(f"[{name}] y_pair must be [B], got {tuple(y_pair.shape)}")
    if mask.dim() != 2:
        raise RuntimeError(f"[{name}] mask must be [B,K], got {tuple(mask.shape)}")

    B = int(pair_id.shape[0])
    if int(y_pair.shape[0]) != B:
        raise RuntimeError(f"[{name}] y_pair shape mismatch: {tuple(y_pair.shape)} vs B={B}")
    if int(mask.shape[0]) != B:
        raise RuntimeError(f"[{name}] mask shape mismatch: {tuple(mask.shape)} vs B={B}")

    if require_tokens:
        if "tokens" not in batch:
            raise KeyError(f"[{name}] missing key: tokens")
        tokens = batch["tokens"]
        if tokens is None:
            # 允许全空 batch（极少见）；若你希望强制非 None，把这里改成 raise
            return
        if tokens.dim() != 3:
            raise RuntimeError(f"[{name}] tokens must be [B,K,D], got {tuple(tokens.shape)}")
        if int(tokens.shape[0]) != B or int(tokens.shape[1]) != int(mask.shape[1]):
            raise RuntimeError(
                f"[{name}] tokens shape mismatch: tokens={tuple(tokens.shape)} mask={tuple(mask.shape)}"
            )
