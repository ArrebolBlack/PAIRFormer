# src/data/pair_tokens_gpu.py
from __future__ import annotations
from typing import Dict, Any, Optional
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from src.models.extractors import get_embedding_and_logit


def build_pair_tokens_on_gpu(
    batch_cpu: Dict[str, Any],
    *,
    instance_model: torch.nn.Module,
    device: torch.device,
    normalize_tokens: bool = False,
    use_amp: bool = False,
    train_instance: bool = False,
) -> Dict[str, Any]:
    """
    batch_cpu from PairBatchBuilderCPU:
      - X: uint8 [B,K,C,L] (CPU)
      - mask: bool [B,K] (CPU)
      - y_pair, pair_id (CPU)

    Return (GPU):
      - inst_emb  : [B,K,D] or None
      - inst_logit: [B,K]   or None
      - mask/y/pair_id on GPU
    """
    pair_id = batch_cpu["pair_id"].to(device, non_blocking=True)
    y_pair = batch_cpu["y_pair"].to(device, non_blocking=True)
    mask = batch_cpu["mask"].to(device, non_blocking=True)

    X_cpu = batch_cpu.get("X", None)
    if X_cpu is None:
        X_cpu = batch_cpu.get("inputs", None)  # 兼容
    if X_cpu is None:
        return {"pair_id": pair_id, "y_pair": y_pair, "mask": mask,
                "inst_emb": None, "inst_logit": None}

    # move to GPU
    X = X_cpu.to(device, non_blocking=True)  # uint8 [B,K,C,L]
    B, K, C, L = X.shape

    X_flat = X.view(B * K, C, L)
    mask_flat = mask.view(B * K)

    valid = torch.nonzero(mask_flat, as_tuple=False).view(-1)
    if valid.numel() == 0:
        return {
            "pair_id": pair_id, "y_pair": y_pair, "mask": mask,
            "inst_emb": None, "inst_logit": None
        }

    X_valid = X_flat.index_select(0, valid).to(dtype=torch.float32)

    # ---- meta (esa_scores / pos) support for CheapCTSNet ----
    esa_cpu = batch_cpu.get("esa_scores", None)
    if esa_cpu is None:
        esa_cpu = batch_cpu.get("esa_score", None)  # 兼容命名
    if esa_cpu is None:
        esa_cpu = batch_cpu.get("esa", None)

    pos_cpu = batch_cpu.get("pos", None)

    esa_valid = None
    pos_valid = None

    if esa_cpu is not None:
        esa = esa_cpu.to(device, non_blocking=True).view(B * K)
        esa_valid = esa.index_select(0, valid).to(dtype=torch.float32)

    if pos_cpu is not None:
        pos = pos_cpu.to(device, non_blocking=True).view(B * K)
        pos_valid = pos.index_select(0, valid).to(dtype=torch.float32)

    grad_ctx = torch.enable_grad() if train_instance else torch.no_grad()
    amp_ctx = torch.cuda.amp.autocast() if (use_amp and device.type == "cuda") else nullcontext()

    with grad_ctx:
        with amp_ctx:
            feat_valid, logit_valid = get_embedding_and_logit(
                instance_model,
                X_valid,
                esa_scores=esa_valid,
                pos=pos_valid,
            )


    if feat_valid is None or logit_valid is None:
        raise RuntimeError("[build_pair_tokens_on_gpu] extractor returned None.")

    if normalize_tokens and feat_valid is not None:
        feat_valid = F.normalize(feat_valid.float(), dim=-1)

    D = int(feat_valid.shape[1])

    emb_flat = torch.zeros((B * K, D), device=device, dtype=feat_valid.dtype)
    log_flat = torch.zeros((B * K,), device=device, dtype=logit_valid.dtype)

    emb_flat.index_copy_(0, valid, feat_valid)
    log_flat.index_copy_(0, valid, logit_valid)

    inst_emb = emb_flat.view(B, K, D)
    inst_logit = log_flat.view(B, K)

    return {
        "pair_id": pair_id,
        "y_pair": y_pair,
        "mask": mask,
        "inst_emb": inst_emb,
        "inst_logit": inst_logit,
    }
