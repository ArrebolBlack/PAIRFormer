from __future__ import annotations

from typing import Any, Dict, List

import torch


def selected_pair_collate(
    batch: List[Dict[str, Any]], *, truncate_k: int | None = None
) -> Dict[str, Any]:
    pair_id = torch.tensor([int(x["pair_id"]) for x in batch], dtype=torch.long)
    y_pair = torch.tensor([float(x["y_pair"]) for x in batch], dtype=torch.float32)
    sel_len = torch.tensor([int(x["sel_len"]) for x in batch], dtype=torch.long)

    esa = torch.stack([x["esa"] for x in batch], dim=0)
    pos = torch.stack([x["pos"] for x in batch], dim=0)
    K = int(esa.shape[1])

    # Truncate to first truncate_k tokens (already sorted by selector score)
    if truncate_k is not None and truncate_k < K:
        esa = esa[:, :truncate_k]
        pos = pos[:, :truncate_k]
        sel_len = torch.clamp(sel_len, max=truncate_k)
        K = truncate_k

    mask = torch.arange(K, dtype=torch.long).unsqueeze(0) < sel_len.unsqueeze(1)

    out: Dict[str, Any] = {
        "pair_id": pair_id,
        "y_pair": y_pair,
        "sel_len": sel_len,
        "mask": mask,
        "X": None,
        "esa": esa,
        "pos": pos,
        "inst_emb": None,
        "inst_logit": None,
    }

    if batch[0].get("X", None) is not None:
        out["X"] = torch.stack([x["X"] for x in batch], dim=0)

    if batch[0].get("cheap_logit", None) is not None:
        out["cheap_logit"] = torch.stack([x["cheap_logit"] for x in batch], dim=0)

    if batch[0].get("cheap_emb", None) is not None:
        out["cheap_emb"] = torch.stack([x["cheap_emb"] for x in batch], dim=0)

    if batch[0].get("inst_emb", None) is not None:
        embs = torch.stack([x["inst_emb"] for x in batch], dim=0)
        if truncate_k is not None and embs.shape[1] > K:
            embs = embs[:, :K, :]
        out["inst_emb"] = embs

    if batch[0].get("inst_logit", None) is not None:
        logits = torch.stack([x["inst_logit"] for x in batch], dim=0)
        if truncate_k is not None and logits.shape[1] > K:
            logits = logits[:, :K]
        out["inst_logit"] = logits

    return out
