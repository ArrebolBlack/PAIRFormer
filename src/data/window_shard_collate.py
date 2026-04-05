from __future__ import annotations

from typing import Any, Dict, List

import torch


def window_shard_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    inputs = torch.stack([x["inputs"] for x in batch], dim=0)
    labels = torch.stack([x["labels"] for x in batch], dim=0).view(-1)
    set_idx = torch.stack([x["set_idx"] for x in batch], dim=0).view(-1)
    esa = torch.stack([x["esa_scores"] for x in batch], dim=0).view(-1)
    pos = torch.stack([x["pos"] for x in batch], dim=0).view(-1)
    valid_mask = torch.stack([x["valid_mask"] for x in batch], dim=0).view(-1)
    out = {
        "inputs": inputs,
        "labels": labels,
        "set_idx": set_idx,
        "esa_scores": esa,
        "pos": pos,
        "valid_mask": valid_mask,
    }
    if "teacher_logit" in batch[0]:
        out["teacher_logit"] = torch.stack([x["teacher_logit"] for x in batch], dim=0).view(-1)
    if "teacher_feat" in batch[0] and batch[0]["teacher_feat"] is not None:
        out["teacher_feat"] = torch.stack([x["teacher_feat"] for x in batch], dim=0)
    return out
