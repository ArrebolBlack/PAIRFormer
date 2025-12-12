# src/data/collate.py
"""
collate.py
======================
本模块提供针对 ChunkedCTSDataset 的标准 collate_fn，用于：

1. 统一 batch 结构为字典 dict：
   {
     "inputs": FloatTensor [B, C, L],
     "labels": FloatTensor [B],
     "set_idx": LongTensor  [B],
   }

2. 在 collate 阶段完成：
   - uint8 -> float32 的转换
   - label / idx 的 squeeze，变成一维向量
   - 留出扩展空间（以后想加入 task_id / sample_weight 等字段也很好加）

这样，Trainer 和模型只需要面向这个统一接口，便于你接入各种模型（TargetNet / Transformer / DiT）。
"""

from typing import List, Tuple, Dict, Any

import torch


def cts_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict[str, Any]:
    """
    针对 ChunkedCTSDataset 的 collate 函数。

    输入
    ----
    batch : list
        每个元素是 Dataset.__getitem__ 返回的四元组 (x, y, set_idx, esa_score)：
        - x: uint8 Tensor,  shape [C, L]
        - y: float Tensor,  shape [1]
        - set_idx: long Tensor, shape [1]
        - esa_score: float Tensor, shape[1]
        - pos: float Tensor, shape[1]
    输出
    ----
    batch_dict : dict
        {
          "inputs": FloatTensor [B, C, L],
          "labels": FloatTensor [B],
          "set_idx": LongTensor  [B],
          "esa_scores" FloatTensor [B],
          "pos" FloatTensor [B],
        }
    """
    # 分别是 length=B 的 tuple
    if len(batch[0]) == 3:
        xs, ys, idxs = zip(*batch)
        esas = None
        pos = None
    elif len(batch[0]) == 4:
        xs, ys, idxs, esas = zip(*batch)
        pos = None
    else:
        xs, ys, idxs, esas, pos = zip(*batch)
    

    # X: [B, C, L]，统一转为 float32，后续模型直接使用
    x = torch.stack(xs, dim=0).float()  # 原来是 uint8 one-hot

    # y: [B, 1] -> [B]
    y = torch.stack(ys, dim=0).view(-1).float()

    # set_idx: [B, 1] -> [B]
    set_idx = torch.stack(idxs, dim=0).view(-1).long()

    batch_dict: Dict[str, Any] = {
        "inputs": x,
        "labels": y,
        "set_idx": set_idx,
    }

    if esas is not None:
        esa_scores = torch.stack(esas, dim=0).view(-1).float()
        batch_dict["esa_scores"] = esa_scores
    
    if pos is not None:
        pos = torch.stack(pos, dim=0).view(-1).float()
        batch_dict["pos"] = pos

    return batch_dict
