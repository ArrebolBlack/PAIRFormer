# src/data/pair_dataset_dynamic.py
from __future__ import annotations

from typing import Dict, Any
import torch

from src.data.dataset import ChunkedCTSDataset


class DynamicPairDataset(torch.utils.data.Dataset):
    """
    Dynamic PairDataset (Stage 4)
    - __getitem__ 只返回 pair_id，保持极轻量
    - y_pair 等其他信息由 BatchBuilder/collate_fn 批量读取（更快）
    """

    def __init__(self, cts_ds: ChunkedCTSDataset):
        if cts_ds.pair_offsets is None or cts_ds.num_pairs is None:
            raise RuntimeError(
                "[DynamicPairDataset] ChunkedCTSDataset has no PairIndex. "
                "Please build pair_index_*.pt first."
            )
        self.cts_ds = cts_ds
        self.num_pairs = int(cts_ds.num_pairs)

    def __len__(self) -> int:
        return self.num_pairs

    def __getitem__(self, pair_id: int) -> Dict[str, Any]:
        return {"pair_id": int(pair_id)}
