from __future__ import annotations

from typing import Any, Dict

import torch

from src.data.selected_pair_cache import SelectedPairCacheReader


class SelectedPairDataset(torch.utils.data.Dataset):
    """
    Pair-level dataset backed by the new compact selected-pair cache.
    """

    def __init__(
        self,
        cache_root: str,
        split: str,
        *,
        cache_type: str = "selected_raw",
        max_pairs: int | None = None,
    ):
        self.reader = SelectedPairCacheReader(cache_root, split=split, cache_type=cache_type)
        self.max_pairs = None if max_pairs is None else int(max_pairs)

    def __len__(self) -> int:
        n = len(self.reader)
        if self.max_pairs is not None:
            return min(n, self.max_pairs)
        return n

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.reader.read_pair(int(idx))
