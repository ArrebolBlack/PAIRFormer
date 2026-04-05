from __future__ import annotations

import bisect
from typing import List, Optional

import numpy as np
import torch

from src.data.teacher_shard_cache import TeacherShardReader, load_teacher_shard_manifest
from src.data.window_shard_cache import WindowShardReader, load_window_shard_manifest


class WindowShardDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        *,
        split: str,
        include_ignore: bool = False,
        max_samples: Optional[int] = None,
        teacher_root: Optional[str] = None,
    ):
        self.root = str(root)
        self.split = str(split)
        self.include_ignore = bool(include_ignore)
        self.manifest = load_window_shard_manifest(root, split=split)
        self.shard_paths: List[str] = [str(x["path"]) for x in self.manifest.shards]
        self.shard_sizes: List[int] = [int(x["num_samples"]) for x in self.manifest.shards]
        self.cum_sizes: List[int] = np.cumsum(self.shard_sizes).tolist() if self.shard_sizes else []
        self.total_size = int(self.cum_sizes[-1]) if self.cum_sizes else 0
        self.max_samples = (None if max_samples is None else int(max_samples))

        self.current_reader: Optional[WindowShardReader] = None
        self.current_shard_idx: int = -1
        self.teacher_root = (None if teacher_root is None else str(teacher_root))
        self.teacher_reader: Optional[TeacherShardReader] = None
        self.teacher_paths: Optional[List[str]] = None
        if self.teacher_root is not None:
            teacher_manifest = load_teacher_shard_manifest(self.teacher_root, split=split)
            self.teacher_paths = [str(x["path"]) for x in teacher_manifest.shards]
            if len(self.teacher_paths) != len(self.shard_paths):
                raise RuntimeError("Teacher shards and window shards count mismatch.")

    def __len__(self) -> int:
        n = self.total_size
        if self.max_samples is not None:
            n = min(n, self.max_samples)
        return int(n)

    def _load_shard(self, shard_idx: int) -> None:
        if self.current_shard_idx == int(shard_idx):
            return
        self.current_reader = WindowShardReader(self.shard_paths[int(shard_idx)])
        if self.teacher_paths is not None:
            self.teacher_reader = TeacherShardReader(self.teacher_paths[int(shard_idx)])
        self.current_shard_idx = int(shard_idx)

    def __getitem__(self, idx: int):
        idx = int(idx)
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        shard_idx = bisect.bisect_right(self.cum_sizes, idx)
        self._load_shard(shard_idx)
        assert self.current_reader is not None
        local_idx = idx if shard_idx == 0 else idx - self.cum_sizes[shard_idx - 1]
        item = self.current_reader.read(local_idx)
        if self.teacher_reader is not None:
            item.update(self.teacher_reader.read(local_idx))
        if self.include_ignore:
            return item
        if bool(item["valid_mask"]):
            return item
        # Rare path: skip ignored sample by probing forward, then backward
        j = idx + 1
        while j < len(self):
            shard_j = bisect.bisect_right(self.cum_sizes, j)
            self._load_shard(shard_j)
            local_j = j if shard_j == 0 else j - self.cum_sizes[shard_j - 1]
            item_j = self.current_reader.read(local_j)
            if bool(item_j["valid_mask"]):
                return item_j
            j += 1
        j = idx - 1
        while j >= 0:
            shard_j = bisect.bisect_right(self.cum_sizes, j)
            self._load_shard(shard_j)
            local_j = j if shard_j == 0 else j - self.cum_sizes[shard_j - 1]
            item_j = self.current_reader.read(local_j)
            if bool(item_j["valid_mask"]):
                return item_j
            j -= 1
        raise RuntimeError("No valid samples found in dataset.")
