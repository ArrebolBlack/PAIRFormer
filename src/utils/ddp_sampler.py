# src/utils/ddp_sampler.py
"""
Distributed sampler utilities for PyTorch DDP training.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, DistributedSampler

from .ddp import barrier, is_ddp


def get_ddp_sampler(
    dataset: Any,
    *,
    shuffle: bool = True,
    drop_last: bool = False,
    seed: Optional[int] = None,
) -> Optional[DistributedSampler]:
    """
    Return DistributedSampler if DDP is active, else None.
    """
    if not is_ddp():
        return None

    return DistributedSampler(
        dataset,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )


def set_epoch_for_sampler(loader: DataLoader, epoch: int) -> None:
    """
    Call sampler.set_epoch(epoch) if DistributedSampler is present.
    Must be called at the beginning of each epoch for proper shuffling.
    """
    if loader is None:
        return

    sampler = getattr(loader, "sampler", None)
    if sampler is not None and isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)


def build_ddp_loader(
    dataset: Any,
    *,
    batch_size: int,
    num_workers: int = 4,
    collate_fn: Optional[Any] = None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    drop_last: bool = False,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> DataLoader:
    """
    Build DataLoader with correct sampler for DDP.
    When DDP is active, sampler controls shuffling (shuffle=False to DataLoader).
    """
    sampler = get_ddp_sampler(
        dataset,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )

    use_persistent = persistent_workers and (num_workers > 0)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        drop_last=drop_last,
        # shuffle is mutually exclusive with sampler
        # when sampler is None, we need to pass shuffle
        **({} if sampler is not None else {"shuffle": shuffle}),
    )


def get_effective_batch_size(batch_size: int) -> int:
    """Get effective batch size considering DDP: batch_size * world_size."""
    from .ddp import get_world_size

    return batch_size * get_world_size()


def disable_persistent_workers_if_ddp(persistent_workers: bool, num_workers: int) -> bool:
    """Disable persistent_workers in DDP mode to avoid stale workers after cache refresh."""
    if is_ddp() and num_workers > 0:
        return False
    return persistent_workers


def get_worker_init_fn(rank: int, base_seed: int = 2020) -> Any:
    """Create worker init function with rank-specific seed."""
    import random

    import numpy as np

    def worker_init_fn(worker_id: int):
        worker_seed = base_seed + (rank * 1000) + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return worker_init_fn
