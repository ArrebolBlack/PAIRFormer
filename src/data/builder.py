# src/data/builder.py
"""High-level data construction interface for train/val/test phases.

This module provides unified builders for constructing (dataset, dataloader) pairs.

Responsibilities:
    - DataConfig: Describes data source paths and ESA encoding settings
    - cache: High-performance caching with block-based construction and meta.json
    - dataset/sampler/collate: ChunkedCTSDataset, ChunkAwareBatchSampler, cts_collate_fn
    - builder: Unified high-level function build_dataset_and_loader(...)

Example:
    from src.config.data_config import DataConfig
    from src.data.builder import build_dataset_and_loader

    train_data_cfg = DataConfig.from_file("config/data/siRNA_HUVK_train.json")

    train_ds, train_loader = build_dataset_and_loader(
        data_cfg=train_data_cfg,
        split_idx="train",
        cache_data_path="cache/HUVK_TrainValidationTest",
        batch_size=1024,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )

    # For Hydra/OmegaConf integration:
    train_data_cfg = DataConfig.from_omegaconf(cfg.data_train)
"""

from __future__ import annotations

from typing import List, Tuple

from torch.utils.data import DataLoader

from src.config.data_config import DataConfig
from src.data.cache import get_or_build_blocks, iter_rows_for_split
from src.data.collate import cts_collate_fn
from src.data.dataset import ChunkedCTSDataset
from src.data.sampler import ChunkAwareBatchSampler


def get_set_labels(data_cfg: DataConfig, split_idx: str) -> List[float]:
    """Read all pair labels for the specified split in set_idx order.

    Args:
        data_cfg: Data configuration object.
        split_idx: Split name (e.g., "train", "val", "test0").

    Returns:
        List of labels where the i-th element corresponds to set_idx == i.
        Length equals the number of pairs (rows) in the split.
    """
    labels: List[float] = []
    for local_idx, line in iter_rows_for_split(data_cfg, split_idx):
        toks = line.rstrip("\n").split("\t")
        if len(toks) < 5:
            continue
        try:
            labels.append(float(toks[4]))
        except (ValueError, TypeError):
            labels.append(0.0)
    return labels


def build_dataset_and_loader(
    data_cfg: DataConfig,
    split_idx: str,
    cache_data_path: str,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = False,
) -> Tuple[ChunkedCTSDataset, DataLoader]:
    """Build dataset and dataloader with ChunkedCTSDataset and ChunkAwareBatchSampler.

    Args:
        data_cfg: Data configuration object. Can be created via:
            - DataConfig.from_file("config/data/xxx.json")
            - DataConfig.from_omegaconf(cfg.data_xxx)
        split_idx: Split name ("train", "val", "test0", "test1", etc.).
        cache_data_path: Cache directory root path for meta.json and .pt blocks.
            Example: "cache/HUVK_TrainValidationTest_64_16_20_origin_align"
        batch_size: Number of samples per batch.
        num_workers: Number of DataLoader workers. Default 4.
        pin_memory: Enable pin_memory for faster CPU->GPU transfer. Default True.
        shuffle: Whether to shuffle samples within chunks. Default True.
        drop_last: Whether to drop the last incomplete batch. Default False.

    Returns:
        Tuple of (dataset, loader):
            - dataset: ChunkedCTSDataset with lazy block loading via meta.json
            - loader: DataLoader with ChunkAwareBatchSampler and cts_collate_fn

    Internal flow:
        1. Call get_or_build_blocks() to ensure meta.json and blocks exist
        2. Construct ChunkedCTSDataset from meta.json
        3. Construct ChunkAwareBatchSampler for chunk-aware sampling
        4. Construct DataLoader with persistent_workers and prefetch_factor
    """
    # 1) Ensure cache + meta.json exist (auto-build if missing)
    get_or_build_blocks(data_cfg, split_idx, cache_data_path)

    # 2) Construct Dataset (depends only on meta.json)
    dataset = ChunkedCTSDataset(cache_data_path, data_cfg, split_idx)

    # 3) Construct chunk-aware batch sampler
    sampler = ChunkAwareBatchSampler(
        dataset.cum_sizes,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
    )

    prefetch_factor = 2
    persistent_workers = num_workers > 0

    if num_workers == 0:
        prefetch_factor = None
        persistent_workers = False

    # 4) Construct DataLoader
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=cts_collate_fn,
    )

    return dataset, loader


from src.data.pair_level_dataset import PairLevelDataset, pair_level_collate_fn
from src.data.sampler import PairLevelChunkBatchSampler


def build_pair_level_dataset_and_loader(
    pair_cfg,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = False,
) -> Tuple[PairLevelDataset, DataLoader]:
    """Build pair-level dataset and dataloader with PairLevelChunkBatchSampler.

    Args:
        pair_cfg: Pair configuration object (typically from cfg.data.pair) with:
            - cache_root: str
            - train_split / val_split / test_split: str
            - max_cts_per_pair: int
            - selection_mode: str
            - pos_in_token: bool
            - order_mode: str
        split: Split name ("train", "val", "test", etc.).
        batch_size: Number of samples per batch.
        num_workers: Number of DataLoader workers. Default 4.
        pin_memory: Enable pin_memory for faster CPU->GPU transfer. Default True.
        shuffle: Whether to shuffle samples. Default True.
        drop_last: Whether to drop the last incomplete batch. Default False.

    Returns:
        Tuple of (dataset, loader) for pair-level training.
    """
    if split == "train":
        split_name = pair_cfg.train_split
    elif split == "val":
        split_name = pair_cfg.val_split
    elif hasattr(pair_cfg, "test_split") and split == "test":
        split_name = pair_cfg.test_split
    else:
        # Fallback to split name as-is
        split_name = split

    dataset = PairLevelDataset(
        cache_root=pair_cfg.cache_root,
        split=split_name,
        max_cts_per_pair=pair_cfg.max_cts_per_pair,
        selection_mode=pair_cfg.selection_mode,
        pos_in_token=pair_cfg.pos_in_token,
        order_mode=pair_cfg.order_mode,
        random_order_seed=pair_cfg.random_order_seed,
        random_select_seed=pair_cfg.random_select_seed,
    )

    sampler = PairLevelChunkBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=pair_level_collate_fn,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return dataset, loader
