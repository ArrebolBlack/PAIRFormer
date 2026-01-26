# src/launch/build_cache_window.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import torch
import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

from src.config.data_config import DataConfig
from src.data.builder import build_dataset_and_loader
from src.utils import set_seeds


def _resolve_cache_root(cfg: DictConfig) -> str:
    """
    复刻 train.py 的 cache_root 解析逻辑：
    - 优先 run.cache_path
    - 否则 paths.cache_root
    - 否则 "cache"
    相对路径相对于 original_cwd
    """
    orig_cwd = Path(get_original_cwd())

    if "paths" in cfg:
        default_cache = cfg.paths.get("cache_root", "cache")
    else:
        default_cache = "cache"

    cache_root_cfg = cfg.run.get("cache_path", default_cache)
    cache_root = Path(cache_root_cfg)
    if not cache_root.is_absolute():
        cache_root = orig_cwd / cache_root
    return str(cache_root)


def _build_one_split(cfg: DictConfig, data_cfg: DataConfig, split: str, cache_root: str):
    batch_size = int(cfg.run.get("batch_size", 1024))
    num_workers = int(cfg.run.get("num_workers", 0))     # 构建 cache 时建议先用 0，避免额外 DataLoader worker 干扰
    pin_memory = bool(cfg.run.get("pin_memory", True))

    ds, ld = build_dataset_and_loader(
        data_cfg=data_cfg,
        split_idx=split,
        cache_data_path=cache_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=False,
    )

    # 轻量触发一次迭代，确保 DataLoader 侧也没问题（不会完整跑完）
    try:
        next(iter(ld))
    except StopIteration:
        pass

    print(f"[CacheBuild] split='{split}' done. len(ds)={len(ds)}")
    return ds


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # 可复用你的 seed 逻辑
    seed = int(cfg.get("seed", 2020))
    set_seeds(seed)

    # window-level only：只需要 DataConfig + build_dataset_and_loader
    data_cfg = DataConfig.from_omegaconf(cfg.data)

    cache_root = _resolve_cache_root(cfg)
    os.makedirs(cache_root, exist_ok=True)
    print(f"[CacheBuild] cache_root={cache_root}")

    # 默认构建 train/val/test；也可用 run.cache_splits 覆盖
    splits: List[str] = list(cfg.run.get("cache_splits", ["train", "val", "test"]))
    for split in splits:
        _build_one_split(cfg, data_cfg, split, cache_root)

    print("[CacheBuild] All requested splits finished.")


if __name__ == "__main__":
    # 如果你之前遇到多进程卡死（尤其 build cache 内部也会开 worker），spawn 通常更稳
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
