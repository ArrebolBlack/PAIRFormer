# src/em/selection_runner.py
from __future__ import annotations

from typing import Optional
from pathlib import Path
from dataclasses import dataclass

from src.config.data_config import DataConfig
from src.em.selector_runner import SelectionCacheRunner, SelectionCacheBuildConfig
from src.selectors.selector_module import SelectorModule


def run_selection_cache(
    *,
    data_cfg: DataConfig,
    dataset_cache_root: str,
    em_cache_root: str,
    selector: SelectorModule,
    kmax: int,
    epoch: int,
    splits: list[str],
    overwrite: bool = True,
    skip_if_ready: bool = False,
    sel_version: Optional[str] = None,
    pair_batch_size: int = 64,    

    candidate_pool_size=None,
    candidate_pool_mode="topn",
    candidate_pool_topn_ratio=1.0,
    candidate_pool_seed=2020,
) -> None:
    runner = SelectionCacheRunner(
        data_cfg=data_cfg,
        dataset_cache_root=str(Path(dataset_cache_root)),
        em_cache_root=str(Path(em_cache_root)),
    )
    cfg = SelectionCacheBuildConfig(
        splits=list(splits),
        overwrite=bool(overwrite),
        skip_if_ready=bool(skip_if_ready),
        epoch=int(epoch),
        pair_batch_size=int(pair_batch_size),
        progress_bar=True,
        candidate_pool_size=candidate_pool_size,
        candidate_pool_mode=candidate_pool_mode,
        candidate_pool_topn_ratio=candidate_pool_topn_ratio,
        candidate_pool_seed=candidate_pool_seed,
    )
    runner.build(selector=selector, kmax=int(kmax), cfg=cfg, sel_version=sel_version)
