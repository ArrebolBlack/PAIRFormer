# src/em/selector_runner.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from src.config.data_config import DataConfig
from src.data.builder import build_dataset_and_loader
from src.data.em_cache import MemmapCacheStore
from src.selectors.selector_module import SelectorModule


def _load_cheap_meta(em_cache_root: Union[str, Path], split: str) -> dict:
    meta_path = Path(str(em_cache_root)) / "em_cache" / split / "cheap" / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"[SelectorRunner] cheap meta not found: {meta_path}")
    with open(meta_path, "r") as f:
        return json.load(f)


@dataclass
class SelectionCacheBuildConfig:
    splits: List[str] = None  # e.g. ["train","val"] or ["test"]
    overwrite: bool = False
    skip_if_ready: bool = True

    # selector control knobs
    epoch: int = 0  # for exploration reproducibility (train only)

    # performance knobs
    pair_batch_size: int = 256  # write_selection in batch; 1 => per-pair
    progress_bar: bool = True

    candidate_pool_size: Optional[int] = None  # n_pool, None=disable
    candidate_pool_mode: str = "topn"  # "topn" or "topn_plus_rand"
    candidate_pool_topn_ratio: float = 1.0  # "topn" or "topn_plus_rand"
    candidate_pool_seed: int = 2020
    candidate_pool_allow_below_kmax: bool = False  # allow n_pool < kmax for Robustness exp

    # DDP sharding
    rank: int = 0
    world_size: int = 1

    def __post_init__(self):
        if self.splits is None:
            self.splits = ["train", "val"]
        # Sanitize: Hydra may pass string "None" instead of Python None
        if self.candidate_pool_size is not None and not isinstance(
            self.candidate_pool_size, (int, float)
        ):
            try:
                self.candidate_pool_size = int(self.candidate_pool_size)
            except (ValueError, TypeError):
                self.candidate_pool_size = None


def _restrict_topn_pool(
    *,
    uids: torch.Tensor,
    pos: torch.Tensor,
    cheap_logit: torch.Tensor,
    cheap_emb: Optional[torch.Tensor],
    n_pool: Optional[int],
    kmax: int,
    mode: str,
    topn_ratio: float,
    seed: int,
    pair_id: int,
    allow_below_kmax: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if n_pool is None:
        return uids, pos, cheap_logit, cheap_emb
    n = int(cheap_logit.numel())
    if n <= 0:
        return uids, pos, cheap_logit, cheap_emb

    # When allow_below_kmax=True (for Robustness experiment), permit n_pool < kmax
    # so the selector picks fewer than K instances from a smaller visible pool.
    if allow_below_kmax:
        n_eff = int(min(n_pool, n))
    else:
        n_eff = int(max(kmax, min(n_pool, n)))
    if n_eff >= n:
        return uids, pos, cheap_logit, cheap_emb

    if mode == "topn":
        idx = torch.topk(cheap_logit, k=n_eff, dim=0, largest=True, sorted=True).indices
    elif mode == "random":
        # Robustness experiment: uniformly sample n_eff candidates from the full pool
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed + pair_id))
        perm = torch.randperm(n, generator=gen)
        idx = perm[:n_eff]
    elif mode == "topn_plus_rand":
        base = int(min(n_eff, round(topn_ratio * n_eff)))
        if not allow_below_kmax:
            base = max(kmax, base)
        base = max(1, min(base, n_eff))
        base_idx = torch.topk(cheap_logit, k=base, dim=0, largest=True, sorted=True).indices

        remain = n_eff - base
        if remain <= 0:
            idx = base_idx
        else:
            # 从 tail 随机补 distractors（可复现：seed + pair_id）
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed + pair_id))

            mask = torch.ones(n, dtype=torch.bool)
            mask[base_idx] = False
            tail = torch.nonzero(mask, as_tuple=False).view(-1)
            if tail.numel() == 0:
                idx = base_idx
            else:
                perm = torch.randperm(tail.numel(), generator=gen)
                pick = tail[perm[: min(remain, tail.numel())]]
                if pick.numel() < remain:
                    pad = pick[-1:].repeat(remain - pick.numel())
                    pick = torch.cat([pick, pad], dim=0)
                idx = torch.cat([base_idx, pick], dim=0)
    else:
        raise ValueError(f"Unknown candidate_pool_mode={mode}")

    uids = uids[idx]
    pos = pos[idx]
    cheap_logit = cheap_logit[idx]
    if cheap_emb is not None:
        cheap_emb = cheap_emb[idx]
    return uids, pos, cheap_logit, cheap_emb


class SelectionCacheRunner:
    """
    Selector step (Stage5 / E-step component):
      - For each split:
          1) load dataset (to access PairIndex + get_pair_slice + batch_gather_by_uid)
          2) open cheap cache (must exist; version from cheap meta)
          3) run selector on each pair_id (CPU-only)
          4) write selection cache (sel_uids/sel_len) and mark ready
    """

    def __init__(
        self,
        *,
        data_cfg: DataConfig,
        dataset_cache_root: Union[str, Path],
        em_cache_root: Union[str, Path],
    ):
        self.data_cfg = data_cfg
        self.dataset_cache_root = str(dataset_cache_root)
        self.em_cache_root = str(em_cache_root)

    @torch.no_grad()
    def build(
        self,
        *,
        selector: SelectorModule,
        kmax: int,
        cfg: SelectionCacheBuildConfig,
        sel_version: Optional[str] = None,
    ) -> None:
        sel_version = str(sel_version) if sel_version is not None else str(selector.version)

        print(f"[SelectorRunner] dataset_cache_root={self.dataset_cache_root}")
        print(f"[SelectorRunner] em_cache_root={self.em_cache_root}")
        print(
            f"[SelectorRunner] splits={cfg.splits} overwrite={cfg.overwrite} skip_if_ready={cfg.skip_if_ready}"
        )
        print(f"[SelectorRunner] sel_version={sel_version} kmax={kmax} epoch={cfg.epoch}")
        print(f"[SelectorRunner] selector_cfg={selector.cfg}")
        print(
            f"[SelectorRunner] candidate_pool_size={cfg.candidate_pool_size} "
            f"mode={cfg.candidate_pool_mode} topn_ratio={cfg.candidate_pool_topn_ratio} seed={cfg.candidate_pool_seed}"
        )

        for split in cfg.splits:
            self._build_one_split(
                split=str(split),
                selector=selector,
                kmax=int(kmax),
                sel_version=sel_version,
                cfg=cfg,
            )

        print("[SelectorRunner] ALL DONE.")

    @torch.no_grad()
    def _build_one_split(
        self,
        *,
        split: str,
        selector: SelectorModule,
        kmax: int,
        sel_version: str,
        cfg: SelectionCacheBuildConfig,
    ) -> None:
        from src.utils.ddp import barrier as ddp_barrier

        rank = int(cfg.rank)
        world_size = int(cfg.world_size)

        # 1) load dataset (only for PairIndex + meta gathering)
        ds, _ = build_dataset_and_loader(
            data_cfg=self.data_cfg,
            split_idx=split,
            cache_data_path=str(Path(self.dataset_cache_root)),
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )

        if getattr(ds, "pair_offsets", None) is None:
            raise RuntimeError(
                f"[SelectorRunner] PairIndex not loaded for split={split}. "
                f"Please rebuild dataset cache to generate pair_index_{split}_*.pt"
            )

        num_pairs = int(ds.num_pairs)
        total_cts_ds = int(len(ds))

        # 2) read cheap meta
        cheap_meta = _load_cheap_meta(self.em_cache_root, split)

        if cheap_meta.get("state", None) != "ready":
            raise RuntimeError(
                f"[SelectorRunner] cheap cache not ready for split={split}: state={cheap_meta.get('state')}. "
                f"Run cheap step first."
            )

        dataset_hash_key = str(cheap_meta["dataset_hash_key"])
        path_hash = str(cheap_meta["path_hash"])
        cheap_version_used = str(cheap_meta["cheap_version"])
        total_cts_cache = int(cheap_meta["total_cts"])
        emb_dim_cache = int(cheap_meta["emb_dim"])

        if total_cts_cache != total_cts_ds:
            raise RuntimeError(
                f"[SelectorRunner] total_cts mismatch for split={split}: "
                f"dataset={total_cts_ds} vs cheap_cache={total_cts_cache}"
            )

        if selector.cfg.use_hash_dedup and emb_dim_cache != 64:
            raise RuntimeError(
                f"[SelectorRunner] Selector requires cheap_emb dim=64 for Axis-SimHash, got {emb_dim_cache}. "
                f"(split={split}, cheap_version={cheap_version_used})"
            )

        # 3) open store: cheap + selection
        store = MemmapCacheStore(
            cache_root=str(self.em_cache_root),
            split=str(split),
            path_hash=str(path_hash),
            dataset_hash_key=str(dataset_hash_key),
        )

        store.create_or_open_cheap(
            total_cts=total_cts_cache,
            emb_dim=emb_dim_cache,
            cheap_version=cheap_version_used,
            dtype_logits=np.float16,
            dtype_emb=np.float16,
            has_entropy=bool(cheap_meta.get("has_entropy", False)),
            overwrite=False,
            require_ready=True,
        )

        # DDP: rank 0 creates selection memmap, others wait then open
        if world_size > 1:
            if rank == 0:
                store.create_or_open_selection(
                    num_pairs=num_pairs,
                    kmax=int(kmax),
                    sel_version=str(sel_version),
                    cheap_version_used=str(cheap_version_used),
                    overwrite=bool(cfg.overwrite),
                )
            ddp_barrier()
            if rank != 0:
                store.create_or_open_selection(
                    num_pairs=num_pairs,
                    kmax=int(kmax),
                    sel_version=str(sel_version),
                    cheap_version_used=str(cheap_version_used),
                    overwrite=False,
                )
        else:
            store.create_or_open_selection(
                num_pairs=num_pairs,
                kmax=int(kmax),
                sel_version=str(sel_version),
                cheap_version_used=str(cheap_version_used),
                overwrite=bool(cfg.overwrite),
            )

        if (
            (not cfg.overwrite)
            and (store.sel_meta is not None)
            and (store.sel_meta.state == "ready")
            and cfg.skip_if_ready
        ):
            store.assert_version_consistent()
            print(f"[SelectorRunner] SKIP split={split} (already ready).")
            return

        # 4) Compute shard range for pair_ids
        pair_start = (rank * num_pairs) // world_size
        pair_end = ((rank + 1) * num_pairs) // world_size
        shard_size = pair_end - pair_start

        if world_size > 1:
            print(
                f"[SelectorRunner:{split}] rank={rank}/{world_size} pair_shard=[{pair_start},{pair_end}) ({shard_size}/{num_pairs})"
            )

        mode = "train" if split == "train" else "eval"
        bs_pair = max(1, int(cfg.pair_batch_size))
        epoch = int(cfg.epoch)

        it = range(pair_start, pair_end)
        if cfg.progress_bar and (world_size == 1 or rank == 0):
            it = tqdm(it, desc=f"[SelectorRunner:{split}:r{rank}]", dynamic_ncols=True)

        pair_ids_buf: List[int] = []
        uids_list: List[torch.Tensor] = []
        pos_list: List[torch.Tensor] = []
        logit_list: List[torch.Tensor] = []
        emb_list: List[Optional[torch.Tensor]] = []

        def _flush_batch() -> None:
            if not pair_ids_buf:
                return
            sel_uids_b, sel_len_b = selector.forward_batch(
                uids_list=uids_list,
                pos_list=pos_list,
                logit_list=logit_list,
                emb_list=emb_list,
                mode=mode,
                epoch=epoch,
                pair_ids=pair_ids_buf,
            )

            start = pair_ids_buf[0]
            store.write_selection_slice(
                start_pair_id=start,
                sel_uids=sel_uids_b,
                sel_len=sel_len_b,
            )

            pair_ids_buf.clear()
            uids_list.clear()
            pos_list.clear()
            logit_list.clear()
            emb_list.clear()

        for pair_id in it:
            s, e = ds.get_pair_slice(int(pair_id))
            n = int(e - s)

            if n <= 0:
                uids = torch.empty((0,), dtype=torch.long, device="cpu")
                pos = torch.empty((0,), dtype=torch.float32, device="cpu")
                cheap_logit = torch.empty((0,), dtype=torch.float32, device="cpu")
                cheap_emb = torch.empty((0, emb_dim_cache), dtype=torch.float16, device="cpu")
            else:
                uids = torch.arange(int(s), int(e), dtype=torch.long, device="cpu")
                cheap_logit, cheap_emb, _ = store.read_cheap_slice(int(s), int(e))
                cheap_logit = cheap_logit.view(-1).to(dtype=torch.float32, device="cpu")
                cheap_emb = cheap_emb.to(device="cpu") if cheap_emb is not None else None
                meta = ds.batch_gather_by_uid(uids, fields=("pos",))
                pos = meta["pos"].view(-1).to(dtype=torch.float32, device="cpu")

                uids, pos, cheap_logit, cheap_emb = _restrict_topn_pool(
                    uids=uids,
                    pos=pos,
                    cheap_logit=cheap_logit,
                    cheap_emb=cheap_emb,
                    n_pool=cfg.candidate_pool_size,
                    kmax=kmax,
                    mode=str(cfg.candidate_pool_mode),
                    topn_ratio=float(cfg.candidate_pool_topn_ratio),
                    seed=int(cfg.candidate_pool_seed),
                    pair_id=int(pair_id),
                    allow_below_kmax=bool(cfg.candidate_pool_allow_below_kmax),
                )

            pair_ids_buf.append(int(pair_id))
            uids_list.append(uids)
            pos_list.append(pos)
            logit_list.append(cheap_logit)
            emb_list.append(cheap_emb)

            if len(pair_ids_buf) >= bs_pair:
                _flush_batch()

        _flush_batch()

        # DDP: every rank flushes, barrier, then rank 0 finalizes
        store.flush_selection()
        if world_size > 1:
            ddp_barrier()

        if rank == 0:
            store.set_selection_ready()
            store.assert_version_consistent()

        out_dir = Path(self.em_cache_root) / "em_cache" / split / "selection"
        print(
            f"[SelectorRunner:{split}] rank={rank} DONE pairs=[{pair_start},{pair_end}) -> {out_dir}"
        )
