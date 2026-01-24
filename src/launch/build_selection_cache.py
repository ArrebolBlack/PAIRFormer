# src/launch/build_selection_cache.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from tqdm import tqdm

from src.config.data_config import DataConfig
from src.data.builder import build_dataset_and_loader
from src.data.em_cache import MemmapCacheStore

from src.selectors.st_selector import STSelectorConfig
from src.selectors.selector_module import SelectorModule


# -----------------------------------------------------------------------------
# helpers: path resolve (match build_cheap_cache / train.py philosophy)
# -----------------------------------------------------------------------------

def _resolve_cache_root_like_train_py(cfg: DictConfig, orig_cwd: Path) -> str:
    """
    Strictly follow train.py-like behavior:
      cache_root = run.cache_path if set else paths.cache_root else "cache"
      if relative -> resolve under original_cwd (NOT hydra run dir)
    """
    if "paths" in cfg and cfg.paths is not None:
        default_cache = cfg.paths.get("cache_root", "cache")
    else:
        default_cache = "cache"

    cache_root_cfg = cfg.run.get("cache_path", default_cache)
    cache_root = Path(str(cache_root_cfg))
    if not cache_root.is_absolute():
        cache_root = orig_cwd / cache_root
    return str(cache_root)


def _resolve_path(p: Optional[str], orig_cwd: Path) -> Optional[str]:
    """
    Resolve CLI-provided relative path under original_cwd.
    Support ~ and env var expansion.
    """
    if p is None:
        return None
    pp = Path(os.path.expandvars(os.path.expanduser(str(p))))
    if not pp.is_absolute():
        pp = orig_cwd / pp
    return str(pp)


def _load_cheap_meta(em_cache_root: str, split: str) -> dict:
    meta_path = Path(em_cache_root) / "em_cache" / split / "cheap" / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"[SelectionCache] cheap meta not found: {meta_path}")
    with open(meta_path, "r") as f:
        return json.load(f)


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    orig_cwd = Path(get_original_cwd())

    # dataset cache root (for build_dataset_and_loader)
    cache_root = _resolve_cache_root_like_train_py(cfg, orig_cwd)

    # em_cache root (where cheap/selection memmaps live)
    em_cache_root = _resolve_path(cfg.get("em_cache_root", cache_root), orig_cwd)
    assert em_cache_root is not None

    data_cfg = DataConfig.from_omegaconf(cfg.data)

    splits: List[str] = list(cfg.get("sel_cache_splits", ["train", "val", "test"]))
    overwrite: bool = bool(cfg.get("sel_cache_overwrite", True))  # False
    skip_if_ready: bool = bool(cfg.get("sel_cache_skip_if_ready", True))

    # -------------------------------------------------------------------------
    # Selector config (NEW spec)
    # -------------------------------------------------------------------------
    # NOTE: base_seed 用于 exploration 的可复现噪声；hash_seed 用于 Axis-SimHash dims 固定采样
    base_seed = int(cfg.get("seed", 2020))
    sel_cfg = STSelectorConfig(
        kmax=int(cfg.get("selector_kmax", 128)), # 512

        # Step A
        k1_ratio=float(cfg.get("selector_k1_ratio", 1)), # 0.5
        score_use_sigmoid=bool(cfg.get("selector_score_use_sigmoid", False)),

        # exploration (train only; default off)
        exploration_sigma=float(cfg.get("selector_exploration_sigma", 0.0)),
        base_seed=int(cfg.get("selector_base_seed", base_seed)),

        # Step B / pos binning
        pos_bin_eps=float(cfg.get("selector_pos_bin_eps", 1e-6)),

        # Step C / hash dedup
        use_hash_dedup=bool(cfg.get("selector_use_hash_dedup", True)),
        hash_seed=int(cfg.get("selector_hash_seed", base_seed)),

        # Step D / balanced quota
        score_norm_z=bool(cfg.get("selector_score_norm_z", True)),
        score_norm_eps=float(cfg.get("selector_score_norm_eps", 1e-6)),
        quota_top_t=int(cfg.get("selector_quota_top_t", 8)),
        quota_tau_w=float(cfg.get("selector_quota_tau_w", 1.0)),
    )

    selector = SelectorModule(sel_cfg)
    sel_version = str(cfg.get("sel_version", selector.version))

    print(f"[SelectionCache] dataset_cache_root={cache_root}")
    print(f"[SelectionCache] em_cache_root={em_cache_root}")
    print(f"[SelectionCache] splits={splits} overwrite={overwrite} skip_if_ready={skip_if_ready}")
    print(f"[SelectionCache] sel_version={sel_version}")
    print(f"[SelectionCache] selector_cfg={sel_cfg}")

    # dataset loader returned is UNUSED in this script.
    # Keep it minimal and deterministic to reduce overhead.
    bs = 1
    nw = 0
    pin = False

    # selector control knobs
    epoch = int(cfg.get("selector_epoch", 0))

    for split in splits:
        ds, _ = build_dataset_and_loader(
            data_cfg=data_cfg,
            split_idx=split,
            cache_data_path=str(Path(cache_root)),
            batch_size=bs,
            num_workers=nw,
            pin_memory=pin,
            shuffle=False,
            drop_last=False,
        )

        if getattr(ds, "pair_offsets", None) is None:
            raise RuntimeError(
                f"[SelectionCache] PairIndex not loaded for split={split}. "
                f"Please rebuild dataset cache to generate pair_index_{split}_*.pt"
            )

        num_pairs = int(ds.num_pairs)
        total_cts_ds = int(len(ds))

        # cheap meta -> reuse dataset_hash_key/path_hash
        cheap_meta = _load_cheap_meta(em_cache_root, split)
        if cheap_meta.get("state", None) != "ready":
            raise RuntimeError(
                f"[SelectionCache] cheap cache not ready for split={split}: state={cheap_meta.get('state')}. "
                f"Run cheap step first."
            )
        
        dataset_hash_key = cheap_meta["dataset_hash_key"]
        path_hash = cheap_meta["path_hash"]
        cheap_version_used = cheap_meta["cheap_version"]
        total_cts_cache = int(cheap_meta["total_cts"])
        emb_dim_cache = int(cheap_meta["emb_dim"])

        if total_cts_cache != total_cts_ds:
            raise RuntimeError(
                f"[SelectionCache] total_cts mismatch for split={split}: "
                f"dataset={total_cts_ds} vs cheap_cache={total_cts_cache}"
            )

        # Axis-SimHash requires emb_dim=64 (per your spec)
        if sel_cfg.use_hash_dedup and emb_dim_cache != 64:
            raise RuntimeError(
                f"[SelectionCache] Selector requires cheap_emb dim=64 for Axis-SimHash, got {emb_dim_cache}. "
                f"(split={split}, cheap_version={cheap_version_used})"
            )

        store = MemmapCacheStore(
            cache_root=str(em_cache_root),
            split=str(split),
            path_hash=str(path_hash),
            dataset_hash_key=str(dataset_hash_key),
        )

        # open cheap cache (must exist & match)
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

        # open/create selection cache
        store.create_or_open_selection(
            num_pairs=num_pairs,
            kmax=int(sel_cfg.kmax),
            sel_version=sel_version,
            cheap_version_used=cheap_version_used,
            overwrite=overwrite,
        )

        if (not overwrite) and (store.sel_meta is not None) and (store.sel_meta.state == "ready") and skip_if_ready:
            store.assert_version_consistent()
            print(f"[SelectionCache] SKIP split={split} (already ready).")
            continue

        mode = "train" if split == "train" else "eval"

        pbar = tqdm(range(num_pairs), desc=f"[SelectionCache:{split}]", dynamic_ncols=True)
        with torch.no_grad():
            for pair_id in pbar:
                s, e = ds.get_pair_slice(pair_id)
                n = int(e - s)

                if n <= 0:
                    sel_uids = torch.full((sel_cfg.kmax,), -1, dtype=torch.int32, device="cpu")
                    sel_len = torch.tensor(0, dtype=torch.int16, device="cpu")
                else:
                    # uids are global CTS ids in [s,e)
                    uids = torch.arange(s, e, dtype=torch.long, device="cpu")

                    # cheap outputs (CPU)
                    cheap_logit, cheap_emb, _ = store.read_cheap_slice(s, e)
                    cheap_logit = cheap_logit.view(-1).float().cpu()
                    cheap_emb = cheap_emb.cpu() if cheap_emb is not None else None

                    # meta: pos (only required by selector)
                    meta = ds.batch_gather_by_uid(uids, fields=("pos",))
                    pos = meta["pos"].view(-1).float().cpu()

                    sel_uids, sel_len = selector(
                        uids=uids,
                        pos=pos,
                        cheap_logit=cheap_logit,
                        cheap_emb=cheap_emb,
                        mode=mode,
                        epoch=epoch,
                        pair_id=int(pair_id),
                    )

                    # be explicit about on-disk dtypes
                    sel_uids = sel_uids.to(dtype=torch.int32, device="cpu")
                    sel_len = sel_len.to(dtype=torch.int16, device="cpu")

                store.write_selection(
                    pair_ids=torch.tensor([pair_id], dtype=torch.long, device="cpu"),
                    sel_uids=sel_uids.view(1, -1),
                    sel_len=sel_len.view(1),
                )

        store.flush_selection()
        store.set_selection_ready()
        store.assert_version_consistent()

        out_dir = Path(em_cache_root) / "em_cache" / split / "selection"
        print(f"[SelectionCache] DONE split={split} -> {out_dir}")

    print("[SelectionCache] ALL DONE.")


if __name__ == "__main__":
    main()
