# src/launch/build_cache_ddp.py
"""
Multi-GPU cache builder for the EM pipeline.

Builds all three EM caches (cheap → selection → instance) in sequence,
with each stage sharded across all available GPUs.

Usage:
    # Single GPU (backward compatible)
    python -m src.launch.build_cache_ddp experiment=MTI_EM_Pipeline

    # Multi-GPU via torchrun
    torchrun --nproc_per_node=4 -m src.launch.build_cache_ddp experiment=MTI_EM_Pipeline

    # Build to /dev/shm then transfer to vePFS
    torchrun --nproc_per_node=4 -m src.launch.build_cache_ddp \
        experiment=MTI_EM_Pipeline \
        em_cache_root=/dev/shm/pairformer_cache \
        cache_transfer_dest=/vepfs-mlp2/.../cache

Reuses the same experiment config (MTI_EM_Pipeline) as train_em.py.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

from src.config.data_config import DataConfig
from src.data.builder import get_or_build_blocks
from src.data.dataset import ChunkedCTSDataset
from src.em.cheap_runner import CheapCacheBuildConfig, CheapCacheRunner, load_ckpt_into_model
from src.em.instance_runner import run_instance_cache
from src.em.selection_runner import run_selection_cache
from src.em.update_policy import UpdatePolicy, UpdatePolicyConfig
from src.models.registry import build_model
from src.utils import set_seeds
from src.utils.ddp import barrier, cleanup_ddp, is_ddp, is_rank0, print_on_rank0, setup_ddp


def _resolve_path(p: Optional[str], orig_cwd: Path) -> Optional[Path]:
    if p is None:
        return None
    pp = Path(os.path.expandvars(os.path.expanduser(str(p))))
    if not pp.is_absolute():
        pp = orig_cwd / pp
    return pp


def _load_ckpt_clean(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    """Load checkpoint with prefix stripping."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    cleaned = {}
    for k, v in sd.items():
        for pref in ("model.", "module.", "net."):
            if k.startswith(pref):
                k = k[len(pref) :]
        cleaned[k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[build_cache_ddp] WARN missing keys: {len(missing)}")
    if unexpected:
        print(f"[build_cache_ddp] WARN unexpected keys: {len(unexpected)}")
    model.to(device).eval()


def _transfer_cache(src_root: str, dest_root: str) -> None:
    """Copy cache tree from src to dest (e.g., /dev/shm → vePFS)."""
    src = Path(src_root)
    dest = Path(dest_root)
    if not src.exists():
        print(f"[build_cache_ddp] WARN src={src} does not exist, skip transfer.")
        return

    print(f"[build_cache_ddp] Transferring cache: {src} -> {dest}")
    os.makedirs(dest.parent, exist_ok=True)
    ret = os.system(f"cp -r {src_root} {dest_root}")
    if ret != 0:
        raise RuntimeError(f"[build_cache_ddp] Transfer FAILED with exit code {ret}")
    print(f"[build_cache_ddp] Transfer complete.")


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # ---- DDP setup ----
    rank, local_rank, world_size = setup_ddp()
    if world_size > 1:
        print_on_rank0(
            f"[build_cache_ddp] DDP mode: rank={rank} local_rank={local_rank} world_size={world_size}"
        )
        device = torch.device(f"cuda:{local_rank}")
        set_seeds(int(cfg.get("seed", 2020)) + rank)
    else:
        print("[build_cache_ddp] Single GPU mode")
        dev_req = str(cfg.get("device", "cuda"))
        device = torch.device(
            "cuda:0" if dev_req == "cuda" and torch.cuda.is_available() else dev_req
        )

    orig_cwd = Path(get_original_cwd())
    em_node = cfg.get("em", {})

    # ---- Paths ----
    default_cache = cfg.get("paths", {}).get("cache_root", "cache")
    run_cfg = cfg.run if ("run" in cfg and cfg.run is not None) else {}
    cache_root_cfg = run_cfg.get("cache_path", default_cache)
    cache_root = _resolve_path(str(cache_root_cfg), orig_cwd)
    assert cache_root is not None

    em_cache_root_cfg = cfg.get("em_cache_root", str(cache_root))
    em_cache_root = str(_resolve_path(str(em_cache_root_cfg), orig_cwd))

    # Optional: transfer destination (for /dev/shm → vePFS)
    cache_transfer_dest = cfg.get("cache_transfer_dest", None)
    if cache_transfer_dest is not None:
        cache_transfer_dest = str(_resolve_path(str(cache_transfer_dest), orig_cwd))

    print_on_rank0(f"[build_cache_ddp] cache_root={cache_root}")
    print_on_rank0(f"[build_cache_ddp] em_cache_root={em_cache_root}")
    print_on_rank0(f"[build_cache_ddp] cache_transfer_dest={cache_transfer_dest}")

    # ---- Data config ----
    data_cfg = DataConfig.from_omegaconf(cfg.data)
    split_train = str(run_cfg.get("split", "train"))
    split_val = str(run_cfg.get("val_split", "val"))
    splits = [split_train] if split_val == split_train else [split_train, split_val]

    # Build dataset blocks
    for sp in splits:
        get_or_build_blocks(data_cfg, sp, str(cache_root))

    # ---- Config extraction ----
    kmax = int(run_cfg.get("kmax", 512))
    num_workers = int(run_cfg.get("num_workers", 8))
    cheap_version = str(em_node.get("cheap_version", cfg.get("cheap_version", "cheap_v0")))
    cheap_emb_dim = int(em_node.get("cheap_emb_dim", cfg.get("cheap_emb_dim", 64)))
    inst_emb_dim = int(cfg.get("em", {}).get("inst_emb_dim", 384))
    inst_version = str(cfg.get("em", {}).get("inst_version", "inst_v0"))

    # Selector setup
    sel_mod_node = (
        OmegaConf.select(cfg, "em.selector_module")
        or OmegaConf.select(cfg, "em.selector")
        or OmegaConf.select(cfg, "selector_module")
        or OmegaConf.select(cfg, "selector")
    )
    if sel_mod_node is None:
        raise KeyError("[build_cache_ddp] Missing selector config")
    selector_module = instantiate(sel_mod_node)

    import hashlib

    sel_mod_container = OmegaConf.to_container(sel_mod_node, resolve=True)
    sel_hash = hashlib.sha1(
        json.dumps(sel_mod_container, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    sel_version_cfg = OmegaConf.select(cfg, "em.sel_version", default=None) or OmegaConf.select(
        cfg, "sel_version", default=None
    )
    sel_version = (
        f"sel_{sel_hash}"
        if sel_version_cfg is None or str(sel_version_cfg).lower() in ("", "none", "null", "auto")
        else str(sel_version_cfg)
    )
    sel_kmax = (
        int(selector_module.cfg.kmax)
        if hasattr(selector_module, "cfg") and hasattr(selector_module.cfg, "kmax")
        else kmax
    )

    sel_node = em_node.get("selection_cache", {})
    sel_pair_batch_size = int(sel_node.get("pair_batch_size", 10240))

    cheap_cache_node = em_node.get("cheap_cache", {})
    inst_node = em_node.get("instance_cache", {})

    # ========================================
    # Stage 1: Cheap Cache (GPU-bound)
    # ========================================
    print_on_rank0(f"\n{'='*60}")
    print_on_rank0(f"[Stage 1/3] Building CHEAP cache ({world_size} GPUs)")
    print_on_rank0(f"{'='*60}")

    cheap_arch_cfg = em_node.get("cheap_model", cfg.get("cheap_model", None))
    if cheap_arch_cfg is None:
        raise KeyError("[build_cache_ddp] Missing cheap_model config")
    cheap_model = build_model(
        str(cheap_arch_cfg.get("arch", cheap_arch_cfg.get("name"))),
        cheap_arch_cfg,
        data_cfg=data_cfg,
    ).to(device)

    cheap_ckpt = _resolve_path(
        str(em_node.get("cheap_ckpt_path", cfg.get("cheap_ckpt_path", None))),
        orig_cwd,
    )
    if cheap_ckpt is None or not cheap_ckpt.exists():
        raise RuntimeError(f"[build_cache_ddp] cheap_ckpt not found: {cheap_ckpt}")
    _load_ckpt_clean(cheap_model, str(cheap_ckpt), device)

    cheap_runner = CheapCacheRunner(
        data_cfg=data_cfg,
        dataset_cache_root=str(cache_root),
        em_cache_root=em_cache_root,
        device=str(device),
    )
    cheap_runner.build(
        cheap_model=cheap_model,
        cheap_version=cheap_version,
        emb_dim=cheap_emb_dim,
        cfg=CheapCacheBuildConfig(
            splits=splits,
            overwrite=True,
            skip_if_ready=False,
            batch_size=int(cheap_cache_node.get("batch_size", 10240)),
            num_workers=int(cheap_cache_node.get("num_workers", num_workers)),
            pin_memory=True,
            amp=bool(cheap_cache_node.get("amp", device.type == "cuda")),
            has_entropy=bool(cheap_cache_node.get("has_entropy", False)),
            rank=rank,
            world_size=world_size,
        ),
    )
    barrier()
    print_on_rank0("[Stage 1/3] CHEAP cache done.")

    # Free cheap model GPU memory
    del cheap_model
    torch.cuda.empty_cache()

    # ========================================
    # Stage 2: Selection Cache (CPU-bound)
    # ========================================
    print_on_rank0(f"\n{'='*60}")
    print_on_rank0(f"[Stage 2/3] Building SELECTION cache ({world_size} procs)")
    print_on_rank0(f"{'='*60}")

    run_selection_cache(
        data_cfg=data_cfg,
        dataset_cache_root=str(cache_root),
        em_cache_root=em_cache_root,
        selector=selector_module,
        kmax=int(sel_kmax),
        epoch=0,
        splits=splits,
        overwrite=True,
        skip_if_ready=False,
        sel_version=str(sel_version),
        pair_batch_size=int(sel_pair_batch_size),
        candidate_pool_size=sel_node.get("candidate_pool_size", None),
        candidate_pool_mode=str(sel_node.get("candidate_pool_mode", "topn")),
        candidate_pool_topn_ratio=float(sel_node.get("candidate_pool_topn_ratio", 1.0)),
        candidate_pool_seed=int(sel_node.get("candidate_pool_seed", 2020)),
        rank=rank,
        world_size=world_size,
    )
    barrier()
    print_on_rank0("[Stage 2/3] SELECTION cache done.")

    # ========================================
    # Stage 3: Instance Cache (GPU-bound)
    # ========================================
    print_on_rank0(f"\n{'='*60}")
    print_on_rank0(f"[Stage 3/3] Building INSTANCE cache ({world_size} GPUs)")
    print_on_rank0(f"{'='*60}")

    inst_cfg = cfg.get("instance_model", cfg.get("cts_model", cfg.get("model_instance")))
    if inst_cfg is None:
        raise KeyError("[build_cache_ddp] Missing instance_model config")
    inst_arch = str(inst_cfg.get("arch", inst_cfg.get("name")))
    instance_model = build_model(inst_arch, inst_cfg, data_cfg=data_cfg).to(device)

    inst_ckpt = _resolve_path(
        str(em_node.get("instance_ckpt_path", cfg.get("instance_ckpt_path", None))),
        orig_cwd,
    )
    if inst_ckpt is None or not inst_ckpt.exists():
        raise RuntimeError(f"[build_cache_ddp] instance_ckpt not found: {inst_ckpt}")
    _load_ckpt_clean(instance_model, str(inst_ckpt), device)

    run_instance_cache(
        data_cfg=data_cfg,
        dataset_cache_root=str(cache_root),
        em_cache_root=em_cache_root,
        instance_model=instance_model,
        inst_version=inst_version,
        emb_dim=inst_emb_dim,
        epoch=0,
        splits=splits,
        overwrite=True,
        skip_if_ready=False,
        batch_size=int(inst_node.get("batch_size", 10240)),
        num_workers=int(inst_node.get("num_workers", num_workers)),
        use_amp=bool(inst_node.get("use_amp", device.type == "cuda")),
        normalize_emb=bool(inst_node.get("normalize_emb", False)),
        rank=rank,
        world_size=world_size,
    )
    barrier()
    print_on_rank0("[Stage 3/3] INSTANCE cache done.")

    # ========================================
    # Optional: Transfer from /dev/shm to vePFS
    # ========================================
    if cache_transfer_dest is not None and rank == 0:
        print_on_rank0(f"\n[Transfer] Copying cache: {em_cache_root} → {cache_transfer_dest}")
        _transfer_cache(em_cache_root, cache_transfer_dest)
        print_on_rank0("[Transfer] Done.")

    print_on_rank0(f"\n{'='*60}")
    print_on_rank0("[build_cache_ddp] ALL DONE.")
    print_on_rank0(f"{'='*60}")

    if is_ddp():
        cleanup_ddp()


if __name__ == "__main__":
    main()
