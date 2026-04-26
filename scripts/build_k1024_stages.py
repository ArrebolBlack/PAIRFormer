#!/usr/bin/env python
"""Build selection + instance cache for K=1024.
Assumes cheap cache already exists at em_cache_root.
"""
from __future__ import annotations
import os, sys, json, hashlib, time
from pathlib import Path

# Ensure project root is on path
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from hydra.utils import instantiate

from src.config.data_config import DataConfig
from src.data.dataset import ChunkedCTSDataset
from src.data.builder import get_or_build_blocks
from src.models.registry import build_model
from src.em.selection_runner import run_selection_cache
from src.em.instance_runner import run_instance_cache
from src.utils import set_seeds


def load_ckpt_clean(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    cleaned = {}
    for k, v in sd.items():
        for pref in ("model.", "module.", "net."):
            if k.startswith(pref):
                k = k[len(pref):]
        cleaned[k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  WARN missing keys: {len(missing)}")
    if unexpected:
        print(f"  WARN unexpected keys: {len(unexpected)}")
    model.to(device).eval()


def main():
    # Config
    experiment = "MTI_EM_K1024"
    em_cache_root = "/dev/shm/pairformer_cache_k1024"
    cache_root = str(PROJECT / "cache")
    device = torch.device("cuda:0")
    seed = 2020
    num_workers = 32
    rank = 0
    world_size = 1

    set_seeds(seed)

    # Load Hydra config
    with initialize_config_dir(config_dir=str(PROJECT / "configs"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={experiment}"])

    data_cfg = DataConfig.from_omegaconf(cfg.data)
    run_cfg = cfg.run if hasattr(cfg, "run") and cfg.run else {}
    em_node = cfg.em if hasattr(cfg, "em") else {}

    split_train = str(run_cfg.get("split", "train"))
    split_val = str(run_cfg.get("val_split", "val"))
    splits = [split_train] if split_val == split_train else [split_train, split_val]

    kmax = int(run_cfg.get("kmax", 1024))
    inst_emb_dim = int(em_node.get("inst_emb_dim", 1536))
    inst_version = str(em_node.get("inst_version", "inst_v0"))

    # Build dataset blocks
    for sp in splits:
        get_or_build_blocks(data_cfg, sp, cache_root)

    # Selector setup
    sel_mod_node = (
        OmegaConf.select(cfg, "em.selector_module")
        or OmegaConf.select(cfg, "em.selector")
    )
    if sel_mod_node is None:
        raise KeyError("Missing selector config")
    selector_module = instantiate(sel_mod_node)
    sel_mod_container = OmegaConf.to_container(sel_mod_node, resolve=True)
    sel_hash = hashlib.sha1(json.dumps(sel_mod_container, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:10]
    sel_version_cfg = OmegaConf.select(cfg, "em.sel_version", default=None)
    sel_version = f"sel_{sel_hash}" if sel_version_cfg is None or str(sel_version_cfg).lower() in ("", "none", "null", "auto") else str(sel_version_cfg)
    sel_kmax = int(selector_module.cfg.kmax) if hasattr(selector_module, "cfg") and hasattr(selector_module.cfg, "kmax") else kmax

    sel_node = em_node.get("selection_cache", {})
    sel_pair_batch_size = int(sel_node.get("pair_batch_size", 10240))

    # Verify cheap cache is ready
    print(f"\n{'='*60}")
    print(f"Verifying CHEAP cache exists at {em_cache_root}")
    print(f"{'='*60}")
    for sp in splits:
        cheap_meta_path = Path(em_cache_root) / "em_cache" / sp / "cheap" / "meta.json"
        if not cheap_meta_path.exists():
            raise FileNotFoundError(f"Cheap cache meta missing: {cheap_meta_path}")
        with open(cheap_meta_path) as f:
            d = json.load(f)
        if d.get("state") != "ready":
            raise RuntimeError(f"Cheap cache not ready for {sp}: state={d.get('state')}")
        print(f"  {sp}: ready (total_cts={d.get('total_cts')}, emb_dim={d.get('emb_dim')})")

    # ========================================
    # Stage 2: Selection Cache
    # ========================================
    print(f"\n{'='*60}")
    print(f"[Stage 2/3] Building SELECTION cache (kmax={sel_kmax})")
    print(f"{'='*60}")

    t0 = time.time()
    run_selection_cache(
        data_cfg=data_cfg,
        dataset_cache_root=cache_root,
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
        candidate_pool_seed=int(sel_node.get("candidate_pool_seed", seed)),
        rank=rank,
        world_size=world_size,
    )
    print(f"[Stage 2/3] SELECTION cache done. ({time.time()-t0:.1f}s)")

    # ========================================
    # Stage 3: Instance Cache
    # ========================================
    print(f"\n{'='*60}")
    print(f"[Stage 3/3] Building INSTANCE cache (kmax={sel_kmax}, emb_dim={inst_emb_dim})")
    print(f"{'='*60}")

    inst_cfg = cfg.get("instance_model", cfg.get("cts_model", cfg.get("model_instance")))
    if inst_cfg is None:
        raise KeyError("Missing instance_model config")
    inst_arch = str(inst_cfg.get("arch", inst_cfg.get("name")))
    instance_model = build_model(inst_arch, inst_cfg, data_cfg=data_cfg).to(device)

    inst_ckpt_path = str(em_node.get("instance_ckpt_path", cfg.get("instance_ckpt_path", "")))
    if not Path(inst_ckpt_path).is_absolute():
        inst_ckpt_path = str(PROJECT / inst_ckpt_path)
    if not Path(inst_ckpt_path).exists():
        raise RuntimeError(f"Instance checkpoint not found: {inst_ckpt_path}")
    load_ckpt_clean(instance_model, inst_ckpt_path, device)

    inst_node = em_node.get("instance_cache", {})

    t0 = time.time()
    run_instance_cache(
        data_cfg=data_cfg,
        dataset_cache_root=cache_root,
        em_cache_root=em_cache_root,
        instance_model=instance_model,
        inst_version=inst_version,
        emb_dim=inst_emb_dim,
        epoch=0,
        splits=splits,
        overwrite=True,
        skip_if_ready=False,
        batch_size=int(inst_node.get("batch_size", 10240)),
        num_workers=num_workers,
        use_amp=bool(inst_node.get("use_amp", True)),
        normalize_emb=bool(inst_node.get("normalize_emb", False)),
        rank=rank,
        world_size=world_size,
    )
    print(f"[Stage 3/3] INSTANCE cache done. ({time.time()-t0:.1f}s)")

    print(f"\n{'='*60}")
    print("[build_k1024] ALL DONE.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
