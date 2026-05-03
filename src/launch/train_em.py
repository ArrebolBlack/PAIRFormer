# src/launch/train_em.py
from __future__ import annotations

import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Iterable, Tuple
from contextlib import contextmanager

import hydra
import torch
import torch.nn as nn
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
import hashlib

from src.config.data_config import DataConfig
from src.data.dataset import ChunkedCTSDataset
from src.data.pair_dataset_dynamic import DynamicPairDataset
from src.data.pair_batch_builder_cpu import PairBatchBuilderCPU, PairBatchBuilderCPUConfig
from src.models.registry import build_model
from src.em.token_provider import TokenProvider, TokenProviderConfig
from src.em.controller import EMPipelineController, EMControllerConfig
from src.em.update_policy import UpdatePolicy, UpdatePolicyConfig

from src.trainer.trainer_em import TrainerEM, TrainerEMConfig
from src.utils import set_seeds
from hydra.utils import instantiate

from src.em.cheap_runner import CheapCacheRunner, CheapCacheBuildConfig
from src.em.selection_runner import run_selection_cache
from src.em.instance_runner import run_instance_cache

import numbers
import numpy as np
from src.evaluator.evaluator import evaluate_with_trainer
from src.data.builder import get_or_build_blocks
from src.launch.train import iter_scalar_metrics, setup_wandb

# DDP imports
from src.utils.ddp import (
    setup_ddp, cleanup_ddp, is_ddp, is_rank0, get_rank, get_world_size,
    print_on_rank0, barrier,
)
from src.utils.ddp_sampler import (
    get_ddp_sampler, set_epoch_for_sampler, disable_persistent_workers_if_ddp,
)


def _apply_sync_batchnorm(model: torch.nn.Module) -> torch.nn.Module:
    """Convert BatchNorm1d layers to SyncBatchNorm for DDP training."""
    return nn.SyncBatchNorm.convert_sync_batchnorm(model)


def _resolve_path(p: Optional[str], orig_cwd: Path) -> Optional[Path]:
    if p is None:
        return None
    pp = Path(os.path.expandvars(os.path.expanduser(str(p))))
    if not pp.is_absolute():
        pp = orig_cwd / pp
    return pp


def _load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r") as f:
        return json.load(f)


def _get_cfg_node(cfg: DictConfig, *cands: str) -> Optional[Any]:
    for k in cands:
        if k in cfg and cfg.get(k) is not None:
            return cfg.get(k)
    return None


def _pick_device(cfg: DictConfig) -> torch.device:
    dev_req = str(cfg.get("device", "cuda"))
    if torch.cuda.is_available() and dev_req != "cpu":
        dev_req = "cuda:0" if dev_req == "cuda" else dev_req
        return torch.device(dev_req)
    return torch.device("cpu")


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # ---- Basic environment ----
    seed = int(cfg.get("seed", 2020))
    set_seeds(seed)

    orig_cwd = Path(get_original_cwd())
    run_dir = Path.cwd()

    # ---- DDP setup ----
    rank, local_rank, world_size = setup_ddp()
    if world_size > 1:
        print_on_rank0(f"[train_em] DDP mode: rank={rank} local_rank={local_rank} world_size={world_size}")
        device = torch.device(f"cuda:{local_rank}")
        set_seeds(seed + rank)  # offset seed per rank for DDP
    else:
        print("[train_em] Single GPU mode")
        device = _pick_device(cfg)

    # ---- ckpt/eval dir ----
    run_cfg = cfg.run if ("run" in cfg and cfg.run is not None) else {}
    ckpt_dir_cfg = run_cfg.get("ckpt_dir", run_cfg.get("ckpt_subdir", "checkpoints"))
    eval_dir_cfg = run_cfg.get("eval_dir", run_cfg.get("eval_subdir", "eval"))
    ckpt_dir = Path(ckpt_dir_cfg)
    eval_dir = Path(eval_dir_cfg)
    if not ckpt_dir.is_absolute():
        ckpt_dir = run_dir / ckpt_dir
    if not eval_dir.is_absolute():
        eval_dir = run_dir / eval_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ---- wandb (rank 0 only) ----
    wandb_run = setup_wandb(cfg)

    # ----------------------------
    # 0) data cfg / dataset
    # ----------------------------
    data_cfg = DataConfig.from_omegaconf(cfg.data)

    split_train = str(run_cfg.get("split", "train"))
    split_val = str(run_cfg.get("val_split", "val"))
    kmax = int(run_cfg.get("kmax", 512))
    batch_size = int(run_cfg.get("batch_size", 256))
    num_workers = int(run_cfg.get("num_workers", 8))
    persistent_workers = bool(run_cfg.get("persistent_workers", True))
    if is_ddp():
        persistent_workers = disable_persistent_workers_if_ddp(persistent_workers, num_workers)

    train_instance_mode = run_cfg.get("train_instance_mode", "cached")
    val_instance_mode = run_cfg.get("val_instance_mode", "cached")

    train_use_inst_cache_default = (train_instance_mode == "cached")
    val_use_inst_cache_default = (val_instance_mode == "cached")

    force_overwrite_bootstrap = bool(run_cfg.get("force_overwrite_bootstrap", True))

    refresh_instance_after_update = bool(
        run_cfg.get("refresh_instance_after_update", (train_use_inst_cache_default or val_use_inst_cache_default))
    )

    print_on_rank0(
        f"[train_em] instance modes: train={train_instance_mode} val={val_instance_mode} | "
        f"force_overwrite_bootstrap={force_overwrite_bootstrap} "
        f"refresh_instance_after_update={refresh_instance_after_update}"
    )

    # cache_root
    default_cache = cfg.get("paths", {}).get("cache_root", "cache")
    cache_root_cfg = run_cfg.get("cache_path", default_cache)
    cache_root = _resolve_path(str(cache_root_cfg), orig_cwd)
    assert cache_root is not None

    em_cache_root_cfg = cfg.get("em_cache_root", str(cache_root))
    em_cache_root = _resolve_path(str(em_cache_root_cfg), orig_cwd)
    assert em_cache_root is not None

    # ---- dataset bootstrap ----
    splits_needed = [split_train] if split_val == split_train else [split_train, split_val]
    for sp in splits_needed:
        get_or_build_blocks(data_cfg, sp, str(cache_root))

    cts_ds_train = ChunkedCTSDataset(str(cache_root), data_cfg, split_train)
    cts_ds_val = ChunkedCTSDataset(str(cache_root), data_cfg, split_val)

    pair_ds_train = DynamicPairDataset(cts_ds_train)
    pair_ds_val = DynamicPairDataset(cts_ds_val)

    _cts_ds_by_split: Dict[str, ChunkedCTSDataset] = {
        split_train: cts_ds_train,
        split_val: cts_ds_val,
    }

    def _get_cts_ds(split: str) -> ChunkedCTSDataset:
        if split in _cts_ds_by_split:
            return _cts_ds_by_split[split]
        get_or_build_blocks(data_cfg, split, str(cache_root))
        ds = ChunkedCTSDataset(str(cache_root), data_cfg, split)
        _cts_ds_by_split[split] = ds
        return ds

    # ----------------------------
    # 2) build models
    # ----------------------------
    inst_cfg = _get_cfg_node(cfg, "instance_model", "cts_model", "model_instance")
    if inst_cfg is None:
        raise KeyError("[train_em] missing instance model config")

    inst_arch = str(inst_cfg.get("arch", inst_cfg.get("name")))
    instance_model = build_model(inst_arch, inst_cfg, data_cfg=data_cfg).to(device)

    if is_ddp():
        instance_model = _apply_sync_batchnorm(instance_model)

    # Load instance pretrained checkpoint
    inst_ckpt = _resolve_path(cfg.get("instance_ckpt_path", None), orig_cwd)
    if inst_ckpt is None:
        raise RuntimeError("[train_em] instance_ckpt is None, do u mean from scratch?")
    if inst_ckpt is not None and inst_ckpt.exists():
        ckpt = torch.load(str(inst_ckpt), map_location="cpu")
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        cleaned = {}
        for k, v in sd.items():
            for pref in ("model.", "module.", "net."):
                if k.startswith(pref):
                    k = k[len(pref):]
            cleaned[k] = v
        missing, unexpected = instance_model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"[train_em] WARN instance missing keys: {len(missing)} (first10): {missing[:10]}")
        if unexpected:
            print(f"[train_em] WARN instance unexpected keys: {len(unexpected)} (first10): {unexpected[:10]}")
        instance_model.to(device)

    agg_cfg = _get_cfg_node(cfg, "model", "agg_model", "pair_model")
    if agg_cfg is None:
        raise KeyError("[train_em] missing aggregator model config")
    agg_arch = str(agg_cfg.get("arch", agg_cfg.get("name")))
    agg_model = build_model(agg_arch, agg_cfg, data_cfg=data_cfg).to(device)

    if is_ddp():
        agg_model = _apply_sync_batchnorm(agg_model)

    # ---------------------------------------------------------
    # 3) UpdatePolicy + Controller + Refresh FNs
    # ---------------------------------------------------------
    em_node = cfg.get("em", {})

    policy_node = em_node.get("policy", None)
    if policy_node is None:
        raise KeyError("[train_em] missing cfg.em.update_policy")
    pol_cfg_dict = OmegaConf.to_container(policy_node, resolve=True)

    policy_train = UpdatePolicy(UpdatePolicyConfig(**pol_cfg_dict))
    policy_eval = UpdatePolicy(UpdatePolicyConfig(
        warmup_epochs=0,
        instance_mode=("cached" if val_use_inst_cache_default else "online"),
        cheap_mode="cached",
    ))

    ctrl_node = em_node.get("controller", None)
    if ctrl_node is None:
        raise KeyError("[train_em] missing cfg.em.controller")
    ctrl_cfg = EMControllerConfig(**OmegaConf.to_container(ctrl_node, resolve=True))

    refresh_enabled = (
        int(pol_cfg_dict.get("refresh_cheap_cache_every_epochs", 0)) > 0
        or int(pol_cfg_dict.get("refresh_selection_cache_every_epochs", 0)) > 0
        or int(pol_cfg_dict.get("refresh_instance_cache_every_epochs", 0)) > 0
    )

    dataset_cache_root = str(cache_root)
    em_cache_root_str = str(em_cache_root)
    refresh_splits = [split_train] if split_val == split_train else [split_train, split_val]

    # =========================================================
    # 3.1 Cheap refresh
    # =========================================================
    cheap_model = None
    cheap_arch_cfg = em_node.get("cheap_model", cfg.get("cheap_model", None))
    if cheap_arch_cfg is not None:
        cheap_arch = str(cheap_arch_cfg.get("arch", cheap_arch_cfg.get("name")))
        cheap_model = build_model(cheap_arch, cheap_arch_cfg, data_cfg=data_cfg).to(device)

        cheap_ckpt = _resolve_path(
            em_node.get("cheap_ckpt_path", cfg.get("cheap_ckpt_path", None)),
            orig_cwd,
        )
        if cheap_ckpt is None:
            raise RuntimeError("[train_em] cheap_ckpt is None, do u mean from scratch?")
        if cheap_ckpt is not None and cheap_ckpt.exists():
            ckpt = torch.load(str(cheap_ckpt), map_location="cpu")
            sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            cleaned = {}
            for k, v in sd.items():
                for pref in ("model.", "module.", "net."):
                    if k.startswith(pref):
                        k = k[len(pref):]
                cleaned[k] = v
            cheap_model.load_state_dict(cleaned, strict=False)
            cheap_model.to(device)

    cheap_version = str(em_node.get("cheap_version", cfg.get("cheap_version", "cheap_v0")))
    cheap_emb_dim = int(em_node.get("cheap_emb_dim", cfg.get("cheap_emb_dim", 64)))

    cheap_cache_node = em_node.get("cheap_cache", {})
    cheap_cache_bs = int(cheap_cache_node.get("batch_size", 256))
    cheap_cache_nw = int(cheap_cache_node.get("num_workers", num_workers))
    cheap_cache_amp = bool(cheap_cache_node.get("amp", (device.type == "cuda")))
    cheap_cache_has_entropy = bool(cheap_cache_node.get("has_entropy", False))

    def cheap_refresh_fn(
        epoch: int,
        *,
        overwrite: bool = True,
        skip_if_ready: bool = False,
        splits: Optional[Iterable[str]] = None,
    ) -> None:
        if cheap_model is None:
            raise RuntimeError("[train_em] cheap_model is None but cheap refresh requested")
        splits_use = list(splits) if splits is not None else list(refresh_splits)
        runner = CheapCacheRunner(
            data_cfg=data_cfg,
            dataset_cache_root=dataset_cache_root,
            em_cache_root=em_cache_root_str,
            device=str(device),
        )
        c = CheapCacheBuildConfig(
            splits=splits_use,
            overwrite=bool(overwrite),
            skip_if_ready=bool(skip_if_ready),
            batch_size=int(cheap_cache_bs),
            num_workers=int(cheap_cache_nw),
            pin_memory=True,
            amp=bool(cheap_cache_amp),
            has_entropy=bool(cheap_cache_has_entropy),
            rank=local_rank,
            world_size=world_size,
        )
        runner.build(
            cheap_model=cheap_model,
            cheap_version=str(cheap_version),
            emb_dim=int(cheap_emb_dim),
            cfg=c,
        )

    # =========================================================
    # 3.2 Selection refresh
    # =========================================================
    def _oc_select(cfg_obj: DictConfig, key: str, default=None):
        try:
            return OmegaConf.select(cfg_obj, key, default=default)
        except Exception:
            return default

    def _stable_cfg_hash(obj: Any) -> str:
        s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

    sel_mod_node = (
        _oc_select(cfg, "em.selector_module")
        or _oc_select(cfg, "em.selector")
        or _oc_select(cfg, "selector_module")
        or _oc_select(cfg, "selector")
    )
    if sel_mod_node is None:
        raise KeyError("[train_em] Missing selector config")

    selector_module = instantiate(sel_mod_node)

    sel_version_cfg = (
        _oc_select(cfg, "em.sel_version", default=None)
        or _oc_select(cfg, "sel_version", default=None)
    )
    sel_mod_container = OmegaConf.to_container(sel_mod_node, resolve=True)
    sel_hash = _stable_cfg_hash(sel_mod_container)

    if sel_version_cfg is None or str(sel_version_cfg).lower() in ("", "none", "null", "auto"):
        sel_version = f"sel_{sel_hash}"
    else:
        sel_version = str(sel_version_cfg)

    sel_node = em_node.get("selection_cache", {})
    sel_pair_batch_size = int(sel_node.get("pair_batch_size", 64))

    if hasattr(selector_module, "cfg") and hasattr(selector_module.cfg, "kmax"):
        sel_kmax = int(selector_module.cfg.kmax)
    else:
        sel_kmax = int(kmax)

    print_on_rank0(f"[train_em] selector: {selector_module.__class__.__name__} sel_version={sel_version} sel_kmax={sel_kmax}")

    def selection_refresh_fn(
        epoch: int,
        *,
        overwrite: bool = True,
        skip_if_ready: bool = False,
        splits: Optional[Iterable[str]] = None,
    ) -> None:
        splits_use = list(splits) if splits is not None else list(refresh_splits)
        run_selection_cache(
            data_cfg=data_cfg,
            dataset_cache_root=dataset_cache_root,
            em_cache_root=em_cache_root_str,
            selector=selector_module,
            kmax=int(sel_kmax),
            epoch=int(epoch),
            splits=splits_use,
            overwrite=bool(overwrite),
            skip_if_ready=bool(skip_if_ready),
            sel_version=str(sel_version),
            pair_batch_size=int(sel_pair_batch_size),
            rank=local_rank,
            world_size=world_size,
        )

    # =========================================================
    # 3.3 Instance refresh
    # =========================================================
    inst_node = em_node.get("instance_cache", {})
    inst_cache_bs = int(inst_node.get("batch_size", 1024))
    inst_cache_nw = int(inst_node.get("num_workers", num_workers))
    inst_cache_use_amp = bool(inst_node.get("use_amp", (device.type == "cuda")))
    inst_cache_norm = bool(inst_node.get("normalize_emb", False))

    inst_emb_dim = int(cfg.get("em", {}).get("inst_emb_dim", 384))
    inst_version = str(cfg.get("em", {}).get("inst_version", "inst_v0"))

    def instance_refresh_fn(
        epoch: int,
        *,
        overwrite: bool = True,
        skip_if_ready: bool = False,
        splits: Optional[Iterable[str]] = None,
    ) -> None:
        splits_use = list(splits) if splits is not None else list(refresh_splits)
        run_instance_cache(
            data_cfg=data_cfg,
            dataset_cache_root=dataset_cache_root,
            em_cache_root=em_cache_root_str,
            instance_model=instance_model,
            inst_version=str(inst_version),
            emb_dim=int(inst_emb_dim),
            epoch=int(epoch),
            splits=splits_use,
            overwrite=bool(overwrite),
            skip_if_ready=bool(skip_if_ready),
            batch_size=int(inst_cache_bs),
            num_workers=int(inst_cache_nw),
            use_amp=bool(inst_cache_use_amp),
            normalize_emb=bool(inst_cache_norm),
            rank=local_rank,
            world_size=world_size,
        )

    def _expected_identity(split: str) -> tuple[str, str]:
        ds = _get_cts_ds(split)
        return str(getattr(ds, "path_hash")), str(getattr(ds, "dataset_hash_key"))

    def _is_stage_compatible(split: str, stage: str) -> bool:
        p = em_cache_root / "em_cache" / split / stage / "meta.json"
        if not p.exists():
            return False
        d = _load_json(p)
        if str(d.get("state", "")) != "ready":
            return False
        exp_path_hash, exp_dataset_hash_key = _expected_identity(split)
        if str(d.get("path_hash", "")) != exp_path_hash:
            return False
        if str(d.get("dataset_hash_key", "")) != exp_dataset_hash_key:
            return False
        return True

    def _load_meta_if_exists(split: str, stage: str) -> Optional[Dict[str, Any]]:
        p = em_cache_root / "em_cache" / split / stage / "meta.json"
        if not p.exists():
            return None
        return _load_json(p)

    def _selection_ready_and_match(split: str) -> bool:
        meta = _load_meta_if_exists(split, "selection")
        if meta is None:
            return False
        if str(meta.get("state", "")) != "ready":
            return False
        if str(meta.get("sel_version", "")) != str(sel_version):
            return False
        if str(meta.get("cheap_version_used", "")) != str(cheap_version):
            return False
        return True

    # =========================================================
    # BOOTSTRAP (build missing caches)
    # DDP: only rank 0 builds caches, then barrier
    # =========================================================
    bootstrap_node = em_node.get("bootstrap", {})
    bootstrap_enabled = bool(bootstrap_node.get("enabled", True))

    bootstrap_overwrite_all = True if force_overwrite_bootstrap else bool(bootstrap_node.get("overwrite_all", True))
    bootstrap_skip_if_ready = False if bootstrap_overwrite_all else bool(bootstrap_node.get("skip_if_ready", True))
    bootstrap_epoch0 = int(bootstrap_node.get("epoch", 0))
    bootstrap_skip_instance = bool(bootstrap_node.get("skip_instance_cache", False))

    if bootstrap_enabled:
        need_cheap = bootstrap_overwrite_all or any(
            not _is_stage_compatible(sp, "cheap") for sp in refresh_splits
        )
        need_sel = bootstrap_overwrite_all or any(
            not _selection_ready_and_match(sp) for sp in refresh_splits
        )
        need_inst = bootstrap_overwrite_all or any(
            not _is_stage_compatible(sp, "instance") for sp in refresh_splits
        )

        if need_cheap:
            need_sel = True
            if not bootstrap_skip_instance:
                need_inst = True
        if need_sel:
            if not bootstrap_skip_instance:
                need_inst = True

        if bootstrap_skip_instance:
            need_inst = False

        if need_cheap or need_sel or need_inst:
            print_on_rank0(
                f"[train_em] BOOTSTRAP: overwrite_all={bootstrap_overwrite_all} "
                f"need_cheap={need_cheap} need_sel={need_sel} need_inst={need_inst} "
                f"splits={refresh_splits}"
            )

        # DDP: only rank 0 builds caches
        if is_rank0():
            if need_cheap:
                cheap_refresh_fn(bootstrap_epoch0, overwrite=bootstrap_overwrite_all, skip_if_ready=bootstrap_skip_if_ready)
            if need_sel:
                selection_refresh_fn(bootstrap_epoch0, overwrite=bootstrap_overwrite_all, skip_if_ready=bootstrap_skip_if_ready)
            if need_inst:
                instance_refresh_fn(bootstrap_epoch0, overwrite=bootstrap_overwrite_all, skip_if_ready=bootstrap_skip_if_ready)

        # DDP: all ranks wait for cache building to complete
        barrier()

    # ----------------------------
    # Read meta
    # ----------------------------
    def read_split_meta(split: str):
        sel_meta_p = em_cache_root / "em_cache" / split / "selection" / "meta.json"
        cheap_meta_p = em_cache_root / "em_cache" / split / "cheap" / "meta.json"
        if (not sel_meta_p.exists()) or (not cheap_meta_p.exists()):
            return None
        return _load_json(sel_meta_p), _load_json(cheap_meta_p)

    train_meta = read_split_meta(split_train)
    if train_meta is None:
        raise FileNotFoundError(f"[train_em] missing meta for split={split_train}")
    sel_meta_train, cheap_meta_train = train_meta

    val_meta = read_split_meta(split_val)
    if val_meta is None:
        print_on_rank0(f"[train_em] WARN val meta not found for split={split_val}, fallback to train meta.")
        sel_meta_val, cheap_meta_val = sel_meta_train, cheap_meta_train
    else:
        sel_meta_val, cheap_meta_val = val_meta

    # ----------------------------
    # 4) DataLoaders
    # ----------------------------
    if refresh_enabled:
        persistent_workers = False
    if num_workers <= 0:
        persistent_workers = False

    use_persistent = persistent_workers and (num_workers > 0)

    def _load_sel_meta(split: str) -> Dict[str, Any]:
        p = em_cache_root / "em_cache" / split / "selection" / "meta.json"
        return _load_json(p)

    def build_train_loader() -> DataLoader:
        meta = _load_sel_meta(split_train)
        cpu_builder = PairBatchBuilderCPU(
            cts_ds=cts_ds_train,
            em_cache_root=em_cache_root,
            split=split_train,
            cfg=PairBatchBuilderCPUConfig(
                kmax=kmax,
                include_pos=bool(run_cfg.get("include_pos", True)),
                include_esa=bool(run_cfg.get("include_esa", True)),
                pin_memory=True,
            ),
            expected_path_hash=str(meta["path_hash"]),
            expected_dataset_hash_key=str(meta["dataset_hash_key"]),
        )
        sampler = get_ddp_sampler(pair_ds_train, shuffle=True, seed=seed)
        return DataLoader(
            pair_ds_train,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=use_persistent,
            collate_fn=cpu_builder,
            drop_last=False,
        )

    def build_val_loader() -> DataLoader:
        meta = _load_sel_meta(split_val)
        cpu_builder = PairBatchBuilderCPU(
            cts_ds=cts_ds_val,
            em_cache_root=em_cache_root,
            split=split_val,
            cfg=PairBatchBuilderCPUConfig(
                kmax=kmax,
                include_pos=bool(run_cfg.get("include_pos", True)),
                include_esa=bool(run_cfg.get("include_esa", True)),
                pin_memory=True,
            ),
            expected_path_hash=str(meta["path_hash"]),
            expected_dataset_hash_key=str(meta["dataset_hash_key"]),
        )
        sampler = get_ddp_sampler(pair_ds_val, shuffle=False, seed=seed)
        return DataLoader(
            pair_ds_val,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=use_persistent,
            collate_fn=cpu_builder,
            drop_last=False,
        )

    def _sync_token_provider_identity(tp: Any, split: str) -> None:
        meta = _load_sel_meta(split)
        if hasattr(tp, "path_hash"):
            tp.path_hash = str(meta["path_hash"])
        if hasattr(tp, "dataset_hash_key"):
            tp.dataset_hash_key = str(meta["dataset_hash_key"])
        if hasattr(tp, "sel_version_used"):
            tp.sel_version_used = str(meta.get("sel_version", getattr(tp, "sel_version_used", "sel_v0")))
        if hasattr(tp, "cheap_version_used"):
            tp.cheap_version_used = str(meta.get("cheap_version_used", getattr(tp, "cheap_version_used", "cheap_v0")))

    def _notify_token_providers_cache_refreshed(plan: Dict[str, bool]) -> None:
        for tp in [token_provider_train, token_provider_val]:
            if hasattr(tp, "on_cache_refreshed"):
                tp.on_cache_refreshed(plan)

    def build_eval_loader_for_split(split: str) -> DataLoader:
        ds = _get_cts_ds(split)
        pair_ds = DynamicPairDataset(ds)
        meta = _load_sel_meta(split)
        cpu_builder = PairBatchBuilderCPU(
            cts_ds=ds,
            em_cache_root=em_cache_root,
            split=split,
            cfg=PairBatchBuilderCPUConfig(
                kmax=kmax,
                include_pos=bool(run_cfg.get("include_pos", True)),
                include_esa=bool(run_cfg.get("include_esa", True)),
                pin_memory=True,
            ),
            expected_path_hash=str(meta["path_hash"]),
            expected_dataset_hash_key=str(meta["dataset_hash_key"]),
        )
        sampler = get_ddp_sampler(pair_ds, shuffle=False, seed=seed)
        return DataLoader(
            pair_ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=use_persistent,
            collate_fn=cpu_builder,
            drop_last=False,
        )

    # Build initial loaders
    train_loader = build_train_loader()
    val_loader = build_val_loader()

    # ----------------------------
    # total_cts sanity checks
    # ----------------------------
    total_cts_train = int(len(cts_ds_train))
    total_cts_val = int(len(cts_ds_val))

    def _assert_total_cts_matches_cheap(split_name: str, expected: int) -> None:
        meta_p = os.path.join(str(em_cache_root), "em_cache", split_name, "cheap", "meta.json")
        if os.path.exists(meta_p):
            with open(meta_p, "r") as f:
                d = json.load(f)
            got = int(d.get("total_cts", -1))
            if got != expected:
                raise RuntimeError(f"[train_em] total_cts mismatch for split={split_name}: dataset={expected} vs cheap_meta={got}")

    _assert_total_cts_matches_cheap(split_train, total_cts_train)
    _assert_total_cts_matches_cheap(split_val, total_cts_val)

    # ----------------------------
    # 5) TokenProvider
    # ----------------------------
    tp_cfg_node = cfg.get("token_provider", cfg.get("em", {}).get("token_provider", None))
    if tp_cfg_node is None:
        raise KeyError("[train_em] missing token_provider config")

    token_provider_cfg = TokenProviderConfig(
        policy=UpdatePolicyConfig(**pol_cfg_dict),
        assemble=tp_cfg_node.assemble,
        use_amp=bool(tp_cfg_node.get("use_amp", False)),
        normalize_tokens=bool(tp_cfg_node.get("normalize_tokens", False)),
        cache_missing=str(tp_cfg_node.get("cache_missing", "error")),
    )

    sel_version_train = str(sel_meta_train.get("sel_version", "sel_v0"))
    cheap_version_used_train = str(sel_meta_train.get("cheap_version_used", cheap_meta_train.get("cheap_version", "cheap_v0")))
    sel_version_val = str(sel_meta_val.get("sel_version", sel_version_train))
    cheap_version_used_val = str(sel_meta_val.get("cheap_version_used", cheap_version_used_train))

    token_provider_train = TokenProvider(
        cfg=token_provider_cfg,
        policy=policy_train,
        instance_model=instance_model,
        device=device,
        em_cache_root=str(em_cache_root),
        split=split_train,
        path_hash=str(sel_meta_train["path_hash"]),
        dataset_hash_key=str(sel_meta_train["dataset_hash_key"]),
        total_cts=total_cts_train,
        inst_emb_dim=inst_emb_dim,
        inst_version=inst_version,
        sel_version_used=sel_version_train,
        cheap_version_used=cheap_version_used_train,
        require_ready=train_use_inst_cache_default,
        num_pairs=int(sel_meta_train.get("num_pairs", 0)),
        inst_kmax=int(sel_meta_train.get("kmax", 0)),
    )

    token_provider_val = TokenProvider(
        cfg=token_provider_cfg,
        policy=policy_eval,
        instance_model=instance_model,
        device=device,
        em_cache_root=str(em_cache_root),
        split=split_val,
        path_hash=str(sel_meta_val["path_hash"]),
        dataset_hash_key=str(sel_meta_val["dataset_hash_key"]),
        total_cts=total_cts_val,
        inst_emb_dim=inst_emb_dim,
        inst_version=inst_version,
        sel_version_used=sel_version_val,
        cheap_version_used=cheap_version_used_val,
        require_ready=val_use_inst_cache_default,
        num_pairs=int(sel_meta_val.get("num_pairs", 0)),
        inst_kmax=int(sel_meta_val.get("kmax", 0)),
    )

    def _read_split_meta_required(split: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        sel_meta_p = em_cache_root / "em_cache" / split / "selection" / "meta.json"
        cheap_meta_p = em_cache_root / "em_cache" / split / "cheap" / "meta.json"
        if (not sel_meta_p.exists()) or (not cheap_meta_p.exists()):
            raise FileNotFoundError(f"[train_em] missing meta for split={split}")
        return _load_json(sel_meta_p), _load_json(cheap_meta_p)

    def build_token_provider_for_split(split: str, *, require_ready: bool = True) -> TokenProvider:
        sel_meta, cheap_meta = _read_split_meta_required(split)
        ds = _get_cts_ds(split)
        total_cts_local = int(len(ds))
        sel_version_used = str(sel_meta.get("sel_version", sel_version_train))
        cheap_version_used = str(sel_meta.get("cheap_version_used", cheap_meta.get("cheap_version", cheap_version)))

        return TokenProvider(
            cfg=token_provider_cfg,
            policy=policy_eval,
            instance_model=instance_model,
            device=device,
            em_cache_root=str(em_cache_root),
            split=split,
            path_hash=str(sel_meta["path_hash"]),
            dataset_hash_key=str(sel_meta["dataset_hash_key"]),
            total_cts=total_cts_local,
            inst_emb_dim=inst_emb_dim,
            inst_version=inst_version,
            sel_version_used=sel_version_used,
            cheap_version_used=cheap_version_used,
            require_ready=bool(require_ready),
            num_pairs=int(sel_meta.get("num_pairs", 0)),
            inst_kmax=int(sel_meta.get("kmax", 0)),
        )

    # ----------------------------
    # 6) TrainerEMConfig
    # ----------------------------
    tr_node = cfg.get("trainer_em", None)
    if tr_node is None:
        raise ValueError("Missing config node: cfg.trainer_em")

    betas = tr_node.get("betas", [0.9, 0.999])
    betas = (float(betas[0]), float(betas[1]))

    tr_cfg = TrainerEMConfig(
        num_epochs=int(tr_node.get("num_epochs", cfg.run.num_epochs if "run" in cfg else 10)),
        log_every=int(tr_node.get("log_every", 50)),
        grad_accum_steps=int(tr_node.get("grad_accum_steps", 1)),
        clip_grad_norm=float(tr_node.get("clip_grad_norm", 0.0)),
        use_amp=bool(tr_node.get("use_amp", False)),
        loss_type=str(tr_node.get("loss_type", "bce")),
        esa_weighting=bool(tr_node.get("esa_weighting", False)),
        focal_alpha=float(tr_node.get("focal_alpha", 0.25)),
        focal_gamma=float(tr_node.get("focal_gamma", 2.0)),
        focal_lambda=float(tr_node.get("focal_lambda", 1.0)),
        label_smoothing=bool(tr_node.get("label_smoothing", False)),
        smooth_neg=float(tr_node.get("smooth_neg", 0.0)),
        smooth_pos=float(tr_node.get("smooth_pos", 1.0)),
        bce_lambda=float(tr_node.get("bce_lambda", 1.0)),
        pos_weight=float(tr_node.get("pos_weight", 1.0)),
        bce_pos_weight=float(tr_node.get("bce_pos_weight", tr_node.get("pos_weight", 1.0))),
        esa_scale=float(tr_node.get("esa_scale", 10.0)),
        esa_lambda_pos=float(tr_node.get("esa_lambda_pos", 1.0)),
        esa_lambda_neg=float(tr_node.get("esa_lambda_neg", 0.5)),
        optimizer=str(tr_node.get("optimizer", "adamw")),
        betas=betas,
        eps=float(tr_node.get("eps", 1e-8)),
        amsgrad=bool(tr_node.get("amsgrad", False)),
        lr_agg=float(tr_node.get("lr_agg", 3e-4)),
        wd_agg=float(tr_node.get("wd_agg", 1e-2)),
        lr_inst=float(tr_node.get("lr_inst", 1e-5)),
        wd_inst=float(tr_node.get("wd_inst", 0.0)),
        scheduler_agg=str(tr_node.get("scheduler_agg", "none")),
        scheduler_inst=str(tr_node.get("scheduler_inst", "none")),
        scheduler_t_max=int(tr_node.get("scheduler_t_max", tr_node.get("num_epochs", 10))),
        scheduler_step_size=int(tr_node.get("scheduler_step_size", 10)),
        scheduler_gamma=float(tr_node.get("scheduler_gamma", 0.1)),
        scheduler_factor=float(tr_node.get("scheduler_factor", 0.2)),
        scheduler_patience=int(tr_node.get("scheduler_patience", 5)),
        monitor=str(tr_node.get("monitor", "loss")),
        greater_is_better=bool(tr_node.get("greater_is_better", False)),
        ema_enabled=bool(tr_node.get("ema_enabled", False)),
        ema_decay=float(tr_node.get("ema_decay", 0.999)),
    )

    trainer = TrainerEM(
        local_rank=local_rank,
        cfg=tr_cfg,
        device=device,
        agg_model=agg_model,
        instance_model=instance_model,
        token_provider=token_provider_train,
        token_provider_val=token_provider_val,
        controller=None,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=wandb_run,
        task_cfg=(cfg.task if "task" in cfg else None),
    )

    # ---- resume ----
    if bool(run_cfg.get("resume", False)) or (run_cfg.get("checkpoint", None) is not None):
        ckpt_path = run_cfg.get("checkpoint", str(ckpt_dir / "best.pt"))
        if ckpt_path is not None and os.path.exists(str(ckpt_path)):
            trainer.load_checkpoint(str(ckpt_path), map_location=device)
            print_on_rank0(f"[train_em] Resumed from checkpoint: {ckpt_path}")
        else:
            print_on_rank0(f"[train_em] No checkpoint found at {ckpt_path}, start from scratch.")

    @contextmanager
    def _swap_token_provider_val(tr: TrainerEM, tp_new: TokenProvider):
        old = tr.token_provider_val
        tr.token_provider_val = tp_new
        try:
            yield
        finally:
            tr.token_provider_val = old

    # ---- Controller ----
    ctrl = EMPipelineController(
        cfg=ctrl_cfg,
        update_policy=policy_train,
        token_providers=[token_provider_train, token_provider_val],
        build_train_loader_fn=build_train_loader,
        on_loader_rebuilt_fn=trainer.set_train_loader,
        cheap_refresh_fn=cheap_refresh_fn,
        selection_refresh_fn=selection_refresh_fn,
        instance_refresh_fn=instance_refresh_fn,
    )

    # ----------------------------
    # 7) Main training loop
    # ----------------------------
    num_epochs = int(tr_cfg.num_epochs)

    warmup_epochs = int(pol_cfg_dict.get("warmup_epochs", 0))
    inst_every = int(pol_cfg_dict.get("instance_update_every_epochs", 0))
    if inst_every < 0:
        inst_every = 0

    val_online_cfg = bool(run_cfg.get("val_online", False))
    if val_online_cfg:
        print_on_rank0("[train_em] WARN run.val_online will be ignored (val always cached).")

    for epoch in range(trainer.state.epoch, num_epochs):
        # DDP: Set sampler epoch for proper shuffling
        set_epoch_for_sampler(train_loader, epoch)
        set_epoch_for_sampler(val_loader, epoch)

        trainer.state.epoch = epoch
        ctrl.on_epoch_begin(epoch)

        # Decide instance-update epoch
        do_inst_epoch = False
        if inst_every > 0 and epoch >= warmup_epochs:
            do_inst_epoch = (((epoch - warmup_epochs) % inst_every) == 0)

        trainer.set_instance_trainable(do_inst_epoch)
        if do_inst_epoch:
            print_on_rank0(f"[TrainEM][epoch={epoch}] Instance-update epoch: UNFREEZE instance_model.")
        else:
            print_on_rank0(f"[TrainEM][epoch={epoch}] Agg-only epoch: FREEZE instance_model.")

        # Force train token plan for the whole epoch
        if do_inst_epoch:
            trainer.set_train_plan_override({"train_instance": True, "use_instance_cache": False})
        else:
            trainer.set_train_plan_override({"train_instance": False, "use_instance_cache": train_use_inst_cache_default})

        # E-step + rebuild train loader
        new_loader = ctrl.maybe_refresh_and_rebuild(epoch=epoch)
        refresh_plan = dict(ctrl.last_refresh_plan)
        if new_loader is not None:
            train_loader = new_loader
            trainer.set_train_loader(train_loader)

        # If selection was refreshed, also rebuild val_loader
        if bool(refresh_plan.get("refresh_selection_cache", False)):
            val_loader = build_val_loader()
            trainer.val_loader = val_loader
            _sync_token_provider_identity(token_provider_train, split_train)
            _sync_token_provider_identity(token_provider_val, split_val)

        # Train
        train_metrics = trainer.train_one_epoch(train_loader)
        train_loss = float(train_metrics.get("loss", 0.0))
        print_on_rank0(f"[Epoch {epoch+1}/{num_epochs}] Train loss = {train_loss:.4f}")

        # Post-update: refresh instance cache if needed
        did_train_inst = float(train_metrics.get("train_inst_pct", 0.0)) > 0.0

        post_update_splits = []
        if train_use_inst_cache_default:
            post_update_splits.append(split_train)
        if (split_val != split_train) and val_use_inst_cache_default:
            post_update_splits.append(split_val)

        if do_inst_epoch and did_train_inst and refresh_instance_after_update and len(post_update_splits) > 0:
            print_on_rank0(f"[TrainEM][epoch={epoch}] Refreshing INSTANCE cache for splits={post_update_splits} after instance update...")
            # DDP: only rank 0 refreshes caches
            if is_rank0():
                instance_refresh_fn(epoch, overwrite=True, skip_if_ready=False, splits=post_update_splits)
            barrier()
            _notify_token_providers_cache_refreshed({"refresh_instance_cache": True})

        # Validate
        val_metrics = trainer.validate_one_epoch(
            val_loader,
            use_ema=True,
            use_instance_cache=val_use_inst_cache_default,
        )

        val_loss = float(val_metrics.get("loss", 0.0))

        # scheduler + best
        trainer.step_schedulers(val_metrics)
        improved = trainer.update_best(val_metrics)

        monitor_name = str(tr_cfg.monitor)
        monitor_value = float(val_metrics.get(monitor_name, val_loss))

        extra = []
        for k in ["f1", "pr_auc", "roc_auc", "accuracy"]:
            if k in val_metrics:
                extra.append(f"{k}={float(val_metrics[k]):.4f}")
        extra_str = " | " + " | ".join(extra) if extra else ""

        print_on_rank0(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"Val loss = {val_loss:.4f} | {monitor_name} = {monitor_value:.4f}{extra_str}"
        )

        # Save checkpoints (rank 0 only)
        trainer.save_checkpoint(str(ckpt_dir / "last.pt"))
        if improved:
            trainer.save_checkpoint(str(ckpt_dir / "best.pt"))
            print_on_rank0(
                f"[Epoch {epoch+1}] Improved {monitor_name}: "
                f"{trainer.state.best_metric:.6f}. Saved best checkpoint."
            )

    # ============================================================
    # Post-training evaluation on val split
    # ============================================================
    val_eval_dir = Path(eval_dir) / "val"
    val_eval_dir.mkdir(parents=True, exist_ok=True)

    _val_pk = {"use_instance_cache": val_use_inst_cache_default}
    val_eval_result = evaluate_with_trainer(
        trainer=trainer,
        loader=val_loader,
        task_cfg=cfg.task,
        logging_cfg=cfg.logging,
        output_dir=str(val_eval_dir),
        set_labels=None,
        aggregate_sets=False,
        tag="val",
        do_threshold_sweep=cfg.eval.do_threshold_sweep,
        sweep_num_thresholds=cfg.eval.sweep_num_thresholds,
        reduction=cfg.run.get("eval_reduction", "max"),
        softmax_temp=cfg.run.get("eval_softmax_temp", 1.0),
        topk=cfg.run.get("eval_topk", 3),
        predict_kwargs=_val_pk,
    )

    best_threshold = val_eval_result.get("best_threshold", None)
    metrics = val_eval_result.get("metrics", {})

    print_on_rank0("\n[TrainEM] Final val metrics:")
    for k, v in iter_scalar_metrics(metrics):
        print_on_rank0(f"  {k}: {v:.4f}")
    if best_threshold is not None:
        print_on_rank0(f"[TrainEM] Best threshold on val = {float(best_threshold):.4f}")

    # ============================================================
    # Optional: evaluate on test splits
    # ============================================================
    if bool(run_cfg.get("eval_test_after_train", False)):
        print_on_rank0("\n[TrainEM] eval_test_after_train=True, start evaluating on test set...")

        eval_with_last = bool(run_cfg.get("eval_test_with_last", True))
        eval_with_best = bool(run_cfg.get("eval_test_with_best", False))

        best_ckpt_path_cfg = run_cfg.get("best_ckpt_path", None)
        default_best_ckpt_path = str(ckpt_dir / "best.pt")
        best_ckpt_path = best_ckpt_path_cfg or default_best_ckpt_path

        test_splits = run_cfg.get("test_splits", ["test"])
        if isinstance(test_splits, str):
            test_splits = [test_splits]
        test_splits = [str(x) for x in list(test_splits)]

        for sp in test_splits:
            get_or_build_blocks(data_cfg, sp, str(cache_root))

        def _refresh_em_caches_for_split(split: str, epoch_for_build: int) -> None:
            cheap_refresh_fn(epoch_for_build, overwrite=True, skip_if_ready=False, splits=[split])
            selection_refresh_fn(epoch_for_build, overwrite=True, skip_if_ready=False, splits=[split])
            instance_refresh_fn(epoch_for_build, overwrite=True, skip_if_ready=False, splits=[split])

        def run_test_eval_for_current_trainer(tag_prefix: str) -> None:
            for split_idx in test_splits:
                print_on_rank0(f"[TrainEM][{tag_prefix}] Building test loader for split='{split_idx}'")

                _refresh_em_caches_for_split(split_idx, epoch_for_build=int(trainer.state.epoch))

                test_loader = build_eval_loader_for_split(split_idx)
                tp_test = build_token_provider_for_split(split_idx, require_ready=True)

                test_root = Path(eval_dir) / "test" / str(split_idx) / tag_prefix
                test_root.mkdir(parents=True, exist_ok=True)

                # (A) fixed threshold = 0.5
                task_fixed = OmegaConf.create(OmegaConf.to_container(cfg.task, resolve=True))
                task_fixed.threshold = 0.5
                out_dir_fixed = test_root / "thr0_5"
                out_dir_fixed.mkdir(parents=True, exist_ok=True)

                print_on_rank0(f"[TrainEM][Test {split_idx}][{tag_prefix}] Eval fixed threshold=0.5")
                with _swap_token_provider_val(trainer, tp_test):
                    res_fixed = evaluate_with_trainer(
                        trainer=trainer,
                        loader=test_loader,
                        task_cfg=task_fixed,
                        logging_cfg=cfg.logging,
                        output_dir=str(out_dir_fixed),
                        set_labels=None,
                        aggregate_sets=False,
                        tag=f"{split_idx}_{tag_prefix}_thr0.5",
                        do_threshold_sweep=False,
                        sweep_num_thresholds=int(cfg.eval.sweep_num_thresholds),
                        reduction=run_cfg.get("test_reduction", run_cfg.get("eval_reduction", "max")),
                        softmax_temp=float(run_cfg.get("test_softmax_temp", run_cfg.get("eval_softmax_temp", 1.0))),
                        topk=int(run_cfg.get("test_topk", run_cfg.get("eval_topk", 3))),
                    )

                # (B) val best_threshold
                res_valbest = None
                if best_threshold is not None:
                    task_valbest = OmegaConf.create(OmegaConf.to_container(cfg.task, resolve=True))
                    task_valbest.threshold = float(best_threshold)
                    out_dir_valbest = test_root / "val_best"
                    out_dir_valbest.mkdir(parents=True, exist_ok=True)
                    print_on_rank0(
                        f"[TrainEM][Test {split_idx}][{tag_prefix}] "
                        f"Eval val best_threshold={float(best_threshold):.4f}"
                    )
                    with _swap_token_provider_val(trainer, tp_test):
                        res_valbest = evaluate_with_trainer(
                            trainer=trainer,
                            loader=test_loader,
                            task_cfg=task_valbest,
                            logging_cfg=cfg.logging,
                            output_dir=str(out_dir_valbest),
                            set_labels=None,
                            aggregate_sets=False,
                            tag=f"{split_idx}_{tag_prefix}_valbest",
                            do_threshold_sweep=False,
                            sweep_num_thresholds=int(cfg.eval.sweep_num_thresholds),
                            reduction=run_cfg.get("test_reduction", run_cfg.get("eval_reduction", "max")),
                            softmax_temp=float(run_cfg.get("test_softmax_temp", run_cfg.get("eval_softmax_temp", 1.0))),
                            topk=int(run_cfg.get("test_topk", run_cfg.get("eval_topk", 3))),
                        )

                # (C) sweep
                task_sweep = OmegaConf.create(OmegaConf.to_container(cfg.task, resolve=True))
                out_dir_sweep = test_root / "sweep"
                out_dir_sweep.mkdir(parents=True, exist_ok=True)
                print_on_rank0(f"[TrainEM][Test {split_idx}][{tag_prefix}] Eval threshold sweep")
                with _swap_token_provider_val(trainer, tp_test):
                    res_sweep = evaluate_with_trainer(
                        trainer=trainer,
                        loader=test_loader,
                        task_cfg=task_sweep,
                        logging_cfg=cfg.logging,
                        output_dir=str(out_dir_sweep),
                        set_labels=None,
                        aggregate_sets=False,
                        tag=f"{split_idx}_{tag_prefix}_sweep",
                        do_threshold_sweep=True,
                        sweep_num_thresholds=int(cfg.eval.sweep_num_thresholds),
                        reduction=run_cfg.get("test_reduction", run_cfg.get("eval_reduction", "max")),
                        softmax_temp=float(run_cfg.get("test_softmax_temp", run_cfg.get("eval_softmax_temp", 1.0))),
                        topk=int(run_cfg.get("test_topk", run_cfg.get("eval_topk", 3))),
                    )

                best_thr_test = res_sweep.get("best_threshold", None)

                metrics_fixed = res_fixed.get("metrics", {})
                print_on_rank0(f"\n[Test {split_idx}][{tag_prefix}] Fixed threshold=0.5 metrics:")
                for k, v in iter_scalar_metrics(metrics_fixed):
                    print_on_rank0(f"  {k}: {v:.4f}")

                metrics_valbest = res_valbest.get("metrics", {}) if res_valbest is not None else None
                if metrics_valbest is not None:
                    print_on_rank0(
                        f"\n[Test {split_idx}][{tag_prefix}] "
                        f"Using val best_threshold={float(best_threshold):.4f} metrics:"
                    )
                    for k, v in iter_scalar_metrics(metrics_valbest):
                        print_on_rank0(f"  {k}: {v:.4f}")

                metrics_sweep = res_sweep.get("metrics_at_best", res_sweep.get("metrics", {}))
                if best_thr_test is not None and "metrics_at_best" in res_sweep:
                    print_on_rank0(
                        f"\n[Test {split_idx}][{tag_prefix}] "
                        f"Sweep metrics (best threshold={float(best_thr_test):.4f}):"
                    )

                for k, v in iter_scalar_metrics(metrics_sweep):
                    print_on_rank0(f"  {k}: {v:.4f}")

                if best_thr_test is not None:
                    print_on_rank0(f"[Test {split_idx}][{tag_prefix}] Best threshold on test = {float(best_thr_test):.4f}")

                if wandb_run is not None:
                    prefix = f"test/{split_idx}/{tag_prefix}"
                    for k, v in iter_scalar_metrics(metrics_fixed):
                        wandb_run.summary[f"{prefix}_thr0.5/{k}"] = v
                    if metrics_valbest is not None:
                        for k, v in iter_scalar_metrics(metrics_valbest):
                            wandb_run.summary[f"{prefix}_valbest/{k}"] = v
                    for k, v in iter_scalar_metrics(metrics_sweep):
                        wandb_run.summary[f"{prefix}_sweep/{k}"] = v
                    if best_thr_test is not None:
                        try:
                            wandb_run.summary[f"{prefix}_sweep/best_threshold"] = float(best_thr_test)
                        except Exception:
                            pass

        if eval_with_last:
            print_on_rank0("\n[TrainEM] Evaluating on test set with LAST checkpoint...")
            run_test_eval_for_current_trainer(tag_prefix="last")

        if eval_with_best:
            if best_ckpt_path is not None and os.path.exists(str(best_ckpt_path)):
                print_on_rank0(f"\n[TrainEM] Loading BEST checkpoint from: {best_ckpt_path}")
                trainer.load_checkpoint(str(best_ckpt_path), map_location=device)
                print_on_rank0("[TrainEM] Evaluating on test set with BEST checkpoint...")
                run_test_eval_for_current_trainer(tag_prefix="best")
            else:
                print_on_rank0(f"\n[TrainEM] Skipped eval_test_with_best: best checkpoint not found at {best_ckpt_path}")

    # wandb finish
    if wandb_run is not None:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    # Cleanup DDP
    if is_ddp():
        cleanup_ddp()


if __name__ == "__main__":
    main()
