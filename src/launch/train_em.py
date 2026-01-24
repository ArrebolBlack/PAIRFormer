# src/launch/train_em.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import os, json

from src.config.data_config import DataConfig
from src.data.dataset import ChunkedCTSDataset
from src.data.pair_dataset_dynamic import DynamicPairDataset
from src.data.pair_batch_builder_cpu import PairBatchBuilderCPU, PairBatchBuilderCPUConfig
from src.models.registry import build_model
from src.em.token_provider import TokenProvider, TokenProviderConfig
from src.em.controller import EMPipelineController, EMControllerConfig
from src.em.update_policy import UpdatePolicy, UpdatePolicyConfig

from src.trainer.trainer_em import TrainerEM, TrainerEMConfig

from src.utils import set_seeds  # 复用原 train.py

from hydra.utils import instantiate

from src.em.cheap_runner import CheapCacheRunner, CheapCacheBuildConfig
from src.em.selection_runner import run_selection_cache
from src.em.instance_runner import run_instance_cache

import numbers
import numpy as np
from src.evaluator.evaluator import evaluate_with_trainer


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


def setup_wandb(cfg: DictConfig):
    """尽量复用原 train.py 的 wandb 初始化语义。"""
    try:
        import wandb  # type: ignore
    except ImportError:
        return None

    if "logging" not in cfg or "wandb" not in cfg.logging:
        return None
    wandb_cfg = cfg.logging.wandb
    if not bool(wandb_cfg.get("enabled", False)):
        return None

    project = wandb_cfg.get("project", "default_project")
    entity = wandb_cfg.get("entity", None)
    mode = wandb_cfg.get("mode", "online")
    group = wandb_cfg.get("group", None)
    tags = wandb_cfg.get("tags", None)
    run_name = cfg.get("experiment_name", None)

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        group=group,
        tags=tags,
        mode=mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    return run


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # ---- 基本环境 ----
    seed = int(cfg.get("seed", 2020))
    set_seeds(seed)

    device = _pick_device(cfg)
    orig_cwd = Path(get_original_cwd())
    run_dir = Path.cwd()

    # ---- ckpt/eval dir（复用原 train.py 风格）----
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

    # ---- wandb ----
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

    # cache_root
    default_cache = cfg.get("paths", {}).get("cache_root", "cache")
    cache_root_cfg = run_cfg.get("cache_path", default_cache)
    cache_root = _resolve_path(str(cache_root_cfg), orig_cwd)
    assert cache_root is not None

    # em_cache_root（默认同 cache_root）
    em_cache_root_cfg = cfg.get("em_cache_root", str(cache_root))
    em_cache_root = _resolve_path(str(em_cache_root_cfg), orig_cwd)
    assert em_cache_root is not None

    cts_ds_train = ChunkedCTSDataset(str(cache_root), data_cfg, split_train)
    cts_ds_val = ChunkedCTSDataset(str(cache_root), data_cfg, split_val)

    pair_ds_train = DynamicPairDataset(cts_ds_train)
    pair_ds_val = DynamicPairDataset(cts_ds_val)

    # ----------------------------
    # 1) read meta（train/val 各自一份；val 不存在就 fallback train）
    # ----------------------------
    def read_split_meta(split: str):
        sel_meta_p = em_cache_root / "em_cache" / split / "selection" / "meta.json"
        cheap_meta_p = em_cache_root / "em_cache" / split / "cheap" / "meta.json"
        if (not sel_meta_p.exists()) or (not cheap_meta_p.exists()):
            return None
        return _load_json(sel_meta_p), _load_json(cheap_meta_p)

    train_meta = read_split_meta(split_train)
    if train_meta is None:
        raise FileNotFoundError(f"[train_em] missing meta under em_cache_root for split={split_train}")
    sel_meta_train, cheap_meta_train = train_meta

    val_meta = read_split_meta(split_val)
    if val_meta is None:
        # 允许 val 复用 train（但不推荐；至少脚本不崩）
        print(f"[train_em] WARN val meta not found for split={split_val}, fallback to train meta.")
        sel_meta_val, cheap_meta_val = sel_meta_train, cheap_meta_train
    else:
        sel_meta_val, cheap_meta_val = val_meta

    inst_emb_dim = int(cfg.get("inst_emb_dim", cfg.get("em", {}).get("inst_emb_dim", 384)))
    inst_version = str(cfg.get("inst_version", cfg.get("em", {}).get("inst_version", "inst_v0")))

    # ----------------------------
    # 2) build models
    # ----------------------------
    inst_cfg = _get_cfg_node(cfg, "instance_model", "cts_model", "model_instance")
    if inst_cfg is None:
        raise KeyError("[train_em] missing instance model config: cfg.instance_model / cfg.cts_model / cfg.model_instance")

    inst_arch = str(inst_cfg.get("arch", inst_cfg.get("name")))
    instance_model = build_model(inst_arch, inst_cfg, data_cfg=data_cfg).to(device)

    # 可选：加载 instance 预训练 ckpt（仍允许后续训练）
    inst_ckpt = _resolve_path(cfg.get("instance_ckpt_path", None), orig_cwd)
    if inst_ckpt is not None and inst_ckpt.exists():
        ckpt = torch.load(str(inst_ckpt), map_location="cpu")
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        # 兼容 prefix
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
        raise KeyError("[train_em] missing aggregator model config: cfg.model (or cfg.agg_model / cfg.pair_model).")
    agg_arch = str(agg_cfg.get("arch", agg_cfg.get("name")))
    agg_model = build_model(agg_arch, agg_cfg, data_cfg=data_cfg).to(device)

    # ---------------------------------------------------------
    # 3) UpdatePolicy + Controller + Refresh FNs
    #     (放在 token_provider / trainer 创建完成之后)
    # ---------------------------------------------------------
    em_node = cfg.get("em", {})

    policy_node = em_node.get("update_policy", None)
    if policy_node is None:
        raise KeyError("[train_em] missing cfg.em.update_policy for scheme-B.")
    pol_cfg_dict = OmegaConf.to_container(policy_node, resolve=True)
    policy = UpdatePolicy(UpdatePolicyConfig(**pol_cfg_dict))

    ctrl_node = em_node.get("controller", None)
    if ctrl_node is None:
        raise KeyError("[train_em] missing cfg.em.controller for scheme-B.")
    ctrl_cfg = EMControllerConfig(**OmegaConf.to_container(ctrl_node, resolve=True))

    # ---- refresh_enabled -> 影响 DataLoader persistent_workers（你已写过，可保留）----
    refresh_enabled = (
        int(pol_cfg_dict.get("refresh_cheap_cache_every_epochs", 0)) > 0
        or int(pol_cfg_dict.get("refresh_selection_cache_every_epochs", 0)) > 0
        or int(pol_cfg_dict.get("refresh_instance_cache_every_epochs", 0)) > 0
    )

    # ---- runner 路径与 splits ----
    dataset_cache_root = str(cache_root)        # window-level dataset cache root（ChunkedCTSDataset 所在）
    em_cache_root_str  = str(em_cache_root)     # EM memmap cache root

    refresh_splits = [split_train] if split_val == split_train else [split_train, split_val]

    # =========================================================
    # 3.1 Cheap refresh：CheapCacheRunner.build(...)
    # =========================================================
    # cheap model 需要你在 cfg 里提供（否则如果 policy 触发 cheap refresh，这里会报错）
    cheap_model = None
    cheap_arch_cfg = em_node.get("cheap_model", cfg.get("cheap_model", None))
    if cheap_arch_cfg is not None:
        cheap_arch = str(cheap_arch_cfg.get("arch", cheap_arch_cfg.get("name")))
        cheap_model = build_model(cheap_arch, cheap_arch_cfg, data_cfg=data_cfg).to(device)

        cheap_ckpt = _resolve_path(
            em_node.get("cheap_ckpt_path", cfg.get("cheap_ckpt_path", None)),
            orig_cwd,
        )
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

    cheap_version = str(em_node.get("cheap_version", cfg.get("cheap_version", cheap_meta_train.get("cheap_version", "cheap_v0"))))
    cheap_emb_dim = int(em_node.get("cheap_emb_dim", cfg.get("cheap_emb_dim", 64)))

    cheap_cache_node = em_node.get("cheap_cache", {})
    cheap_cache_bs = int(cheap_cache_node.get("batch_size", 256))
    cheap_cache_nw = int(cheap_cache_node.get("num_workers", num_workers))
    cheap_cache_amp = bool(cheap_cache_node.get("amp", (device.type == "cuda")))
    cheap_cache_has_entropy = bool(cheap_cache_node.get("has_entropy", False))

    def cheap_refresh_fn(epoch: int) -> None:
        if cheap_model is None:
            raise RuntimeError(
                "[train_em] UpdatePolicy requests cheap refresh, but cheap_model is None. "
                "Provide cfg.em.cheap_model (and optionally cfg.em.cheap_ckpt_path)."
            )
        runner = CheapCacheRunner(
            data_cfg=data_cfg,
            dataset_cache_root=dataset_cache_root,
            em_cache_root=em_cache_root_str,
            device=str(device),
        )
        c = CheapCacheBuildConfig(
            splits=list(refresh_splits),
            overwrite=True,
            skip_if_ready=False,
            batch_size=int(cheap_cache_bs),
            num_workers=int(cheap_cache_nw),
            pin_memory=True,
            amp=bool(cheap_cache_amp),
            has_entropy=bool(cheap_cache_has_entropy),
        )
        runner.build(
            cheap_model=cheap_model,
            cheap_version=str(cheap_version),
            emb_dim=int(cheap_emb_dim),
            cfg=c,
        )

    # =========================================================
    # 3.2 Selection refresh：run_selection_cache(...)
    # =========================================================
    # selector_module 你必须能构造出来；这里提供一个 hydra instantiate 的默认路径
    selector_module = None
    sel_mod_node = em_node.get("selector_module", em_node.get("selector", cfg.get("selector_module", cfg.get("selector", None))))
    if sel_mod_node is not None:
        selector_module = instantiate(sel_mod_node)

    sel_version = str(em_node.get("sel_version", cfg.get("sel_version", sel_meta_train.get("sel_version", "sel_v0"))))
    sel_node = em_node.get("selection_cache", {})
    sel_pair_batch_size = int(sel_node.get("pair_batch_size", 64))

    def selection_refresh_fn(epoch: int) -> None:
        if selector_module is None:
            raise RuntimeError(
                "[train_em] UpdatePolicy requests selection refresh, but selector_module is None. "
                "Provide cfg.em.selector_module (or cfg.em.selector / cfg.selector)."
            )
        run_selection_cache(
            data_cfg=data_cfg,
            dataset_cache_root=dataset_cache_root,
            em_cache_root=em_cache_root_str,
            selector=selector_module,
            kmax=int(kmax),
            epoch=int(epoch),
            splits=list(refresh_splits),
            overwrite=True,
            skip_if_ready=False,
            sel_version=str(sel_version),
            pair_batch_size=int(sel_pair_batch_size),
        )

    # =========================================================
    # 3.3 Instance refresh：run_instance_cache(...)
    # =========================================================
    inst_node = em_node.get("instance_cache", {})
    inst_cache_bs = int(inst_node.get("batch_size", 1024))
    inst_cache_nw = int(inst_node.get("num_workers", num_workers))
    inst_cache_use_amp = bool(inst_node.get("use_amp", (device.type == "cuda")))
    inst_cache_norm = bool(inst_node.get("normalize_emb", False))

    def instance_refresh_fn(epoch: int) -> None:
        # 注意：instance_runner 会检查 selection/cheap meta ready + version consistency
        run_instance_cache(
            data_cfg=data_cfg,
            dataset_cache_root=dataset_cache_root,
            em_cache_root=em_cache_root_str,
            instance_model=instance_model,
            inst_version=str(inst_version),
            emb_dim=int(inst_emb_dim),
            epoch=int(epoch),
            splits=list(refresh_splits),
            overwrite=True,
            skip_if_ready=False,
            batch_size=int(inst_cache_bs),
            num_workers=int(inst_cache_nw),
            use_amp=bool(inst_cache_use_amp),
            normalize_emb=bool(inst_cache_norm),
        )


    # ----------------------------
    # 4) DataLoaders（E-step 开启时：persistent_workers=False）
    # ----------------------------
    # refresh_enabled 由 policy 的 refresh_* 决定
    persistent_workers = bool(run_cfg.get("persistent_workers", True))
    if refresh_enabled:
        persistent_workers = False
    if num_workers <= 0:
        persistent_workers = False

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
        return DataLoader(
            pair_ds_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
            collate_fn=cpu_builder,
            drop_last=False,
        )

    def build_val_loader() -> DataLoader:
        meta = _load_sel_meta(split_val)  # <-- 关键：每次都 reload 最新 selection meta
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
        return DataLoader(
            pair_ds_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
            collate_fn=cpu_builder,
            drop_last=False,
        )

    def _sync_token_provider_identity(tp: Any, split: str) -> None:
        meta = _load_sel_meta(split)
        # 以下字段名按你 TokenProvider 构造参数推断；用 hasattr 防止字段名不一致直接崩
        if hasattr(tp, "path_hash"):
            tp.path_hash = str(meta["path_hash"])
        if hasattr(tp, "dataset_hash_key"):
            tp.dataset_hash_key = str(meta["dataset_hash_key"])
        if hasattr(tp, "sel_version_used"):
            tp.sel_version_used = str(meta.get("sel_version", getattr(tp, "sel_version_used", "sel_v0")))
        if hasattr(tp, "cheap_version_used"):
            tp.cheap_version_used = str(meta.get("cheap_version_used", getattr(tp, "cheap_version_used", "cheap_v0")))

    def _is_stage_ready(split: str, stage: str) -> bool:
        p = em_cache_root / "em_cache" / split / stage / "meta.json"
        if not p.exists():
            return False
        d = _load_json(p)
        return str(d.get("state", "")) == "ready"


    def _notify_token_providers_cache_refreshed(plan: Dict[str, bool]) -> None:
        for tp in [token_provider_train, token_provider_val]:
            if hasattr(tp, "on_cache_refreshed"):
                tp.on_cache_refreshed(plan)


    train_loader = build_train_loader()
    val_loader = build_val_loader()
    #TODO：我坚持 refresh val，那就要给 controller 增加 build_val_loader_fn 并在 do_sel 时同时 rebuild val_loader（Trainer 也要 set_val_loader）

    # ---------------------------------------------------------
    # total_cts MUST be split-local uid space size (uid in [0, total_cts))
    # i.e., number of CTS instances in that split.
    # ---------------------------------------------------------
    total_cts_train = int(len(cts_ds_train))  # == cts_ds_train.total_size
    total_cts_val   = int(len(cts_ds_val))    # == cts_ds_val.total_size

    # (Optional but recommended) sanity-check vs cheap cache meta, if exists
    def _assert_total_cts_matches_cheap(split_name: str, expected: int) -> None:
        meta_p = os.path.join(str(em_cache_root), "em_cache", split_name, "cheap", "meta.json")
        if os.path.exists(meta_p):
            with open(meta_p, "r") as f:
                d = json.load(f)
            got = int(d.get("total_cts", -1))
            if got != expected:
                raise RuntimeError(f"[train_em] total_cts mismatch for split={split_name}: dataset={expected} vs cheap_meta={got} ({meta_p})")

    _assert_total_cts_matches_cheap(split_train, total_cts_train)
    _assert_total_cts_matches_cheap(split_val, total_cts_val)

    # ----------------------------
    # 5) TokenProvider（train/val 各一份，避免 split 混用）
    # ----------------------------
    tp_cfg_node = cfg.get("token_provider", cfg.get("em", {}).get("token_provider", None))
    if tp_cfg_node is None:
        raise KeyError("[train_em] missing token_provider config. Expected cfg.token_provider or cfg.em.token_provider")

    token_provider_cfg = TokenProviderConfig(
        policy=tp_cfg_node.policy,
        assemble=tp_cfg_node.assemble,
        use_amp=bool(tp_cfg_node.get("use_amp", False)),
        normalize_tokens=bool(tp_cfg_node.get("normalize_tokens", False)),
        cache_missing=str(tp_cfg_node.get("cache_missing", "error")),
    )

    # 从 meta 里拿版本串（instance cache identity 需要）
    sel_version_train = str(sel_meta_train.get("sel_version", "sel_v0"))
    cheap_version_used_train = str(sel_meta_train.get("cheap_version_used", cheap_meta_train.get("cheap_version", "cheap_v0")))

    sel_version_val = str(sel_meta_val.get("sel_version", sel_version_train))
    cheap_version_used_val = str(sel_meta_val.get("cheap_version_used", cheap_version_used_train))

    token_provider_train = TokenProvider(
        cfg=token_provider_cfg,
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
    )

    token_provider_val = TokenProvider(
        cfg=token_provider_cfg,
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
    )

    # ---- 共享同一个 policy ----
    token_provider_train.policy = policy
    token_provider_val.policy = policy


    # ----------------------------
    # 6) TrainerEMConfig（兼容 cfg.trainer_em 或 cfg.em.trainer_em）
    # ----------------------------
    tr_node = cfg.get("trainer_em", cfg.get("em", {}).get("trainer_em", None))
    if tr_node is None:
        tr_cfg = TrainerEMConfig(
            num_epochs=int(run_cfg.get("num_epochs", 10)),
            lr_agg=float(cfg.get("train", {}).get("lr", 1e-4)) if "train" in cfg else 1e-4,
            lr_inst=float(cfg.get("train", {}).get("lr_instance", 1e-5)) if "train" in cfg else 1e-5,
            wd_agg=float(cfg.get("train", {}).get("weight_decay", 0.0)) if "train" in cfg else 0.0,
            wd_inst=float(cfg.get("train", {}).get("weight_decay_instance", 0.0)) if "train" in cfg else 0.0,
            use_amp=bool(cfg.get("train", {}).get("use_amp", False)) if "train" in cfg else False,
        )
    else:
        tr_cfg = TrainerEMConfig(
            num_epochs=int(tr_node.get("num_epochs", run_cfg.get("num_epochs", 10))),
            log_every=int(tr_node.get("log_every", 50)),
            grad_accum_steps=int(tr_node.get("grad_accum_steps", 1)),
            clip_grad_norm=float(tr_node.get("clip_grad_norm", 0.0)),
            use_amp=bool(tr_node.get("use_amp", False)),
            lr_agg=float(tr_node.get("lr_agg", 1e-4)),
            wd_agg=float(tr_node.get("wd_agg", 0.0)),
            lr_inst=float(tr_node.get("lr_inst", 1e-5)),
            wd_inst=float(tr_node.get("wd_inst", 0.0)),
            monitor=str(tr_node.get("monitor", "loss")),
            greater_is_better=bool(tr_node.get("greater_is_better", False)),
            ema_enabled=bool(tr_node.get("ema_enabled", False)),
            ema_decay=float(tr_node.get("ema_decay", 0.999)),
        )

    trainer = TrainerEM(
        cfg=tr_cfg,
        device=device,
        agg_model=agg_model,
        instance_model=instance_model,
        token_provider=token_provider_train,
        token_provider_val=token_provider_val,
        controller=None,  # 我们在入口脚本里按 epoch 控制 E-step，风格更接近原 train.py
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
            print(f"[train_em] Resumed from checkpoint: {ckpt_path}")
        else:
            print(f"[train_em] No checkpoint found at {ckpt_path}, start from scratch.")

    # ---- Controller：注入 refresh_fn（只走 python，不走 cmd）----
    ctrl = EMPipelineController(
        cfg=ctrl_cfg,
        update_policy=policy,
        token_providers=[token_provider_train, token_provider_val],
        build_train_loader_fn=build_train_loader,          # selection refresh 后重建 train loader
        on_loader_rebuilt_fn=trainer.set_train_loader,     # 直接让 trainer 换 loader
        cheap_refresh_fn=cheap_refresh_fn,
        selection_refresh_fn=selection_refresh_fn,
        instance_refresh_fn=instance_refresh_fn,
    )

    # ---------------------------------------------------------
    # BOOTSTRAP: if instance cache is missing/not-ready, build once before training
    # ---------------------------------------------------------
    need_inst_train = not _is_stage_ready(split_train, "instance")
    need_inst_val = (split_val != split_train) and (not _is_stage_ready(split_val, "instance"))

    if need_inst_train or need_inst_val:
        print(f"[train_em] BOOTSTRAP instance cache: train_need={need_inst_train} val_need={need_inst_val}")

        # selection 必须 ready（否则 instance_runner 会失败）
        need_sel_train = not _is_stage_ready(split_train, "selection")
        need_sel_val = (split_val != split_train) and (not _is_stage_ready(split_val, "selection"))
        if need_sel_train or need_sel_val:
            print(f"[train_em] BOOTSTRAP selection cache (required by instance): train_need={need_sel_train} val_need={need_sel_val}")
            selection_refresh_fn(0)

        # build instance once (overwrite=True in your instance_refresh_fn)
        instance_refresh_fn(0)

        # 让 token_provider reopen memmap
        _notify_token_providers_cache_refreshed({
            "refresh_cheap_cache": False,
            "refresh_selection_cache": bool(need_sel_train or need_sel_val),
            "refresh_instance_cache": True,
        })

    # ----------------------------
    # 7) 主训练循环（复用原 train.py 风格）
    # ----------------------------
    num_epochs = int(tr_cfg.num_epochs)

    for epoch in range(trainer.state.epoch, num_epochs):
        trainer.state.epoch = epoch
        # 只调用一次：包含 policy.on_epoch_begin + token_providers.on_epoch_begin
        ctrl.on_epoch_begin(epoch)

        # （可选）如果你想在 refresh 前先拿到计划，用于后续重建 val loader
        refresh_plan = policy.refresh_plan(epoch)

        # E-step + rebuild train loader（单卡）
        new_loader = ctrl.maybe_refresh_and_rebuild(epoch=epoch)
        if new_loader is not None:
            train_loader = new_loader
            # 这一句其实 controller 已经通过 on_loader_rebuilt_fn 调过了；保留也无害
            trainer.set_train_loader(train_loader)

        # 若本 epoch 做了 selection refresh，你坚持 refresh val：这里重建 val_loader
        if bool(refresh_plan.get("refresh_selection_cache", False)):
            val_loader = build_val_loader()
            trainer.val_loader = val_loader  # TrainerEM 里就是 self.val_loader，直接赋值即可

        if bool(refresh_plan.get("refresh_selection_cache", False)):
            _sync_token_provider_identity(token_provider_train, split_train)
            _sync_token_provider_identity(token_provider_val, split_val)



        train_metrics = trainer.train_one_epoch(train_loader)
        train_loss = float(train_metrics.get("loss", 0.0))
        print(f"[Epoch {epoch+1}/{num_epochs}] Train loss = {train_loss:.4f}")

        prev_best = trainer.state.best_metric
        val_metrics = trainer.validate_one_epoch(val_loader, use_ema=True)
        val_loss = float(val_metrics.get("loss", 0.0))

        # scheduler + best
        trainer.step_schedulers(val_metrics)
        improved = trainer.update_best(val_metrics)

        monitor_name = str(tr_cfg.monitor)
        monitor_value = float(val_metrics.get(monitor_name, val_loss))

        # 可选打印常用指标（存在就打印）
        extra = []
        for k in ["f1", "pr_auc", "roc_auc", "accuracy"]:
            if k in val_metrics:
                extra.append(f"{k}={float(val_metrics[k]):.4f}")
        extra_str = " | " + " | ".join(extra) if extra else ""

        print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"Val loss = {val_loss:.4f} | {monitor_name} = {monitor_value:.4f}{extra_str}"
        )

        # save last
        trainer.save_checkpoint(str(ckpt_dir / "last.pt"))

        # save best
        if improved:
            trainer.save_checkpoint(str(ckpt_dir / "best.pt"))
            print(
                f"[Epoch {epoch+1}] Improved {monitor_name}: "
                f"{prev_best:.6f} → {trainer.state.best_metric:.6f}. Saved best checkpoint."
            )

    # wandb finish
    if wandb_run is not None:
        try:
            import wandb  # type: ignore
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
