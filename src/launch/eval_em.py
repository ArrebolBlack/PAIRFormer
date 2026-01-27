# src/launch/eval_em.py
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
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Iterable, Tuple, List

import hydra
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.config.data_config import DataConfig
from src.data.dataset import ChunkedCTSDataset
from src.data.pair_dataset_dynamic import DynamicPairDataset
from src.data.pair_batch_builder_cpu import PairBatchBuilderCPU, PairBatchBuilderCPUConfig
from src.data.builder import get_or_build_blocks

from src.models.registry import build_model

from src.em.token_provider import TokenProvider, TokenProviderConfig
from src.em.update_policy import UpdatePolicy, UpdatePolicyConfig

from src.trainer.trainer_em import TrainerEM, TrainerEMConfig
from src.evaluator.evaluator import evaluate_with_trainer

from src.em.cheap_runner import CheapCacheRunner, CheapCacheBuildConfig
from src.em.selection_runner import run_selection_cache
from src.em.instance_runner import run_instance_cache

from src.utils import set_seeds
from src.launch.train import iter_scalar_metrics, setup_wandb


# -----------------------------------------------------------------------------
# Helpers (keep consistent with train_em.py)
# -----------------------------------------------------------------------------

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


def _oc_select(cfg: DictConfig, key: str, default=None):
    try:
        return OmegaConf.select(cfg, key, default=default)
    except Exception:
        return default


def _stable_cfg_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def _strip_prefix_state_dict(sd: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for k, v in sd.items():
        kk = k
        for pref in ("model.", "module.", "net."):
            if kk.startswith(pref):
                kk = kk[len(pref):]
        cleaned[kk] = v
    return cleaned


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # ---- Basic env ----
    seed = int(cfg.get("seed", 2020))
    set_seeds(seed)

    device = _pick_device(cfg)
    orig_cwd = Path(get_original_cwd())
    run_dir = Path.cwd()

    run_cfg = cfg.run if ("run" in cfg and cfg.run is not None) else {}

    # ---- dirs ----
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

    # ---- data cfg ----
    data_cfg = DataConfig.from_omegaconf(cfg.data)

    # ---- eval splits ----
    # 兼容：run.eval_splits / run.eval_split / run.val_split / run.split
    eval_splits = run_cfg.get("test_splits", None)

    if len(eval_splits) == 0:
        raise ValueError("[eval_em] eval_splits is empty.")

    # ---- main runtime knobs ----
    kmax = int(run_cfg.get("kmax", 512))
    batch_size = int(run_cfg.get("batch_size", 256))
    num_workers = int(run_cfg.get("num_workers", 8))

    # eval instance mode (cached/online)
    eval_instance_mode = str(run_cfg.get("eval_instance_mode", run_cfg.get("val_instance_mode", "cached"))).lower()
    eval_use_inst_cache_default = (eval_instance_mode == "cached")

    # 强制入口 overwrite cache（你要求的：避免 cross-run contamination）
    force_overwrite_bootstrap = bool(run_cfg.get("force_overwrite_bootstrap", True))

    print(
        f"[eval_em] eval_splits={eval_splits} | "
        f"eval_instance_mode={eval_instance_mode} (use_cache={eval_use_inst_cache_default}) | "
        f"force_overwrite_bootstrap={force_overwrite_bootstrap}"
    )

    # ---- cache roots ----
    default_cache = cfg.get("paths", {}).get("cache_root", "cache")
    cache_root_cfg = run_cfg.get("cache_path", default_cache)
    cache_root = _resolve_path(str(cache_root_cfg), orig_cwd)
    assert cache_root is not None

    em_cache_root_cfg = cfg.get("em_cache_root", str(cache_root))
    em_cache_root = _resolve_path(str(em_cache_root_cfg), orig_cwd)
    assert em_cache_root is not None

    dataset_cache_root = str(cache_root)
    em_cache_root_str = str(em_cache_root)

    # ---- ensure window-level blocks exist (train.py semantics) ----
    for sp in eval_splits:
        get_or_build_blocks(data_cfg, sp, str(cache_root))

    # ---- build datasets lazily by split ----
    _cts_ds_by_split: Dict[str, ChunkedCTSDataset] = {}

    def _get_cts_ds(split: str) -> ChunkedCTSDataset:
        if split in _cts_ds_by_split:
            return _cts_ds_by_split[split]
        # blocks are ensured already, but keep semantics
        get_or_build_blocks(data_cfg, split, str(cache_root))
        ds = ChunkedCTSDataset(str(cache_root), data_cfg, split)
        _cts_ds_by_split[split] = ds
        return ds

    # -----------------------------------------------------------------------------
    # Build models (instance + agg + optional cheap + selector)
    # -----------------------------------------------------------------------------
    inst_cfg = _get_cfg_node(cfg, "instance_model", "cts_model", "model_instance")
    if inst_cfg is None:
        raise KeyError("[eval_em] missing instance model config: cfg.instance_model / cfg.cts_model / cfg.model_instance")
    inst_arch = str(inst_cfg.get("arch", inst_cfg.get("name")))
    instance_model = build_model(inst_arch, inst_cfg, data_cfg=data_cfg).to(device)

    agg_cfg = _get_cfg_node(cfg, "model", "agg_model", "pair_model")
    if agg_cfg is None:
        raise KeyError("[eval_em] missing aggregator model config: cfg.model (or cfg.agg_model / cfg.pair_model).")
    agg_arch = str(agg_cfg.get("arch", agg_cfg.get("name")))
    agg_model = build_model(agg_arch, agg_cfg, data_cfg=data_cfg).to(device)

    em_node = cfg.get("em", {})

    # ---- cheap model (required if we rebuild cheap cache) ----
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
            cheap_model.load_state_dict(_strip_prefix_state_dict(sd), strict=False)
            cheap_model.to(device)

    cheap_version = str(em_node.get("cheap_version", cfg.get("cheap_version", "cheap_v0")))
    cheap_emb_dim = int(em_node.get("cheap_emb_dim", cfg.get("cheap_emb_dim", 64)))

    # ---- selector module (Hydra instantiate) ----
    sel_mod_node = (
        _oc_select(cfg, "em.selector_module")
        or _oc_select(cfg, "em.selector")
        or _oc_select(cfg, "selector_module")
        or _oc_select(cfg, "selector")
    )
    if sel_mod_node is None:
        raise KeyError(
            "[eval_em] Missing selector config. Provide cfg.em.selector_module "
            "(recommended) or cfg.em.selector / cfg.selector_module / cfg.selector, "
            "and it must be hydra-instantiateable (contain _target_)."
        )
    selector_module = instantiate(sel_mod_node)

    # ---- sel_version (auto hash by selector node) ----
    sel_version_cfg = (_oc_select(cfg, "em.sel_version", default=None) or _oc_select(cfg, "sel_version", default=None))
    sel_mod_container = OmegaConf.to_container(sel_mod_node, resolve=True)
    sel_hash = _stable_cfg_hash(sel_mod_container)
    if sel_version_cfg is None or str(sel_version_cfg).lower() in ("", "none", "null", "auto"):
        sel_version = f"sel_{sel_hash}"
    else:
        sel_version = str(sel_version_cfg)

    # selector kmax is source of truth
    if hasattr(selector_module, "cfg") and hasattr(selector_module.cfg, "kmax"):
        sel_kmax = int(selector_module.cfg.kmax)
    else:
        sel_kmax = int(kmax)

    print(f"[eval_em] selector instantiated: {selector_module.__class__.__name__}")
    if hasattr(selector_module, "version"):
        print(f"[eval_em] selector.version: {getattr(selector_module, 'version')}")
    print(f"[eval_em] sel_version: {sel_version} (hash={sel_hash}) sel_kmax={sel_kmax}")

    # -----------------------------------------------------------------------------
    # Build refresh fns (cheap / selection / instance)
    # -----------------------------------------------------------------------------
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
            raise RuntimeError(
                "[eval_em] Need to rebuild cheap cache but cheap_model is None. "
                "Provide cfg.em.cheap_model (and optionally cfg.em.cheap_ckpt_path)."
            )
        splits_use = list(splits) if splits is not None else list(eval_splits)
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
        )
        runner.build(
            cheap_model=cheap_model,
            cheap_version=str(cheap_version),
            emb_dim=int(cheap_emb_dim),
            cfg=c,
        )

    sel_node = em_node.get("selection_cache", {})
    sel_pair_batch_size = int(sel_node.get("pair_batch_size", 64))

    def selection_refresh_fn(
        epoch: int,
        *,
        overwrite: bool = True,
        skip_if_ready: bool = False,
        splits: Optional[Iterable[str]] = None,
    ) -> None:
        splits_use = list(splits) if splits is not None else list(eval_splits)
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
        )

    inst_node = em_node.get("instance_cache", {})
    inst_cache_bs = int(inst_node.get("batch_size", 1024))
    inst_cache_nw = int(inst_node.get("num_workers", num_workers))
    inst_cache_use_amp = bool(inst_node.get("use_amp", (device.type == "cuda")))
    inst_cache_norm = bool(inst_node.get("normalize_emb", False))

    inst_emb_dim = int(em_node.get("inst_emb_dim", cfg.get("em", {}).get("inst_emb_dim", 384)))
    inst_version = str(em_node.get("inst_version", cfg.get("em", {}).get("inst_version", "inst_v0")))

    def instance_refresh_fn(
        epoch: int,
        *,
        overwrite: bool = True,
        skip_if_ready: bool = False,
        splits: Optional[Iterable[str]] = None,
    ) -> None:
        splits_use = list(splits) if splits is not None else list(eval_splits)
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
        )

    # -----------------------------------------------------------------------------
    # BOOTSTRAP at entry (hard overwrite by default)
    # -----------------------------------------------------------------------------
    bootstrap_epoch0 = int(em_node.get("bootstrap", {}).get("epoch", 0))
    bootstrap_overwrite_all = True if force_overwrite_bootstrap else bool(em_node.get("bootstrap", {}).get("overwrite_all", True))
    bootstrap_skip_if_ready = False if bootstrap_overwrite_all else bool(em_node.get("bootstrap", {}).get("skip_if_ready", True))

    print(
        "[eval_em] BOOTSTRAP plan: "
        f"overwrite_all={bootstrap_overwrite_all} skip_if_ready={bootstrap_skip_if_ready} splits={eval_splits}"
    )

    # 入口刷新 cheap + selection（instance 会在加载 ckpt 后再刷新一次，确保匹配）
    cheap_refresh_fn(bootstrap_epoch0, overwrite=bootstrap_overwrite_all, skip_if_ready=bootstrap_skip_if_ready, splits=eval_splits)
    selection_refresh_fn(bootstrap_epoch0, overwrite=bootstrap_overwrite_all, skip_if_ready=bootstrap_skip_if_ready, splits=eval_splits)

    # -----------------------------------------------------------------------------
    # Build DataLoader(s) after selection is ready
    # -----------------------------------------------------------------------------
    # refresh cache 时不建议 persistent_workers
    persistent_workers = bool(run_cfg.get("persistent_workers", True))
    if num_workers <= 0:
        persistent_workers = False
    # selection refresh/instance refresh 都会动 cache；eval 通常也不需要 persistent
    persistent_workers = False

    def _load_sel_meta(split: str) -> Dict[str, Any]:
        p = em_cache_root / "em_cache" / split / "selection" / "meta.json"
        return _load_json(p)

    def build_eval_loader_for_split(split: str) -> DataLoader:
        ds = _get_cts_ds(split)
        pair_ds = DynamicPairDataset(ds)
        meta = _load_sel_meta(split)
        cpu_builder = PairBatchBuilderCPU(
            cts_ds=ds,
            em_cache_root=em_cache_root,
            split=split,
            cfg=PairBatchBuilderCPUConfig(
                kmax=int(kmax),
                include_pos=bool(run_cfg.get("include_pos", True)),
                include_esa=bool(run_cfg.get("include_esa", True)),
                pin_memory=True,
            ),
            expected_path_hash=str(meta["path_hash"]),
            expected_dataset_hash_key=str(meta["dataset_hash_key"]),
        )
        return DataLoader(
            pair_ds,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=True,
            persistent_workers=persistent_workers,
            collate_fn=cpu_builder,
            drop_last=False,
        )

    # -----------------------------------------------------------------------------
    # TokenProvider configs
    # -----------------------------------------------------------------------------
    policy_node = em_node.get("policy", None)
    if policy_node is None:
        raise KeyError("[eval_em] missing cfg.em.policy (UpdatePolicyConfig node).")
    pol_cfg_dict = OmegaConf.to_container(policy_node, resolve=True)

    tp_cfg_node = cfg.get("token_provider", cfg.get("em", {}).get("token_provider", None))
    if tp_cfg_node is None:
        raise KeyError("[eval_em] missing token_provider config. Expected cfg.token_provider or cfg.em.token_provider")

    # 临时 policy：为了先 load_checkpoint，不要求 instance cache ready
    policy_tmp_online = UpdatePolicy(UpdatePolicyConfig(
        warmup_epochs=0,
        instance_mode="online",
        cheap_mode="cached",
    ))

    token_provider_cfg = TokenProviderConfig(
        policy=UpdatePolicyConfig(**pol_cfg_dict),
        assemble=tp_cfg_node.assemble,
        use_amp=bool(tp_cfg_node.get("use_amp", False)),
        normalize_tokens=bool(tp_cfg_node.get("normalize_tokens", False)),
        cache_missing=str(tp_cfg_node.get("cache_missing", "error")),
    )

    def _read_split_meta_required(split: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        sel_meta_p = em_cache_root / "em_cache" / split / "selection" / "meta.json"
        cheap_meta_p = em_cache_root / "em_cache" / split / "cheap" / "meta.json"
        if (not sel_meta_p.exists()) or (not cheap_meta_p.exists()):
            raise FileNotFoundError(f"[eval_em] missing meta under em_cache_root for split={split}")
        return _load_json(sel_meta_p), _load_json(cheap_meta_p)

    def build_token_provider_for_split(
        split: str,
        *,
        policy: UpdatePolicy,
        require_ready: bool,
    ) -> TokenProvider:
        sel_meta, cheap_meta = _read_split_meta_required(split)
        ds = _get_cts_ds(split)
        total_cts_local = int(len(ds))

        sel_version_used = str(sel_meta.get("sel_version", "sel_v0"))
        cheap_version_used = str(sel_meta.get("cheap_version_used", cheap_meta.get("cheap_version", cheap_version)))

        return TokenProvider(
            cfg=token_provider_cfg,
            policy=policy,
            instance_model=instance_model,
            device=device,
            em_cache_root=str(em_cache_root),
            split=split,
            path_hash=str(sel_meta["path_hash"]),
            dataset_hash_key=str(sel_meta["dataset_hash_key"]),
            total_cts=total_cts_local,
            inst_emb_dim=int(inst_emb_dim),
            inst_version=str(inst_version),
            sel_version_used=sel_version_used,
            cheap_version_used=cheap_version_used,
            require_ready=bool(require_ready),
        )

    # -----------------------------------------------------------------------------
    # Trainer config (reuse trainer_em node)
    # -----------------------------------------------------------------------------
    tr_node = cfg.get("trainer_em", None)
    if tr_node is None:
        raise ValueError("[eval_em] Missing config node: cfg.trainer_em.")

    betas = tr_node.get("betas", [0.9, 0.999])
    betas = (float(betas[0]), float(betas[1]))

    tr_cfg = TrainerEMConfig(
        # loop (eval doesn't need full loop, but keep for compatibility)
        num_epochs=int(tr_node.get("num_epochs", run_cfg.get("num_epochs", 1))),
        log_every=int(tr_node.get("log_every", 50)),
        grad_accum_steps=int(tr_node.get("grad_accum_steps", 1)),
        clip_grad_norm=float(tr_node.get("clip_grad_norm", 0.0)),
        use_amp=bool(tr_node.get("use_amp", False)),

        # loss
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

        # optimizer (won't be used in eval, but needed for ckpt compatibility)
        optimizer=str(tr_node.get("optimizer", "adamw")),
        betas=betas,
        eps=float(tr_node.get("eps", 1e-8)),
        amsgrad=bool(tr_node.get("amsgrad", False)),
        lr_agg=float(tr_node.get("lr_agg", 3e-4)),
        wd_agg=float(tr_node.get("wd_agg", 1e-2)),
        lr_inst=float(tr_node.get("lr_inst", 1e-5)),
        wd_inst=float(tr_node.get("wd_inst", 0.0)),

        # scheduler
        scheduler_agg=str(tr_node.get("scheduler_agg", "none")),
        scheduler_inst=str(tr_node.get("scheduler_inst", "none")),
        scheduler_t_max=int(tr_node.get("scheduler_t_max", tr_node.get("num_epochs", 10))),
        scheduler_step_size=int(tr_node.get("scheduler_step_size", 10)),
        scheduler_gamma=float(tr_node.get("scheduler_gamma", 0.1)),
        scheduler_factor=float(tr_node.get("scheduler_factor", 0.2)),
        scheduler_patience=int(tr_node.get("scheduler_patience", 5)),

        # monitor/ema
        monitor=str(tr_node.get("monitor", "loss")),
        greater_is_better=bool(tr_node.get("greater_is_better", False)),
        ema_enabled=bool(tr_node.get("ema_enabled", False)),
        ema_decay=float(tr_node.get("ema_decay", 0.999)),
    )

    # -----------------------------------------------------------------------------
    # Load checkpoint (agg + inst) BEFORE building instance cache (but we need a trainer)
    # We'll create a temporary online token_provider to avoid requiring instance cache.
    # -----------------------------------------------------------------------------
    ckpt_path_cfg = (
        run_cfg.get("checkpoint", None)
        or run_cfg.get("eval_checkpoint", None)
        or run_cfg.get("ckpt_path", None)
    )
    if ckpt_path_cfg is None:
        default = ckpt_dir / "best.pt"
        ckpt_path = default if default.exists() else (ckpt_dir / "last.pt")
    else:
        ckpt_path = _resolve_path(str(ckpt_path_cfg), orig_cwd) or Path(str(ckpt_path_cfg))

    if not ckpt_path.exists():
        raise FileNotFoundError(f"[eval_em] checkpoint not found: {ckpt_path}")

    print(f"[eval_em] Will load checkpoint: {ckpt_path}")

    # -----------------------------------------------------------------------------
    # Evaluate each split
    # -----------------------------------------------------------------------------
    use_ema = bool(run_cfg.get("use_ema", True))

    for sp in eval_splits:
        print(f"\n[eval_em] ===================== split={sp} =====================")

        eval_loader = build_eval_loader_for_split(sp)

        # temp token provider (online, no require_ready) to allow load_checkpoint
        tp_tmp = build_token_provider_for_split(sp, policy=policy_tmp_online, require_ready=False)

        trainer = TrainerEM(
            cfg=tr_cfg,
            device=device,
            agg_model=agg_model,
            instance_model=instance_model,
            token_provider=tp_tmp,
            token_provider_val=tp_tmp,
            controller=None,
            train_loader=eval_loader,  # dummy
            val_loader=eval_loader,
            logger=wandb_run,
            task_cfg=(cfg.task if "task" in cfg else None),
        )

        trainer.load_checkpoint(str(ckpt_path), map_location=device)
        print(f"[eval_em] Loaded checkpoint into TrainerEM: {ckpt_path}")

        # After loading weights: refresh instance cache to match ckpt weights (关键)
        if eval_use_inst_cache_default or bool(run_cfg.get("always_build_instance_cache", True)):
            print(f"[eval_em] Rebuilding INSTANCE cache (match ckpt weights) for split={sp} ...")
            instance_refresh_fn(epoch=bootstrap_epoch0, overwrite=True, skip_if_ready=False, splits=[sp])
            # try notify tp (if implemented)
            if hasattr(tp_tmp, "on_cache_refreshed"):
                tp_tmp.on_cache_refreshed({"refresh_instance_cache": True})

        # build final eval token provider
        policy_eval = UpdatePolicy(UpdatePolicyConfig(
            warmup_epochs=0,
            instance_mode=("cached" if eval_use_inst_cache_default else "online"),
            cheap_mode="cached",
        ))
        tp_eval = build_token_provider_for_split(sp, policy=policy_eval, require_ready=eval_use_inst_cache_default)

        trainer.token_provider = tp_eval
        trainer.token_provider_val = tp_eval

        # quick validate (epoch-like)
        val_metrics = trainer.validate_one_epoch(
            eval_loader,
            use_ema=use_ema,
            use_instance_cache=eval_use_inst_cache_default,
        )
        val_loss = float(val_metrics.get("loss", 0.0))
        extra = []
        for k in ["f1", "pr_auc", "roc_auc", "accuracy"]:
            if k in val_metrics:
                extra.append(f"{k}={float(val_metrics[k]):.4f}")
        extra_str = " | " + " | ".join(extra) if extra else ""
        print(f"[eval_em][split={sp}] validate_one_epoch: loss={val_loss:.4f}{extra_str}")

        # full evaluation output (mirrors train_em final val eval)
        out_dir = Path(eval_dir) / "eval" / str(sp)
        out_dir.mkdir(parents=True, exist_ok=True)

        res = evaluate_with_trainer(
            trainer=trainer,
            loader=eval_loader,
            task_cfg=cfg.task,
            logging_cfg=cfg.logging,
            output_dir=str(out_dir),
            set_labels=None,
            aggregate_sets=False,
            tag=f"eval_{sp}",
            do_threshold_sweep=bool(cfg.eval.do_threshold_sweep),
            sweep_num_thresholds=int(cfg.eval.sweep_num_thresholds),
            reduction=run_cfg.get("eval_reduction", "max"),
            softmax_temp=float(run_cfg.get("eval_softmax_temp", 1.0)),
            topk=int(run_cfg.get("eval_topk", 3)),
        )

        best_threshold = res.get("best_threshold", None)
        metrics = res.get("metrics", {})

        print(f"\n[eval_em][split={sp}] Final metrics:")
        for k, v in iter_scalar_metrics(metrics):
            print(f"  {k}: {v:.4f}")
        if best_threshold is not None:
            print(f"[eval_em][split={sp}] Best threshold = {float(best_threshold):.4f}")

        # wandb summary
        if wandb_run is not None:
            prefix = f"eval/{sp}"
            for k, v in iter_scalar_metrics(metrics):
                wandb_run.summary[f"{prefix}/{k}"] = v
            if best_threshold is not None:
                try:
                    wandb_run.summary[f"{prefix}/best_threshold"] = float(best_threshold)
                except Exception:
                    pass

    # finish wandb
    if wandb_run is not None:
        try:
            import wandb  # type: ignore
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
