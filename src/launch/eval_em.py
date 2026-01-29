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
from contextlib import contextmanager

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
# Helpers (COPY from train_em.py)
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


def _resolve_ckpt_path(p: Optional[str], *, run_dir: Path, orig_cwd: Path, ckpt_dir: Path) -> Path:
    """
    与 train_em.py 行为一致：优先 run_dir 相对路径，其次 orig_cwd。
    若 p=None：默认 ckpt_dir/best.pt。
    """
    if p is None:
        cand_best = ckpt_dir / "best.pt"
        if cand_best.exists():
            return cand_best
        cand_last = ckpt_dir / "last.pt"
        return cand_last

    raw = Path(os.path.expandvars(os.path.expanduser(str(p))))
    if raw.is_absolute():
        return raw

    cand1 = run_dir / raw
    if cand1.exists():
        return cand1

    cand2 = orig_cwd / raw
    return cand2


def _load_em_checkpoint_into_models(
    ckpt_path: Path,
    *,
    agg_model: torch.nn.Module,
    instance_model: torch.nn.Module,
    device: torch.device,
) -> None:
    """
    目的：避免依赖 TrainerEM/token_provider，严格做到“bootstrap 后最后构建 Trainer”。
    但又不能猜测 ckpt 结构 -> 这里实现一个“兼容常见格式”的 deterministic loader：
      - 若 ckpt 含明确字段：agg_state_dict / instance_state_dict / agg_model / instance_model
      - 否则尝试从 ckpt["state_dict"] 或 ckpt 本体 (dict) 里按 key overlap 自动分配
    若都失败：直接 raise（让你回去看 TrainerEM.save_checkpoint 的真实格式）。
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"[eval_em] checkpoint is not a dict: {ckpt_path}")

    # 1) direct keys
    cand_agg = None
    cand_inst = None

    for k in ("agg_state_dict", "agg_model_state_dict", "aggregator_state_dict", "agg_model", "aggregator"):
        if k in ckpt and isinstance(ckpt[k], dict):
            cand_agg = ckpt[k]
            break

    for k in ("instance_state_dict", "inst_state_dict", "instance_model_state_dict", "instance_model", "inst_model", "cts_model"):
        if k in ckpt and isinstance(ckpt[k], dict):
            cand_inst = ckpt[k]
            break

    # 2) fallback: unified state_dict
    if cand_agg is None or cand_inst is None:
        sd_all = None
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd_all = ckpt["state_dict"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            sd_all = ckpt["model_state_dict"]
        else:
            # sometimes ckpt itself is a state_dict-like dict of tensors
            if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                sd_all = ckpt

        if sd_all is not None:
            sd_all = _strip_prefix_state_dict(sd_all)

            agg_keys = set(agg_model.state_dict().keys())
            inst_keys = set(instance_model.state_dict().keys())

            agg_sd: Dict[str, Any] = {}
            inst_sd: Dict[str, Any] = {}

            # first pass: direct match
            for k, v in sd_all.items():
                if k in agg_keys:
                    agg_sd[k] = v
                if k in inst_keys:
                    inst_sd[k] = v

            # second pass: prefix patterns (agg_model./instance_model.)
            if len(agg_sd) == 0 or len(inst_sd) == 0:
                for k, v in sd_all.items():
                    if k.startswith("agg_model.") or k.startswith("aggregator."):
                        kk = k.split(".", 1)[1]
                        if kk in agg_keys:
                            agg_sd[kk] = v
                    if k.startswith("instance_model.") or k.startswith("inst_model.") or k.startswith("cts_model."):
                        kk = k.split(".", 1)[1]
                        if kk in inst_keys:
                            inst_sd[kk] = v

            if cand_agg is None and len(agg_sd) > 0:
                cand_agg = agg_sd
            if cand_inst is None and len(inst_sd) > 0:
                cand_inst = inst_sd

    if cand_agg is None or cand_inst is None:
        raise RuntimeError(
            "[eval_em] Cannot locate both agg/instance state_dict in ckpt. "
            f"ckpt_path={ckpt_path}. Please check TrainerEM.save_checkpoint format."
        )

    cand_agg = _strip_prefix_state_dict(cand_agg)
    cand_inst = _strip_prefix_state_dict(cand_inst)

    miss_a, unexp_a = agg_model.load_state_dict(cand_agg, strict=False)
    miss_i, unexp_i = instance_model.load_state_dict(cand_inst, strict=False)

    if miss_a:
        print(f"[eval_em] WARN agg missing keys: {len(miss_a)} (first10): {miss_a[:10]}")
    if unexp_a:
        print(f"[eval_em] WARN agg unexpected keys: {len(unexp_a)} (first10): {unexp_a[:10]}")
    if miss_i:
        print(f"[eval_em] WARN inst missing keys: {len(miss_i)} (first10): {miss_i[:10]}")
    if unexp_i:
        print(f"[eval_em] WARN inst unexpected keys: {len(unexp_i)} (first10): {unexp_i[:10]}")

    agg_model.to(device)
    instance_model.to(device)


@contextmanager
def _swap_token_provider_val(tr: TrainerEM, tp_new: TokenProvider):
    old = tr.token_provider_val
    tr.token_provider_val = tp_new
    try:
        yield
    finally:
        tr.token_provider_val = old


# -----------------------------------------------------------------------------
# Main (structure mirrors train_em.py; difference: no training, load ckpt, go test)
# -----------------------------------------------------------------------------
@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # ---- 基本环境 ----
    seed = int(cfg.get("seed", 2020))
    set_seeds(seed)

    device = _pick_device(cfg)
    orig_cwd = Path(get_original_cwd())
    run_dir = Path.cwd()

    # ---- ckpt/eval dir（复用 train_em.py 风格）----
    run_cfg = cfg.run if ("run" in cfg and cfg.run is not None) else {}
    ckpt_dir_cfg = run_cfg.get("ckpt_dir", run_cfg.get("ckpt_subdir", "checkpoints"))
    eval_dir_cfg = run_cfg.get("eval_dir", run_cfg.get("eval_subdir", "eval"))
    ckpt_dir = Path(str(ckpt_dir_cfg))
    eval_dir = Path(str(eval_dir_cfg))
    if not ckpt_dir.is_absolute():
        ckpt_dir = run_dir / ckpt_dir
    if not eval_dir.is_absolute():
        eval_dir = run_dir / eval_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ---- wandb ----
    wandb_run = setup_wandb(cfg)

    # ----------------------------
    # 0) data cfg / runtime knobs (ONLY use keys from your config)
    # ----------------------------
    data_cfg = DataConfig.from_omegaconf(cfg.data)

    kmax = int(run_cfg.get("kmax", 512))
    batch_size = int(run_cfg.get("batch_size", 256))
    num_workers = int(run_cfg.get("num_workers", 8))

    force_overwrite_bootstrap = bool(run_cfg.get("force_overwrite_bootstrap", True))

    test_splits = run_cfg.get("test_splits", ["test"])
    if isinstance(test_splits, str):
        test_splits = [test_splits]
    test_splits = [str(x) for x in list(test_splits)]
    if len(test_splits) == 0:
        raise ValueError("[eval_em] run.test_splits is empty.")

    print(
        f"[eval_em] test_splits={test_splits} | "
        f"force_overwrite_bootstrap={force_overwrite_bootstrap} | "
        f"kmax={kmax} batch_size={batch_size} num_workers={num_workers}"
    )

    # ----------------------------
    # cache roots (same as train_em.py)
    # ----------------------------
    default_cache = cfg.get("paths", {}).get("cache_root", "cache")
    cache_root_cfg = run_cfg.get("cache_path", default_cache)
    cache_root = _resolve_path(str(cache_root_cfg), orig_cwd)
    assert cache_root is not None

    em_cache_root_cfg = cfg.get("em_cache_root", str(cache_root))
    em_cache_root = _resolve_path(str(em_cache_root_cfg), orig_cwd)
    assert em_cache_root is not None

    dataset_cache_root = str(cache_root)
    em_cache_root_str = str(em_cache_root)

    # ensure window-level blocks exist (train.py semantics)
    for sp in test_splits:
        get_or_build_blocks(data_cfg, sp, str(cache_root))

    # lazy datasets
    _cts_ds_by_split: Dict[str, ChunkedCTSDataset] = {}
    def _get_cts_ds(split: str) -> ChunkedCTSDataset:
        if split in _cts_ds_by_split:
            return _cts_ds_by_split[split]
        get_or_build_blocks(data_cfg, split, str(cache_root))
        ds = ChunkedCTSDataset(str(cache_root), data_cfg, split)
        _cts_ds_by_split[split] = ds
        return ds

    # ----------------------------
    # 2) build models (same nodes as train_em.py)
    # ----------------------------
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

    # ----------------------------
    # 3) policy + runners + selector (COPY train_em.py)
    # ----------------------------
    em_node = cfg.get("em", {})

    policy_node = em_node.get("policy", None)
    if policy_node is None:
        raise KeyError("[eval_em] missing cfg.em.policy")
    pol_cfg_dict = OmegaConf.to_container(policy_node, resolve=True)

    # eval policy: ALWAYS cached for test (train_em test routine uses require_ready=True)
    policy_eval = UpdatePolicy(UpdatePolicyConfig(
        warmup_epochs=0,
        instance_mode="cached",
        cheap_mode="cached",
    ))

    # ---- cheap model ----
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
            raise RuntimeError("[eval_em] cheap_ckpt_path is None (required to rebuild cheap cache).")
        if cheap_ckpt.exists():
            ckpt = torch.load(str(cheap_ckpt), map_location="cpu")
            sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            cheap_model.load_state_dict(_strip_prefix_state_dict(sd), strict=False)
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
            raise RuntimeError("[eval_em] cheap_model is None, cannot build cheap cache.")
        splits_use = list(splits) if splits is not None else list(test_splits)
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

    # ---- selector module + sel_version ----
    sel_mod_node = (
        _oc_select(cfg, "em.selector_module")
        or _oc_select(cfg, "em.selector")
        or _oc_select(cfg, "selector_module")
        or _oc_select(cfg, "selector")
    )
    if sel_mod_node is None:
        raise KeyError("[eval_em] Missing selector config: cfg.em.selector_module (must contain _target_).")

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

    if hasattr(selector_module, "cfg") and hasattr(selector_module.cfg, "kmax"):
        sel_kmax = int(selector_module.cfg.kmax)
    else:
        sel_kmax = int(kmax)

    print(f"[eval_em] selector instantiated: {selector_module.__class__.__name__}")
    if hasattr(selector_module, "version"):
        print(f"[eval_em] selector.version: {getattr(selector_module, 'version')}")
    print(f"[eval_em] sel_version: {sel_version} (hash={sel_hash}) sel_kmax={sel_kmax}")

    sel_node = em_node.get("selection_cache", {})
    sel_pair_batch_size = int(sel_node.get("pair_batch_size", 64))

    def selection_refresh_fn(
        epoch: int,
        *,
        overwrite: bool = True,
        skip_if_ready: bool = False,
        splits: Optional[Iterable[str]] = None,
    ) -> None:
        splits_use = list(splits) if splits is not None else list(test_splits)

        # >>>>>> 关键：把 candidate pool 参数读出来 <<<<<<
        cand_pool_size = sel_node.get("candidate_pool_size", None)
        cand_pool_mode = str(sel_node.get("candidate_pool_mode", "topn"))
        cand_pool_topn_ratio = float(sel_node.get("candidate_pool_topn_ratio", 1.0))
        cand_pool_seed = int(sel_node.get("candidate_pool_seed", seed))

        print(
            f"[eval_em] selection_refresh_fn: candidate_pool_size={cand_pool_size} "
            f"mode={cand_pool_mode} topn_ratio={cand_pool_topn_ratio} seed={cand_pool_seed}"
        )

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

            # >>>>>> 关键：传进去 <<<<<<
            candidate_pool_size=cand_pool_size,
            candidate_pool_mode=cand_pool_mode,
            candidate_pool_topn_ratio=cand_pool_topn_ratio,
            candidate_pool_seed=cand_pool_seed,
        )


    # ---- instance cache runner knobs ----
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
        splits_use = list(splits) if splits is not None else list(test_splits)
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

    # ----------------------------
    # 4) TokenProviderConfig + builders (COPY train_em.py style)
    # ----------------------------
    tp_cfg_node = cfg.get("token_provider", cfg.get("em", {}).get("token_provider", None))
    if tp_cfg_node is None:
        raise KeyError("[eval_em] missing token_provider config: cfg.token_provider")

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

    def build_token_provider_for_split(split: str, *, require_ready: bool = True) -> TokenProvider:
        sel_meta, cheap_meta = _read_split_meta_required(split)
        ds = _get_cts_ds(split)
        total_cts_local = int(len(ds))

        sel_version_used = str(sel_meta.get("sel_version", "sel_v0"))
        cheap_version_used = str(sel_meta.get("cheap_version_used", cheap_meta.get("cheap_version", cheap_version)))

        tp = TokenProvider(
            cfg=token_provider_cfg,
            policy=policy_eval,
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
        return tp

    def _load_sel_meta(split: str) -> Dict[str, Any]:
        p = em_cache_root / "em_cache" / split / "selection" / "meta.json"
        return _load_json(p)

    # eval loader builder (COPY train_em.py)
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
        return DataLoader(
            pair_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False,  # eval 阶段别用 persistent
            collate_fn=cpu_builder,
            drop_last=False,
        )

    # ----------------------------
    # 5) TrainerEMConfig (COPY train_em.py)
    # ----------------------------
    tr_node = cfg.get("trainer_em", None)
    if tr_node is None:
        raise ValueError("[eval_em] Missing config node: cfg.trainer_em.")

    betas = tr_node.get("betas", [0.9, 0.999])
    betas = (float(betas[0]), float(betas[1]))

    tr_cfg = TrainerEMConfig(
        num_epochs=int(tr_node.get("num_epochs", run_cfg.get("num_epochs", 1))),
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

    # ----------------------------
    # 6) Resolve checkpoint (MUST use run.checkpoint)
    # ----------------------------
    ckpt_path_cfg = run_cfg.get("checkpoint", None)
    ckpt_path = _resolve_ckpt_path(ckpt_path_cfg, run_dir=run_dir, orig_cwd=orig_cwd, ckpt_dir=ckpt_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"[eval_em] checkpoint not found: {ckpt_path}")

    tag_prefix = str(run_cfg.get("test_tag", f"ckpt_{ckpt_path.stem}"))
    print(f"[eval_em] Will load checkpoint: {ckpt_path}")
    print(f"[eval_em] Test tag_prefix: {tag_prefix}")

    # ----------------------------
    # 7) BOOTSTRAP for TEST (cheap + selection only)
    #    IMPORTANT: do NOT build instance cache before loading ckpt weights
    # ----------------------------
    bootstrap_epoch0 = int(em_node.get("bootstrap", {}).get("epoch", 0))
    bootstrap_overwrite_all = True if force_overwrite_bootstrap else bool(em_node.get("bootstrap", {}).get("overwrite_all", True))
    bootstrap_skip_if_ready = False if bootstrap_overwrite_all else bool(em_node.get("bootstrap", {}).get("skip_if_ready", True))

    print(
        "[eval_em] BOOTSTRAP plan (pre-ckpt): "
        f"overwrite_all={bootstrap_overwrite_all} skip_if_ready={bootstrap_skip_if_ready} splits={test_splits}"
    )
    # build cheap + selection (instance after ckpt)
    cheap_refresh_fn(bootstrap_epoch0, overwrite=bootstrap_overwrite_all, skip_if_ready=bootstrap_skip_if_ready, splits=test_splits)
    selection_refresh_fn(bootstrap_epoch0, overwrite=bootstrap_overwrite_all, skip_if_ready=bootstrap_skip_if_ready, splits=test_splits)

    # ----------------------------
    # 8) Load ckpt into (agg, inst) weights (NO trainer yet)
    # ----------------------------
    _load_em_checkpoint_into_models(
        ckpt_path,
        agg_model=agg_model,
        instance_model=instance_model,
        device=device,
    )
    print(f"[eval_em] Loaded checkpoint weights into models: {ckpt_path}")

    # ----------------------------
    # 9) Build INSTANCE cache for test splits (must match sel_version from step7)
    # ----------------------------
    print(
        "[eval_em] BOOTSTRAP plan (post-ckpt): "
        f"build_instance_cache=True overwrite=True splits={test_splits}"
    )
    instance_refresh_fn(epoch=bootstrap_epoch0, overwrite=True, skip_if_ready=False, splits=test_splits)

    # ----------------------------
    # 10) Build Trainer AFTER bootstrap+ckpt+instance_cache (as you要求)
    # ----------------------------
    bootstrap_split = test_splits[0]
    test_loader_boot = build_eval_loader_for_split(bootstrap_split)
    tp_boot = build_token_provider_for_split(bootstrap_split, require_ready=True)

    trainer = TrainerEM(
        cfg=tr_cfg,
        device=device,
        agg_model=agg_model,
        instance_model=instance_model,
        token_provider=tp_boot,
        token_provider_val=tp_boot,
        controller=None,
        train_loader=test_loader_boot,  # dummy
        val_loader=test_loader_boot,    # dummy
        logger=wandb_run,
        task_cfg=(cfg.task if "task" in cfg else None),
    )

    # ----------------------------
    # 11) TEST routine (COPY train_em.py test section)
    # ----------------------------
    # optional val-best threshold from run.best_threshold (if user provides)
    best_threshold_cfg = run_cfg.get("best_threshold", None)
    best_threshold = float(best_threshold_cfg) if best_threshold_cfg is not None else None

    def _refresh_em_caches_for_split(split: str, epoch_for_build: int) -> None:
        # 强制用“当前模型权重快照”重建 test caches，确保一致
        cheap_refresh_fn(epoch_for_build, overwrite=True, skip_if_ready=False, splits=[split])
        selection_refresh_fn(epoch_for_build, overwrite=True, skip_if_ready=False, splits=[split])
        instance_refresh_fn(epoch_for_build, overwrite=True, skip_if_ready=False, splits=[split])

    def run_test_eval_for_current_trainer(tag_prefix_local: str) -> None:
        for split_idx in test_splits:
            print(f"[eval_em][{tag_prefix_local}] Building test loader/token_provider for split='{split_idx}'")

            _refresh_em_caches_for_split(split_idx, epoch_for_build=int(bootstrap_epoch0))

            test_loader = build_eval_loader_for_split(split_idx)
            tp_test = build_token_provider_for_split(split_idx, require_ready=True)

            test_root = Path(eval_dir) / "test" / str(split_idx) / tag_prefix_local
            test_root.mkdir(parents=True, exist_ok=True)

            # ---------- (A) fixed threshold = 0.5 ----------
            task_fixed = OmegaConf.create(OmegaConf.to_container(cfg.task, resolve=True))
            task_fixed.threshold = 0.5
            out_dir_fixed = test_root / "thr0_5"
            out_dir_fixed.mkdir(parents=True, exist_ok=True)

            print(f"[eval_em][Test {split_idx}][{tag_prefix_local}] Eval fixed threshold=0.5")
            with _swap_token_provider_val(trainer, tp_test):
                res_fixed = evaluate_with_trainer(
                    trainer=trainer,
                    loader=test_loader,
                    task_cfg=task_fixed,
                    logging_cfg=cfg.logging,
                    output_dir=str(out_dir_fixed),
                    set_labels=None,
                    aggregate_sets=False,
                    tag=f"{split_idx}_{tag_prefix_local}_thr0.5",
                    do_threshold_sweep=False,
                    sweep_num_thresholds=int(cfg.eval.sweep_num_thresholds),
                    reduction=run_cfg.get("test_reduction", run_cfg.get("eval_reduction", "max")),
                    softmax_temp=float(run_cfg.get("test_softmax_temp", run_cfg.get("eval_softmax_temp", 1.0))),
                    topk=int(run_cfg.get("test_topk", run_cfg.get("eval_topk", 3))),
                )

            # ---------- (B) val best_threshold (optional) ----------
            res_valbest = None
            if best_threshold is not None:
                task_valbest = OmegaConf.create(OmegaConf.to_container(cfg.task, resolve=True))
                task_valbest.threshold = float(best_threshold)
                out_dir_valbest = test_root / "val_best"
                out_dir_valbest.mkdir(parents=True, exist_ok=True)
                print(
                    f"[eval_em][Test {split_idx}][{tag_prefix_local}] "
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
                        tag=f"{split_idx}_{tag_prefix_local}_valbest",
                        do_threshold_sweep=False,
                        sweep_num_thresholds=int(cfg.eval.sweep_num_thresholds),
                        reduction=run_cfg.get("test_reduction", run_cfg.get("eval_reduction", "max")),
                        softmax_temp=float(run_cfg.get("test_softmax_temp", run_cfg.get("eval_softmax_temp", 1.0))),
                        topk=int(run_cfg.get("test_topk", run_cfg.get("eval_topk", 3))),
                    )
            else:
                print(f"[eval_em][Test {split_idx}][{tag_prefix_local}] Skip val-best eval because run.best_threshold is None.")

            # ---------- (C) sweep on test ----------
            task_sweep = OmegaConf.create(OmegaConf.to_container(cfg.task, resolve=True))
            out_dir_sweep = test_root / "sweep"
            out_dir_sweep.mkdir(parents=True, exist_ok=True)

            print(f"[eval_em][Test {split_idx}][{tag_prefix_local}] Eval threshold sweep on test")
            with _swap_token_provider_val(trainer, tp_test):
                res_sweep = evaluate_with_trainer(
                    trainer=trainer,
                    loader=test_loader,
                    task_cfg=task_sweep,
                    logging_cfg=cfg.logging,
                    output_dir=str(out_dir_sweep),
                    set_labels=None,
                    aggregate_sets=False,
                    tag=f"{split_idx}_{tag_prefix_local}_sweep",
                    do_threshold_sweep=True,
                    sweep_num_thresholds=int(cfg.eval.sweep_num_thresholds),
                    reduction=run_cfg.get("test_reduction", run_cfg.get("eval_reduction", "max")),
                    softmax_temp=float(run_cfg.get("test_softmax_temp", run_cfg.get("eval_softmax_temp", 1.0))),
                    topk=int(run_cfg.get("test_topk", run_cfg.get("eval_topk", 3))),
                )

            best_thr_test = res_sweep.get("best_threshold", None)

            metrics_fixed = res_fixed.get("metrics", {})
            print(f"\n[Test {split_idx}][{tag_prefix_local}] Fixed threshold=0.5 metrics:")
            for k, v in iter_scalar_metrics(metrics_fixed):
                print(f"  {k}: {v:.4f}")

            metrics_valbest = res_valbest.get("metrics", {}) if res_valbest is not None else None
            if metrics_valbest is not None:
                print(
                    f"\n[Test {split_idx}][{tag_prefix_local}] "
                    f"Using val best_threshold={float(best_threshold):.4f} metrics:"
                )
                for k, v in iter_scalar_metrics(metrics_valbest):
                    print(f"  {k}: {v:.4f}")

            metrics_sweep = res_sweep.get("metrics_at_best", res_sweep.get("metrics", {}))
            if best_thr_test is not None and "metrics_at_best" in res_sweep:
                print(f"\n[Test {split_idx}][{tag_prefix_local}] Sweep metrics (best threshold={float(best_thr_test):.4f}):")
            else:
                try:
                    base_thr = float(task_sweep.threshold)
                except Exception:
                    base_thr = float("nan")
                print(f"\n[Test {split_idx}][{tag_prefix_local}] Sweep metrics (base threshold={base_thr:.4f}):")

            for k, v in iter_scalar_metrics(metrics_sweep):
                print(f"  {k}: {v:.4f}")

            if best_thr_test is not None:
                print(f"[Test {split_idx}][{tag_prefix_local}] Best threshold on test = {float(best_thr_test):.4f}")

            # wandb summary
            if wandb_run is not None:
                prefix = f"test/{split_idx}/{tag_prefix_local}"
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

    print("\n[eval_em] Start evaluating on test set...")
    run_test_eval_for_current_trainer(tag_prefix_local=tag_prefix)

    # wandb finish
    if wandb_run is not None:
        try:
            import wandb  # type: ignore
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
