# src/launch/bench_compute_vs_k.py
from __future__ import annotations

import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

import os
import json
import csv
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Iterable, List, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, instantiate
from torch.utils.data import DataLoader, Subset

from src.config.data_config import DataConfig
from src.data.dataset import ChunkedCTSDataset
from src.data.pair_dataset_dynamic import DynamicPairDataset
from src.data.pair_batch_builder_cpu import PairBatchBuilderCPU, PairBatchBuilderCPUConfig
from src.data.builder import get_or_build_blocks

from src.models.registry import build_model

from src.em.token_provider import TokenProvider, TokenProviderConfig
from src.em.update_policy import UpdatePolicy, UpdatePolicyConfig

from src.em.cheap_runner import CheapCacheRunner, CheapCacheBuildConfig
from src.em.selection_runner import run_selection_cache
from src.em.instance_runner import run_instance_cache  # optional, only if you benchmark cached-instance mode

from src.utils import set_seeds


# -------------------------
# Small utilities
# -------------------------

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


def _oc_select(cfg: DictConfig, key: str, default=None):
    try:
        return OmegaConf.select(cfg, key, default=default)
    except Exception:
        return default


def _stable_cfg_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


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


def _meta_path(em_cache_root: Path, split: str, stage: str) -> Path:
    return em_cache_root / "em_cache" / split / stage / "meta.json"


def _meta_exists_ready(p: Path) -> bool:
    if not p.exists():
        return False
    d = _load_json(p)
    return str(d.get("state", "")) == "ready"


def _expected_identity(ds: ChunkedCTSDataset) -> Tuple[str, str]:
    return str(getattr(ds, "path_hash")), str(getattr(ds, "dataset_hash_key"))


def _stage_compatible(em_cache_root: Path, split: str, stage: str, ds: ChunkedCTSDataset) -> bool:
    p = _meta_path(em_cache_root, split, stage)
    if not p.exists():
        return False
    d = _load_json(p)
    if str(d.get("state", "")) != "ready":
        return False
    exp_path_hash, exp_dataset_hash_key = _expected_identity(ds)
    if str(d.get("path_hash", "")) != exp_path_hash:
        return False
    if str(d.get("dataset_hash_key", "")) != exp_dataset_hash_key:
        return False
    return True


def _selection_compatible(
    em_cache_root: Path,
    split: str,
    ds: ChunkedCTSDataset,
    *,
    sel_version: str,
    cheap_version: str,
    min_kmax: int,
) -> bool:
    p = _meta_path(em_cache_root, split, "selection")
    if not p.exists():
        return False
    d = _load_json(p)
    if str(d.get("state", "")) != "ready":
        return False

    # identity
    exp_path_hash, exp_dataset_hash_key = _expected_identity(ds)
    if str(d.get("path_hash", "")) != exp_path_hash:
        return False
    if str(d.get("dataset_hash_key", "")) != exp_dataset_hash_key:
        return False

    # version constraints
    if str(d.get("sel_version", "")) != str(sel_version):
        return False
    if str(d.get("cheap_version_used", "")) != str(cheap_version):
        return False

    # kmax coverage (if meta contains it)
    kmax_got = int(d.get("kmax", d.get("K", -1))) if isinstance(d, dict) else -1
    if kmax_got > 0 and kmax_got < int(min_kmax):
        return False
    return True


def _make_or_load_pair_subset(out_dir: Path, split: str, n_pairs: int, seed: int, total_pairs: int) -> List[int]:
    out_p = out_dir / f"bench_pairs_{split}_{n_pairs}_seed{seed}.txt"
    if out_p.exists():
        ids: List[int] = []
        with open(out_p, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.append(int(line))
        return ids

    n = min(int(n_pairs), int(total_pairs))
    rng = np.random.RandomState(int(seed))
    ids = rng.choice(np.arange(total_pairs, dtype=np.int64), size=n, replace=False).tolist()
    ids = [int(x) for x in ids]
    ids.sort()

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        for x in ids:
            f.write(f"{x}\n")
    return ids


@torch.inference_mode()
def _bench_loop(
    *,
    loader: DataLoader,
    token_provider: TokenProvider,
    agg_model: torch.nn.Module,
    device: torch.device,
    amp: bool,
    warmup_batches: int,
    max_batches: Optional[int],
    plan: Dict[str, Any],
) -> Dict[str, float]:
    """
    End-to-end: loader -> token_provider.build_tokens -> agg forward
    Includes ONLINE instance encode if plan.use_instance_cache=False.
    """
    device_is_cuda = (device.type == "cuda")
    if device_is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    agg_model.eval()

    # warmup
    it = iter(loader)
    n_warm = 0
    global_step = 0
    while n_warm < int(warmup_batches):
        try:
            batch_cpu = next(it)
        except StopIteration:
            break

        out = token_provider.build_tokens(
            batch_cpu,
            epoch=0,
            global_step=global_step,
            plan=plan,
        )
        global_step += 1
        tokens = out.get("tokens", None)
        mask = out.get("mask", None)
        y = out.get("y_pair", None)
        if tokens is None or y is None:
            continue

        with torch.amp.autocast(device_type="cuda", enabled=bool(amp and device_is_cuda)):
            _ = agg_model(tokens, attn_mask=mask)

        n_warm += 1

    if device_is_cuda:
        torch.cuda.synchronize(device)

    # timed
    total_pairs = 0
    nb = 0
    t0 = time.perf_counter()

    while True:
        if max_batches is not None and nb >= int(max_batches):
            break
        try:
            batch_cpu = next(it)
        except StopIteration:
            break

        out = token_provider.build_tokens(
            batch_cpu,
            epoch=0,
            global_step=global_step,
            plan=plan,
        )
        global_step += 1

        tokens = out.get("tokens", None)
        mask = out.get("mask", None)
        y = out.get("y_pair", None)
        if tokens is None or y is None:
            continue

        with torch.amp.autocast(device_type="cuda", enabled=bool(amp and device_is_cuda)):
            _ = agg_model(tokens, attn_mask=mask)

        total_pairs += int(y.numel())
        nb += 1

    if device_is_cuda:
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    elapsed = max(t1 - t0, 1e-9)

    peak_alloc_gb = 0.0
    peak_reserved_gb = 0.0
    if device_is_cuda:
        peak_alloc_gb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
        peak_reserved_gb = float(torch.cuda.max_memory_reserved(device) / (1024 ** 3))

    return {
        "pairs_per_s": float(total_pairs / elapsed),
        "elapsed_s": float(elapsed),
        "total_pairs": float(total_pairs),
        "peak_alloc_gb": float(peak_alloc_gb),
        "peak_reserved_gb": float(peak_reserved_gb),
        "timed_batches": float(nb),
    }


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # ---- seed/device ----
    seed = int(cfg.get("seed", 2020))
    set_seeds(seed)
    device = _pick_device(cfg)
    orig_cwd = Path(get_original_cwd())
    run_dir = Path.cwd()

    # ---- run cfg ----
    run_cfg = cfg.run if ("run" in cfg and cfg.run is not None) else {}
    batch_size = int(run_cfg.get("batch_size", 64))
    num_workers = int(run_cfg.get("num_workers", 0))
    include_pos = bool(run_cfg.get("include_pos", True))
    include_esa = bool(run_cfg.get("include_esa", True))

    # ---- eval dir (reuse train.py semantics) ----
    eval_dir_cfg = run_cfg.get("eval_dir", run_cfg.get("eval_subdir", "eval"))
    eval_dir = Path(eval_dir_cfg)
    if not eval_dir.is_absolute():
        eval_dir = run_dir / eval_dir
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ---- benchmark node (independent; use +bench.xxx overrides) ----
    bench = cfg.get("bench", {}) or {}
    bench_enabled = bool(bench.get("enabled", True))
    if not bench_enabled:
        print("[bench] bench.enabled=false -> exit.")
        return

    bench_split = str(bench.get("split", run_cfg.get("val_split", "val")))
    bench_pairs = int(bench.get("pairs", 1024))
    bench_seed = int(bench.get("seed", seed))
    repeats = int(bench.get("repeats", 3))
    warmup_batches = int(bench.get("warmup_batches", 20))
    max_batches = bench.get("max_batches", 100)
    max_batches = None if (max_batches is None) else int(max_batches)

    # instance mode for benchmark:
    # - "online" : include expensive instance encoder (recommended for Fig3)
    # - "cached" : read instance cache (requires instance cache ready)
    inst_mode = str(bench.get("instance_mode", "online")).lower().strip()
    if inst_mode not in ("online", "cached"):
        raise ValueError(f"[bench] bench.instance_mode must be 'online' or 'cached', got: {inst_mode}")

    # K grid
    k_list = bench.get("k_list", None)
    if k_list is None:
        k_list = [8, 16, 32, 64, 128, 256, 512]
    if isinstance(k_list, str):
        k_list = [int(x) for x in k_list.replace(" ", "").split(",") if x]
    k_list = [int(x) for x in list(k_list)]
    k_list = sorted(list(dict.fromkeys([k for k in k_list if k > 0])))
    maxK = max(k_list)

    # AMP
    amp = bool(bench.get("amp", bool(device.type == "cuda")))

    # dataloader stability knobs for benchmark
    bench_batch_size = int(bench.get("batch_size", batch_size))
    bench_num_workers = int(bench.get("num_workers", 0))  # recommend 0
    drop_last = bool(bench.get("drop_last", True))

    # ---- cache roots ----
    data_cfg = DataConfig.from_omegaconf(cfg.data)

    default_cache = cfg.get("paths", {}).get("cache_root", "cache")
    cache_root_cfg = run_cfg.get("cache_path", default_cache)
    cache_root = _resolve_path(str(cache_root_cfg), orig_cwd)
    assert cache_root is not None

    em_cache_root_cfg = cfg.get("em_cache_root", str(cache_root))
    em_cache_root = _resolve_path(str(em_cache_root_cfg), orig_cwd)
    assert em_cache_root is not None

    dataset_cache_root = str(cache_root)
    em_cache_root_str = str(em_cache_root)

    # ---- ensure dataset blocks exist for bench_split ----
    get_or_build_blocks(data_cfg, bench_split, str(cache_root))
    cts_ds = ChunkedCTSDataset(str(cache_root), data_cfg, bench_split)
    pair_ds_full = DynamicPairDataset(cts_ds)

    # ---- build models: instance + aggregator ----
    inst_cfg = _get_cfg_node(cfg, "instance_model", "cts_model", "model_instance")
    if inst_cfg is None:
        raise KeyError("[bench] missing instance model config: cfg.instance_model / cfg.cts_model / cfg.model_instance")
    inst_arch = str(inst_cfg.get("arch", inst_cfg.get("name")))
    instance_model = build_model(inst_arch, inst_cfg, data_cfg=data_cfg).to(device)

    inst_ckpt = _resolve_path(cfg.get("instance_ckpt_path", None), orig_cwd)
    if inst_ckpt is None or (not inst_ckpt.exists()):
        raise RuntimeError(
            f"[bench] instance_ckpt_path missing or not found: {inst_ckpt}. "
            "Benchmark expects a valid instance model checkpoint."
        )
    ckpt = torch.load(str(inst_ckpt), map_location="cpu")
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    cleaned = {}
    for k, v in sd.items():
        for pref in ("model.", "module.", "net."):
            if k.startswith(pref):
                k = k[len(pref):]
        cleaned[k] = v
    instance_model.load_state_dict(cleaned, strict=False)
    instance_model.to(device)

    agg_cfg = _get_cfg_node(cfg, "model", "agg_model", "pair_model")
    if agg_cfg is None:
        raise KeyError("[bench] missing aggregator model config: cfg.model (or cfg.agg_model / cfg.pair_model).")
    agg_arch = str(agg_cfg.get("arch", agg_cfg.get("name")))
    agg_model = build_model(agg_arch, agg_cfg, data_cfg=data_cfg).to(device)

    # optional: load agg ckpt if provided (recommended)
    agg_ckpt = _resolve_path(cfg.get("agg_ckpt_path", cfg.get("aggregator_ckpt_path", None)), orig_cwd)
    if agg_ckpt is not None and agg_ckpt.exists():
        ckptA = torch.load(str(agg_ckpt), map_location="cpu")
        sdA = ckptA.get("state_dict", ckptA) if isinstance(ckptA, dict) else ckptA
        cleanedA = {}
        for k, v in sdA.items():
            for pref in ("model.", "module.", "net.", "agg_state_dict.", "agg_state_dict"):
                if k.startswith(pref):
                    k = k[len(pref):]
            cleanedA[k] = v
        agg_model.load_state_dict(cleanedA, strict=False)
        agg_model.to(device)
        print(f"[bench] Loaded agg ckpt: {agg_ckpt}")
    else:
        print("[bench] WARN no agg_ckpt_path provided; using randomly initialized aggregator weights.")

    # ---- selector module (required if we need to build selection cache) ----
    em_node = cfg.get("em", {}) or {}

    sel_mod_node = (
        _oc_select(cfg, "em.selector_module")
        or _oc_select(cfg, "em.selector")
        or _oc_select(cfg, "selector_module")
        or _oc_select(cfg, "selector")
    )
    if sel_mod_node is None:
        raise KeyError(
            "[bench] Missing selector config. Provide cfg.em.selector_module (recommended) "
            "or cfg.em.selector / cfg.selector_module / cfg.selector (must contain _target_)."
        )
    selector_module = instantiate(sel_mod_node)

    # sel_version = stable hash on selector config (auto)
    sel_mod_container = OmegaConf.to_container(sel_mod_node, resolve=True)
    sel_hash = _stable_cfg_hash(sel_mod_container)
    sel_version = str(bench.get("sel_version", em_node.get("sel_version", f"sel_{sel_hash}")))
    if str(sel_version).lower() in ("", "none", "null", "auto"):
        sel_version = f"sel_{sel_hash}"

    # selection kmax must cover maxK
    if hasattr(selector_module, "cfg") and hasattr(selector_module.cfg, "kmax"):
        sel_kmax = int(selector_module.cfg.kmax)
    else:
        # fallback: use run.kmax
        sel_kmax = int(run_cfg.get("kmax", maxK))

    if sel_kmax < maxK:
        raise RuntimeError(
            f"[bench] selector.cfg.kmax={sel_kmax} < maxK={maxK}. "
            f"Set run.kmax={maxK} and bind em.selector_module.cfg.kmax=${{run.kmax}}, then rerun."
        )

    # ---- cheap model (only needed if we bootstrap cheap cache) ----
    cheap_model = None
    cheap_arch_cfg = em_node.get("cheap_model", cfg.get("cheap_model", None))
    cheap_version = str(em_node.get("cheap_version", cfg.get("cheap_version", "cheap_v0")))
    cheap_emb_dim = int(em_node.get("cheap_emb_dim", cfg.get("cheap_emb_dim", 64)))

    if cheap_arch_cfg is not None:
        cheap_arch = str(cheap_arch_cfg.get("arch", cheap_arch_cfg.get("name")))
        cheap_model = build_model(cheap_arch, cheap_arch_cfg, data_cfg=data_cfg).to(device)

        cheap_ckpt = _resolve_path(
            em_node.get("cheap_ckpt_path", cfg.get("cheap_ckpt_path", None)),
            orig_cwd,
        )
        if cheap_ckpt is not None and cheap_ckpt.exists():
            ckptC = torch.load(str(cheap_ckpt), map_location="cpu")
            sdC = ckptC.get("state_dict", ckptC) if isinstance(ckptC, dict) else ckptC
            cleanedC = {}
            for k, v in sdC.items():
                for pref in ("model.", "module.", "net."):
                    if k.startswith(pref):
                        k = k[len(pref):]
                cleanedC[k] = v
            cheap_model.load_state_dict(cleanedC, strict=False)
            cheap_model.to(device)
            print(f"[bench] Loaded cheap ckpt: {cheap_ckpt}")

    # ---- bootstrap caches if needed ----
    bootstrap = bench.get("bootstrap", {}) or {}
    bootstrap_enabled = bool(bootstrap.get("enabled", True))
    overwrite_all = bool(bootstrap.get("overwrite_all", False))
    skip_if_ready = bool(bootstrap.get("skip_if_ready", (not overwrite_all)))

    # If benchmark uses cached instance, we may need instance cache too
    need_inst_cache = (inst_mode == "cached")

    if bootstrap_enabled:
        need_cheap = overwrite_all or (not _stage_compatible(em_cache_root, bench_split, "cheap", cts_ds))
        need_sel = overwrite_all or (not _selection_compatible(
            em_cache_root, bench_split, cts_ds, sel_version=sel_version, cheap_version=cheap_version, min_kmax=maxK
        ))

        need_inst = False
        if need_inst_cache:
            inst_ok = _stage_compatible(em_cache_root, bench_split, "instance", cts_ds)
            need_inst = overwrite_all or (not inst_ok)

        # dependency closure
        if need_cheap:
            need_sel = True
            if need_inst_cache:
                need_inst = True
        if need_sel and need_inst_cache:
            need_inst = True

        if need_cheap or need_sel or need_inst:
            print(
                "[bench] BOOTSTRAP plan: "
                f"split={bench_split} overwrite_all={overwrite_all} skip_if_ready={skip_if_ready} | "
                f"need_cheap={need_cheap} need_sel={need_sel} need_inst={need_inst}"
            )

        # ---- cheap build ----
        if need_cheap:
            if cheap_model is None:
                raise RuntimeError(
                    "[bench] Need to build cheap cache but cheap_model is None. "
                    "Provide cfg.em.cheap_model (and optionally cfg.em.cheap_ckpt_path)."
                )
            cheap_cache_node = em_node.get("cheap_cache", {}) or {}
            cheap_cache_bs = int(cheap_cache_node.get("batch_size", 256))
            cheap_cache_nw = int(cheap_cache_node.get("num_workers", 0))
            cheap_cache_amp = bool(cheap_cache_node.get("amp", bool(device.type == "cuda")))
            cheap_cache_has_entropy = bool(cheap_cache_node.get("has_entropy", False))

            runner = CheapCacheRunner(
                data_cfg=data_cfg,
                dataset_cache_root=dataset_cache_root,
                em_cache_root=em_cache_root_str,
                device=str(device),
            )
            c = CheapCacheBuildConfig(
                splits=[bench_split],
                overwrite=bool(overwrite_all),
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

        # ---- selection build ----
        if need_sel:
            sel_node = em_node.get("selection_cache", {}) or {}
            sel_pair_batch_size = int(sel_node.get("pair_batch_size", 64))
            run_selection_cache(
                data_cfg=data_cfg,
                dataset_cache_root=dataset_cache_root,
                em_cache_root=em_cache_root_str,
                selector=selector_module,
                kmax=int(maxK),
                epoch=0,
                splits=[bench_split],
                overwrite=bool(overwrite_all),
                skip_if_ready=bool(skip_if_ready),
                sel_version=str(sel_version),
                pair_batch_size=int(sel_pair_batch_size),
            )

        # ---- instance build (optional) ----
        if need_inst_cache:
            inst_node = em_node.get("instance_cache", {}) or {}
            inst_cache_bs = int(inst_node.get("batch_size", 1024))
            inst_cache_nw = int(inst_node.get("num_workers", 0))
            inst_cache_use_amp = bool(inst_node.get("use_amp", bool(device.type == "cuda")))
            inst_cache_norm = bool(inst_node.get("normalize_emb", False))

            inst_emb_dim = int(em_node.get("inst_emb_dim", 384))
            inst_version = str(em_node.get("inst_version", "inst_v0"))

            if need_inst:
                run_instance_cache(
                    data_cfg=data_cfg,
                    dataset_cache_root=dataset_cache_root,
                    em_cache_root=em_cache_root_str,
                    instance_model=instance_model,
                    inst_version=str(inst_version),
                    emb_dim=int(inst_emb_dim),
                    epoch=0,
                    splits=[bench_split],
                    overwrite=bool(overwrite_all),
                    skip_if_ready=bool(skip_if_ready),
                    batch_size=int(inst_cache_bs),
                    num_workers=int(inst_cache_nw),
                    use_amp=bool(inst_cache_use_amp),
                    normalize_emb=bool(inst_cache_norm),
                )

    # After bootstrap, hard-check required metas
    if not _stage_compatible(em_cache_root, bench_split, "cheap", cts_ds):
        raise FileNotFoundError(f"[bench] cheap cache not ready for split={bench_split} under {em_cache_root}")
    if not _selection_compatible(em_cache_root, bench_split, cts_ds, sel_version=sel_version, cheap_version=cheap_version, min_kmax=maxK):
        raise FileNotFoundError(f"[bench] selection cache not ready/matched for split={bench_split} under {em_cache_root}")

    if inst_mode == "cached" and (not _stage_compatible(em_cache_root, bench_split, "instance", cts_ds)):
        raise FileNotFoundError(f"[bench] instance cache not ready for split={bench_split} under {em_cache_root}")

    # ---- build TokenProvider (split-specific) ----
    tp_cfg_node = cfg.get("token_provider", cfg.get("em", {}).get("token_provider", None))
    if tp_cfg_node is None:
        raise KeyError("[bench] missing token_provider config. Expected cfg.token_provider or cfg.em.token_provider")

    # Minimal policy for benchmark (we override plan anyway)
    policy_cfg = UpdatePolicyConfig(
        warmup_epochs=0,
        instance_mode=("cached" if inst_mode == "cached" else "online"),
        cheap_mode="cached",
    )
    policy = UpdatePolicy(policy_cfg)

    token_provider_cfg = TokenProviderConfig(
        policy=policy_cfg,
        assemble=tp_cfg_node.assemble,
        use_amp=bool(tp_cfg_node.get("use_amp", False)),
        normalize_tokens=bool(tp_cfg_node.get("normalize_tokens", False)),
        cache_missing=str(tp_cfg_node.get("cache_missing", "error")),
    )

    sel_meta = _load_json(_meta_path(em_cache_root, bench_split, "selection"))
    cheap_meta = _load_json(_meta_path(em_cache_root, bench_split, "cheap"))

    sel_version_used = str(sel_meta.get("sel_version", sel_version))
    cheap_version_used = str(sel_meta.get("cheap_version_used", cheap_meta.get("cheap_version", cheap_version)))

    inst_emb_dim = int(em_node.get("inst_emb_dim", 384))
    inst_version = str(em_node.get("inst_version", "inst_v0"))

    tp = TokenProvider(
        cfg=token_provider_cfg,
        policy=policy,
        instance_model=instance_model,
        device=device,
        em_cache_root=str(em_cache_root),
        split=bench_split,
        path_hash=str(sel_meta["path_hash"]),
        dataset_hash_key=str(sel_meta["dataset_hash_key"]),
        total_cts=int(len(cts_ds)),
        inst_emb_dim=int(inst_emb_dim),
        inst_version=str(inst_version),
        sel_version_used=str(sel_version_used),
        cheap_version_used=str(cheap_version_used),
        require_ready=bool(inst_mode == "cached"),  # cached-mode requires ready instance cache
    )

    # ---- build stable pair subset ----
    pair_ids = _make_or_load_pair_subset(eval_dir, bench_split, bench_pairs, bench_seed, total_pairs=int(len(pair_ds_full)))
    pair_ds = Subset(pair_ds_full, pair_ids)

    # ---- plan for benchmark ----
    # IMPORTANT:
    # - "online" includes expensive instance encoding for selected K, matching Fig3.
    # - disable writeback to avoid disk pollution (TokenProvider should ignore unknown keys safely).
    if inst_mode == "online":
        plan = {
            "train_instance": False,
            "use_instance_cache": False,
            "write_back": False,
            "train_cheap": False,
            "use_cheap_cache": True,
        }
    else:
        plan = {
            "train_instance": False,
            "use_instance_cache": True,
            "write_back": False,
            "train_cheap": False,
            "use_cheap_cache": True,
        }

    # ---- run grid ----
    out_csv = Path(bench.get("output_csv", str(eval_dir / "bench_compute_vs_k.csv")))
    if not out_csv.is_absolute():
        out_csv = run_dir / out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print("[bench] Figure3 Compute vs K")
    print(f"[bench] split={bench_split} instance_mode={inst_mode} pairs={len(pair_ids)} repeats={repeats}")
    print(f"[bench] K grid={k_list}")
    print(f"[bench] loader: batch_size={bench_batch_size} num_workers={bench_num_workers} drop_last={drop_last}")
    print(f"[bench] device={device} amp={amp}")
    print(f"[bench] output_csv={out_csv}")
    print("=" * 90 + "\n")

    rows: List[Dict[str, Any]] = []

    for K in k_list:
        # 每个 K 使用 PairBatchBuilderCPUConfig(kmax=K) -> 直接切片 selection 的 topK
        meta = _load_json(_meta_path(em_cache_root, bench_split, "selection"))
        cpu_builder = PairBatchBuilderCPU(
            cts_ds=cts_ds,
            em_cache_root=em_cache_root,
            split=bench_split,
            cfg=PairBatchBuilderCPUConfig(
                kmax=int(K),
                include_pos=bool(include_pos),
                include_esa=bool(include_esa),
                pin_memory=True,
            ),
            expected_path_hash=str(meta["path_hash"]),
            expected_dataset_hash_key=str(meta["dataset_hash_key"]),
        )

        loader = DataLoader(
            pair_ds,
            batch_size=int(bench_batch_size),
            shuffle=False,
            num_workers=int(bench_num_workers),
            pin_memory=True,
            persistent_workers=False,
            collate_fn=cpu_builder,
            drop_last=bool(drop_last),
        )

        for r in range(repeats):
            if device.type == "cuda":
                torch.cuda.empty_cache()

            m = _bench_loop(
                loader=loader,
                token_provider=tp,
                agg_model=agg_model,
                device=device,
                amp=amp,
                warmup_batches=warmup_batches,
                max_batches=max_batches,
                plan=plan,
            )

            row = {
                "split": bench_split,
                "instance_mode": inst_mode,
                "K": int(K),
                "repeat": int(r),
                "pairs_per_s": float(m["pairs_per_s"]),
                "elapsed_s": float(m["elapsed_s"]),
                "total_pairs": float(m["total_pairs"]),
                "timed_batches": float(m["timed_batches"]),
                "peak_alloc_gb": float(m["peak_alloc_gb"]),
                "peak_reserved_gb": float(m["peak_reserved_gb"]),
                "batch_size": int(bench_batch_size),
                "num_workers": int(bench_num_workers),
                "amp": bool(amp),
                "device": str(device),
                "sel_version": str(sel_version_used),
                "cheap_version_used": str(cheap_version_used),
            }
            rows.append(row)

            print(
                f"[bench] K={K:<4d} rep={r}  "
                f"pairs/s={row['pairs_per_s']:.2f}  "
                f"VRAM(alloc)={row['peak_alloc_gb']:.2f}GB  "
                f"VRAM(resv)={row['peak_reserved_gb']:.2f}GB  "
                f"elapsed={row['elapsed_s']:.2f}s"
            )

    # ---- write CSV ----
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"\n[bench] Done. CSV saved to: {out_csv}")

    # optional summary
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        agg = df.groupby("K")[["pairs_per_s", "peak_alloc_gb", "peak_reserved_gb"]].agg(["mean", "std"])
        print("\n[bench] Summary (mean±std):")
        print(agg)
    except Exception:
        pass


if __name__ == "__main__":
    main()
