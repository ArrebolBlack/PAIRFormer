# src/launch/inference_targetnet_like_online.py
from __future__ import annotations

import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import os
import json
import hashlib
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable, Iterator

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.config.data_config import DataConfig
from src.data.dataset import ChunkedCTSDataset
from src.data.pair_dataset_dynamic import DynamicPairDataset
from src.data.pair_batch_builder_cpu import PairBatchBuilderCPU, PairBatchBuilderCPUConfig
from src.models.registry import build_model
from src.models.extractors import get_embedding_and_logit

from src.launch.bench_utils import (
    set_seeds,
    ensure_dir,
    select_pair_ids_subset,
    compute_n_full_stats,
    BenchConfig,
    run_benchmark_once,
    summarize_repeats_to_records,
    append_records_to_csv,
)


"""
推荐运行（你的 config 里默认已有 test split）：

python -m src.launch.inference_targetnet_like_online \
  experiment=miRAW_EM_Pipeline \
  run.kmax=32 \
  run.batch_size=64 \
  run.num_workers=4 \
  +bench.num_pairs_subset=1024 \
  +bench.measure_iters=100 \
  +bench.warmup_iters=20 \
  +bench.repeats=3 \
  +bench.amp=true
"""


# -----------------------------
# Helpers
# -----------------------------

def _as_path_from_orig(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (Path(get_original_cwd()) / pp).resolve()


def _cfg_get(cfg: DictConfig, key: str, default: Any) -> Any:
    try:
        v: Any = cfg
        for k in key.split("."):
            v = v[k]
        return v
    except Exception:
        return default


def _resolve_cache_root(cfg: DictConfig) -> Path:
    """
    Use your real config fields:
      - cfg.run.cache_path (default: cfg.paths.cache_root)
      - relative paths resolved under original_cwd
    """
    default_cache = str(_cfg_get(cfg, "paths.cache_root", "cache"))
    cache_root_cfg = str(_cfg_get(cfg, "run.cache_path", default_cache))
    return _as_path_from_orig(cache_root_cfg)


def _resolve_em_cache_root(cfg: DictConfig, fallback_cache_root: Path) -> Path:
    """
    Your config has em_cache_root at _global_:
      em_cache_root: "${paths.cache_root}"
    """
    em_cache_cfg = _cfg_get(cfg, "em_cache_root", None)
    if em_cache_cfg is None:
        return fallback_cache_root
    return _as_path_from_orig(str(em_cache_cfg))


def _compute_path_hash(data_cfg: DataConfig, split: str) -> str:
    # alignment may exist; otherwise keep consistent default
    data_file_path = str(data_cfg.get_path(split))
    alignment = getattr(data_cfg, "alignment", "extended_seed_alignment")
    hash_key = f"{data_file_path}|{alignment}"
    return hashlib.md5(hash_key.encode("utf-8")).hexdigest()[:8]


class PairIdListDataset(torch.utils.data.Dataset):
    def __init__(self, pair_ids: List[int]):
        self.pair_ids = [int(x) for x in pair_ids]

    def __len__(self) -> int:
        return len(self.pair_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"pair_id": int(self.pair_ids[idx])}


class InfiniteLoader:
    """
    Wrap a finite DataLoader into an infinite iterable:
      - avoids StopIteration when warmup+measure iters exceed num_batches.
    """
    def __init__(self, loader: Iterable[Any]):
        self.loader = loader

    def __iter__(self) -> Iterator[Any]:
        while True:
            for b in self.loader:
                yield b


def _build_pair_batch_builder(
    *,
    cts_ds: ChunkedCTSDataset,
    em_cache_root: Path,
    split: str,
    kmax: int,
    pin_memory: bool,
    include_pos: bool,
    include_esa: bool,
    expected_path_hash: Optional[str],
    expected_dataset_hash_key: Optional[str],
) -> Any:
    """
    Robust builder: adapt to your actual PairBatchBuilderCPU signature at runtime.
    """
    # 1) build cfg (PairBatchBuilderCPUConfig may have different field names; try best)
    cfg_kwargs = {}
    cfg_sig = inspect.signature(PairBatchBuilderCPUConfig)
    for name, val in {
        "kmax": kmax,
        "pin_memory": pin_memory,
        "include_pos": include_pos,
        "include_esa": include_esa,
    }.items():
        if name in cfg_sig.parameters:
            cfg_kwargs[name] = val
    bb_cfg = PairBatchBuilderCPUConfig(**cfg_kwargs)

    # 2) construct PairBatchBuilderCPU with introspection
    sig = inspect.signature(PairBatchBuilderCPU)
    kwargs: Dict[str, Any] = {}
    if "cts_ds" in sig.parameters:
        kwargs["cts_ds"] = cts_ds
    if "em_cache_root" in sig.parameters:
        kwargs["em_cache_root"] = str(em_cache_root)
    if "split" in sig.parameters:
        kwargs["split"] = str(split)
    if "cfg" in sig.parameters:
        kwargs["cfg"] = bb_cfg
    if "expected_path_hash" in sig.parameters and expected_path_hash is not None:
        kwargs["expected_path_hash"] = str(expected_path_hash)
    if "expected_dataset_hash_key" in sig.parameters and expected_dataset_hash_key is not None:
        kwargs["expected_dataset_hash_key"] = str(expected_dataset_hash_key)

    # Some variants may accept path_hash/dataset_hash_key directly
    if "path_hash" in sig.parameters and expected_path_hash is not None:
        kwargs["path_hash"] = str(expected_path_hash)
    if "dataset_hash_key" in sig.parameters and expected_dataset_hash_key is not None:
        kwargs["dataset_hash_key"] = str(expected_dataset_hash_key)

    # fallback: try minimal required args
    try:
        return PairBatchBuilderCPU(**kwargs)
    except TypeError:
        # progressively remove identity args if mismatch
        for drop in ["expected_dataset_hash_key", "dataset_hash_key", "expected_path_hash", "path_hash"]:
            if drop in kwargs:
                kwargs.pop(drop, None)
                try:
                    return PairBatchBuilderCPU(**kwargs)
                except TypeError:
                    pass
        # final: try positional minimal
        return PairBatchBuilderCPU(cts_ds, str(em_cache_root), str(split), bb_cfg)


def _resolve_split(cfg: DictConfig) -> str:
    # priority:
    #  1) bench.split (if user adds +bench.split=xxx)
    #  2) run.test_splits[0] (your config has ["test"])
    #  3) run.split
    #  4) "test"
    bsplit = _cfg_get(cfg, "bench.split", None)
    if bsplit is not None:
        return str(bsplit)

    test_splits = _cfg_get(cfg, "run.test_splits", None)
    if isinstance(test_splits, (list, tuple)) and len(test_splits) > 0:
        return str(test_splits[0])

    rs = _cfg_get(cfg, "run.split", None)
    if rs is not None:
        return str(rs)
    return "test"


# -----------------------------
# Main
# -----------------------------

@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # -------- config (use your real fields; bench.* is optional override) --------
    seed = int(_cfg_get(cfg, "seed", 2020))
    device_str = str(_cfg_get(cfg, "device", "cuda"))
    device = torch.device(device_str if (device_str != "cuda" or torch.cuda.is_available()) else "cpu")

    split = _resolve_split(cfg)

    K = int(_cfg_get(cfg, "run.kmax", 64))
    batch_size = int(_cfg_get(cfg, "run.batch_size", 64))
    num_workers = int(_cfg_get(cfg, "run.num_workers", 0))
    pin_memory = bool(_cfg_get(cfg, "run.pin_memory", device.type == "cuda"))

    include_pos = bool(_cfg_get(cfg, "run.include_pos", False))
    include_esa = bool(_cfg_get(cfg, "run.include_esa", False))

    # benchmark-only knobs (not in your experiment yaml; allow +bench.xxx overrides)
    num_pairs_subset = int(_cfg_get(cfg, "bench.num_pairs_subset", 1024))
    warmup_iters = int(_cfg_get(cfg, "bench.warmup_iters", 20))
    measure_iters = int(_cfg_get(cfg, "bench.measure_iters", 100))
    repeats = int(_cfg_get(cfg, "bench.repeats", 3))

    # amp default: prefer your em.instance_cache.use_amp if present
    amp_default = bool(_cfg_get(cfg, "em.instance_cache.use_amp", True))
    use_amp = bool(_cfg_get(cfg, "bench.amp", amp_default))

    set_seeds(seed)

    # -------- resolve paths --------
    cache_root = _resolve_cache_root(cfg)
    em_cache_root = _resolve_em_cache_root(cfg, cache_root)
    ensure_dir(cache_root)
    ensure_dir(em_cache_root)

    out_dir = Path(os.getcwd())  # hydra run dir
    subset_path = Path(str(_cfg_get(cfg, "bench.pair_ids_path", out_dir / f"pair_ids_{split}_{num_pairs_subset}.json")))
    out_csv = Path(str(_cfg_get(cfg, "bench.out_csv", out_dir / "fig3_bench.csv")))

    # -------- build DataConfig + CTS dataset --------
    if hasattr(DataConfig, "from_omegaconf"):
        data_cfg = DataConfig.from_omegaconf(cfg.data)  # type: ignore
    else:
        data_cfg = DataConfig(**OmegaConf.to_container(cfg.data, resolve=True))  # type: ignore

    cts_ds = ChunkedCTSDataset(str(cache_root), data_cfg, split)

    # -------- pair subset --------
    all_pairs_ds = DynamicPairDataset(cts_ds, drop_empty_pairs=True)
    pair_ids = select_pair_ids_subset(
        all_pairs_ds,
        num_pairs_subset=num_pairs_subset,
        seed=seed,
        out_path=subset_path,
        allow_overwrite=False,
    )

    n_full_list = [int(cts_ds.get_pair_num_cts(int(pid))) for pid in pair_ids]
    avg_n, median_n, p95_n = compute_n_full_stats(n_full_list)

    # -------- DataLoader (selection memmap is used by PairBatchBuilderCPU) --------
    expected_path_hash = _compute_path_hash(data_cfg, split)
    expected_dataset_hash_key = getattr(cts_ds, "dataset_hash_key", None)

    batch_builder = _build_pair_batch_builder(
        cts_ds=cts_ds,
        em_cache_root=em_cache_root,
        split=split,
        kmax=K,
        pin_memory=pin_memory,
        include_pos=include_pos,
        include_esa=include_esa,
        expected_path_hash=expected_path_hash,
        expected_dataset_hash_key=str(expected_dataset_hash_key) if expected_dataset_hash_key is not None else None,
    )

    subset_ds = PairIdListDataset(pair_ids)

    loader_kwargs: Dict[str, Any] = dict(
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
        collate_fn=batch_builder,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(_cfg_get(cfg, "bench.prefetch_factor", 2))

    finite_loader = DataLoader(subset_ds, **loader_kwargs)
    loader = InfiniteLoader(finite_loader)

    # -------- instance model (your config uses _global_.instance_model + instance_ckpt_path) --------
    inst_cfg = _cfg_get(cfg, "instance_model", None)
    if inst_cfg is None:
        raise RuntimeError("Cannot find instance_model in config (expected _global_.instance_model).")

    instance_model = build_model(inst_cfg)
    instance_model.to(device)
    instance_model.eval()

    # ckpt path from your experiment yaml
    ckpt_path = str(_cfg_get(cfg, "instance_ckpt_path", ""))
    if ckpt_path:
        p = _as_path_from_orig(ckpt_path)
        obj = torch.load(str(p), map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "state_dict" in obj:
            sd = obj["state_dict"]
        elif isinstance(obj, dict) and "model" in obj:
            sd = obj["model"]
        else:
            sd = obj
        missing, unexpected = instance_model.load_state_dict(sd, strict=False)
        print(f"[Bench] Loaded instance ckpt: {p}")
        if missing:
            print(f"[Bench] Warning: missing keys (first20): {missing[:20]}")
        if unexpected:
            print(f"[Bench] Warning: unexpected keys (first20): {unexpected[:20]}")

    # -------- forward_fn (strict online instance) --------
    def forward_fn(batch_cpu: Dict[str, Any], timer) -> Dict[str, Any]:
        pair_id = batch_cpu["pair_id"].to(device, non_blocking=True)
        y_pair = batch_cpu["y_pair"].to(device, non_blocking=True)
        mask = batch_cpu["mask"].to(device, non_blocking=True)

        X_cpu = batch_cpu.get("X", None)
        if X_cpu is None:
            return {"pair_id": pair_id, "y_pair": y_pair, "mask": mask, "pair_logit": None}

        with timer("t_h2d"):
            X = X_cpu.to(device, non_blocking=True)  # uint8 [B,K,C,L]
            B, K2, C, L = X.shape
            X_flat = X.view(B * K2, C, L)
            mask_flat = mask.view(B * K2)

            valid = torch.nonzero(mask_flat, as_tuple=False).view(-1)
            if valid.numel() == 0:
                return {"pair_id": pair_id, "y_pair": y_pair, "mask": mask, "pair_logit": None}

            X_valid = X_flat.index_select(0, valid).to(dtype=torch.float32)

        with timer("t_inst_k"):
            feat_valid, logit_valid = get_embedding_and_logit(instance_model, X_valid)
            if feat_valid is None or logit_valid is None:
                raise RuntimeError("[Bench] extractor returned None.")

        with timer("t_pool"):
            log_fill = torch.full((B * K2,), -1e9, device=device, dtype=logit_valid.dtype)
            log_fill.index_copy_(0, valid, logit_valid.view(-1))
            inst_logit = log_fill.view(B, K2)
            pair_logit = inst_logit.max(dim=1).values  # [B]

        return {"pair_id": pair_id, "y_pair": y_pair, "mask": mask, "pair_logit": pair_logit}

    # -------- benchmark --------
    bench_cfg = BenchConfig(
        seed=int(seed),
        warmup_iters=int(warmup_iters),
        measure_iters=int(measure_iters),
        repeats=int(repeats),
        amp=bool(use_amp),
        sync_cuda=True,
        device=str(device),
        reset_peak_mem_after_warmup=True,
    )

    repeat_stats = []
    for _ in range(int(repeats)):
        st = run_benchmark_once(loader=loader, forward_fn=forward_fn, bench_cfg=bench_cfg)
        repeat_stats.append(st)

    records = summarize_repeats_to_records(
        pipeline="targetnet_like_online",
        split=str(split),
        K=int(K),
        repeat_stats=repeat_stats,
        seed=int(seed),
        num_pairs_subset=int(num_pairs_subset),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        amp=bool(use_amp),
        avg_n_full=float(avg_n),
        median_n_full=float(median_n),
        p95_n_full=float(p95_n),
    )

    append_records_to_csv(out_csv, records)
    print(f"[Bench] wrote CSV: {out_csv}")
    print(f"[Bench] subset pair_ids: {subset_path}")
    print(f"[Bench] n_full stats on subset: mean={avg_n:.2f} median={median_n:.2f} p95={p95_n:.2f}")


if __name__ == "__main__":
    main()
