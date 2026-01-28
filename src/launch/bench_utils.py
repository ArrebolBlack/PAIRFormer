# src/launch/bench_utils.py
from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union, Mapping

import numpy as np
import torch


# -----------------------------
# Basic utilities
# -----------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_gpu_name() -> str:
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception:
            return "cuda"
    return "cpu"


def ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_iso() -> str:
    # stable timestamp for filenames/logging
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def to_gb(num_bytes: int) -> float:
    return float(num_bytes) / (1024.0 ** 3)


# -----------------------------
# Timing (segment + total)
# -----------------------------

class SegmentTimer:
    """
    A lightweight segment timer that supports:
      - CUDA sync timing for correctness (optional)
      - nested usage via context manager

    Example:
      timer = SegmentTimer(sync_cuda=True)
      with timer("t_cheap_n"):
          ...
      with timer("t_select"):
          ...
      total = timer.total_time()
      seg = timer.segments()
    """
    def __init__(self, *, sync_cuda: bool = True) -> None:
        self.sync_cuda = sync_cuda and torch.cuda.is_available()
        self._acc: Dict[str, float] = {}
        self._stack: List[Tuple[str, float]] = []

    def _sync(self) -> None:
        if self.sync_cuda:
            torch.cuda.synchronize()

    def __call__(self, name: str):
        return _SegmentCtx(self, name)

    def _start(self, name: str) -> None:
        self._sync()
        self._stack.append((name, time.perf_counter()))

    def _stop(self, name: str) -> None:
        self._sync()
        if not self._stack:
            raise RuntimeError(f"SegmentTimer.stop({name}) called without start")
        last_name, t0 = self._stack.pop()
        if last_name != name:
            raise RuntimeError(f"SegmentTimer segment mismatch: stop({name}) but last start was {last_name}")
        dt = time.perf_counter() - t0
        self._acc[name] = self._acc.get(name, 0.0) + dt

    def reset(self) -> None:
        self._acc.clear()
        self._stack.clear()

    def segments(self) -> Dict[str, float]:
        return dict(self._acc)

    def total_time(self) -> float:
        # total of recorded segments (not necessarily equals wall time if user doesn't wrap everything)
        return float(sum(self._acc.values()))


class _SegmentCtx:
    def __init__(self, timer: SegmentTimer, name: str) -> None:
        self.timer = timer
        self.name = name

    def __enter__(self):
        self.timer._start(self.name)

    def __exit__(self, exc_type, exc, tb):
        self.timer._stop(self.name)
        return False


# -----------------------------
# Subset selection
# -----------------------------

def _read_jsonl_or_json(path: Path) -> Any:
    s = path.read_text(encoding="utf-8").strip()
    if not s:
        return None
    if s[0] == "{":
        return json.loads(s)
    # allow jsonl of integers
    out = []
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _flatten_int_list(obj: Any) -> List[int]:
    """
    Accept:
      - [1,2,3]
      - [[1,2,3]]  (common mistake)
      - [[1,2],[3,4]] (jsonl-style aggregation)
    Return: flat List[int]
    """
    if isinstance(obj, dict) and "pair_ids" in obj:
        obj = obj["pair_ids"]

    if not isinstance(obj, list):
        raise RuntimeError(f"pair_ids must be list/dict, got {type(obj)}")

    # list-of-lists -> flatten
    if len(obj) > 0 and isinstance(obj[0], list):
        flat: List[int] = []
        for sub in obj:
            if not isinstance(sub, list):
                raise RuntimeError("mixed list structure in pair_ids file")
            flat.extend(sub)
        obj = flat

    return [int(x) for x in obj]


def _infer_num_pairs(pair_index: Any) -> Optional[int]:
    """
    Best-effort infer total number of pairs.
    Priority:
      1) dict/object with pair_offsets: len(pair_offsets)-1
      2) attribute num_pairs
      3) __len__ (only if looks like a real PairIndex object, not dict)
    """
    # dict with pair_offsets
    if isinstance(pair_index, dict) and "pair_offsets" in pair_index:
        off = pair_index["pair_offsets"]
        if torch.is_tensor(off):
            return int(off.numel()) - 1
        return int(len(off)) - 1

    # object with pair_offsets
    if hasattr(pair_index, "pair_offsets") and getattr(pair_index, "pair_offsets") is not None:
        off = getattr(pair_index, "pair_offsets")
        if torch.is_tensor(off):
            return int(off.numel()) - 1
        return int(len(off)) - 1

    # object with num_pairs
    if hasattr(pair_index, "num_pairs"):
        try:
            return int(getattr(pair_index, "num_pairs"))
        except Exception:
            pass

    # avoid len(dict) trap
    if isinstance(pair_index, dict):
        return None

    # fallback
    try:
        return int(len(pair_index))
    except Exception:
        return None


def select_pair_ids_subset(
    pair_index: Any,
    *,
    num_pairs_subset: int,
    seed: int,
    out_path: Optional[Union[str, Path]] = None,
    allow_overwrite: bool = False,
) -> List[int]:
    """
    Returns a list of pair_ids in [0, num_pairs).
    Robust to old/bad json formats.
    """
    out_path_p = Path(out_path) if out_path is not None else None

    num_pairs_total = _infer_num_pairs(pair_index)

    # 1) reuse existing file (if allowed)
    if out_path_p is not None and out_path_p.exists() and not allow_overwrite:
        obj = _read_jsonl_or_json(out_path_p)
        subset = _flatten_int_list(obj)

        # range check if we can infer num_pairs
        if num_pairs_total is not None:
            bad = [x for x in subset if x < 0 or x >= num_pairs_total]
            if bad:
                raise RuntimeError(
                    f"pair_ids file contains out-of-range ids (likely CTS uids). "
                    f"num_pairs={num_pairs_total}, example_bad={bad[:5]}, file={out_path_p}\n"
                    f"Fix: delete this file or rerun with allow_overwrite=True, and pass an ABSOLUTE path."
                )

        return subset

    # 2) generate new subset
    if num_pairs_total is None:
        raise RuntimeError(
            "Cannot infer num_pairs from pair_index. Expected dict/object with pair_offsets "
            "or object with __len__/num_pairs."
        )

    if num_pairs_subset > num_pairs_total:
        raise RuntimeError(f"num_pairs_subset={num_pairs_subset} > available pairs={num_pairs_total}")

    all_ids: Sequence[int] = list(range(num_pairs_total))
    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(num_pairs_total)
    subset = [int(all_ids[i]) for i in perm[:num_pairs_subset]]

    if out_path_p is not None:
        _write_json(out_path_p, {"seed": int(seed), "num_pairs_subset": int(num_pairs_subset), "pair_ids": subset})

    return subset

# -----------------------------
# Batch size accounting
# -----------------------------

def infer_num_pairs_in_batch(batch: Any) -> int:
    """
    Try hard to infer how many pairs are in a batch.

    Supported patterns:
      - dict with 'pair_ids' / 'pair_id' / 'pair_idx' as 1D tensor/list
      - dict with 'pair_batch' or 'pairs' list
      - object with attribute 'pair_ids'
    """
    if batch is None:
        return 0

    # dict-like
    if isinstance(batch, Mapping):
        for k in ("pair_ids", "pair_id", "pair_idx", "pair_indices"):
            if k in batch:
                v = batch[k]
                if isinstance(v, torch.Tensor):
                    return int(v.numel())
                if isinstance(v, (list, tuple)):
                    return int(len(v))
        for k in ("pairs", "pair_batch"):
            if k in batch:
                v = batch[k]
                if isinstance(v, (list, tuple)):
                    return int(len(v))
                if isinstance(v, torch.Tensor):
                    # ambiguous; assume first dim is pairs
                    return int(v.shape[0])

    # attribute-like
    if hasattr(batch, "pair_ids"):
        v = getattr(batch, "pair_ids")
        if isinstance(v, torch.Tensor):
            return int(v.numel())
        if isinstance(v, (list, tuple)):
            return int(len(v))

    # last resort: if batch has a tensor field, use leading dim
    if isinstance(batch, Mapping):
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.ndim >= 1:
                return int(v.shape[0])

    raise RuntimeError("Cannot infer num_pairs in batch; please add 'pair_ids' to batch dict.")


# -----------------------------
# Benchmark output schema
# -----------------------------

@dataclass
class BenchRecord:
    pipeline: str
    split: str
    K: int
    repeat_id: int

    seed: int
    num_pairs_subset: int
    batch_size: int
    num_workers: int
    amp: bool
    gpu_name: str

    # dataset stats on subset (optional but strongly recommended)
    avg_n_full: float = float("nan")
    median_n_full: float = float("nan")
    p95_n_full: float = float("nan")

    # timing (seconds): mean/std over measured iters
    t_total_mean: float = float("nan")
    t_total_std: float = float("nan")

    # throughput
    throughput_pairs_per_s_mean: float = float("nan")
    throughput_pairs_per_s_std: float = float("nan")

    # peak memory
    peak_vram_gb_mean: float = float("nan")
    peak_vram_gb_std: float = float("nan")

    # segment times (optional; pipeline-dependent)
    # store as json string for flexibility
    segment_time_mean_json: str = "{}"
    segment_time_std_json: str = "{}"

    timestamp: str = ""


def append_records_to_csv(path: Union[str, Path], records: List[BenchRecord]) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    rows = []
    for r in records:
        if is_dataclass(r):
            rows.append(asdict(r))
        elif isinstance(r, Mapping):
            rows.append(dict(r))
        elif hasattr(r, "__dict__"):
            rows.append(dict(vars(r)))
        else:
            rows.append({"record": str(r)})
    if not rows:
        return
    fieldnames = list(rows[0].keys())

    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# -----------------------------
# Core benchmark loop
# -----------------------------

@dataclass
class BenchConfig:
    seed: int = 2020
    warmup_iters: int = 20
    measure_iters: int = 100
    repeats: int = 3
    amp: bool = True
    sync_cuda: bool = True  # enforce cuda sync around timing
    device: str = "cuda"

    # memory stats
    reset_peak_mem_after_warmup: bool = True


@dataclass
class BenchStats:
    # per-iter totals
    t_total_list: List[float]
    # per-iter segment dict
    seg_list: List[Dict[str, float]]
    # per-iter pairs processed
    pairs_list: List[int]
    # peak memory for this run
    peak_vram_bytes: int


def _mean_std(xs: Sequence[float]) -> Tuple[float, float]:
    if len(xs) == 0:
        return float("nan"), float("nan")
    arr = np.asarray(xs, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def _merge_segment_stats(seg_list: List[Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    seg_list: list of dict(name -> seconds) per iter
    Returns mean/std dicts over iters for each key (missing treated as 0).
    """
    keys = sorted({k for d in seg_list for k in d.keys()})
    if not keys:
        return {}, {}
    mat = {k: [] for k in keys}
    for d in seg_list:
        for k in keys:
            mat[k].append(float(d.get(k, 0.0)))
    mean = {k: _mean_std(v)[0] for k, v in mat.items()}
    std = {k: _mean_std(v)[1] for k, v in mat.items()}
    return mean, std




def _infinite_loader(loader: Iterable[Any]) -> Iterator[Any]:
    while True:
        for batch in loader:
            yield batch


def run_benchmark_once(
    *,
    loader: Iterable[Any],
    forward_fn,
    bench_cfg: BenchConfig,
    amp_dtype: Optional[torch.dtype] = None,
) -> BenchStats:
    """
    forward_fn(batch, timer: SegmentTimer) -> Any
    Total time now INCLUDES dataloader fetch (next batch).
    Also records 't_next_batch' in segment dict.
    """
    device = bench_cfg.device
    use_cuda = (device.startswith("cuda") and torch.cuda.is_available())
    sync_cuda = bench_cfg.sync_cuda and use_cuda

    amp_enabled = bool(bench_cfg.amp and use_cuda)
    if amp_dtype is None:
        amp_dtype = torch.float16

    def _sync():
        if sync_cuda:
            torch.cuda.synchronize()

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    it = _infinite_loader(loader)

    # warmup (include fetch + forward, but don't record)
    for _ in range(int(bench_cfg.warmup_iters)):
        batch = next(it)
        timer = SegmentTimer(sync_cuda=sync_cuda)
        with torch.inference_mode():
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    _ = forward_fn(batch, timer)
            else:
                _ = forward_fn(batch, timer)

    if use_cuda and bench_cfg.reset_peak_mem_after_warmup:
        torch.cuda.reset_peak_memory_stats()

    t_total_list: List[float] = []
    seg_list: List[Dict[str, float]] = []
    pairs_list: List[int] = []

    for _ in range(int(bench_cfg.measure_iters)):
        # ---- include dataloader fetch time ----
        t0 = time.perf_counter()
        batch = next(it)
        t_fetch = time.perf_counter() - t0

        n_pairs = infer_num_pairs_in_batch(batch)
        timer = SegmentTimer(sync_cuda=sync_cuda)

        # forward timing
        _sync()
        t1 = time.perf_counter()

        with torch.inference_mode():
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    _ = forward_fn(batch, timer)
            else:
                _ = forward_fn(batch, timer)

        _sync()
        t2 = time.perf_counter()

        t_total_list.append(float((t2 - t0)))  # fetch + forward (+ sync)
        seg = timer.segments()
        seg["t_next_batch"] = float(t_fetch)
        seg_list.append(seg)
        pairs_list.append(int(n_pairs))

    peak = int(torch.cuda.max_memory_allocated()) if use_cuda else 0

    return BenchStats(
        t_total_list=t_total_list,
        seg_list=seg_list,
        pairs_list=pairs_list,
        peak_vram_bytes=peak,
    )


def summarize_repeats_to_records(
    *,
    pipeline: str,
    split: str,
    K: int,
    repeat_stats: List[BenchStats],
    seed: int,
    num_pairs_subset: int,
    batch_size: int,
    num_workers: int,
    amp: bool,
    avg_n_full: float,
    median_n_full: float,
    p95_n_full: float,
    repeat_id_base: int = 0,
) -> List[BenchRecord]:
    """
    Produces one BenchRecord per repeat (keep raw repeat-level results),
    so later plotting can compute mean±std across repeats, or you can
    aggregate here if you prefer.
    """
    gpu_name = get_gpu_name()
    records: List[BenchRecord] = []

    for j, st in enumerate(repeat_stats):
        t_mean, t_std = _mean_std(st.t_total_list)

        # throughput per iter: pairs / time
        per_iter_tp = []
        for pairs, dt in zip(st.pairs_list, st.t_total_list):
            if dt <= 0:
                continue
            per_iter_tp.append(float(pairs) / float(dt))
        tp_mean, tp_std = _mean_std(per_iter_tp)

        peak_gb = to_gb(st.peak_vram_bytes)

        seg_mean, seg_std = _merge_segment_stats(st.seg_list)

        rec = BenchRecord(
            pipeline=pipeline,
            split=split,
            K=int(K),
            repeat_id=int(repeat_id_base + j),
            seed=int(seed),
            num_pairs_subset=int(num_pairs_subset),
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            amp=bool(amp),
            gpu_name=gpu_name,
            avg_n_full=float(avg_n_full),
            median_n_full=float(median_n_full),
            p95_n_full=float(p95_n_full),
            t_total_mean=float(t_mean),
            t_total_std=float(t_std),
            throughput_pairs_per_s_mean=float(tp_mean),
            throughput_pairs_per_s_std=float(tp_std),
            peak_vram_gb_mean=float(peak_gb),
            peak_vram_gb_std=0.0,  # per-repeat single value; you can fill later if you aggregate
            segment_time_mean_json=json.dumps(seg_mean, ensure_ascii=False),
            segment_time_std_json=json.dumps(seg_std, ensure_ascii=False),
            timestamp=now_iso(),
        )
        records.append(rec)

    return records


def compute_n_full_stats(
    n_full_list: Sequence[int],
) -> Tuple[float, float, float]:
    """
    n_full_list: list of 'n' (CTS count) per pair in subset.
    Returns (mean, median, p95).
    """
    if not n_full_list:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(n_full_list, dtype=np.float64)
    mean = float(arr.mean())
    median = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    return mean, median, p95
