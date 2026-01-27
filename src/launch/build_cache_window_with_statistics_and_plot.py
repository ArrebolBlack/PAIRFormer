# src/launch/build_cache_window_with_statistics_and_plot.py
from __future__ import annotations

import os
import json
import gc
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

# headless-safe matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config.data_config import DataConfig
from src.data.builder import build_dataset_and_loader
from src.utils import set_seeds


# ----------------------------
# cache root resolve
# ----------------------------
def _resolve_cache_root(cfg: DictConfig) -> str:
    """
    复刻 train.py 的 cache_root 解析逻辑：
    - 优先 run.cache_path
    - 否则 paths.cache_root
    - 否则 "cache"
    相对路径相对于 original_cwd
    """
    orig_cwd = Path(get_original_cwd())

    if "paths" in cfg:
        default_cache = cfg.paths.get("cache_root", "cache")
    else:
        default_cache = "cache"

    cache_root_cfg = cfg.run.get("cache_path", default_cache)
    cache_root = Path(cache_root_cfg)
    if not cache_root.is_absolute():
        cache_root = orig_cwd / cache_root
    return str(cache_root)


def _path_hash_for_split(data_cfg: DataConfig, split: str) -> str:
    """
    与 ChunkedCTSDataset 里一致的 path_hash 逻辑：data_path + alignment
    """
    data_file_path = str(data_cfg.get_path(split))
    alignment = getattr(data_cfg, "alignment", "extended_seed_alignment")
    hash_key = f"{data_file_path}|{alignment}"
    return hashlib.md5(hash_key.encode("utf-8")).hexdigest()[:8]


# ----------------------------
# fallback scan when PairIndex missing
# ----------------------------
def _scan_counts_by_blocks(ds) -> np.ndarray:
    """
    fallback：当 PairIndex 不存在时，通过扫描每个 block 的 set_idxs 统计每个 pair 的 CTS 数量。
    注意：这会额外遍历所有 blocks（I/O 重），但不需要逐样本 __getitem__。
    """
    counts = np.zeros(1024, dtype=np.int64)  # 动态扩容
    max_pid = -1
    n_blocks = len(getattr(ds, "chunk_files", []))
    if n_blocks == 0:
        return np.zeros(0, dtype=np.int64)

    for bi, p in enumerate(ds.chunk_files):
        obj = torch.load(p, map_location="cpu", weights_only=False)
        set_idxs = obj["set_idxs"].view(-1).to(dtype=torch.long).cpu().numpy()
        if set_idxs.size == 0:
            del obj
            continue

        block_max = int(set_idxs.max())
        if block_max > max_pid:
            max_pid = block_max

        need = max_pid + 1
        if need > counts.size:
            new_size = counts.size
            while new_size < need:
                new_size *= 2
            counts = np.pad(counts, (0, new_size - counts.size), mode="constant")

        bc = np.bincount(set_idxs, minlength=need).astype(np.int64)
        counts[:need] += bc

        del obj
        if (bi + 1) % 20 == 0:
            gc.collect()
            print(f"[PairStats][scan] scanned blocks {bi+1}/{n_blocks} ...")

    if max_pid < 0:
        return np.zeros(0, dtype=np.int64)
    return counts[: max_pid + 1]


def _get_num_cts_per_pair(ds) -> Tuple[np.ndarray, str]:
    """
    返回 (counts_np, source)
    source 表示 counts 来源：pair_counts / pair_offsets / scan_blocks
    """
    pair_counts = getattr(ds, "pair_counts", None)
    if pair_counts is not None:
        pc = pair_counts.view(-1).to(dtype=torch.long).cpu().numpy()
        return pc, "pair_counts"

    pair_offsets = getattr(ds, "pair_offsets", None)
    if pair_offsets is not None:
        po = pair_offsets.view(-1).to(dtype=torch.long)
        if po.numel() >= 2:
            pc = (po[1:] - po[:-1]).cpu().numpy()
            return pc, "pair_offsets"

    pc = _scan_counts_by_blocks(ds)
    return pc, "scan_blocks"


# ----------------------------
# stats
# ----------------------------
def _summarize_counts(counts: np.ndarray) -> Dict[str, Any]:
    counts = counts.astype(np.int64)
    if counts.size == 0:
        return {
            "n_pairs": 0,
            "sum_cts": 0,
            "mean": None,
            "var": None,
            "std": None,
            "min": None,
            "max": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "p99": None,
        }

    p05, p25, p50, p75, p95, p99 = np.percentile(counts, [5, 25, 50, 75, 95, 99])

    mean = float(counts.mean())
    var = float(counts.var(ddof=0))   # population variance
    std = float(counts.std(ddof=0))

    return {
        "n_pairs": int(counts.size),
        "sum_cts": int(counts.sum()),
        "mean": mean,
        "var": var,
        "std": std,
        "min": int(counts.min()),
        "max": int(counts.max()),
        "p05": float(p05),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p95": float(p95),
        "p99": float(p99),
    }


def _print_stats(split: str, stats: Dict[str, Any], source: str):
    print(f"\n[PairStats] split='{split}' (source={source})")
    for k in ["n_pairs", "sum_cts", "mean", "var", "std", "min", "max", "p05", "p25", "p50", "p75", "p95", "p99"]:
        print(f"  - {k}: {stats.get(k)}")


# ----------------------------
# build cache per split
# ----------------------------
def _build_one_split(cfg: DictConfig, data_cfg: DataConfig, split: str, cache_root: str):
    batch_size = int(cfg.run.get("batch_size", 1024))
    num_workers = int(cfg.run.get("num_workers", 0))     # 构建 cache 时建议先用 0，避免额外 DataLoader worker 干扰
    pin_memory = bool(cfg.run.get("pin_memory", True))

    ds, ld = build_dataset_and_loader(
        data_cfg=data_cfg,
        split_idx=split,
        cache_data_path=cache_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=False,
    )

    # 轻量触发一次迭代，确保 DataLoader 侧也没问题（不会完整跑完）
    try:
        next(iter(ld))
    except StopIteration:
        pass

    print(f"[CacheBuild] split='{split}' done. len(ds)={len(ds)}")
    return ds


# ----------------------------
# plot
# ----------------------------
def _make_bins(counts: np.ndarray, max_bins: int = 80):
    """
    根据最大值自适应 bins：
    - 若 max 较小（<=1024），用整数 bins（每个整数一个桶）
    - 若 max 很大，用 log-spaced bins（更适合长尾）
    """
    if counts.size == 0:
        return np.array([0, 1], dtype=float)

    xmax = int(counts.max())
    if xmax <= 1024:
        return np.arange(0, xmax + 2, dtype=float) - 0.5

    edges = np.unique(np.logspace(0, np.log10(xmax + 1), max_bins).astype(int))
    edges = edges[edges >= 1]
    edges = np.r_[0, edges]
    edges = np.unique(edges).astype(float) - 0.5
    if edges[-1] < xmax + 0.5:
        edges = np.r_[edges, xmax + 0.5]
    return edges


def _pick_three_splits(splits: List[str]) -> List[str]:
    """
    目标：画“一张图 3 子图”。优先 train/val/test；不足则用 splits 的前几个补齐。
    """
    preferred = ["train", "val", "test"]
    picked = [s for s in preferred if s in splits]
    if len(picked) < 3:
        for s in splits:
            if s not in picked:
                picked.append(s)
            if len(picked) == 3:
                break
    return picked[:3]


def plot_three_splits(pair_stats_dir: str, splits: List[str], save_name: str = "num_cts_per_pair_dist.png"):
    pair_stats_dir = Path(pair_stats_dir)
    plot_splits = _pick_three_splits(list(splits))

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)

    for ax, sp in zip(axes, plot_splits):
        npy_path = pair_stats_dir / f"num_cts_per_pair__{sp}.npy"
        if not npy_path.exists():
            ax.set_title(f"{sp} (missing)")
            ax.axis("off")
            continue

        counts = np.load(npy_path).astype(np.int64)
        bins = _make_bins(counts)

        ax.hist(counts, bins=bins, log=True)  # y 轴 log：更容易看长尾
        med = float(np.percentile(counts, 50)) if counts.size else 0.0
        p95 = float(np.percentile(counts, 95)) if counts.size else 0.0

        ax.axvline(med, linestyle="--", linewidth=1.0)
        ax.axvline(p95, linestyle=":", linewidth=1.0)

        ax.set_title(f"{sp} (n_pairs={counts.size})")
        ax.set_xlabel("num_cts_per_pair")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    axes[0].set_ylabel("frequency (log scale)")
    fig.tight_layout()

    out_path = pair_stats_dir / save_name
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[PairStats][plot] Saved figure to: {out_path}")


# ----------------------------
# main
# ----------------------------
@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    seed = int(cfg.get("seed", 2020))
    set_seeds(seed)

    data_cfg = DataConfig.from_omegaconf(cfg.data)

    cache_root = _resolve_cache_root(cfg)
    os.makedirs(cache_root, exist_ok=True)
    print(f"[CacheBuild] cache_root={cache_root}")

    # 输出：改为在 cache_root 下建立文件夹保存
    out_dir = Path(cache_root) / "pair_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[PairStats] out_dir={out_dir}")

    splits: List[str] = list(cfg.run.get("cache_splits", ["train", "val", "test"]))

    all_stats: Dict[str, Any] = {
        "seed": seed,
        "splits": splits,
        "data_alignment": getattr(data_cfg, "alignment", None),
        "by_split": {},
    }

    # 逐 split：构建 + 统计 + 落盘
    for split in splits:
        ds = _build_one_split(cfg, data_cfg, split, cache_root)

        counts_np, source = _get_num_cts_per_pair(ds)
        stats = _summarize_counts(counts_np)
        _print_stats(split, stats, source=source)

        npy_path = out_dir / f"num_cts_per_pair__{split}.npy"
        json_path = out_dir / f"num_cts_per_pair__{split}.json"
        np.save(npy_path, counts_np)

        ph = _path_hash_for_split(data_cfg, split)
        payload = {
            "split": split,
            "source": source,
            "cache_root": cache_root,
            "path_hash": ph,
            "data_path": str(data_cfg.get_path(split)),
            "alignment": getattr(data_cfg, "alignment", None),
            "stats": stats,
            "npy_path": str(npy_path),
        }
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)

        all_stats["by_split"][split] = payload

    # 汇总落盘
    with open(out_dir / "num_cts_per_pair__ALL.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    # 绘图：一张图 3 子图（优先 train/val/test）
    plot_three_splits(str(out_dir), splits=splits, save_name="num_cts_per_pair_dist.png")

    print("\n[CacheBuild] All requested splits finished.")
    print(f"[PairStats] Saved to: {out_dir}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
