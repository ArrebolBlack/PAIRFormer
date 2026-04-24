#!/usr/bin/env python
"""Fast relabel window shards using teacher logits - V3 pipeline.

Reads from cache_window_esa_v2_compact_r4_relabel (which has both window shards
and teacher shards built from v2 encoder).
Writes to cache_window_esa_v2_compact_r4_v3relbl.

Usage:
    python scripts/fast_relabel_shards_v3.py --topr 4
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.data.window_shard_cache import WindowShardReader, load_window_shard_manifest, write_window_shard_manifest
from src.data.teacher_shard_cache import TeacherShardReader, load_teacher_shard_manifest

CACHE_ROOT = "cache_window_esa_v2_compact_r4_relabel"
OUT_ROOT = "cache_window_esa_v2_compact_r4_v3relbl"
SHM_ROOT = "/dev/shm/relabel_v3_build"


def fast_relabel(topr=4):
    split = "train"

    w_manifest = load_window_shard_manifest(CACHE_ROOT, split=split)
    t_manifest = load_teacher_shard_manifest(CACHE_ROOT, split=split)

    if len(w_manifest.shards) != len(t_manifest.shards):
        raise RuntimeError(f"Shard count mismatch: window={len(w_manifest.shards)}, teacher={len(t_manifest.shards)}")

    # Prepare output dirs
    shm_dir = Path(SHM_ROOT)
    shm_dir.mkdir(parents=True, exist_ok=True)

    out_dir = Path(OUT_ROOT) / "window_shards" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_infos = []
    total_samples = 0

    for ws, ts in tqdm(zip(w_manifest.shards, t_manifest.shards), total=len(w_manifest.shards), desc="Relabeling"):
        shard_id = int(ws["shard_id"])
        num_samples = int(ws["num_samples"])

        win_shard_dir = Path(CACHE_ROOT) / "window_shards" / split / f"shard_{shard_id:05d}"
        teacher_shard_dir = Path(CACHE_ROOT) / "teacher_shards" / split / f"shard_{shard_id:05d}"

        win_reader = WindowShardReader(win_shard_dir)
        teacher_reader = TeacherShardReader(teacher_shard_dir)

        labels = np.array(win_reader.label, dtype=np.int8, copy=True)
        pair_ids = np.array(win_reader.pair_id, dtype=np.int32, copy=True)
        teacher_logits = np.array(teacher_reader.logit, dtype=np.float32, copy=True)

        sorted_idx = np.argsort(pair_ids)
        sorted_pair_ids = pair_ids[sorted_idx]
        unique_pairs, pair_starts, pair_counts = np.unique(sorted_pair_ids, return_index=True, return_counts=True)

        new_labels = labels.copy()

        for i in range(len(unique_pairs)):
            pid = unique_pairs[i]
            start = pair_starts[i]
            count = pair_counts[i]
            orig_indices = sorted_idx[start:start + count]

            pair_label = int(labels[orig_indices[0]])

            if pair_label <= 0:
                continue

            pair_logits = teacher_logits[orig_indices]
            r = max(1, min(topr, len(orig_indices)))
            top_k_local = np.argsort(pair_logits)[::-1][:r]
            top_indices = orig_indices[top_k_local]

            new_labels[orig_indices] = -1
            new_labels[top_indices] = 1

        shard_out_shm = shm_dir / f"shard_{shard_id:05d}"
        shard_out_shm.mkdir(parents=True, exist_ok=True)

        label_mmap = np.memmap(shard_out_shm / "label.i8.mmap", mode="w+", dtype=np.int8, shape=(num_samples,))
        label_mmap[:] = new_labels
        del label_mmap

        meta = {
            "state": "ready",
            "split": split,
            "shard_id": shard_id,
            "max_samples": num_samples,
            "num_samples": num_samples,
            "channels": int(win_reader.meta.channels),
            "seq_len": int(win_reader.meta.seq_len),
            "has_esa": True,
            "has_pos": True,
        }
        with open(shard_out_shm / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

        num_pos = int((new_labels == 1).sum())
        num_neg = int((new_labels == 0).sum())
        num_ign = int((new_labels == -1).sum())
        print(f"  Shard {shard_id}: {num_samples} samples -> pos={num_pos}, neg={num_neg}, ignore={num_ign}")

        shard_infos.append({
            "shard_id": shard_id,
            "path": str(out_dir / f"shard_{shard_id:05d}"),
            "num_samples": num_samples,
        })
        total_samples += num_samples

    manifest = write_window_shard_manifest(
        OUT_ROOT,
        split=split,
        source_path=w_manifest.source_path,
        label_policy=f"teacher_topr_{topr}",
        esa_min_score=float(w_manifest.esa_min_score),
        total_pairs=int(w_manifest.total_pairs),
        shards=shard_infos,
    )
    print(f"[DONE] Relabeled {total_samples} samples in {len(shard_infos)} shards")
    print(f"  Manifest: {manifest}")

    print("Moving to VEPFS...")
    for shard_info in shard_infos:
        shard_id = shard_info["shard_id"]
        shard_name = f"shard_{shard_id:05d}"

        src_win = Path(CACHE_ROOT) / "window_shards" / split / shard_name
        src_shm = shm_dir / shard_name
        dst = out_dir / shard_name
        dst.mkdir(parents=True, exist_ok=True)

        for fname in ["X.u8.mmap", "esa.f16.mmap", "pos.f16.mmap", "pair_id.i32.mmap"]:
            src_file = src_win / fname
            dst_file = dst / fname
            if not dst_file.exists():
                try:
                    os.link(src_file, dst_file)
                except OSError:
                    shutil.copy2(src_file, dst_file)

        shutil.copy2(src_shm / "label.i8.mmap", dst / "label.i8.mmap")
        shutil.copy2(src_shm / "meta.json", dst / "meta.json")

        print(f"  {shard_name}: done")

    print(f"[MOVED] All shards to {OUT_ROOT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topr", type=int, default=4)
    args = parser.parse_args()
    fast_relabel(args.topr)
