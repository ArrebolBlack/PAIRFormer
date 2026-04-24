#!/usr/bin/env python
"""Build v4 teacher shards + relabel in one pass.

Memory strategy: all intermediate writes go to /dev/shm, then bulk copy to VepFS.

Step 1: Run v3 encoder on all training window shards -> teacher logits/feats -> /dev/shm
Step 2: Relabel top-r per positive pair using new teacher logits -> /dev/shm
Step 3: Bulk copy to VepFS (cache_window_esa_v2_compact_r4_v4relbl)

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/build_v4_relabel.py --gpu 0 --topr 4
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.data.window_shard_cache import WindowShardReader, load_window_shard_manifest, write_window_shard_manifest
from src.models.registry import build_model
from src.models.extractors import get_embedding_and_logit
from src.config.data_config import DataConfig

# ── Paths ──
WINDOW_SRC = "cache_window_esa_v2_compact_r4_relabel"  # source window shards
TEACHER_CKPT = "checkpoints/MTI_v3_xlarge_resume/best.pt"
VEPFS_OUT = "cache_window_esa_v2_compact_r4_v4relbl"
SHM_TEACHER = "/dev/shm/v4_teacher_build"
SHM_RELABEL = "/dev/shm/v4_relabel_build"

# ── Model config (stem + 3 stages: [64,64,128,128]/[3,3,3,3], emb=1536) ──
MODEL_CFG = {
    "name": "targetnet_optimized",
    "arch": "TargetNet_Optimized",
    "num_channels": [64, 64, 128, 128],
    "num_blocks": [3, 3, 3, 3],
    "pool_size": 3,
    "stem_kernel_size": 5,
    "block_kernel_size": 3,
    "skip_connection": True,
    "dropout": 0.1,
    "multi_scale": True,
    "se_type": "cbam",
    "se_reduction": 8,
    "use_bn": True,
    "target_output_length": 12,
}


def build_teacher_shards(split, batch_size, gpu):
    """Step 1: Run v3 encoder -> teacher shards in /dev/shm."""
    device = torch.device(f"cuda:{gpu}")
    ckpt = torch.load(TEACHER_CKPT, map_location=device, weights_only=False)
    print(f"[Teacher] Loaded v3 encoder: best_f1={ckpt.get('best_metric', 0):.4f}, epoch={ckpt.get('epoch', '?')}")

    from omegaconf import OmegaConf
    data_cfg = DataConfig.from_omegaconf(OmegaConf.create({
        "name": "mirna_MTI",
        "path": {"train": "", "val": "", "test": ""},
        "with_esa": True,
    }))

    teacher = build_model("TargetNet_Optimized", MODEL_CFG, data_cfg=data_cfg).to(device)
    teacher.load_state_dict(ckpt["state_dict"])
    teacher.eval()

    manifest = load_window_shard_manifest(WINDOW_SRC, split=split)
    emb_dim = 12 * 128  # 1536

    shm_dir = Path(SHM_TEACHER) / split
    shm_dir.mkdir(parents=True, exist_ok=True)

    shard_infos = []
    t0 = time.time()

    for shard_info in tqdm(manifest.shards, desc=f"[Teacher] Building {split}"):
        shard_id = int(shard_info["shard_id"])
        num_samples = int(shard_info["num_samples"])

        out_dir = shm_dir / f"shard_{shard_id:05d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        shard_dir = Path(WINDOW_SRC) / "window_shards" / split / f"shard_{shard_id:05d}"
        reader = WindowShardReader(shard_dir)

        logit_mmap = np.memmap(out_dir / "teacher_logit.f16.mmap", mode="w+", dtype=np.float16, shape=(num_samples,))
        feat_mmap = np.memmap(out_dir / "teacher_feat.f16.mmap", mode="w+", dtype=np.float16, shape=(num_samples, emb_dim))

        start = 0
        while start < num_samples:
            end = min(start + batch_size, num_samples)
            x_batch = torch.from_numpy(np.array(reader.X[start:end], dtype=np.float32, copy=True))
            x_batch = x_batch.to(device, non_blocking=True)

            with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=True):
                feat, logit = get_embedding_and_logit(teacher, x_batch)

            logit_mmap[start:end] = logit.detach().cpu().numpy().astype(np.float16)
            feat_mmap[start:end] = feat.detach().cpu().numpy().astype(np.float16)
            start = end

        del logit_mmap, feat_mmap

        meta = {
            "state": "ready", "split": split, "shard_id": shard_id,
            "num_samples": num_samples, "emb_dim": emb_dim, "has_feat": True,
        }
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

        print(f"  Shard {shard_id}: {num_samples} samples done ({time.time()-t0:.0f}s)")
        shard_infos.append({"shard_id": shard_id, "num_samples": num_samples})

    manifest_data = {
        "state": "ready", "split": split, "num_shards": len(shard_infos),
        "total_samples": sum(s["num_samples"] for s in shard_infos),
        "emb_dim": emb_dim, "has_feat": True, "source_window_root": WINDOW_SRC,
        "shards": [{"shard_id": s["shard_id"], "num_samples": s["num_samples"],
                     "path": f"{VEPFS_OUT}/teacher_shards/{split}/shard_{s['shard_id']:05d}"} for s in shard_infos],
    }
    with open(shm_dir / "manifest.json", "w") as f:
        json.dump(manifest_data, f, indent=2, sort_keys=True)

    print(f"[Teacher] DONE: {len(shard_infos)} shards in {time.time()-t0:.0f}s")
    return shard_infos


def relabel_shards(split, topr):
    """Step 2: Relabel using v4 teacher logits -> /dev/shm."""
    shm_teacher = Path(SHM_TEACHER) / split
    shm_relabel = Path(SHM_RELABEL) / split
    shm_relabel.mkdir(parents=True, exist_ok=True)

    window_manifest = load_window_shard_manifest(WINDOW_SRC, split=split)

    shard_infos = []
    total_samples = 0

    for shard_info in tqdm(window_manifest.shards, desc=f"[Relabel] {split}"):
        shard_id = int(shard_info["shard_id"])
        num_samples = int(shard_info["num_samples"])

        # Read source window data
        win_dir = Path(WINDOW_SRC) / "window_shards" / split / f"shard_{shard_id:05d}"
        reader = WindowShardReader(win_dir)

        labels = np.array(reader.label, dtype=np.int8, copy=True)
        pair_ids = np.array(reader.pair_id, dtype=np.int32, copy=True)

        # Read v4 teacher logits from /dev/shm
        teacher_logit = np.fromfile(
            shm_teacher / f"shard_{shard_id:05d}" / "teacher_logit.f16.mmap", dtype=np.float16
        ).astype(np.float32)

        # Sort by pair_id for efficient grouping
        sorted_idx = np.argsort(pair_ids)
        sorted_pair_ids = pair_ids[sorted_idx]
        unique_pairs, pair_starts, pair_counts = np.unique(sorted_pair_ids, return_index=True, return_counts=True)

        new_labels = labels.copy()

        for i in range(len(unique_pairs)):
            start = pair_starts[i]
            count = pair_counts[i]
            orig_indices = sorted_idx[start:start + count]
            pair_label = int(labels[orig_indices[0]])

            if pair_label <= 0:
                continue

            pair_logits = teacher_logit[orig_indices]
            r = max(1, min(topr, len(orig_indices)))
            top_k_local = np.argsort(pair_logits)[::-1][:r]
            top_indices = orig_indices[top_k_local]

            new_labels[orig_indices] = -1
            new_labels[top_indices] = 1

        # Write relabeled shard to /dev/shm
        out_dir = shm_relabel / f"shard_{shard_id:05d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        label_mmap = np.memmap(out_dir / "label.i8.mmap", mode="w+", dtype=np.int8, shape=(num_samples,))
        label_mmap[:] = new_labels
        del label_mmap

        num_pos = int((new_labels == 1).sum())
        num_neg = int((new_labels == 0).sum())
        num_ign = int((new_labels == -1).sum())
        print(f"  Shard {shard_id}: {num_samples} -> pos={num_pos}, neg={num_neg}, ignore={num_ign}")

        meta = {
            "state": "ready", "split": split, "shard_id": shard_id,
            "max_samples": num_samples, "num_samples": num_samples,
            "channels": int(reader.meta.channels), "seq_len": int(reader.meta.seq_len),
            "has_esa": True, "has_pos": True,
        }
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

        shard_infos.append({"shard_id": shard_id, "num_samples": num_samples})
        total_samples += num_samples

    # Write relabel manifest
    write_window_shard_manifest(
        VEPFS_OUT, split=split,
        source_path=window_manifest.source_path,
        label_policy=f"v4_teacher_topr_{topr}",
        esa_min_score=float(window_manifest.esa_min_score),
        total_pairs=int(window_manifest.total_pairs),
        shards=[{"shard_id": s["shard_id"], "num_samples": s["num_samples"],
                 "path": f"{VEPFS_OUT}/window_shards/{split}/shard_{s['shard_id']:05d}"} for s in shard_infos],
    )
    print(f"[Relabel] DONE: {total_samples} samples in {len(shard_infos)} shards")


def copy_to_vepfs(split):
    """Step 3: Bulk copy from /dev/shm to VepFS."""
    shm_teacher = Path(SHM_TEACHER) / split
    shm_relabel = Path(SHM_RELABEL) / split

    # 3a. Copy teacher shards
    print("[Copy] Teacher shards -> VepFS...")
    vepfs_teacher = Path(VEPFS_OUT) / "teacher_shards" / split
    vepfs_teacher.mkdir(parents=True, exist_ok=True)

    for shard_dir in sorted(shm_teacher.glob("shard_*")):
        dst = vepfs_teacher / shard_dir.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(shard_dir, dst)
        print(f"  {shard_dir.name}: done")

    shutil.copy2(shm_teacher / "manifest.json", vepfs_teacher / "manifest.json")

    # 3b. Copy relabeled window shards (label + meta from shm, rest hardlinked from source)
    print("[Copy] Window shards -> VepFS...")
    vepfs_window = Path(VEPFS_OUT) / "window_shards" / split
    vepfs_window.mkdir(parents=True, exist_ok=True)

    for shard_dir in sorted(shm_relabel.glob("shard_*")):
        shard_name = shard_dir.name
        dst = vepfs_window / shard_name
        dst.mkdir(parents=True, exist_ok=True)

        # Hardlink immutable files from source (X, esa, pos, pair_id)
        src_win = Path(WINDOW_SRC) / "window_shards" / split / shard_name
        for fname in ["X.u8.mmap", "esa.f16.mmap", "pos.f16.mmap", "pair_id.i32.mmap"]:
            src_file = src_win / fname
            dst_file = dst / fname
            if not dst_file.exists():
                try:
                    os.link(src_file, dst_file)
                except OSError:
                    shutil.copy2(src_file, dst_file)

        # Copy new label + meta from shm
        shutil.copy2(shard_dir / "label.i8.mmap", dst / "label.i8.mmap")
        shutil.copy2(shard_dir / "meta.json", dst / "meta.json")
        print(f"  {shard_name}: done")

    # Copy manifest
    src_manifest = Path(VEPFS_OUT) / "window_shards" / split / "manifest.json"
    # manifest was already written by write_window_shard_manifest
    if not src_manifest.exists():
        print("[WARN] window manifest not found, copying from source")
        src_m = Path(WINDOW_SRC) / "window_shards" / split / "manifest.json"
        shutil.copy2(src_m, src_manifest)

    print(f"[Copy] DONE: {VEPFS_OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--topr", type=int, default=4)
    args = parser.parse_args()

    t_start = time.time()

    print("=" * 60)
    print("V4 RELABEL PIPELINE")
    print(f"  Teacher: {TEACHER_CKPT}")
    print(f"  Source:  {WINDOW_SRC}")
    print(f"  Output:  {VEPFS_OUT}")
    print(f"  topr:    {args.topr}")
    print(f"  GPU:     {args.gpu}")
    print("=" * 60)

    # Step 1: Build teacher shards in /dev/shm
    build_teacher_shards(args.split, args.batch_size, args.gpu)

    # Step 2: Relabel in /dev/shm
    relabel_shards(args.split, args.topr)

    # Step 3: Bulk copy to VepFS
    copy_to_vepfs(args.split)

    print(f"\n[ALL DONE] Total time: {time.time()-t_start:.0f}s")
