#!/usr/bin/env python
"""Fast teacher shard builder - writes to /dev/shm then moves to VEPFS.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/fast_build_teacher_shards.py \
        --split train --gpu 0 --skip_existing
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.data.window_shard_cache import WindowShardReader, load_window_shard_manifest
from src.models.registry import build_model
from src.models.extractors import get_embedding_and_logit
from src.config.data_config import DataConfig

CACHE_ROOT = "cache_window_esa_v2_compact_r4"
SHM_ROOT = "/dev/shm/teacher_shards_build"


def fast_build(split, batch_size=8192, gpu=0, skip_existing=False, shard_start=0):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    ckpt_path = "checkpoints/MTI_TargetNet_Optimized_shard_v1_compact_r4_v2_xlarge_cosine/best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    from omegaconf import OmegaConf
    data_cfg = DataConfig.from_omegaconf(OmegaConf.create({
        "name": "mirna_MTI",
        "path": {"train": "", "val": "", "test": ""},
        "with_esa": True,
        "split_column": 6,
        "split_map": {"train": "train", "val": "val", "test": "test"},
    }))
    model_cfg = {
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
    teacher = build_model("TargetNet_Optimized", model_cfg, data_cfg=data_cfg).to(device)
    teacher.load_state_dict(ckpt["state_dict"])
    teacher.eval()
    print(f"Loaded teacher: best_f1={ckpt['best_metric']:.4f}, epoch={ckpt['epoch']}")

    manifest = load_window_shard_manifest(CACHE_ROOT, split=split)
    emb_dim = 12 * 128  # 1536
    has_feat = True

    shard_infos = []
    for shard_info in tqdm(manifest.shards, desc=f"Building {split}"):
        shard_id = int(shard_info["shard_id"])
        num_samples = int(shard_info["num_samples"])

        if shard_id < shard_start:
            continue

        # Check if already done in final location
        final_dir = Path(CACHE_ROOT) / "teacher_shards" / split / f"shard_{shard_id:05d}"
        if skip_existing and final_dir.exists():
            meta_path = final_dir / "meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta.get("state") == "ready":
                    print(f"  Skipping shard {shard_id} (already ready)")
                    shard_infos.append({"shard_id": shard_id, "path": str(final_dir), "num_samples": num_samples})
                    continue

        # Write to /dev/shm
        shm_dir = Path(SHM_ROOT) / split / f"shard_{shard_id:05d}"
        shm_dir.mkdir(parents=True, exist_ok=True)

        shard_dir = Path(CACHE_ROOT) / "window_shards" / split / f"shard_{shard_id:05d}"
        reader = WindowShardReader(shard_dir)

        logit_path = shm_dir / "teacher_logit.f16.mmap"
        feat_path = shm_dir / "teacher_feat.f16.mmap"

        logit_mmap = np.memmap(logit_path, mode="w+", dtype=np.float16, shape=(num_samples,))
        feat_mmap = np.memmap(feat_path, mode="w+", dtype=np.float16, shape=(num_samples, emb_dim))

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

        # Flush and close mmaps
        del logit_mmap, feat_mmap

        # Write meta
        meta = {
            "state": "ready",
            "split": split,
            "shard_id": shard_id,
            "num_samples": num_samples,
            "emb_dim": emb_dim,
            "has_feat": has_feat,
        }
        with open(shm_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

        print(f"  Shard {shard_id}: {num_samples} samples done (in /dev/shm)")

        shard_infos.append({"shard_id": shard_id, "path": str(final_dir), "num_samples": num_samples})

    # Write manifest to shm
    manifest_dir = Path(SHM_ROOT) / split
    manifest_data = {
        "state": "ready",
        "split": split,
        "num_shards": len(shard_infos),
        "total_samples": sum(s["num_samples"] for s in shard_infos),
        "emb_dim": emb_dim,
        "has_feat": has_feat,
        "source_window_root": CACHE_ROOT,
        "shards": shard_infos,
    }
    with open(manifest_dir / "manifest.json", "w") as f:
        json.dump(manifest_data, f, indent=2, sort_keys=True)

    print(f"[DONE] split={split}, {len(shard_infos)} shards in {SHM_ROOT}/{split}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--shard_start", type=int, default=0)
    args = parser.parse_args()
    fast_build(args.split, args.batch_size, args.gpu, args.skip_existing, args.shard_start)
