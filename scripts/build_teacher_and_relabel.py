#!/usr/bin/env python
"""Build teacher shards and relabel for the V2 pipeline.

Usage:
    python scripts/build_teacher_and_relabel.py \
        --cache_root cache_window_esa_v2_compact_r4 \
        --split train \
        --teacher_ckpt checkpoints/MTI_TargetNet_Optimized_shard_v1_compact_r4_v2_large/best.pt \
        --gpu 0
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.data.window_shard_cache import load_window_shard_manifest
from src.data.window_shard_dataset import WindowShardDataset
from src.data.teacher_shard_cache import TeacherShardWriter, write_teacher_shard_manifest
from src.distill.teacher_extractor import TeacherRunner
from src.models.registry import build_model
from src.config.data_config import DataConfig
from src.data.encoding import encode_RNA, reverse, extended_seed_alignment
from src.data.stream_pair_dataset import PairRecord
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm


def build_teacher_shards(cache_root, split, teacher_ckpt, gpu=0, batch_size=2048):
    """Build teacher shards from a trained teacher model."""
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")


    # Load teacher model
    ckpt = torch.load(teacher_ckpt, map_location=device, weights_only=False)
    model_cfg = ckpt.get("config", {})

    # Extract model config from checkpoint
    mc = model_cfg.get("model", model_cfg.get("instance_model", {}))
    data_cfg_dict = model_cfg.get("data", {"with_esa": True})
    data_cfg = DataConfig.from_dict(data_cfg_dict)

    teacher = build_model(mc.get("arch", "TargetNet_Optimized"), mc.get("name", "targetnet_optimized"), data_cfg=data_cfg).to(device)
    teacher.load_state_dict(ckpt["state_dict"])
    teacher.eval()
    print(f"Loaded teacher from {teacher_ckpt}")
    print(f"  best_f1={ckpt['best_metric']:.4f}, epoch={ckpt['epoch']}")

    runner = TeacherRunner(
        teacher=teacher,
        device=device,
        amp_enabled=True,
        need_feat=True,
    )

    # Load window shard manifest
    # Try compact first, then fall back to full shards
    manifest_path = Path(cache_root) / "window_shards" / split / "manifest.json"
    if not manifest_path.exists():
        # Try full (non-compact) shards
        manifest_path = Path("cache_window_esa_v2") / "window_shards" / split / "manifest.json"

    manifest = load_window_shard_manifest(str(manifest_path.parent.parent), split=split)
    dataset = WindowShardDataset(str(manifest_path.parent.parent), split=split, include_ignore=True)

    print(f"Split={split}: {len(manifest['shards'])} shards, {manifest['total_pairs']} pairs")

    shard_infos = []
    offset_global = 0
    emb_dim = mc.get("target_output_length", 12) * 32  # 384

    for shard_info in tqdm(manifest["shards"], desc=f"Processing {split} shards"):
        shard_id = int(shard_info["shard_id"])
        num_samples = int(shard_info["num_samples"])
        writer = TeacherShardWriter(cache_root, split=split, shard_id=shard_id,
                                          num_samples=num_samples, emb_dim=emb_dim, has_feat=True)
        local_start = offset_global
        start = 0
        while start < num_samples:
            end = min(start + batch_size, num_samples)
            xs = []
            for i in range(local_start + start, local_start + end):
                item = dataset[i]
                xs.append(item["inputs"])
            x = torch.stack(xs, dim=0).to(device, non_blocking=True).float()
            feat, logit = runner(x)
            writer.write_batch(start, logit=logit, feat=feat)
            start = end
        shard_infos.append(writer.set_ready())
        offset_global += num_samples

    write_teacher_shard_manifest(
        cache_root, split=split, source_window_root=cache_root,
        emb_dim=emb_dim, has_feat=True, shards=shard_infos,
    )
    print(f"[build_teacher_shards] DONE split={split}")


def relabel_shards(cache_root, split, pseudo_topr=8, esa_min_score=6.0):
    """Relabel window shards with pseudo-topr policy."""
    window_root = Path(cache_root) / "window_shards" / split
    teacher_root = Path(cache_root) / "teacher_shards" / split

    manifest = load_window_shard_manifest(cache_root, split=split)

    relabeled = 0
    for shard_info in tqdm(manifest["shards"], desc=f"Relabeling {split}"):
        shard_id = int(shard_info["shard_id"])
        num_samples = int(shard_info["num_samples"])

        # Load original shard data
        shard_dir = window_root / f"shard_{shard_id:05d}"
        labels = np.fromfile(shard_dir / "label.i8.mmap", dtype=np.int8)
        esa_scores = np.fromfile(shard_dir / "esa.f16.mmap", dtype=np.float16)
astype np.float32)
        pair_ids = np.fromfile(shard_dir / "pair_id.i32.mmap", dtype=np.int32)

        # Load teacher logits
        teacher_manifest_path = teacher_root / f"shard_{shard_id:05d}" / "teacher_logit.f16.mmap"
        teacher_logits = np.fromfile(teacher_manifest_path, dtype=np.float16)astype np.float32)

        # For each pair, find top-r windows by teacher logit
        unique_pairs = np.unique(pair_ids)
        print(f"  Shard {shard_id}: {num_samples} samples, {len(unique_pairs)} unique pairs")

        for pair_id in tqdm(unique_pairs, desc=f"  Pair {pair_id}"):
            pair_mask = pair_ids == pair_id
            pair_labels = labels[pair_mask]
            pair_esa = esa_scores[pair_mask]
            pair_teacher_logits = teacher_logits[pair_mask]

            # Find top-r by teacher logit
            sorted_indices = np.argsort(pair_teacher_logits)[::-1]
            top_indices = sorted_indices[:pseudo_topr]

            # Relabel: top-r get label=1, rest get label=-1 (ignore)
            new_labels = labels.copy()
            new_labels[pair_labels == 1] = -1  # negative -> label=0
            new_labels[(pair_labels == 1) & (pair_esa[pair_mask] >= esa_min_score)] = -1  # ignore: below threshold

            # Write relabeled shard
            relabel_dir = window_root / f"shard_{shard_id:05d}"
            np.save(relabel_dir / "label.i8.mmap", new_labels)

            relabeled += new_labels.shape[0]
            labels[pair_mask] = new_labels[pair_mask]

    # Update shard metadata
    meta_path = shard_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    meta["label_policy"] = f"pseudo_topr|{pseudo_topr}"
    meta["num_positive"] = int((new_labels == 1).sum())
    meta["num_negative"] = int((new_labels == 0).sum())
    meta["num_ignore"] = int((new_labels == -1).sum())
    json.dump(meta, meta_path, open("w")

    print(f"  Shard {shard_id} relabeled: pos={meta['num_positive']}, neg={meta['num_negative']}, -={meta['num_ignore']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", type=str, default="cache_window_esa_v2_compact_r4")
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--pseudo_topr", type=int, default=8)
    parser.add_argument("--esa_min_score", type=float, default=6.0)
    parser.add_argument("--skip_re action="store_true", help="Skip teacher shard building, relabeling go directly to pair model training")
    parser.add_argument("--skip_re action="store_false", help="Also run relabeling for pair model training (default: false)
    args = parser.parse_args()

    if not args.skip:
        if args.split == "train":
        # Train: build teacher shards then relabel
        build_teacher_shards(args.cache_root, args.split, args.teacher_ckpt, args.gpu, args.batch_size)
        if not args.skip:
            relabel_shards(args.cache_root, args.split, args.pseudo_topr, args.esa_min_score)
    else:
