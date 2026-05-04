from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.data.window_shard_cache import (
    WindowShardWriter,
    load_window_shard_manifest,
    write_window_shard_manifest,
)
from src.data.window_shard_dataset import WindowShardDataset


def _resolve_keep_probability(labels: np.ndarray, mode: str, neg_pos_ratio: float) -> float:
    valid = labels >= 0
    pos = labels == 1
    neg = labels == 0
    num_pos = int(pos.sum())
    num_neg = int(neg.sum())
    if str(mode) == "valid_only":
        return 1.0
    if str(mode) != "balanced_neg":
        raise ValueError(f"Unknown compact mode: {mode}")
    if num_neg <= 0:
        return 1.0
    target_neg = int(max(0.0, float(neg_pos_ratio)) * max(1, num_pos))
    if target_neg <= 0:
        return 0.0
    return float(min(1.0, target_neg / max(1, num_neg)))


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    src_root = str(cfg.scalable.cache_root)
    dst_root = str(cfg.run.output_cache_root)
    split = str(cfg.run.split)
    mode = str(cfg.run.get("compact_mode", "valid_only"))
    neg_pos_ratio = float(cfg.run.get("neg_pos_ratio", 4.0))
    max_samples_per_shard = int(cfg.run.get("max_samples_per_shard", 2_000_000))
    seed = int(cfg.get("seed", 2020))

    src_manifest = load_window_shard_manifest(src_root, split=split)
    ds = WindowShardDataset(src_root, split=split, include_ignore=True)

    labels = np.zeros((len(ds),), dtype=np.int8)
    cursor = 0
    for shard in src_manifest.shards:
        shard_ds = WindowShardDataset(src_root, split=split, include_ignore=True)
        break
    del shard_ds

    keep_prob = 1.0
    if mode == "balanced_neg":
        offset = 0
        for shard_info in src_manifest.shards:
            shard_path = str(shard_info["path"])
            with open(Path(shard_path) / "meta.json", "r") as f:
                pass
        # Scan labels sequentially from dataset readers via direct shard readers.
        from src.data.window_shard_cache import (  # local import to avoid wider dependency here
            WindowShardReader,
        )

        labels_all = []
        for shard_info in tqdm(src_manifest.shards, desc=f"scan_labels_{split}"):
            reader = WindowShardReader(str(shard_info["path"]))
            labels_all.append(np.array(reader.label, copy=False))
        labels = np.concatenate(labels_all, axis=0) if labels_all else np.empty((0,), dtype=np.int8)
        keep_prob = _resolve_keep_probability(labels, mode=mode, neg_pos_ratio=neg_pos_ratio)
        print(
            f"[compact_window_shards] split={split} mode={mode} "
            f"num_pos={(labels == 1).sum()} num_neg={(labels == 0).sum()} keep_prob_neg={keep_prob:.6f}"
        )
    else:
        print(f"[compact_window_shards] split={split} mode={mode}")

    rng = np.random.default_rng(seed)
    shard_id = 0
    writer = WindowShardWriter(
        dst_root, split=split, shard_id=shard_id, max_samples=max_samples_per_shard
    )
    shard_infos = []
    kept_samples = 0

    from src.data.window_shard_cache import WindowShardReader  # local import

    global_offset = 0
    for shard_info in tqdm(src_manifest.shards, desc=f"compact_{split}"):
        reader = WindowShardReader(str(shard_info["path"]))
        n = len(reader)
        label_arr = np.array(reader.label, copy=False)
        valid_mask = label_arr >= 0
        if mode == "balanced_neg":
            neg_mask = label_arr == 0
            keep_neg = rng.random(size=n) < keep_prob
            keep_mask = valid_mask & (~neg_mask | keep_neg)
        else:
            keep_mask = valid_mask
        keep_idx = np.flatnonzero(keep_mask)
        if keep_idx.size == 0:
            global_offset += n
            continue

        X = torch.from_numpy(np.array(reader.X[keep_idx], copy=True))
        esa = torch.from_numpy(
            np.array(reader.esa[keep_idx], copy=True).astype(np.float32, copy=False)
        )
        pos = torch.from_numpy(
            np.array(reader.pos[keep_idx], copy=True).astype(np.float32, copy=False)
        )
        label = torch.from_numpy(
            np.array(reader.label[keep_idx], copy=True).astype(np.int8, copy=False)
        )
        pair_id = torch.from_numpy(
            np.array(reader.pair_id[keep_idx], copy=True).astype(np.int32, copy=False)
        )

        offset = 0
        total = int(X.shape[0])
        while offset < total:
            room = int(writer.remaining)
            if room <= 0:
                shard_infos.append(writer.set_ready())
                shard_id += 1
                writer = WindowShardWriter(
                    dst_root, split=split, shard_id=shard_id, max_samples=max_samples_per_shard
                )
                room = int(writer.remaining)
            take = min(room, total - offset)
            writer.write_batch(
                X=X[offset : offset + take],
                esa=esa[offset : offset + take],
                pos=pos[offset : offset + take],
                label=label[offset : offset + take],
                pair_id=pair_id[offset : offset + take],
            )
            offset += take
        kept_samples += total
        global_offset += n

    if writer.num_samples > 0 or not shard_infos:
        shard_infos.append(writer.set_ready())

    label_policy = f"{src_manifest.label_policy}|compact:{mode}"
    if mode == "balanced_neg":
        label_policy += f"|neg_pos_ratio:{neg_pos_ratio}"
    write_window_shard_manifest(
        dst_root,
        split=split,
        source_path=str(src_manifest.source_path),
        label_policy=label_policy,
        esa_min_score=float(src_manifest.esa_min_score),
        total_pairs=int(src_manifest.total_pairs),
        shards=shard_infos,
    )
    print(
        f"[compact_window_shards] DONE split={split} kept_samples={kept_samples} num_shards={len(shard_infos)}"
    )


if __name__ == "__main__":
    main()
