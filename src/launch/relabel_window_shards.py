from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
from omegaconf import DictConfig

from src.data.teacher_shard_cache import TeacherShardReader, load_teacher_shard_manifest
from src.data.window_shard_cache import WindowShardWriter, load_window_shard_manifest, write_window_shard_manifest
from src.data.window_shard_dataset import WindowShardDataset
from src.utils import set_seeds


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_seeds(int(cfg.get("seed", 2020)))
    split = str(cfg.run.get("split", "train"))
    src_root = str(cfg.scalable.cache_root)
    teacher_root = str(cfg.run.get("teacher_cache_root", src_root))
    out_root = str(cfg.run.get("output_cache_root", src_root + "_relabel"))
    topr = int(cfg.run.get("pseudo_topr", 8))

    w_manifest = load_window_shard_manifest(src_root, split=split)
    t_manifest = load_teacher_shard_manifest(teacher_root, split=split)
    if len(w_manifest.shards) != len(t_manifest.shards):
        raise RuntimeError("Window/teacher shard count mismatch.")

    ds = WindowShardDataset(src_root, split=split, include_ignore=True)
    offset = 0
    shard_infos: List[Dict[str, int | str]] = []
    for ws, ts in zip(w_manifest.shards, t_manifest.shards):
        shard_id = int(ws["shard_id"])
        n = int(ws["num_samples"])
        writer = WindowShardWriter(out_root, split=split, shard_id=shard_id, max_samples=n)
        teacher_reader = TeacherShardReader(ts["path"])

        xs = []
        esas = []
        poss = []
        labels = []
        pair_ids = []
        teacher_logits = []
        for i in range(offset, offset + n):
            item = ds[i]
            local = i - offset
            t = teacher_reader.read(local)
            xs.append(item["inputs"])
            esas.append(float(item["esa_scores"]))
            poss.append(float(item["pos"]))
            labels.append(int(item["labels"].item()))
            pair_ids.append(int(item["set_idx"].item()))
            teacher_logits.append(float(t["teacher_logit"].item()))

        pair_to_idx: Dict[int, List[int]] = {}
        for i, pid in enumerate(pair_ids):
            pair_to_idx.setdefault(int(pid), []).append(i)

        new_labels = np.full((n,), -1, dtype=np.int8)
        for pid, idxs in pair_to_idx.items():
            pair_label = labels[idxs[0]]
            if pair_label <= 0:
                for i in idxs:
                    new_labels[i] = 0
                continue
            order = sorted(idxs, key=lambda i: teacher_logits[i], reverse=True)
            keep = set(order[: max(1, min(topr, len(order)))])
            for i in idxs:
                new_labels[i] = 1 if i in keep else -1

        import torch
        writer.write_batch(
            X=torch.stack(xs, dim=0),
            esa=torch.tensor(esas, dtype=torch.float32),
            pos=torch.tensor(poss, dtype=torch.float32),
            label=torch.tensor(new_labels, dtype=torch.int8),
            pair_id=torch.tensor(pair_ids, dtype=torch.int32),
        )
        shard_infos.append(writer.set_ready())
        offset += n
        print(f"[relabel_window_shards] split={split} shard={shard_id} samples={n}")

    out = write_window_shard_manifest(
        out_root,
        split=split,
        source_path=w_manifest.source_path,
        label_policy=f"teacher_topr_{topr}",
        esa_min_score=float(w_manifest.esa_min_score),
        total_pairs=int(w_manifest.total_pairs),
        shards=shard_infos,
    )
    print(f"[relabel_window_shards] DONE split={split} manifest={out}")


if __name__ == "__main__":
    main()
