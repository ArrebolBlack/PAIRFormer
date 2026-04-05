from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from src.config.data_config import DataConfig
from src.data.teacher_shard_cache import TeacherShardWriter, write_teacher_shard_manifest
from src.data.window_shard_cache import load_window_shard_manifest
from src.data.window_shard_dataset import WindowShardDataset
from src.distill.teacher_extractor import TeacherRunner
from src.launch.train import load_model_state
from src.models.registry import build_model
from src.utils import set_seeds


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_seeds(int(cfg.get("seed", 2020)))
    device = torch.device("cuda" if torch.cuda.is_available() and str(cfg.get("device", "cuda")) != "cpu" else "cpu")
    split = str(cfg.run.get("split", "train"))
    root = str(cfg.scalable.cache_root)
    data_cfg = DataConfig.from_omegaconf(cfg.data)

    teacher = build_model(str(cfg.run.distill_teacher_arch), cfg.run.distill_teacher_model, data_cfg=data_cfg).to(device)
    load_model_state(teacher, str(cfg.run.distill_teacher_ckpt), device)
    runner = TeacherRunner(
        teacher=teacher,
        device=device,
        amp_enabled=bool(cfg.run.get("distill_teacher_amp", True)),
        need_feat=bool(cfg.run.get("distill_need_feat", True)),
    )

    manifest = load_window_shard_manifest(root, split=split)
    dataset = WindowShardDataset(root, split=split, include_ignore=True)
    shard_infos = []
    offset_global = 0
    emb_dim = int(cfg.run.get("distill_teacher_feat_dim", 384))
    has_feat = bool(cfg.run.get("distill_need_feat", True))

    for shard in manifest.shards:
        shard_id = int(shard["shard_id"])
        num_samples = int(shard["num_samples"])
        writer = TeacherShardWriter(root, split=split, shard_id=shard_id, num_samples=num_samples, emb_dim=emb_dim, has_feat=has_feat)
        local_start = offset_global
        batch_size = int(cfg.run.get("batch_size", 2048))
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
        print(f"[build_teacher_shards] split={split} shard={shard_id} samples={num_samples}")

    out = write_teacher_shard_manifest(
        root,
        split=split,
        source_window_root=root,
        emb_dim=emb_dim,
        has_feat=has_feat,
        shards=shard_infos,
    )
    print(f"[build_teacher_shards] DONE split={split} manifest={out}")


if __name__ == "__main__":
    main()
