from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.config.data_config import DataConfig
from src.data.stream_pair_dataset import StreamPairDataset
from src.precompute.window_shard_builder import WindowShardBuildConfig, WindowShardBuilderParallel
from src.utils import set_seeds


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed = int(cfg.get("seed", 2020))
    set_seeds(seed)

    split = str(cfg.run.get("split", "train"))
    data_cfg = DataConfig.from_omegaconf(cfg.data)
    max_pairs = cfg.scalable.get("max_pairs", None)
    pair_ds = StreamPairDataset(data_cfg, split, max_pairs=(None if max_pairs is None else int(max_pairs)))

    shard_cfg = WindowShardBuildConfig(
        esa_min_score=float(cfg.scalable.get("esa_min_score", 6.0)),
        num_workers=int(cfg.scalable.get("num_workers", 8)),
        task_pairs=int(cfg.scalable.get("task_pairs", 16)),
        mp_start_method=str(cfg.scalable.get("mp_start_method", "spawn")),
        max_samples_per_shard=int(cfg.scalable.get("max_samples_per_shard", 2_000_000)),
        seed=seed,
        label_policy=str(cfg.scalable.get("label_policy", "all_positive")),
        pseudo_topr=int(cfg.scalable.get("pseudo_topr", 8)),
        positive_score=str(cfg.scalable.get("positive_score", "esa")),
        drop_ignore_at_build=bool(cfg.scalable.get("drop_ignore_at_build", False)),
        negative_keep_prob=float(cfg.scalable.get("negative_keep_prob", 1.0)),
    )
    builder = WindowShardBuilderParallel(
        root=str(cfg.scalable.cache_root),
        split=split,
        source_path=str(data_cfg.get_path(split)),
        cfg=shard_cfg,
    )
    print(
        f"[build_window_shards] split={split} cache_root={cfg.scalable.cache_root} "
        f"label_policy={shard_cfg.label_policy} workers={shard_cfg.num_workers} "
        f"task_pairs={shard_cfg.task_pairs} max_samples_per_shard={shard_cfg.max_samples_per_shard} "
        f"drop_ignore_at_build={shard_cfg.drop_ignore_at_build} "
        f"negative_keep_prob={shard_cfg.negative_keep_prob}"
    )
    builder.build(pair_ds.iter_records(), data_cfg)


if __name__ == "__main__":
    main()
