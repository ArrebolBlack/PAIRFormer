# src/data/builder.py
"""
builder.py
======================
本模块提供「高层数据构建接口」，用于在训练 / 验证 / 测试阶段快速获得：

    (dataset, dataloader)

职责划分
--------
- DataConfig (src.config.data_config.DataConfig)
    负责描述“数据从哪里来、是否使用 ESA 编码”等元信息；
    对上游（train.py / hydra 主入口）暴露统一接口：
        - .with_esa
        - .path
        - .get_path(split)

- cache (src.data.cache)
    负责高性能缓存、按块构建、meta.json 管理：
        - get_or_build_blocks(...)
        - build_and_cache_dataset_parallel(...)

- dataset / sampler / collate
    * ChunkedCTSDataset: 只依赖 meta.json 的按块加载 Dataset
    * ChunkAwareBatchSampler: 按 chunk 优先的采样策略，减少跨块抖动
    * cts_collate_fn: 统一的 batch 拼接逻辑，确保 Trainer 输入干净一致

- builder (本文件)
    提供一个统一的高层函数：
        build_dataset_and_loader(...)

典型用法
--------
    from src.config.data_config import DataConfig
    from src.data.builder import build_dataset_and_loader

    train_data_cfg = DataConfig.from_file("config/data/siRNA_HUVK_train.json")

    train_ds, train_loader = build_dataset_and_loader(
        data_cfg=train_data_cfg,
        split_idx="train",
        cache_data_path="cache/HUVK_TrainValidationTest",
        batch_size=1024,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )

    val_ds, val_loader = build_dataset_and_loader(
        data_cfg=train_data_cfg,
        split_idx="val",
        cache_data_path="cache/HUVK_TrainValidationTest",
        batch_size=1024,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

未来若接 Hydra / OmegaConf，只需把 DataConfig 构造方式替换为:
    train_data_cfg = DataConfig.from_omegaconf(cfg.data_train)
    eval_data_cfg  = DataConfig.from_omegaconf(cfg.data_eval)

上层对 build_dataset_and_loader 的调用完全不变。
"""

from __future__ import annotations

from typing import Tuple, List

from torch.utils.data import DataLoader

from src.config.data_config import DataConfig
from src.data.cache import get_or_build_blocks, iter_rows_for_split
from src.data.dataset import ChunkedCTSDataset
from src.data.sampler import ChunkAwareBatchSampler
from src.data.collate import cts_collate_fn


def get_set_labels(data_cfg: DataConfig, split_idx: str) -> List[float]:
    """
    读取指定 split 下所有 pair（行）的标签，顺序与构建 cache 时的 set_idx 一致。

    返回的列表长度 = 该 split 内的“行数”（pair 数），
    且第 i 个元素对应 set_idx == i 的那一行。
    """
    labels: List[float] = []
    for local_idx, line in iter_rows_for_split(data_cfg, split_idx):
        toks = line.rstrip("\n").split("\t")
        if len(toks) < 5:
            continue
        try:
            labels.append(float(toks[4]))
        except (ValueError, TypeError):
            labels.append(0.0)
    return labels


def build_dataset_and_loader(
    data_cfg: DataConfig,
    split_idx: str,
    cache_data_path: str,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = False,
) -> Tuple[ChunkedCTSDataset, DataLoader]:
    """
    高层统一接口：基于 DataConfig + cache + ChunkedCTSDataset + ChunkAwareBatchSampler
    一次性构建好 Dataset 与 DataLoader。

    参数
    ----
    data_cfg : DataConfig
        数据配置对象。可以来自:
            - DataConfig.from_file("config/data/xxx.json")
            - DataConfig.from_omegaconf(cfg.data_xxx)
        内部会使用:
            - data_cfg.get_path(split_idx) 找到原始 txt 数据文件
            - data_cfg.with_esa 控制 encode_RNA 的模式（在 cache 层使用）

    split_idx : str
        数据 split 名：
            - "train", "val"（会用 tokens[5] == split_idx 过滤）
            - "test0", "test1" 等（只要求列数 >= 5）

    cache_data_path : str
        缓存目录根路径。所有 meta.json 和 .pt block 会放在这里。
        示例：
            "cache/HUVK_TrainValidationTest_64_16_20_origin_align"

    batch_size : int
        单个 batch 的样本数。

    num_workers : int, default=4
        DataLoader 的 worker 数量。大数据集建议设置为 CPU 核数的一半或更多。

    pin_memory : bool, default=True
        是否开启 pin_memory，加速 CPU→GPU 的数据拷贝。

    shuffle : bool, default=True
        是否在 chunk 内 shuffle 样本：
            - 对 "train" 通常设为 True
            - 对 "val"/"test" 通常设为 False

    drop_last : bool, default=False
        是否丢弃最后不足一个 batch 的样本。

    返回
    ----
    (dataset, loader) : Tuple[ChunkedCTSDataset, DataLoader]
        dataset : ChunkedCTSDataset
            基于 meta.json 的懒加载按块 Dataset。
        loader : DataLoader
            已配好 ChunkAwareBatchSampler + cts_collate_fn 的 DataLoader。

    内部流程
    --------
    1. 调用 get_or_build_blocks(data_cfg, split_idx, cache_data_path)，
       确保 meta.json 和所有 block .pt 文件存在。
    2. 使用 ChunkedCTSDataset(cache_data_path, data_cfg, split_idx) 构造 Dataset。
    3. 构造 ChunkAwareBatchSampler(dataset.cum_sizes, batch_size, drop_last, shuffle)。
    4. 构造 torch.utils.data.DataLoader：
        - batch_sampler = sampler
        - num_workers   = num_workers
        - pin_memory    = pin_memory
        - persistent_workers = True if num_workers > 0 else False
        - prefetch_factor   = 2
        - collate_fn        = cts_collate_fn
    """
    # 1) 确保 cache + meta.json 存在（若不存在会自动并行构建）
    get_or_build_blocks(data_cfg, split_idx, cache_data_path)

    # 2) 构造 Dataset（仅依赖 meta.json）
    dataset = ChunkedCTSDataset(cache_data_path, data_cfg, split_idx)

    # 3) 构造 Chunk-aware 的 batch sampler
    sampler = ChunkAwareBatchSampler(
        dataset.cum_sizes,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
    )

    # 4) 构造 DataLoader
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2,
        collate_fn=cts_collate_fn,
    )

    return dataset, loader



from src.data.pair_level_dataset import PairLevelDataset, pair_level_collate_fn
from src.data.sampler import PairLevelChunkBatchSampler

def build_pair_level_dataset_and_loader(
    pair_cfg,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = False,
) -> Tuple[PairLevelDataset, DataLoader]:
    """
    Pair-level 专用 builder：基于 PairLevelDataset + PairLevelChunkBatchSampler
    一次性构建好 Dataset 与 DataLoader。

    参数
    ----
    pair_cfg :
        一般来自 cfg.data.pair，要求包含：
            - cache_root: str
            - train_split / val_split / test_split: str
            - max_cts_per_pair: int
            - selection_mode: str
            - pos_in_token: bool
            - order_mode: str

    split : str
        "train" / "val" / "test" 等，调用方负责传入具体值。

    其余参数与 CTS builder 类似。
    """
    if split == "train":
        split_name = pair_cfg.train_split
    elif split == "val":
        split_name = pair_cfg.val_split
    elif hasattr(pair_cfg, "test_split") and split == "test":
        split_name = pair_cfg.test_split
    else:
        # 可根据你实际的命名灵活处理，这里给一个保守兜底
        split_name = split

    dataset = PairLevelDataset(
        cache_root=pair_cfg.cache_root,
        split=split_name,
        max_cts_per_pair=pair_cfg.max_cts_per_pair,
        selection_mode=pair_cfg.selection_mode,
        pos_in_token=pair_cfg.pos_in_token,
        order_mode=pair_cfg.order_mode,
        random_order_seed=pair_cfg.random_order_seed,
        random_select_seed=pair_cfg.random_select_seed,
    )

    sampler = PairLevelChunkBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=pair_level_collate_fn,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return dataset, loader
