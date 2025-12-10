# src/data/dataset.py
"""
dataset.py
======================
本模块提供 ChunkedCTSDataset：

- 一种只依赖 meta.json 文件的懒加载 Dataset 实现。
- 每个样本存储在若干 .pt block 中，Dataset 通过元信息知道：
  * 每个 block 的路径
  * block 内的样本数量
  * 全局索引如何映射到具体 block 与 block 内局部 index

核心特点：
- 初始化快速：不需要逐个读取所有 .pt block，只需读取单个 meta.json。
- 内存友好：只在访问样本时加载当前 block 的数据，切换 block 时主动释放旧数据并触发 GC。
- 适合大规模数据：block 大小可控（由 BLOCK_SIZE 和编码大小决定）。

典型用法
--------
    from src.data.cache import get_or_build_blocks
    from src.data.dataset import ChunkedCTSDataset

    get_or_build_blocks(data_cfg, "train", cache_path)  # 确保 meta.json 存在
    ds = ChunkedCTSDataset(cache_path, data_cfg, "train")

    len(ds)             # 总样本数
    x, y, set_idx = ds[0]
"""

import os
import json
import hashlib
import gc
from typing import List

import numpy as np
import torch


class ChunkedCTSDataset(torch.utils.data.Dataset):
    """
    支持按块懒加载的 Dataset 实现，专为 CTS/miRNA 类大规模数据设计。

    初始化只依赖 meta.json 文件，不需要在构造时把所有 block 文件读入内存。
    当 __getitem__ 被调用时，才在需要时加载对应 block，并在 block 切换时主动释放旧 block。

    属性
    ----
    chunk_files : List[str]
        所有 block 文件的路径列表
    chunk_sizes : List[int]
        每个 block 内的样本数
    cum_sizes   : List[int]
        累积样本数（用于通过二分查找确定 idx 所在的 block）
    total_size  : int
        总样本数
    current_chunk_idx : int
        当前已经加载到内存的 block 索引
    current_chunk     : dict or None
        当前 block 对应的内容（torch.load 得到的 dict）

    返回样本格式
    ------------
    __getitem__ 返回三元组：
        x       : torch.Tensor,  one-hot 特征（uint8），形状约 (C, L)
        y       : torch.Tensor,  形状 (1,) 的标签
        set_idx : torch.Tensor,  形状 (1,) 的 long，表示在当前 split 内的连续编号

    上层可以在 collate_fn 中将其转换为 dict，例如：
        {"x": x.float(), "label": y.squeeze(-1), "set_idx": set_idx.squeeze(-1)}
    """

    def __init__(self, cache_data_path: str, data_cfg, split_idx: str):
        """
        构造函数：根据 data_cfg 和 split_idx 找到 meta.json，并初始化元信息。

        参数
        ----
        cache_data_path : str
            缓存目录路径
        data_cfg : DataConfig
            原始数据配置（至少包含 path 字段）
        split_idx : str
            当前使用的数据 split（如 "train", "val", "test0"）
        """
        data_file_path = str(data_cfg.path[split_idx])

        # ✅ 新增：把 alignment 拼进 hash key，区分不同 alignment 下的 cache
        alignment = getattr(data_cfg, "alignment", "extended_seed_alignment")
        hash_key = f"{data_file_path}|{alignment}"
        path_hash = hashlib.md5(hash_key.encode("utf-8")).hexdigest()[:8]
        
        meta_filename = f"cache_{split_idx}_{path_hash}_meta.json"
        meta_filepath = os.path.join(cache_data_path, meta_filename)

        if not os.path.exists(meta_filepath):
            raise FileNotFoundError(
                f"Meta file not found: {meta_filepath}. "
                f"请先调用 get_or_build_blocks(data_cfg, '{split_idx}', '{cache_data_path}') 生成缓存。"
            )

        with open(meta_filepath, "r") as f:
            block_metadata = json.load(f)

        # 保证顺序与构建时一致
        block_metadata.sort(key=lambda x: (x["block_idx"], x["shard_idx"]))

        self.chunk_files: List[str] = [m["path"] for m in block_metadata]
        self.chunk_sizes: List[int] = [m["size"] for m in block_metadata]
        self.cum_sizes: List[int] = np.cumsum(self.chunk_sizes).tolist()
        self.total_size: int = self.cum_sizes[-1] if self.cum_sizes else 0

        self.current_chunk_idx: int = -1
        self.current_chunk = None

        print(
            f"Initialized ChunkedCTSDataset(split={split_idx}) "
            f"with {self.total_size} samples across {len(self.chunk_files)} blocks."
        )

    def __len__(self) -> int:
        """
        返回数据集中样本总数。
        """
        return self.total_size

    def __getitem__(self, idx: int):
        """
        根据全局索引 idx 返回单个样本 (x, y, set_idx)。

        实现细节
        --------
        1. 使用 bisect_right 在 self.cum_sizes 中找到 idx 对应的 block_idx。
        2. 若当前 block 与上一次访问不同，则释放旧 block（current_chunk）并加载新 block。
           - 加载时显式指定 weights_only=False，以避免 PyTorch 未来更改默认行为。
        3. 计算 idx 在当前 block 内的局部索引 local_idx（如果是第 0 块则 local_idx=idx，否则减去前一块累积大小）。
        4. 从当前 block 的 dict 中取出 X、labels、set_idxs 的对应条目。

        参数
        ----
        idx : int
            全局样本索引（0 <= idx < len(self)）

        返回
        ----
        (x, y, set_idx, esa_score) : tuple
            - x       : torch.Tensor, one-hot 特征（uint8）
            - y       : torch.Tensor, 标签 (1,)
            - set_idx : torch.Tensor, 原始行号 (1,)
            - esa_score: torch.Tensor, esa_score (1,)
            - pos     : torch.Tensor, pos (1,)
        """
        if idx < 0 or idx >= self.total_size:
            raise IndexError("Index out of range")

        import bisect

        # 找到 block 索引
        chunk_idx = bisect.bisect_right(self.cum_sizes, idx)

        # 如果需要，切换到对应 block（懒加载）
        if self.current_chunk_idx != chunk_idx:
            if self.current_chunk is not None:
                # 显式释放旧 block 并触发 GC，有助于降低内存峰值
                del self.current_chunk
                gc.collect()

            self.current_chunk = torch.load(
                self.chunk_files[chunk_idx],
                map_location="cpu",
                weights_only=False,  # 我们加载的是数据，不是模型权重
            )
            self.current_chunk_idx = chunk_idx

        # 计算该样本在当前 block 内的局部索引
        local_idx = idx if chunk_idx == 0 else idx - self.cum_sizes[chunk_idx - 1]

        x = self.current_chunk["X"][local_idx]
        y = self.current_chunk["labels"][local_idx]
        set_idx = self.current_chunk["set_idxs"][local_idx]
        if "esa_scores" in self.current_chunk:
            esa_score = self.current_chunk["esa_scores"][local_idx]
        else:
            esa_score = torch.tensor([0.0], dtype=torch.float32)

        if "pos" in self.current_chunk:
            pos = self.current_chunk["pos"][local_idx]
        else:
            pos = torch.tensor([0.5], dtype=torch.float32)
        return x, y, set_idx, esa_score, pos

    def __repr__(self) -> str:
        """
        返回可读性较好的字符串表示，用于打印调试。
        """
        return (
            f"ChunkedCTSDataset(\n"
            f"  total_samples={self.total_size},\n"
            f"  num_blocks={len(self.chunk_files)}\n"
            f")"
        )
