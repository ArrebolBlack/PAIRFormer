# src/data/sampler.py
"""
sampler.py
======================
本模块提供 ChunkAwareBatchSampler：

- 针对按块存储的数据集（如 ChunkedCTSDataset）设计的 batch 采样器。
- 目标是在保持随机性的前提下，尽量让一个 batch 内的样本来自尽可能少的块（chunk），
  从而降低频繁切换 block 带来的 I/O 抖动和内存抖动。

核心思想
--------
1. 预先根据 cum_sizes 计算每个 chunk 的 [start, end) 索引范围。
2. 每次 __iter__：
   - 先得到所有 chunk_id 列表；
   - 如 shuffle=True，则打乱 chunk_id 顺序；
   - 对于每个 chunk：
       * 在其 [start, end) 范围内生成所有索引；
       * 根据需要 shuffle；
       * 再按 batch_size 切分为一个个 batch 产出。

这样得到的 batch：
- 在一个 chunk 内是随机打乱的；
- chunk 本身出场顺序也可打乱；
- 但不会跨 chunk 织成“大乱序”，有利于按块高效加载。
"""

import math
import random
from typing import List, Tuple, Iterator

from torch.utils.data import Sampler


class ChunkAwareBatchSampler(Sampler[List[int]]):
    """
    Chunk-aware 的 batch 采样器，实现细节见模块 docstring。

    参数
    ----
    cum_sizes : List[int]
        每个 chunk 的累积样本数列表（例如 [n0, n0+n1, n0+n1+n2, ...]），
        通常由 ChunkedCTSDataset.cum_sizes 提供。
    batch_size : int
        每个 batch 的目标样本数。
    drop_last : bool
        若 True，则丢弃不足 batch_size 的最后一批样本；
        若 False，则保留最后一批（可能小于 batch_size）。
    shuffle : bool
        若 True，则：
            - 打乱 chunk 的出场顺序；
            - 打乱 chunk 内的索引顺序。
        若 False，则保持严格顺序（chunk 顺序与样本顺序均不打乱）。
    """

    def __init__(
        self,
        cum_sizes: List[int],
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
    ):
        self.cum = cum_sizes
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        # 将累积大小转换为每个 chunk 的 [start, end) 范围
        self.ranges: List[Tuple[int, int]] = []
        prev = 0
        for c in self.cum:
            self.ranges.append((prev, c))
            prev = c

    def __iter__(self) -> Iterator[List[int]]:
        """
        迭代产生一个个 batch 索引列表，每个列表约有 batch_size 个索引。

        实现细节
        --------
        1. 根据 shuffle 标志，确定 chunk_ids 的遍历顺序。
        2. 对每个 chunk：
            - 生成 [start, end) 内所有索引；
            - 若 shuffle=True，则在 chunk 内打乱索引顺序；
            - 再按 batch_size 切分为多个 batch：
                * 若 drop_last=True，则丢弃最后不足 batch_size 的 batch。
                * 否则保留最后一个小 batch。
        """
        chunk_ids = list(range(len(self.ranges)))
        if self.shuffle:
            random.shuffle(chunk_ids)

        for cid in chunk_ids:
            s, e = self.ranges[cid]
            idxs = list(range(s, e))
            if self.shuffle:
                random.shuffle(idxs)

            # 将该 chunk 的样本按 batch_size 切分
            for i in range(0, len(idxs), self.batch_size):
                batch = idxs[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        """
        返回采样器能够产生的 batch 数量估计值。

        计算方式
        --------
        - 对每个 chunk 单独估算其 batch 数，然后求和：
            * 若 drop_last=True，则使用 n // batch_size
            * 若 drop_last=False，则使用 ceil(n / batch_size)
        """
        total_batches = 0
        prev = 0
        for c in self.cum:
            n = c - prev
            prev = c
            if self.drop_last:
                total_batches += n // self.batch_size
            else:
                total_batches += math.ceil(n / self.batch_size)
        return total_batches

class PairLevelChunkBatchSampler(Sampler[List[int]]):
    """
    Pair-level 专用的 chunk-aware BatchSampler。

    约定：
    - dataset 必须是 PairLevelDataset，且已经加载 chunked cache：
        cache_root/{split}_meta.json
        cache_root/{split}_shardXXXX.pt

    目标：
    - 与 CTS 的 ChunkAwareBatchSampler 一样：
        * batch 尽量来自同一个 shard 的连续 index 段；
        * shard 顺序可 shuffle；
        * shard 内的 pair 顺序也可 shuffle。
    - 配合 PairLevelDataset 中“只保留一个 current_chunk”的实现，
      减少频繁切换 shard 时的 torch.load 开销。
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        # PairLevelDataset 里会设定 mode="chunked"
        if not hasattr(dataset, "mode") or dataset.mode != "chunked":
            raise ValueError(
                "PairLevelChunkBatchSampler 只能用于 chunked 模式的 PairLevelDataset "
                "(需要有 dataset.mode == 'chunked')"
            )
        if not hasattr(dataset, "cum_sizes") or not hasattr(dataset, "chunk_sizes"):
            raise ValueError(
                "PairLevelChunkBatchSampler 需要 dataset.cum_sizes 和 dataset.chunk_sizes 属性，"
                "请确认使用的是 PairLevelDataset 的分块 cache。"
            )

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

        # PairLevelDataset.__init__ 已经构造了 chunk_sizes / cum_sizes
        self.chunk_sizes: List[int] = list(dataset.chunk_sizes)
        self.cum_sizes: List[int] = list(dataset.cum_sizes)

        # 将 cum_sizes 转为每个 shard 的 [start, end) 段
        self.ranges: List[Tuple[int, int]] = []
        prev = 0
        for c in self.cum_sizes:
            self.ranges.append((prev, c))
            prev = c

    def __iter__(self) -> Iterator[List[int]]:
        # 先确定 shard 的出场顺序
        chunk_ids = list(range(len(self.ranges)))
        if self.shuffle:
            random.shuffle(chunk_ids)

        for cid in chunk_ids:
            start, end = self.ranges[cid]
            idxs = list(range(start, end))
            if self.shuffle:
                random.shuffle(idxs)

            # 和 CTS 一样，按 batch_size 切分当前 shard
            n = len(idxs)
            for i in range(0, n, self.batch_size):
                batch = idxs[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        # 估算总 batch 数：对每个 shard 单独算
        total_batches = 0
        for n in self.chunk_sizes:
            if self.drop_last:
                total_batches += n // self.batch_size
            else:
                total_batches += math.ceil(n / self.batch_size)
        return total_batches