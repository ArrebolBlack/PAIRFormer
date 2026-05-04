from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import get_context
from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import torch

from src.data.stream_pair_dataset import PairRecord
from src.data.window_shard_cache import WindowShardWriter, write_window_shard_manifest
from src.precompute.online_cts_generator import iter_cts_candidates


@dataclass
class WindowShardBuildConfig:
    esa_min_score: float = 6.0
    num_workers: int = 8
    task_pairs: int = 16
    mp_start_method: str = "spawn"
    max_samples_per_shard: int = 2_000_000
    seed: int = 2020
    label_policy: str = "all_positive"  # all_positive | pseudo_topr
    pseudo_topr: int = 8
    positive_score: str = "esa"  # esa
    drop_ignore_at_build: bool = False
    negative_keep_prob: float = 1.0


@dataclass
class PairWindowsChunk:
    X: torch.Tensor
    esa: torch.Tensor
    pos: torch.Tensor
    label: torch.Tensor
    pair_id: torch.Tensor
    num_pairs: int


def _chunk_records(records: Iterable[PairRecord], chunk_size: int) -> Iterator[List[PairRecord]]:
    buf: List[PairRecord] = []
    for rec in records:
        buf.append(rec)
        if len(buf) >= int(chunk_size):
            yield buf
            buf = []
    if buf:
        yield buf


def _assign_window_labels(
    pair_label: float, esa_vals: List[float], *, label_policy: str, pseudo_topr: int
) -> List[int]:
    n = len(esa_vals)
    if n == 0:
        return []
    if float(pair_label) <= 0:
        return [0] * n
    if str(label_policy) == "all_positive":
        return [1] * n
    if str(label_policy) == "pseudo_topr":
        r = max(1, min(int(pseudo_topr), n))
        order = sorted(range(n), key=lambda i: float(esa_vals[i]), reverse=True)
        keep = set(order[:r])
        return [1 if i in keep else -1 for i in range(n)]
    raise ValueError(f"Unknown label_policy: {label_policy}")


def _process_pair_chunk(
    task: Tuple[Sequence[PairRecord], object, WindowShardBuildConfig]
) -> PairWindowsChunk:
    records, data_cfg, cfg = task
    xs_all: List[torch.Tensor] = []
    esa_all: List[float] = []
    pos_all: List[float] = []
    label_all: List[int] = []
    pair_id_all: List[int] = []

    for rec in records:
        xs_pair: List[torch.Tensor] = []
        esa_pair: List[float] = []
        pos_pair: List[float] = []
        for cand in iter_cts_candidates(rec, data_cfg, esa_min_score=float(cfg.esa_min_score)):
            xs_pair.append(cand.x_u8)
            esa_pair.append(float(cand.esa_score))
            pos_pair.append(float(cand.pos))
        labels_pair = _assign_window_labels(
            float(rec.label),
            esa_pair,
            label_policy=str(cfg.label_policy),
            pseudo_topr=int(cfg.pseudo_topr),
        )
        rng = np.random.default_rng(int(cfg.seed) ^ int(rec.pair_id))
        for x, esa, pos, lab in zip(xs_pair, esa_pair, pos_pair, labels_pair):
            if bool(cfg.drop_ignore_at_build) and int(lab) < 0:
                continue
            if int(lab) == 0 and float(cfg.negative_keep_prob) < 1.0:
                if float(rng.random()) >= float(cfg.negative_keep_prob):
                    continue
            xs_all.append(x)
            esa_all.append(float(esa))
            pos_all.append(float(pos))
            label_all.append(int(lab))
            pair_id_all.append(int(rec.pair_id))

    if xs_all:
        X = torch.stack(xs_all, dim=0).to(torch.uint8)
        esa = torch.tensor(esa_all, dtype=torch.float32)
        pos = torch.tensor(pos_all, dtype=torch.float32)
        label = torch.tensor(label_all, dtype=torch.int8)
        pair_id = torch.tensor(pair_id_all, dtype=torch.int32)
    else:
        X = torch.empty((0, 10, 50), dtype=torch.uint8)
        esa = torch.empty((0,), dtype=torch.float32)
        pos = torch.empty((0,), dtype=torch.float32)
        label = torch.empty((0,), dtype=torch.int8)
        pair_id = torch.empty((0,), dtype=torch.int32)
    return PairWindowsChunk(
        X=X, esa=esa, pos=pos, label=label, pair_id=pair_id, num_pairs=len(records)
    )


class WindowShardBuilderParallel:
    def __init__(self, *, root: str, split: str, source_path: str, cfg: WindowShardBuildConfig):
        self.root = str(root)
        self.split = str(split)
        self.source_path = str(source_path)
        self.cfg = cfg

    def build(self, records: Iterable[PairRecord], data_cfg) -> None:
        ctx = get_context(str(self.cfg.mp_start_method))
        task_iter = (
            (chunk, data_cfg, self.cfg)
            for chunk in _chunk_records(records, int(self.cfg.task_pairs))
        )

        shard_id = 0
        writer = WindowShardWriter(
            self.root,
            split=self.split,
            shard_id=shard_id,
            max_samples=int(self.cfg.max_samples_per_shard),
        )
        shard_infos: List[dict] = []
        total_pairs = 0

        with ctx.Pool(processes=int(self.cfg.num_workers)) as pool:
            for chunk in pool.imap_unordered(_process_pair_chunk, task_iter, chunksize=1):
                total_pairs += int(chunk.num_pairs)
                offset = 0
                n = int(chunk.X.shape[0])
                while offset < n:
                    room = int(writer.remaining)
                    if room <= 0:
                        shard_infos.append(writer.set_ready())
                        shard_id += 1
                        writer = WindowShardWriter(
                            self.root,
                            split=self.split,
                            shard_id=shard_id,
                            max_samples=int(self.cfg.max_samples_per_shard),
                        )
                        room = int(writer.remaining)
                    take = min(room, n - offset)
                    writer.write_batch(
                        X=chunk.X[offset : offset + take],
                        esa=chunk.esa[offset : offset + take],
                        pos=chunk.pos[offset : offset + take],
                        label=chunk.label[offset : offset + take],
                        pair_id=chunk.pair_id[offset : offset + take],
                    )
                    offset += take

        if writer.num_samples > 0 or not shard_infos:
            shard_infos.append(writer.set_ready())

        write_window_shard_manifest(
            self.root,
            split=self.split,
            source_path=self.source_path,
            label_policy=str(self.cfg.label_policy),
            esa_min_score=float(self.cfg.esa_min_score),
            total_pairs=int(total_pairs),
            shards=shard_infos,
        )
