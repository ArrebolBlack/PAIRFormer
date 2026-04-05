from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class WindowShardMeta:
    state: str
    split: str
    shard_id: int
    max_samples: int
    num_samples: int
    channels: int = 10
    seq_len: int = 50
    has_esa: bool = True
    has_pos: bool = True


@dataclass
class WindowShardManifest:
    state: str
    split: str
    num_shards: int
    total_samples: int
    total_pairs: int
    source_path: str
    label_policy: str
    esa_min_score: float
    shards: List[Dict[str, int | str]]


class WindowShardWriter:
    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        shard_id: int,
        max_samples: int,
        channels: int = 10,
        seq_len: int = 50,
    ):
        self.root = Path(root)
        self.split = str(split)
        self.shard_id = int(shard_id)
        self.max_samples = int(max_samples)
        self.channels = int(channels)
        self.seq_len = int(seq_len)
        self.shard_dir = self.root / "window_shards" / self.split / f"shard_{self.shard_id:05d}"
        self.shard_dir.mkdir(parents=True, exist_ok=True)

        self.meta = WindowShardMeta(
            state="building",
            split=self.split,
            shard_id=self.shard_id,
            max_samples=self.max_samples,
            num_samples=0,
            channels=self.channels,
            seq_len=self.seq_len,
        )

        shape_x = (self.max_samples, self.channels, self.seq_len)
        shape_1 = (self.max_samples,)
        self.X = np.memmap(self.shard_dir / "X.u8.mmap", mode="w+", dtype=np.uint8, shape=shape_x)
        self.esa = np.memmap(self.shard_dir / "esa.f16.mmap", mode="w+", dtype=np.float16, shape=shape_1)
        self.pos = np.memmap(self.shard_dir / "pos.f16.mmap", mode="w+", dtype=np.float16, shape=shape_1)
        self.label = np.memmap(self.shard_dir / "label.i8.mmap", mode="w+", dtype=np.int8, shape=shape_1)
        self.pair_id = np.memmap(self.shard_dir / "pair_id.i32.mmap", mode="w+", dtype=np.int32, shape=shape_1)

        self._cursor = 0
        self._write_meta()

    def _write_meta(self) -> None:
        with open(self.shard_dir / "meta.json", "w") as f:
            json.dump(asdict(self.meta), f, indent=2, sort_keys=True)

    @property
    def remaining(self) -> int:
        return int(self.max_samples - self._cursor)

    @property
    def num_samples(self) -> int:
        return int(self._cursor)

    def write_batch(
        self,
        *,
        X: torch.Tensor,
        esa: torch.Tensor,
        pos: torch.Tensor,
        label: torch.Tensor,
        pair_id: torch.Tensor,
    ) -> int:
        n = int(X.shape[0])
        if n <= 0:
            return 0
        if n > self.remaining:
            raise RuntimeError(f"Shard full: shard={self.shard_id} need={n} remaining={self.remaining}")

        s, e = self._cursor, self._cursor + n
        self.X[s:e] = X.detach().cpu().numpy().astype(np.uint8, copy=False)
        self.esa[s:e] = esa.detach().cpu().numpy().astype(np.float16, copy=False)
        self.pos[s:e] = pos.detach().cpu().numpy().astype(np.float16, copy=False)
        self.label[s:e] = label.detach().cpu().numpy().astype(np.int8, copy=False)
        self.pair_id[s:e] = pair_id.detach().cpu().numpy().astype(np.int32, copy=False)
        self._cursor = e
        self.meta.num_samples = int(self._cursor)
        return n

    def flush(self) -> None:
        self.X.flush()
        self.esa.flush()
        self.pos.flush()
        self.label.flush()
        self.pair_id.flush()
        self._write_meta()

    def set_ready(self) -> Dict[str, int | str]:
        self.meta.state = "ready"
        self.meta.num_samples = int(self._cursor)
        self.flush()
        return {
            "shard_id": int(self.shard_id),
            "path": str(self.shard_dir),
            "num_samples": int(self._cursor),
        }


def write_window_shard_manifest(
    root: str | Path,
    *,
    split: str,
    source_path: str,
    label_policy: str,
    esa_min_score: float,
    total_pairs: int,
    shards: List[Dict[str, int | str]],
) -> Path:
    root = Path(root)
    split_dir = root / "window_shards" / str(split)
    split_dir.mkdir(parents=True, exist_ok=True)
    total_samples = int(sum(int(x["num_samples"]) for x in shards))
    manifest = WindowShardManifest(
        state="ready",
        split=str(split),
        num_shards=int(len(shards)),
        total_samples=total_samples,
        total_pairs=int(total_pairs),
        source_path=str(source_path),
        label_policy=str(label_policy),
        esa_min_score=float(esa_min_score),
        shards=shards,
    )
    out = split_dir / "manifest.json"
    with open(out, "w") as f:
        json.dump(asdict(manifest), f, indent=2, sort_keys=True)
    return out


class WindowShardReader:
    def __init__(self, shard_dir: str | Path):
        self.shard_dir = Path(shard_dir)
        with open(self.shard_dir / "meta.json", "r") as f:
            self.meta = WindowShardMeta(**json.load(f))
        n = int(self.meta.num_samples)
        c = int(self.meta.channels)
        l = int(self.meta.seq_len)
        self.X = np.memmap(self.shard_dir / "X.u8.mmap", mode="r", dtype=np.uint8, shape=(n, c, l))
        self.esa = np.memmap(self.shard_dir / "esa.f16.mmap", mode="r", dtype=np.float16, shape=(n,))
        self.pos = np.memmap(self.shard_dir / "pos.f16.mmap", mode="r", dtype=np.float16, shape=(n,))
        self.label = np.memmap(self.shard_dir / "label.i8.mmap", mode="r", dtype=np.int8, shape=(n,))
        self.pair_id = np.memmap(self.shard_dir / "pair_id.i32.mmap", mode="r", dtype=np.int32, shape=(n,))

    def __len__(self) -> int:
        return int(self.meta.num_samples)

    def read(self, idx: int) -> Dict[str, torch.Tensor]:
        i = int(idx)
        return {
            "inputs": torch.from_numpy(np.array(self.X[i], copy=True)),
            "labels": torch.tensor(float(self.label[i]), dtype=torch.float32),
            "set_idx": torch.tensor(int(self.pair_id[i]), dtype=torch.long),
            "esa_scores": torch.tensor(float(self.esa[i]), dtype=torch.float32),
            "pos": torch.tensor(float(self.pos[i]), dtype=torch.float32),
            "valid_mask": torch.tensor(bool(int(self.label[i]) >= 0), dtype=torch.bool),
        }


def load_window_shard_manifest(root: str | Path, *, split: str) -> WindowShardManifest:
    path = Path(root) / "window_shards" / str(split) / "manifest.json"
    with open(path, "r") as f:
        return WindowShardManifest(**json.load(f))
