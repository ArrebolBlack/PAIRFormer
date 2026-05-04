from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


@dataclass
class TeacherShardMeta:
    state: str
    split: str
    shard_id: int
    num_samples: int
    emb_dim: int
    has_feat: bool


@dataclass
class TeacherShardManifest:
    state: str
    split: str
    num_shards: int
    total_samples: int
    emb_dim: int
    has_feat: bool
    source_window_root: str
    shards: List[Dict[str, int | str]]


class TeacherShardWriter:
    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        shard_id: int,
        num_samples: int,
        emb_dim: int,
        has_feat: bool,
    ):
        self.root = Path(root)
        self.split = str(split)
        self.shard_id = int(shard_id)
        self.num_samples = int(num_samples)
        self.emb_dim = int(emb_dim)
        self.has_feat = bool(has_feat)
        self.shard_dir = self.root / "teacher_shards" / self.split / f"shard_{self.shard_id:05d}"
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.logit = np.memmap(
            self.shard_dir / "teacher_logit.f16.mmap",
            mode="w+",
            dtype=np.float16,
            shape=(self.num_samples,),
        )
        self.feat = None
        if self.has_feat:
            self.feat = np.memmap(
                self.shard_dir / "teacher_feat.f16.mmap",
                mode="w+",
                dtype=np.float16,
                shape=(self.num_samples, self.emb_dim),
            )
        self.meta = TeacherShardMeta(
            state="building",
            split=self.split,
            shard_id=self.shard_id,
            num_samples=self.num_samples,
            emb_dim=self.emb_dim,
            has_feat=self.has_feat,
        )
        self._write_meta()

    def _write_meta(self) -> None:
        with open(self.shard_dir / "meta.json", "w") as f:
            json.dump(asdict(self.meta), f, indent=2, sort_keys=True)

    def write_batch(
        self, start: int, *, logit: torch.Tensor, feat: torch.Tensor | None = None
    ) -> None:
        s = int(start)
        e = s + int(logit.shape[0])
        self.logit[s:e] = logit.detach().cpu().numpy().astype(np.float16, copy=False)
        if self.feat is not None and feat is not None:
            self.feat[s:e] = feat.detach().cpu().numpy().astype(np.float16, copy=False)

    def set_ready(self) -> Dict[str, int | str]:
        self.logit.flush()
        if self.feat is not None:
            self.feat.flush()
        self.meta.state = "ready"
        self._write_meta()
        return {
            "shard_id": self.shard_id,
            "path": str(self.shard_dir),
            "num_samples": self.num_samples,
        }


def write_teacher_shard_manifest(
    root: str | Path,
    *,
    split: str,
    source_window_root: str,
    emb_dim: int,
    has_feat: bool,
    shards: List[Dict[str, int | str]],
) -> Path:
    root = Path(root)
    split_dir = root / "teacher_shards" / str(split)
    split_dir.mkdir(parents=True, exist_ok=True)
    total_samples = int(sum(int(x["num_samples"]) for x in shards))
    manifest = TeacherShardManifest(
        state="ready",
        split=str(split),
        num_shards=int(len(shards)),
        total_samples=total_samples,
        emb_dim=int(emb_dim),
        has_feat=bool(has_feat),
        source_window_root=str(source_window_root),
        shards=shards,
    )
    out = split_dir / "manifest.json"
    with open(out, "w") as f:
        json.dump(asdict(manifest), f, indent=2, sort_keys=True)
    return out


class TeacherShardReader:
    def __init__(self, shard_dir: str | Path):
        self.shard_dir = Path(shard_dir)
        with open(self.shard_dir / "meta.json", "r") as f:
            self.meta = TeacherShardMeta(**json.load(f))
        n = int(self.meta.num_samples)
        self.logit = np.memmap(
            self.shard_dir / "teacher_logit.f16.mmap", mode="r", dtype=np.float16, shape=(n,)
        )
        self.feat = None
        if bool(self.meta.has_feat):
            self.feat = np.memmap(
                self.shard_dir / "teacher_feat.f16.mmap",
                mode="r",
                dtype=np.float16,
                shape=(n, int(self.meta.emb_dim)),
            )

    def read(self, idx: int) -> Dict[str, torch.Tensor | None]:
        i = int(idx)
        out: Dict[str, torch.Tensor | None] = {
            "teacher_logit": torch.tensor(float(self.logit[i]), dtype=torch.float32),
            "teacher_feat": None,
        }
        if self.feat is not None:
            out["teacher_feat"] = torch.from_numpy(
                np.array(self.feat[i], dtype=np.float32, copy=True)
            )
        return out


def load_teacher_shard_manifest(root: str | Path, *, split: str) -> TeacherShardManifest:
    path = Path(root) / "teacher_shards" / str(split) / "manifest.json"
    with open(path, "r") as f:
        return TeacherShardManifest(**json.load(f))
