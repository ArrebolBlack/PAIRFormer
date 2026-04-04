from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


@dataclass
class SelectedPairCacheMeta:
    state: str
    split: str
    cache_type: str
    num_pairs: int
    kmax: int
    channels: int = 10
    seq_len: int = 50
    has_cheap_logit: bool = False
    has_cheap_emb: bool = False
    cheap_emb_dim: int = 64
    has_inst_logit: bool = False
    has_inst_emb: bool = False
    inst_emb_dim: int = 384


class _BaseSelectedPairCacheWriter:
    def __init__(self, root: str | Path, *, split: str, cache_type: str, meta: SelectedPairCacheMeta):
        self.root = Path(root)
        self.split = str(split)
        self.cache_dir = self.root / "selected_pair_cache" / self.split / str(cache_type)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.meta = meta
        self.label = np.memmap(self.cache_dir / "label.f32.mmap", mode="w+", dtype=np.float32, shape=(int(meta.num_pairs),))
        self.sel_len = np.memmap(self.cache_dir / "sel_len.i16.mmap", mode="w+", dtype=np.int16, shape=(int(meta.num_pairs),))
        self.esa = np.memmap(self.cache_dir / "esa.f16.mmap", mode="w+", dtype=np.float16, shape=(int(meta.num_pairs), int(meta.kmax)))
        self.pos = np.memmap(self.cache_dir / "pos.f16.mmap", mode="w+", dtype=np.float16, shape=(int(meta.num_pairs), int(meta.kmax)))
        self.label[:] = 0
        self.sel_len[:] = 0
        self.esa[:] = 0
        self.pos[:] = 0
        self._write_meta()

    def _write_meta(self) -> None:
        with open(self.cache_dir / "meta.json", "w") as f:
            json.dump(asdict(self.meta), f, indent=2, sort_keys=True)

    def flush_common(self) -> None:
        self.label.flush()
        self.sel_len.flush()
        self.esa.flush()
        self.pos.flush()

    def set_ready(self) -> None:
        self.flush()
        self.meta.state = "ready"
        self._write_meta()

    def flush(self) -> None:
        self.flush_common()


class SelectedPairCacheWriter(_BaseSelectedPairCacheWriter):
    """
    Writer for `selected_raw` cache.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        num_pairs: int,
        kmax: int,
        channels: int = 10,
        seq_len: int = 50,
        cheap_emb_dim: int = 64,
        has_cheap_logit: bool = True,
        has_cheap_emb: bool = False,
    ):
        meta = SelectedPairCacheMeta(
            state="building",
            split=str(split),
            cache_type="selected_raw",
            num_pairs=int(num_pairs),
            kmax=int(kmax),
            channels=int(channels),
            seq_len=int(seq_len),
            has_cheap_logit=bool(has_cheap_logit),
            has_cheap_emb=bool(has_cheap_emb),
            cheap_emb_dim=int(cheap_emb_dim),
            has_inst_logit=False,
            has_inst_emb=False,
        )
        super().__init__(root, split=split, cache_type="selected_raw", meta=meta)

        P, K, C, L = int(num_pairs), int(kmax), int(channels), int(seq_len)
        self.X = np.memmap(self.cache_dir / "X.u8.mmap", mode="w+", dtype=np.uint8, shape=(P, K, C, L))
        self.X[:] = 0

        self.cheap_logit = None
        if meta.has_cheap_logit:
            self.cheap_logit = np.memmap(self.cache_dir / "cheap_logit.f16.mmap", mode="w+", dtype=np.float16, shape=(P, K))
            self.cheap_logit[:] = 0

        self.cheap_emb = None
        if meta.has_cheap_emb:
            self.cheap_emb = np.memmap(
                self.cache_dir / "cheap_emb.f16.mmap",
                mode="w+",
                dtype=np.float16,
                shape=(P, K, int(cheap_emb_dim)),
            )
            self.cheap_emb[:] = 0

    def write_pair(
        self,
        pair_id: int,
        *,
        X: torch.Tensor,
        esa: torch.Tensor,
        pos: torch.Tensor,
        label: float,
        sel_len: int,
        cheap_logit: Optional[torch.Tensor] = None,
        cheap_emb: Optional[torch.Tensor] = None,
    ) -> None:
        pid = int(pair_id)
        n = int(sel_len)
        self.sel_len[pid] = n
        self.label[pid] = float(label)
        if n <= 0:
            return

        self.X[pid, :n] = X[:n].detach().cpu().numpy().astype(np.uint8, copy=False)
        self.esa[pid, :n] = esa[:n].detach().cpu().numpy().astype(np.float16, copy=False)
        self.pos[pid, :n] = pos[:n].detach().cpu().numpy().astype(np.float16, copy=False)
        if self.cheap_logit is not None and cheap_logit is not None:
            self.cheap_logit[pid, :n] = cheap_logit[:n].detach().cpu().numpy().astype(np.float16, copy=False)
        if self.cheap_emb is not None and cheap_emb is not None:
            self.cheap_emb[pid, :n] = cheap_emb[:n].detach().cpu().numpy().astype(np.float16, copy=False)

    def flush(self) -> None:
        self.flush_common()
        self.X.flush()
        if self.cheap_logit is not None:
            self.cheap_logit.flush()
        if self.cheap_emb is not None:
            self.cheap_emb.flush()


class SelectedInstPairCacheWriter(_BaseSelectedPairCacheWriter):
    """
    Writer for `selected_inst` cache.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        num_pairs: int,
        kmax: int,
        inst_emb_dim: int = 384,
        has_inst_logit: bool = True,
    ):
        meta = SelectedPairCacheMeta(
            state="building",
            split=str(split),
            cache_type="selected_inst",
            num_pairs=int(num_pairs),
            kmax=int(kmax),
            has_inst_logit=bool(has_inst_logit),
            has_inst_emb=True,
            inst_emb_dim=int(inst_emb_dim),
        )
        super().__init__(root, split=split, cache_type="selected_inst", meta=meta)

        P, K, D = int(num_pairs), int(kmax), int(inst_emb_dim)
        self.inst_emb = np.memmap(self.cache_dir / "inst_emb.f16.mmap", mode="w+", dtype=np.float16, shape=(P, K, D))
        self.inst_emb[:] = 0
        self.inst_logit = None
        if meta.has_inst_logit:
            self.inst_logit = np.memmap(self.cache_dir / "inst_logit.f16.mmap", mode="w+", dtype=np.float16, shape=(P, K))
            self.inst_logit[:] = 0

    def write_pair(
        self,
        pair_id: int,
        *,
        inst_emb: torch.Tensor,
        inst_logit: Optional[torch.Tensor],
        esa: torch.Tensor,
        pos: torch.Tensor,
        label: float,
        sel_len: int,
    ) -> None:
        pid = int(pair_id)
        n = int(sel_len)
        self.sel_len[pid] = n
        self.label[pid] = float(label)
        if n <= 0:
            return

        self.inst_emb[pid, :n] = inst_emb[:n].detach().cpu().numpy().astype(np.float16, copy=False)
        self.esa[pid, :n] = esa[:n].detach().cpu().numpy().astype(np.float16, copy=False)
        self.pos[pid, :n] = pos[:n].detach().cpu().numpy().astype(np.float16, copy=False)
        if self.inst_logit is not None and inst_logit is not None:
            self.inst_logit[pid, :n] = inst_logit[:n].detach().cpu().numpy().astype(np.float16, copy=False)

    def flush(self) -> None:
        self.flush_common()
        self.inst_emb.flush()
        if self.inst_logit is not None:
            self.inst_logit.flush()


class SelectedPairCacheReader:
    def __init__(self, root: str | Path, *, split: str, cache_type: str = "selected_raw"):
        self.root = Path(root)
        self.split = str(split)
        self.cache_type = str(cache_type)
        self.cache_dir = self.root / "selected_pair_cache" / self.split / self.cache_type

        with open(self.cache_dir / "meta.json", "r") as f:
            self.meta = SelectedPairCacheMeta(**json.load(f))

        P = int(self.meta.num_pairs)
        K = int(self.meta.kmax)

        self.label = np.memmap(self.cache_dir / "label.f32.mmap", mode="r", dtype=np.float32, shape=(P,))
        self.sel_len = np.memmap(self.cache_dir / "sel_len.i16.mmap", mode="r", dtype=np.int16, shape=(P,))
        self.esa = np.memmap(self.cache_dir / "esa.f16.mmap", mode="r", dtype=np.float16, shape=(P, K))
        self.pos = np.memmap(self.cache_dir / "pos.f16.mmap", mode="r", dtype=np.float16, shape=(P, K))

        self.X = None
        self.cheap_logit = None
        self.cheap_emb = None
        self.inst_emb = None
        self.inst_logit = None

        if self.meta.cache_type == "selected_raw":
            C, L = int(self.meta.channels), int(self.meta.seq_len)
            self.X = np.memmap(self.cache_dir / "X.u8.mmap", mode="r", dtype=np.uint8, shape=(P, K, C, L))
            if self.meta.has_cheap_logit:
                self.cheap_logit = np.memmap(self.cache_dir / "cheap_logit.f16.mmap", mode="r", dtype=np.float16, shape=(P, K))
            if self.meta.has_cheap_emb:
                self.cheap_emb = np.memmap(
                    self.cache_dir / "cheap_emb.f16.mmap",
                    mode="r",
                    dtype=np.float16,
                    shape=(P, K, int(self.meta.cheap_emb_dim)),
                )
        elif self.meta.cache_type == "selected_inst":
            self.inst_emb = np.memmap(
                self.cache_dir / "inst_emb.f16.mmap",
                mode="r",
                dtype=np.float16,
                shape=(P, K, int(self.meta.inst_emb_dim)),
            )
            if self.meta.has_inst_logit:
                self.inst_logit = np.memmap(self.cache_dir / "inst_logit.f16.mmap", mode="r", dtype=np.float16, shape=(P, K))
        else:
            raise ValueError(f"Unknown selected pair cache_type: {self.meta.cache_type}")

    def __len__(self) -> int:
        return int(self.meta.num_pairs)

    def read_pair(self, pair_id: int) -> Dict[str, torch.Tensor | int | float | None]:
        pid = int(pair_id)
        out: Dict[str, torch.Tensor | int | float | None] = {
            "pair_id": pid,
            "y_pair": float(self.label[pid]),
            "sel_len": int(self.sel_len[pid]),
            "esa": torch.from_numpy(np.array(self.esa[pid], dtype=np.float32, copy=True)),
            "pos": torch.from_numpy(np.array(self.pos[pid], dtype=np.float32, copy=True)),
            "X": None,
            "cheap_logit": None,
            "cheap_emb": None,
            "inst_emb": None,
            "inst_logit": None,
        }
        if self.X is not None:
            out["X"] = torch.from_numpy(np.array(self.X[pid], copy=True))
        if self.cheap_logit is not None:
            out["cheap_logit"] = torch.from_numpy(np.array(self.cheap_logit[pid], dtype=np.float32, copy=True))
        if self.cheap_emb is not None:
            out["cheap_emb"] = torch.from_numpy(np.array(self.cheap_emb[pid], dtype=np.float32, copy=True))
        if self.inst_emb is not None:
            out["inst_emb"] = torch.from_numpy(np.array(self.inst_emb[pid], dtype=np.float32, copy=True))
        if self.inst_logit is not None:
            out["inst_logit"] = torch.from_numpy(np.array(self.inst_logit[pid], dtype=np.float32, copy=True))
        return out
