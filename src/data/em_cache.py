# src/data/em_cache.py
from __future__ import annotations

import json
import os
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Iterable, Dict, Any, List, Union

import numpy as np
import torch


def _dtype_str(x) -> str:
    return str(np.dtype(x))

def _dtype_from_str(s: str) -> np.dtype:
    return np.dtype(s)

def _check_meta_identity(kind: str, meta: Dict[str, Any], *, split: str, path_hash: str, dataset_hash_key: str) -> None:
    if str(meta.get("split")) != str(split):
        raise RuntimeError(f"[{kind}] meta.split={meta.get('split')} != store.split={split}")
    if str(meta.get("path_hash")) != str(path_hash):
        raise RuntimeError(f"[{kind}] meta.path_hash={meta.get('path_hash')} != store.path_hash={path_hash}")
    if str(meta.get("dataset_hash_key")) != str(dataset_hash_key):
        raise RuntimeError(f"[{kind}] meta.dataset_hash_key={meta.get('dataset_hash_key')} != store.dataset_hash_key={dataset_hash_key}")

def _check_state(kind: str, meta_state: str, *, require_ready: bool) -> None:
    if require_ready and str(meta_state) != "ready":
        raise RuntimeError(f"[{kind}] cache not ready: state={meta_state}")

def _check_version(kind: str, meta_val: str, req_val: str, field: str) -> None:
    if str(meta_val) != str(req_val):
        raise RuntimeError(f"[{kind}] meta.{field}={meta_val} != requested {field}={req_val}")

def _check_file_size(path: Path, expected_bytes: int, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"[{kind}] memmap file missing: {path}")
    actual = path.stat().st_size
    if actual != int(expected_bytes):
        raise RuntimeError(f"[{kind}] memmap file size mismatch: {path} actual={actual} expected={expected_bytes}")

def _nbytes(dtype: np.dtype, shape: tuple) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return int(dtype.itemsize * n)


def _atomic_write_json(obj: Dict[str, Any], path: Union[str, Path]) -> None:
    path = str(path)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)



class FileLock:
    """
    极简文件锁：创建 lock 文件（O_EXCL）作为互斥。
    适用于“单机单进程构建 cache”，避免误并发写坏文件。
    """
    def __init__(self, lock_path: Union[str, Path]):
        self.lock_path = str(lock_path)
        self.fd = None

    def __enter__(self):
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            self.fd = os.open(self.lock_path, flags)
            os.write(self.fd, f"pid={os.getpid()} time={time.time()}\n".encode("utf-8"))
            return self
        except FileExistsError:
            raise RuntimeError(f"Lock exists: {self.lock_path}. Another process maybe building this cache.")

    def __exit__(self, exc_type, exc, tb):
        try: 
            if self.fd is not None:
                os.close(self.fd)
            if os.path.exists(self.lock_path):
                os.remove(self.lock_path)
        finally:
            self.fd = None


def _ensure_dir(p: Union[str, Path]) -> str:
    p = str(p)
    os.makedirs(p, exist_ok=True)
    return p

def _compute_hash(s: str, n: int = 16) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:n]


@dataclass
class CheapMeta:
    state: str  # "building" | "ready"
    cheap_version: str
    dataset_hash_key: str
    path_hash: str
    split: str
    total_cts: int
    emb_dim: int
    dtype_logits: str
    dtype_emb: str
    has_entropy: bool
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class SelectionMeta:
    state: str  # "building" | "ready"
    sel_version: str
    cheap_version_used: str
    dataset_hash_key: str
    path_hash: str
    split: str
    num_pairs: int
    kmax: int
    dtype_uids: str
    dtype_len: str
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class InstanceMeta:
    state: str  # "building" | "ready"
    inst_version: str
    cheap_version_used: str
    sel_version_used: str
    dataset_hash_key: str
    path_hash: str
    split: str
    total_cts: int
    emb_dim: int
    dtype_logits: str
    dtype_emb: str
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


class MemmapCacheStore:
    """
    一个最小可用的 CacheStore：
    - CheapCache：按 cts_uid 连续 slice 写入/读取
    - SelectionCache：按 pair_id 写入/读取 [num_pairs, Kmax]
    - meta.json 记录版本与一致性
    """

    def __init__(self, cache_root: str, split: str, path_hash: str, dataset_hash_key: str):
        self.cache_root = str(cache_root)
        self.split = str(split)
        self.path_hash = str(path_hash)
        self.dataset_hash_key = str(dataset_hash_key)

        self.cheap_dir = _ensure_dir(Path(cache_root) / "em_cache" / split / "cheap")
        self.sel_dir = _ensure_dir(Path(cache_root) / "em_cache" / split / "selection")

        self._cheap_meta_path = str(Path(self.cheap_dir) / "meta.json")
        self._sel_meta_path = str(Path(self.sel_dir) / "meta.json")

        # memmap handles
        self._logits = None
        self._emb = None
        self._entropy = None

        self._sel_uids = None
        self._sel_len = None

        # cached meta
        self.cheap_meta: Optional[CheapMeta] = None
        self.sel_meta: Optional[SelectionMeta] = None

        # instance cache
        self.inst_dir = _ensure_dir(Path(cache_root) / "em_cache" / split / "instance")
        self._inst_meta_path = str(Path(self.inst_dir) / "meta.json")

        self._inst_logit = None
        self._inst_emb = None

        self.inst_meta: Optional[InstanceMeta] = None

    
    # ------------------- CheapCache -------------------
        
    def create_or_open_cheap(
        self,
        total_cts: int,
        emb_dim: int,
        cheap_version: str,
        *,
        dtype_logits=np.float16,
        dtype_emb=np.float16,
        has_entropy: bool = False,
        overwrite: bool = False,
        require_ready: bool = False,
    ) -> None:
        logits_path = Path(self.cheap_dir) / "cheap_logits.f16.mmap"
        emb_path = Path(self.cheap_dir) / "cheap_emb.f16.mmap"
        entropy_path = Path(self.cheap_dir) / "cheap_entropy.f16.mmap"

        lock_path = Path(self.cheap_dir) / ".build.lock"

        if os.path.exists(self._cheap_meta_path) and not overwrite:
            # open existing
            with open(self._cheap_meta_path, "r") as f:
                d = json.load(f)

            # identity + state
            _check_meta_identity("CheapCache", d, split=self.split, path_hash=self.path_hash, dataset_hash_key=self.dataset_hash_key)
            _check_state("CheapCache", d.get("state", ""), require_ready=require_ready)

            # version
            _check_version("CheapCache", d.get("cheap_version", ""), cheap_version, "cheap_version")

            # shape
            if int(d.get("total_cts")) != int(total_cts) or int(d.get("emb_dim")) != int(emb_dim):
                raise RuntimeError(
                    f"CheapCache exists but shape mismatch: meta(total_cts={d.get('total_cts')}, emb_dim={d.get('emb_dim')}) "
                    f"vs requested(total_cts={total_cts}, emb_dim={emb_dim})."
                )

            # dtype（必须与 meta 一致；否则 memmap 读错）
            meta_dtype_logits = _dtype_from_str(d.get("dtype_logits"))
            meta_dtype_emb = _dtype_from_str(d.get("dtype_emb"))
            if _dtype_str(dtype_logits) != str(meta_dtype_logits):
                raise RuntimeError(f"[CheapCache] dtype_logits mismatch: meta={meta_dtype_logits} requested={_dtype_str(dtype_logits)}")
            if _dtype_str(dtype_emb) != str(meta_dtype_emb):
                raise RuntimeError(f"[CheapCache] dtype_emb mismatch: meta={meta_dtype_emb} requested={_dtype_str(dtype_emb)}")

            # has_entropy 一致性
            meta_has_entropy = bool(d.get("has_entropy", False))
            if bool(has_entropy) != meta_has_entropy:
                raise RuntimeError(f"[CheapCache] has_entropy mismatch: meta={meta_has_entropy} requested={bool(has_entropy)}")

            # 文件完整性（存在 + size 正确）
            _check_file_size(logits_path, _nbytes(meta_dtype_logits, (int(total_cts),)), "CheapCache")
            _check_file_size(emb_path, _nbytes(meta_dtype_emb, (int(total_cts), int(emb_dim))), "CheapCache")
            if meta_has_entropy:
                _check_file_size(entropy_path, _nbytes(meta_dtype_logits, (int(total_cts),)), "CheapCache")

            # 通过校验后再创建 memmap（用 meta dtype 打开）
            self.cheap_meta = CheapMeta(**d)
            self._logits = np.memmap(logits_path, mode="r+", dtype=meta_dtype_logits, shape=(total_cts,))
            self._emb = np.memmap(emb_path, mode="r+", dtype=meta_dtype_emb, shape=(total_cts, emb_dim))
            self._entropy = np.memmap(entropy_path, mode="r+", dtype=meta_dtype_logits, shape=(total_cts,)) if meta_has_entropy else None
            return

        
        # create new
        with FileLock(lock_path):
            meta = CheapMeta(
                state="building",
                cheap_version=str(cheap_version),
                dataset_hash_key=self.dataset_hash_key,
                path_hash=self.path_hash,
                split=self.split,
                total_cts=int(total_cts),
                emb_dim=int(emb_dim),
                dtype_logits=str(np.dtype(dtype_logits)),
                dtype_emb=str(np.dtype(dtype_emb)),
                has_entropy=bool(has_entropy),
                created_at=time.time(),
            )
            _atomic_write_json(meta.to_dict(), self._cheap_meta_path)
            self.cheap_meta = meta

            self._logits = np.memmap(logits_path, mode="w+", dtype=dtype_logits, shape=(total_cts,))
            self._emb = np.memmap(emb_path, mode="w+", dtype=dtype_emb, shape=(total_cts, emb_dim))
            if has_entropy:
                self._entropy = np.memmap(entropy_path, mode="w+", dtype=dtype_logits, shape=(total_cts,))
            else:
                self._entropy = None

            # init (optional): fill NaN helps detect missing writes
            self._logits[:] = np.nan
            self._emb[:] = 0
            if self._entropy is not None:
                self._entropy[:] = np.nan

            self.flush_cheap()

    def set_cheap_ready(self) -> None:
        if self.cheap_meta is None:
            raise RuntimeError("CheapCache not initialized.")
        self.cheap_meta.state = "ready"
        _atomic_write_json(self.cheap_meta.to_dict(), self._cheap_meta_path)

    def write_cheap_slice(
        self,
        start: int, 
        end: int, 
        logits: torch.Tensor,
        emb: torch.Tensor,
        entropy: Optional[torch.Tensor] = None,
    ) -> None:
        if self._logits is None or self._emb is None:
            raise RuntimeError("CheapCache not opened.")
        start = int(start); end = int(end)
        n = end - start
        if logits.numel() != n:
            raise ValueError(f"logits shape mismatch: expect {n}, got {logits.numel()}")
        if emb.shape[0] != n or emb.shape[1] != self._emb.shape[1]:
            raise ValueError(f"emb shape mismatch: expect ({n},{self._emb.shape[1]}), got {tuple(emb.shape)}")

        self._logits[start:end] = logits.detach().cpu().numpy()
        self._emb[start:end, :] = emb.detach().cpu().numpy()

        if self._entropy is not None:
            if entropy is None:
                raise ValueError("CheapCache has entropy enabled, but entropy is None.")
            if entropy.numel() != n:
                raise ValueError(f"entropy shape mismatch: expect {n}, got {entropy.numel()}")
            self._entropy[start:end] = entropy.detach().cpu().numpy()

    def read_cheap_slice(self, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self._logits is None or self._emb is None:
            raise RuntimeError("CheapCache not opened.")
        start = int(start); end = int(end)
        logits = torch.from_numpy(np.asarray(self._logits[start:end]))
        emb = torch.from_numpy(np.asarray(self._emb[start:end, :]))
        entropy = None
        if self._entropy is not None:
            entropy = torch.from_numpy(np.asarray(self._entropy[start:end]))
        return logits, emb, entropy

    def flush_cheap(self) -> None:
        if self._logits is not None:
            self._logits.flush()
        if self._emb is not None:
            self._emb.flush()
        if self._entropy is not None:
            self._entropy.flush()


    # ------------------- SelectionCache -------------------

    def create_or_open_selection(
        self,
        num_pairs: int,
        kmax: int,
        sel_version: str,
        cheap_version_used: str,
        *,
        dtype_uids=np.int32,
        dtype_len=np.int16,
        overwrite: bool = False,
        require_ready: bool = False,
    ) -> None:
        uids_path = Path(self.sel_dir) / "sel_uids.i32.mmap"
        len_path = Path(self.sel_dir) / "sel_len.i16.mmap"
        lock_path = Path(self.sel_dir) / ".build.lock"

        if os.path.exists(self._sel_meta_path) and not overwrite:
            with open(self._sel_meta_path, "r") as f:
                d = json.load(f)

            _check_meta_identity("SelectionCache", d, split=self.split, path_hash=self.path_hash, dataset_hash_key=self.dataset_hash_key)
            _check_state("SelectionCache", d.get("state",""), require_ready=require_ready)

            _check_version("SelectionCache", d.get("sel_version",""), sel_version, "sel_version")
            _check_version("SelectionCache", d.get("cheap_version_used",""), cheap_version_used, "cheap_version_used")

            if int(d.get("num_pairs")) != int(num_pairs) or int(d.get("kmax")) != int(kmax):
                raise RuntimeError(
                    f"SelectionCache exists but shape mismatch: meta(num_pairs={d.get('num_pairs')}, kmax={d.get('kmax')}) "
                    f"vs requested(num_pairs={num_pairs}, kmax={kmax})."
                )

            meta_dtype_uids = _dtype_from_str(d.get("dtype_uids"))
            meta_dtype_len = _dtype_from_str(d.get("dtype_len"))
            if _dtype_str(dtype_uids) != str(meta_dtype_uids):
                raise RuntimeError(f"[SelectionCache] dtype_uids mismatch: meta={meta_dtype_uids} requested={_dtype_str(dtype_uids)}")
            if _dtype_str(dtype_len) != str(meta_dtype_len):
                raise RuntimeError(f"[SelectionCache] dtype_len mismatch: meta={meta_dtype_len} requested={_dtype_str(dtype_len)}")

            _check_file_size(uids_path, _nbytes(meta_dtype_uids, (int(num_pairs), int(kmax))), "SelectionCache")
            _check_file_size(len_path, _nbytes(meta_dtype_len, (int(num_pairs),)), "SelectionCache")

            self.sel_meta = SelectionMeta(**d)
            self._sel_uids = np.memmap(uids_path, mode="r+", dtype=meta_dtype_uids, shape=(num_pairs, kmax))
            self._sel_len = np.memmap(len_path, mode="r+", dtype=meta_dtype_len, shape=(num_pairs,))
            return


        with FileLock(lock_path):
            meta = SelectionMeta(
                state="building",
                sel_version=str(sel_version),
                cheap_version_used=str(cheap_version_used),
                dataset_hash_key=self.dataset_hash_key,
                path_hash=self.path_hash,
                split=self.split,
                num_pairs=int(num_pairs),
                kmax=int(kmax),
                dtype_uids=str(np.dtype(dtype_uids)),
                dtype_len=str(np.dtype(dtype_len)),
                created_at=time.time(),
            )
            _atomic_write_json(meta.to_dict(), self._sel_meta_path)
            self.sel_meta = meta

            self._sel_uids = np.memmap(uids_path, mode="w+", dtype=dtype_uids, shape=(num_pairs, kmax))
            self._sel_len = np.memmap(len_path, mode="w+", dtype=dtype_len, shape=(num_pairs,))

            # init: uids=-1 表示无效，len=0
            self._sel_uids[:] = -1
            self._sel_len[:] = 0
            self.flush_selection()

    def set_selection_ready(self) -> None:
        if self.sel_meta is None:
            raise RuntimeError("SelectionCache not initialized.")
        self.sel_meta.state = "ready"
        _atomic_write_json(self.sel_meta.to_dict(), self._sel_meta_path)

    def write_selection(self, pair_ids: torch.Tensor, sel_uids: torch.Tensor, sel_len: torch.Tensor) -> None:
        if self._sel_uids is None or self._sel_len is None:
            raise RuntimeError("SelectionCache not opened.")
        pair_ids = pair_ids.detach().cpu().numpy().astype(np.int64)
        self._sel_uids[pair_ids, :] = sel_uids.detach().cpu().numpy()
        self._sel_len[pair_ids] = sel_len.detach().cpu().numpy()

    def read_selection(self, pair_ids: Union[List[int], torch.Tensor], K: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._sel_uids is None or self._sel_len is None:
            raise RuntimeError("SelectionCache not opened.")
        K = int(K)
        if isinstance(pair_ids, torch.Tensor):
            pair_ids_np = pair_ids.detach().cpu().numpy().astype(np.int64)
        else:
            pair_ids_np = np.asarray(pair_ids, dtype=np.int64)

        u = torch.from_numpy(np.asarray(self._sel_uids[pair_ids_np, :K]))
        l = torch.from_numpy(np.asarray(self._sel_len[pair_ids_np]))
        # mask: [B,K], True 表示有效
        mask = torch.arange(K).view(1, K) < l.view(-1, 1)
        return u, mask

    def flush_selection(self) -> None:
        if self._sel_uids is not None:
            self._sel_uids.flush()
        if self._sel_len is not None:
            self._sel_len.flush()

    # ------------------- Version Consistency -------------------

    def assert_version_consistent(self) -> None:
        if self.cheap_meta is None:
            raise RuntimeError("Cheap meta not loaded.")
        if self.sel_meta is None:
            raise RuntimeError("Selection meta not loaded.")
        if self.sel_meta.cheap_version_used != self.cheap_meta.cheap_version:
            raise RuntimeError(
                f"Version mismatch: selection uses cheap_version={self.sel_meta.cheap_version_used}, "
                f"but current cheap_version={self.cheap_meta.cheap_version}."
            )
        

    def assert_instance_version_consistent(
        self,
        inst_version: str,
        sel_version_used: str,
        cheap_version_used: str,
    ) -> None:
        m = self.inst_meta
        if m is None:
            raise RuntimeError("Instance meta not loaded.")

        if str(m.inst_version) != str(inst_version):
            raise RuntimeError(
                f"[InstanceCache] inst_version mismatch: meta={m.inst_version} requested={inst_version}"
            )
        if str(m.sel_version_used) != str(sel_version_used):
            raise RuntimeError(
                f"[InstanceCache] sel_version_used mismatch: meta={m.sel_version_used} requested={sel_version_used}"
            )
        if str(m.cheap_version_used) != str(cheap_version_used):
            raise RuntimeError(
                f"[InstanceCache] cheap_version_used mismatch: meta={m.cheap_version_used} requested={cheap_version_used}"
            )


    # ------------------- InstanceCache -------------------

    def create_or_open_instance(
        self,
        total_cts: int,
        emb_dim: int,
        inst_version: str,
        sel_version_used: str,
        cheap_version_used: str,
        *,
        dtype_logits=np.float16,
        dtype_emb=np.float16,
        overwrite: bool = False,
        require_ready: bool = False,
    ) -> None:
        logit_path = Path(self.inst_dir) / "inst_logit.f16.mmap"
        emb_path = Path(self.inst_dir) / "inst_emb.f16.mmap"
        lock_path = Path(self.inst_dir) / ".build.lock"

        if os.path.exists(self._inst_meta_path) and not overwrite:
            with open(self._inst_meta_path, "r") as f:
                d = json.load(f)

            _check_meta_identity("InstanceCache", d, split=self.split, path_hash=self.path_hash, dataset_hash_key=self.dataset_hash_key)
            _check_state("InstanceCache", d.get("state",""), require_ready=require_ready)

            _check_version("InstanceCache", d.get("inst_version", ""), inst_version, "inst_version")
            _check_version("InstanceCache", d.get("sel_version_used", ""), sel_version_used, "sel_version_used")
            _check_version("InstanceCache", d.get("cheap_version_used", ""), cheap_version_used, "cheap_version_used")


            if int(d.get("total_cts")) != int(total_cts) or int(d.get("emb_dim")) != int(emb_dim):
                raise RuntimeError(
                    f"InstanceCache exists but shape mismatch: meta(total_cts={d.get('total_cts')}, emb_dim={d.get('emb_dim')}) "
                    f"vs requested(total_cts={total_cts}, emb_dim={emb_dim})."
                )

            meta_dtype_logits = _dtype_from_str(d.get("dtype_logits"))
            meta_dtype_emb = _dtype_from_str(d.get("dtype_emb"))
            if _dtype_str(dtype_logits) != str(meta_dtype_logits):
                raise RuntimeError(f"[InstanceCache] dtype_logits mismatch: meta={meta_dtype_logits} requested={_dtype_str(dtype_logits)}")
            if _dtype_str(dtype_emb) != str(meta_dtype_emb):
                raise RuntimeError(f"[InstanceCache] dtype_emb mismatch: meta={meta_dtype_emb} requested={_dtype_str(dtype_emb)}")

            _check_file_size(logit_path, _nbytes(meta_dtype_logits, (int(total_cts),)), "InstanceCache")
            _check_file_size(emb_path, _nbytes(meta_dtype_emb, (int(total_cts), int(emb_dim))), "InstanceCache")

            self.inst_meta = InstanceMeta(**d)
            self._inst_logit = np.memmap(logit_path, mode="r+", dtype=meta_dtype_logits, shape=(total_cts,))
            self._inst_emb = np.memmap(emb_path, mode="r+", dtype=meta_dtype_emb, shape=(total_cts, emb_dim))

            # self.assert_instance_version_consistent(inst_version, sel_version_used, cheap_version_used)

            return


        with FileLock(lock_path):
            meta = InstanceMeta(
                state="building",
                inst_version=str(inst_version),
                sel_version_used=str(sel_version_used),      
                cheap_version_used=str(cheap_version_used),  
                dataset_hash_key=self.dataset_hash_key,
                path_hash=self.path_hash,
                split=self.split,
                total_cts=int(total_cts),
                emb_dim=int(emb_dim),
                dtype_logits=str(np.dtype(dtype_logits)),
                dtype_emb=str(np.dtype(dtype_emb)),
                created_at=time.time(),
            )
            _atomic_write_json(meta.to_dict(), self._inst_meta_path)
            self.inst_meta = meta

            self._inst_logit = np.memmap(logit_path, mode="w+", dtype=dtype_logits, shape=(total_cts,))
            self._inst_emb = np.memmap(emb_path, mode="w+", dtype=dtype_emb, shape=(total_cts, emb_dim))

            # init: NaN 表示未写入
            self._inst_logit[:] = np.nan
            self._inst_emb[:] = 0
            self.flush_instance()

    def set_instance_ready(self) -> None:
        if self.inst_meta is None:
            raise RuntimeError("InstanceCache not initialized.")
        self.inst_meta.state = "ready"
        _atomic_write_json(self.inst_meta.to_dict(), self._inst_meta_path)

    def flush_instance(self) -> None:
        if self._inst_logit is not None:
            self._inst_logit.flush()
        if self._inst_emb is not None:
            self._inst_emb.flush()

    def write_instance_by_uids(
        self,
        uids: torch.Tensor,          # [N] long on CPU
        logit: torch.Tensor,         # [N] float16/32 on CPU
        emb: torch.Tensor,           # [N,D] float16/32 on CPU
    ) -> None:
        if self._inst_logit is None or self._inst_emb is None:
            raise RuntimeError("InstanceCache not opened.")
        u = uids.detach().cpu().numpy().astype(np.int64)
        self._inst_logit[u] = logit.detach().cpu().numpy()
        self._inst_emb[u, :] = emb.detach().cpu().numpy()

    def read_instance_by_uids(
        self,
        uids: torch.Tensor,          # [N] long/int on CPU
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          emb:   [N,D] float16
          logit: [N]   float16
          ok:    [N]   bool, True 表示 cache 命中（logit 非 NaN）
        """
        if self._inst_logit is None or self._inst_emb is None:
            raise RuntimeError("InstanceCache not opened.")
        u = uids.detach().cpu().numpy().astype(np.int64)
        log_np = np.asarray(self._inst_logit[u])
        emb_np = np.asarray(self._inst_emb[u, :])

        # 命中判定：logit 不是 NaN
        ok_np = ~np.isnan(log_np.astype(np.float32))
        emb = torch.from_numpy(emb_np)
        logit = torch.from_numpy(log_np)
        ok = torch.from_numpy(ok_np)

        return emb, logit, ok


