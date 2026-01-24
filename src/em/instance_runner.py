# src/em/instance_runner.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.config.data_config import DataConfig
from src.data.dataset import ChunkedCTSDataset
from src.data.em_cache import MemmapCacheStore
from src.models.extractors import get_embedding_and_logit

from src.data.collate import cts_collate_fn


def _load_json(p: Union[str, Path]) -> Dict[str, Any]:
    with open(str(p), "r") as f:
        return json.load(f)


def _open_selection_uids_mmap(em_cache_root: Union[str, Path], split: str, sel_meta: Dict[str, Any]) -> np.memmap:
    root = Path(str(em_cache_root))
    sel_dir = root / "em_cache" / split / "selection"
    uids_path = sel_dir / "sel_uids.i32.mmap"
    num_pairs = int(sel_meta["num_pairs"])
    kmax = int(sel_meta["kmax"])
    return np.memmap(uids_path, mode="r", dtype=np.int32, shape=(num_pairs, kmax))


def _build_selected_uids(
    *,
    sel_uids_mmap: np.memmap,   # [num_pairs,kmax] int32, padded -1
    total_cts: int,
    chunk_pairs: int = 4096,
) -> np.ndarray:
    num_pairs, _ = sel_uids_mmap.shape
    selected = np.zeros((int(total_cts),), dtype=np.bool_)

    for s in range(0, num_pairs, int(chunk_pairs)):
        e = min(num_pairs, s + int(chunk_pairs))
        u = np.asarray(sel_uids_mmap[s:e, :]).reshape(-1)
        u = u[u >= 0]
        if u.size == 0:
            continue
        u = u[u < total_cts]
        selected[u] = True

    return np.nonzero(selected)[0].astype(np.int64)


class UIDWrappedDataset(Dataset):
    def __init__(self, base: ChunkedCTSDataset, uids: np.ndarray):
        self.base = base
        self.uids = np.asarray(uids, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.uids.shape[0])

    def __getitem__(self, i: int):
        uid = int(self.uids[i])
        raw = self.base[uid]  # ChunkedCTSDataset returns tuple (x,y,set_idx,esa,pos) or similar

        # 兼容：如果未来 base 返回 dict，也转成 tuple
        if isinstance(raw, dict):
            x = raw.get("inputs", raw.get("X"))
            y = raw.get("labels", raw.get("y"))
            set_idx = raw.get("set_idx", raw.get("set_idxs"))
            esa = raw.get("esa_scores", None)
            pos = raw.get("pos", None)
            base_tuple = (x, y, set_idx) if esa is None else ((x, y, set_idx, esa) if pos is None else (x, y, set_idx, esa, pos))
        else:
            base_tuple = tuple(raw)

        # 返回 “原始样本tuple + uid”
        return (*base_tuple, torch.tensor(uid, dtype=torch.long))

def uid_cts_collate_fn(batch: List[Tuple[Any, ...]]) -> Dict[str, Any]:
    """
    batch元素形如:
      (x,y,set_idx,esa,pos,uid)  或  (x,y,set_idx,uid) 等
    做法:
      - 取出最后一位 uid
      - 剩余部分交给 cts_collate_fn 生成 {"inputs","labels","set_idx",...}
      - 再把 uid 作为 [B] long 拼回去
    """
    base_batch = []
    uids = []

    for item in batch:
        if not isinstance(item, (tuple, list)) or len(item) < 4:
            raise TypeError(f"[uid_cts_collate_fn] bad item type/len: {type(item)} len={getattr(item,'__len__',lambda:None)()}")
        uids.append(item[-1])
        base_batch.append(tuple(item[:-1]))

    out = cts_collate_fn(base_batch)

    # uids -> [B] long
    uid_t = torch.stack([torch.as_tensor(u).view(-1)[0].long() for u in uids], dim=0).view(-1)
    out["uid"] = uid_t
    return out


@dataclass
class InstanceCacheBuildConfig:
    splits: List[str]

    overwrite: bool = True
    skip_if_ready: bool = False

    # loader
    batch_size: int = 1024
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True

    # compute
    use_amp: bool = True
    normalize_emb: bool = False

    # selection scan
    scan_chunk_pairs: int = 4096


class InstanceCacheRunner:
    """
    Build instance cache for all CTS uids that appear in selection cache.
    Pure python callable from controller: no shell, no hydra main.
    """

    def __init__(self, *, data_cfg: DataConfig, dataset_cache_root: str, em_cache_root: str):
        self.data_cfg = data_cfg
        self.dataset_cache_root = str(Path(dataset_cache_root))
        self.em_cache_root = str(Path(em_cache_root))

    @torch.no_grad()
    def build_from_model(
        self,
        *,
        instance_model: torch.nn.Module,
        inst_version: str,
        emb_dim: int,
        cfg: InstanceCacheBuildConfig,
        sel_expected_version: Optional[str] = None,
    ) -> None:
        dev = next(instance_model.parameters()).device
        use_amp = bool(cfg.use_amp) and (dev.type == "cuda")

        # 临时切 eval，结束后恢复
        was_training = instance_model.training
        instance_model.eval()

        try:
            for split in cfg.splits:
                self._build_one_split(
                    split=str(split),
                    instance_model=instance_model,
                    inst_version=str(inst_version),
                    emb_dim=int(emb_dim),
                    cfg=cfg,
                    use_amp=use_amp,
                    sel_expected_version=sel_expected_version,
                )
        finally:
            instance_model.train(was_training)

    @torch.no_grad()
    def _build_one_split(
        self,
        *,
        split: str,
        instance_model: torch.nn.Module,
        inst_version: str,
        emb_dim: int,
        cfg: InstanceCacheBuildConfig,
        use_amp: bool,
        sel_expected_version: Optional[str],
    ) -> None:
        cts_ds = ChunkedCTSDataset(self.dataset_cache_root, self.data_cfg, split)
        total_cts = int(len(cts_ds))

        sel_meta_path = Path(self.em_cache_root) / "em_cache" / split / "selection" / "meta.json"
        cheap_meta_path = Path(self.em_cache_root) / "em_cache" / split / "cheap" / "meta.json"
        if not sel_meta_path.exists():
            raise FileNotFoundError(f"[InstanceRunner] selection meta missing: {sel_meta_path}")
        if not cheap_meta_path.exists():
            raise FileNotFoundError(f"[InstanceRunner] cheap meta missing: {cheap_meta_path}")

        sel_meta = _load_json(sel_meta_path)
        cheap_meta = _load_json(cheap_meta_path)

        if sel_meta.get("state", "") != "ready":
            raise RuntimeError(f"[InstanceRunner] selection not ready: split={split} state={sel_meta.get('state')}")
        if cheap_meta.get("state", "") != "ready":
            raise RuntimeError(f"[InstanceRunner] cheap not ready: split={split} state={cheap_meta.get('state')}")

        if str(sel_meta.get("cheap_version_used", "")) != str(cheap_meta.get("cheap_version", "")):
            raise RuntimeError(
                f"[InstanceRunner] cheap_version mismatch: selection uses {sel_meta.get('cheap_version_used')} "
                f"but cheap meta is {cheap_meta.get('cheap_version')} (split={split})"
            )

        if sel_expected_version is not None and str(sel_meta.get("sel_version", "")) != str(sel_expected_version):
            raise RuntimeError(
                f"[InstanceRunner] sel_version mismatch: expected={sel_expected_version} got={sel_meta.get('sel_version')} (split={split})"
            )

        # scan selected uids
        sel_uids_mmap = _open_selection_uids_mmap(self.em_cache_root, split, sel_meta)
        selected_uids = _build_selected_uids(
            sel_uids_mmap=sel_uids_mmap,
            total_cts=total_cts,
            chunk_pairs=int(cfg.scan_chunk_pairs),
        )
        num_selected = int(selected_uids.shape[0])
        print(f"[InstanceRunner:{split}] total_cts={total_cts} num_selected={num_selected} inst_version={inst_version}")

        # open instance cache via MemmapCacheStore (保持与 TokenProvider 一致)
        store = MemmapCacheStore(
            cache_root=str(self.em_cache_root),
            split=str(split),
            path_hash=str(sel_meta["path_hash"]),
            dataset_hash_key=str(sel_meta["dataset_hash_key"]),
        )
        store.create_or_open_instance(
            total_cts=int(total_cts),
            emb_dim=int(emb_dim),
            inst_version=str(inst_version),
            sel_version_used=str(sel_meta["sel_version"]),
            cheap_version_used=str(sel_meta["cheap_version_used"]),
            overwrite=bool(cfg.overwrite),
        )


        # 可选：跳过
        if (not cfg.overwrite) and getattr(store, "inst_meta", None) is not None:
            meta = store.inst_meta
            if getattr(meta, "state", "") == "ready" and bool(cfg.skip_if_ready):
                store.assert_instance_version_consistent(
                    inst_version=str(inst_version),
                    sel_version_used=str(sel_meta["sel_version"]),
                    cheap_version_used=str(sel_meta["cheap_version_used"]),
                )
                print(f"[InstanceRunner] SKIP split={split} (already ready).")
                return

        wrapped = UIDWrappedDataset(cts_ds, selected_uids)

        bs = int(cfg.batch_size)
        nw = int(cfg.num_workers)
        pin = bool(cfg.pin_memory)
        persistent = bool(cfg.persistent_workers) and (nw > 0)

        loader = DataLoader(
            wrapped,
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=pin,
            persistent_workers=persistent,
            drop_last=False,
            collate_fn=uid_cts_collate_fn,
        )

        dev = next(instance_model.parameters()).device
        pbar = tqdm(loader, desc=f"[InstanceRunner:{split}]", dynamic_ncols=True)

        written = 0
        for batch in pbar:
            
            if "uid" not in batch:
                raise KeyError(f"[InstanceRunner] batch missing 'uid'. got={list(batch.keys())}")

            x_cpu = batch.get("inputs", None)
            if x_cpu is None:
                x_cpu = batch.get("X", None)
            if x_cpu is None:
                raise KeyError(f"[InstanceRunner] batch missing 'inputs'/'X'. got={list(batch.keys())}")

            uids = batch["uid"].view(-1).to(dtype=torch.long, device="cpu")  # [N] CPU long

            # x_cpu 已经在 cts_collate_fn 里 float() 过了；这里再保证一次 dtype
            x = x_cpu.to(dev, non_blocking=True)
            if x.dtype != torch.float32:
                x = x.to(dtype=torch.float32)


            if dev.type == "cuda":
                with torch.autocast(device_type="cuda", enabled=bool(use_amp)):
                    feat, logit = get_embedding_and_logit(instance_model, x)
            else:
                feat, logit = get_embedding_and_logit(instance_model, x)

            if cfg.normalize_emb:
                feat = F.normalize(feat.float(), dim=-1)

            emb_cpu = feat.detach().float().to("cpu", non_blocking=False).to(torch.float16).contiguous()   # [N,D]
            log_cpu = logit.detach().float().view(-1).to("cpu", non_blocking=False).to(torch.float16).contiguous()  # [N]

            # 写回
            store.write_instance_by_uids(
                uids=uids,        # torch.Tensor [N] long CPU
                logit=log_cpu,    # torch.Tensor [N] float16 CPU
                emb=emb_cpu,      # torch.Tensor [N,D] float16 CPU
            )

            written += int(uids.numel())
            pbar.set_postfix(written=written, total=num_selected)

        store.flush_instance()
        store.set_instance_ready()
        print(f"[InstanceRunner] DONE split={split} (written={written})")


def run_instance_cache(
    *,
    data_cfg: DataConfig,
    dataset_cache_root: str,
    em_cache_root: str,
    instance_model: torch.nn.Module,
    inst_version: str,
    emb_dim: int,
    epoch: int,
    splits: list[str],
    overwrite: bool = True,
    skip_if_ready: bool = False,
    batch_size: int = 1024,
    num_workers: int = 8,
    use_amp: bool = True,
    normalize_emb: bool = False,
) -> None:
    runner = InstanceCacheRunner(
        data_cfg=data_cfg,
        dataset_cache_root=str(Path(dataset_cache_root)),
        em_cache_root=str(Path(em_cache_root)),
    )
    cfg = InstanceCacheBuildConfig(
        splits=list(splits),
        overwrite=bool(overwrite),
        skip_if_ready=bool(skip_if_ready),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        use_amp=bool(use_amp),
        normalize_emb=bool(normalize_emb),
    )
    runner.build_from_model(
        instance_model=instance_model,
        inst_version=str(inst_version),
        emb_dim=int(emb_dim),
        cfg=cfg,
    )


