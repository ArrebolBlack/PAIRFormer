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


def _open_selection_len_mmap(em_cache_root: Union[str, Path], split: str, sel_meta: Dict[str, Any]) -> np.memmap:
    root = Path(str(em_cache_root))
    sel_dir = root / "em_cache" / split / "selection"
    len_path = sel_dir / "sel_len.i16.mmap"
    num_pairs = int(sel_meta["num_pairs"])
    return np.memmap(len_path, mode="r", dtype=np.int16, shape=(num_pairs,))


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

    # pair-indexed build
    pair_chunk_size: int = 256  # pairs per iteration

    # DDP sharding
    rank: int = 0
    world_size: int = 1


class InstanceCacheRunner:
    """
    Build pair-indexed instance cache for all selected CTS per pair.
    Stores inst_emb[num_pairs, kmax, emb_dim] and inst_logit[num_pairs, kmax].
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
        from src.utils.ddp import barrier as ddp_barrier

        rank = int(cfg.rank)
        world_size = int(cfg.world_size)

        cts_ds = ChunkedCTSDataset(self.dataset_cache_root, self.data_cfg, split)

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

        num_pairs = int(sel_meta["num_pairs"])
        kmax = int(sel_meta["kmax"])

        # Open selection memmaps
        sel_uids_mmap = _open_selection_uids_mmap(self.em_cache_root, split, sel_meta)
        sel_len_mmap = _open_selection_len_mmap(self.em_cache_root, split, sel_meta)

        # Shard pair range across ranks
        pair_start = (rank * num_pairs) // world_size
        pair_end = ((rank + 1) * num_pairs) // world_size
        shard_size = pair_end - pair_start

        print(f"[InstanceRunner:{split}] rank={rank}/{world_size} num_pairs={num_pairs} kmax={kmax} "
              f"pair_shard=[{pair_start},{pair_end}) ({shard_size}) inst_version={inst_version}")

        # Open instance cache store (pair-indexed)
        store = MemmapCacheStore(
            cache_root=str(self.em_cache_root),
            split=str(split),
            path_hash=str(sel_meta["path_hash"]),
            dataset_hash_key=str(sel_meta["dataset_hash_key"]),
        )

        # DDP: rank 0 creates, others wait then open existing
        if world_size > 1:
            if rank == 0:
                store.create_or_open_instance_pair_indexed(
                    num_pairs=num_pairs,
                    kmax=kmax,
                    emb_dim=emb_dim,
                    inst_version=inst_version,
                    sel_version_used=str(sel_meta["sel_version"]),
                    cheap_version_used=str(sel_meta["cheap_version_used"]),
                    overwrite=bool(cfg.overwrite),
                )
            ddp_barrier()
            if rank != 0:
                store.create_or_open_instance_pair_indexed(
                    num_pairs=num_pairs,
                    kmax=kmax,
                    emb_dim=emb_dim,
                    inst_version=inst_version,
                    sel_version_used=str(sel_meta["sel_version"]),
                    cheap_version_used=str(sel_meta["cheap_version_used"]),
                    overwrite=False,
                )
        else:
            store.create_or_open_instance_pair_indexed(
                num_pairs=num_pairs,
                kmax=kmax,
                emb_dim=emb_dim,
                inst_version=inst_version,
                sel_version_used=str(sel_meta["sel_version"]),
                cheap_version_used=str(sel_meta["cheap_version_used"]),
                overwrite=bool(cfg.overwrite),
            )

        # Skip logic
        if (not cfg.overwrite) and store.inst_meta is not None:
            if store.inst_meta.state == "ready" and cfg.skip_if_ready:
                print(f"[InstanceRunner] SKIP split={split} (already ready).")
                return

        dev = next(instance_model.parameters()).device
        pair_chunk_size = int(cfg.pair_chunk_size)

        written_pairs = 0
        written_cts = 0
        t0 = __import__("time").time()

        pbar = tqdm(
            range(pair_start, pair_end, pair_chunk_size),
            desc=f"[InstanceRunner:{split}:r{rank}]",
            dynamic_ncols=True,
            disable=(world_size > 1 and rank != 0),
        )

        for chunk_start in pbar:
            chunk_end = min(chunk_start + pair_chunk_size, pair_end)
            C = chunk_end - chunk_start

            # Read selection for this chunk of pairs
            sel_u = np.asarray(sel_uids_mmap[chunk_start:chunk_end])  # [C, K]
            sel_l = np.asarray(sel_len_mmap[chunk_start:chunk_end])   # [C]

            # Gather all valid UIDs
            all_uids = []
            uid_counts = []
            for i in range(C):
                n = int(sel_l[i])
                uids_i = sel_u[i, :n]
                uids_i = uids_i[uids_i >= 0]
                all_uids.append(uids_i)
                uid_counts.append(len(uids_i))

            total_valid = sum(uid_counts)
            if total_valid == 0:
                # No valid CTS for this chunk, skip
                continue

            flat_uids = np.concatenate(all_uids).astype(np.int64)
            # batch_gather_by_uid expects a torch tensor
            flat_uids_t = torch.from_numpy(flat_uids)

            # Gather CTS inputs by UID (chunk stores 'X' not 'inputs')
            batch_data = cts_ds.batch_gather_by_uid(flat_uids_t, fields=["X", "esa_scores", "pos"])
            x = batch_data.get("X", None)
            if x is None:
                x = batch_data.get("inputs", None)
            esa = batch_data.get("esa_scores", None)
            pos = batch_data.get("pos", None)

            # Convert to torch tensors if needed
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            if isinstance(esa, np.ndarray):
                esa = torch.from_numpy(esa)
            if isinstance(pos, np.ndarray):
                pos = torch.from_numpy(pos)

            x = x.to(dev, non_blocking=True)
            if x.dtype != torch.float32:
                x = x.to(dtype=torch.float32)
            if esa is not None:
                esa = esa.to(dev, non_blocking=True).float()
            if pos is not None:
                pos = pos.to(dev, non_blocking=True).float()

            # Run instance model forward
            if dev.type == "cuda":
                with torch.autocast(device_type="cuda", enabled=bool(use_amp)):
                    feat, logit = get_embedding_and_logit(instance_model, x)
            else:
                feat, logit = get_embedding_and_logit(instance_model, x)

            if cfg.normalize_emb:
                feat = F.normalize(feat.float(), dim=-1)

            feat_cpu = feat.detach().float().cpu()
            logit_cpu = logit.detach().float().view(-1).cpu()

            # Reshape flat results back to [C, K, D] and [C, K]
            D = feat_cpu.shape[1]
            emb_chunk = torch.zeros(C, kmax, D, dtype=torch.float16)
            logit_chunk = torch.zeros(C, kmax, dtype=torch.float16)

            offset = 0
            for i in range(C):
                n = uid_counts[i]
                if n > 0:
                    emb_chunk[i, :n] = feat_cpu[offset:offset + n].to(torch.float16)
                    logit_chunk[i, :n] = logit_cpu[offset:offset + n].to(torch.float16)
                offset += n

            # Write to store
            pair_ids = torch.arange(chunk_start, chunk_end, dtype=torch.long)
            sel_len_t = torch.from_numpy(sel_l.astype(np.int32))

            store.write_instance_by_pairs(
                pair_ids=pair_ids,
                logit=logit_chunk,
                emb=emb_chunk,
                sel_len=sel_len_t,
            )

            written_pairs += C
            written_cts += total_valid
            pbar.set_postfix(pairs=written_pairs, cts=written_cts, shard=shard_size)

        # Flush + barrier + finalize
        store.flush_instance()
        if world_size > 1:
            ddp_barrier()

        if rank == 0:
            store.set_instance_ready()

        dt = __import__("time").time() - t0
        print(f"[InstanceRunner:{split}] rank={rank} DONE pairs={written_pairs}/{shard_size} "
              f"cts={written_cts} time={dt:.1f}s")


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
    rank: int = 0,
    world_size: int = 1,
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
        rank=int(rank),
        world_size=int(world_size),
    )
    runner.build_from_model(
        instance_model=instance_model,
        inst_version=str(inst_version),
        emb_dim=int(emb_dim),
        cfg=cfg,
    )
