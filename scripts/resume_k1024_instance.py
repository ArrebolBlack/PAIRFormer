#!/usr/bin/env python
"""Resume K=1024 instance cache build from a given pair offset.

Assumes:
  - instance cache files already exist (partially written)
  - meta.json exists with state="building"
  - selection + cheap cache are ready
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)

import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config.data_config import DataConfig
from src.data.dataset import ChunkedCTSDataset
from src.data.em_cache import MemmapCacheStore
from src.models.registry import build_model
from src.models.extractors import get_embedding_and_logit
from src.data.collate import cts_collate_fn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from hydra.utils import instantiate
from src.utils import set_seeds


def load_ckpt_clean(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    cleaned = {}
    for k, v in sd.items():
        for pref in ("model.", "module.", "net."):
            if k.startswith(pref):
                k = k[len(pref):]
        cleaned[k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  WARN missing keys: {len(missing)}")
    if unexpected:
        print(f"  WARN unexpected keys: {len(unexpected)}")
    model.to(device).eval()


def resume_instance_split(
    *,
    store: MemmapCacheStore,
    em_cache_root: str,
    split: str,
    sel_meta: dict,
    cts_ds: ChunkedCTSDataset,
    instance_model: torch.nn.Module,
    kmax: int,
    start_pair: int,
    num_pairs: int,
    emb_dim: int,
    pair_chunk_size: int = 256,
    use_amp: bool = True,
    normalize_emb: bool = False,
):
    """Resume instance cache build from start_pair."""
    dev = next(instance_model.parameters()).device

    # Open selection memmaps
    sel_dir = Path(em_cache_root) / "em_cache" / split / "selection"
    uids_path = sel_dir / "sel_uids.i32.mmap"
    len_path = sel_dir / "sel_len.i16.mmap"
    sel_uids_mmap = np.memmap(uids_path, mode="r", dtype=np.int32, shape=(num_pairs, kmax))
    sel_len_mmap = np.memmap(len_path, mode="r", dtype=np.int16, shape=(num_pairs,))

    pair_end = num_pairs
    shard_size = pair_end - start_pair

    print(f"[Resume:{split}] Resuming from pair {start_pair}/{num_pairs} ({shard_size} remaining)")

    written_pairs = 0
    written_cts = 0
    t0 = time.time()

    pbar = tqdm(
        range(start_pair, pair_end, pair_chunk_size),
        desc=f"[Resume:{split}]",
        dynamic_ncols=True,
    )

    for chunk_start in pbar:
        chunk_end = min(chunk_start + pair_chunk_size, pair_end)
        C = chunk_end - chunk_start

        sel_u = np.asarray(sel_uids_mmap[chunk_start:chunk_end])
        sel_l = np.asarray(sel_len_mmap[chunk_start:chunk_end])

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
            continue

        flat_uids = np.concatenate(all_uids).astype(np.int64)
        flat_uids_t = torch.from_numpy(flat_uids)

        batch_data = cts_ds.batch_gather_by_uid(flat_uids_t, fields=["X", "esa_scores", "pos"])
        x = batch_data.get("X", batch_data.get("inputs"))
        esa = batch_data.get("esa_scores")
        pos = batch_data.get("pos")

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(esa, np.ndarray):
            esa = torch.from_numpy(esa)
        if isinstance(pos, np.ndarray):
            pos = torch.from_numpy(pos)

        x = x.to(dev, non_blocking=True).float()
        if esa is not None:
            esa = esa.to(dev, non_blocking=True).float()
        if pos is not None:
            pos = pos.to(dev, non_blocking=True).float()

        with torch.no_grad():
            if dev.type == "cuda":
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    feat, logit = get_embedding_and_logit(instance_model, x)
            else:
                feat, logit = get_embedding_and_logit(instance_model, x)

        if normalize_emb:
            feat = F.normalize(feat.float(), dim=-1)

        feat_cpu = feat.detach().float().cpu()
        logit_cpu = logit.detach().float().view(-1).cpu()

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
        pbar.set_postfix(pairs=f"{written_pairs}/{shard_size}", cts=written_cts)

    store.flush_instance()
    store.set_instance_ready()

    dt = time.time() - t0
    print(f"[Resume:{split}] DONE pairs={written_pairs}/{shard_size} cts={written_cts} time={dt:.1f}s")


def main():
    experiment = "MTI_EM_K1024"
    em_cache_root = "/dev/shm/pairformer_cache_k1024"
    cache_root = str(PROJECT / "cache")
    device = torch.device("cuda:0")
    seed = 2020
    start_pair_train = 278016  # Where the last build stopped

    set_seeds(seed)

    with initialize_config_dir(config_dir=str(PROJECT / "configs"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment={experiment}"])

    data_cfg = DataConfig.from_omegaconf(cfg.data)
    em_node = cfg.em if hasattr(cfg, "em") else {}
    inst_emb_dim = int(em_node.get("inst_emb_dim", 1536))
    inst_version = str(em_node.get("inst_version", "inst_v0"))

    # ========================================
    # Resume train instance cache
    # ========================================
    print(f"\n{'='*60}")
    print(f"[Resume] Resuming TRAIN instance cache from pair {start_pair_train}")
    print(f"{'='*60}")

    sel_meta_path = Path(em_cache_root) / "em_cache" / "train" / "selection" / "meta.json"
    with open(sel_meta_path) as f:
        sel_meta_train = json.load(f)
    num_pairs_train = int(sel_meta_train["num_pairs"])
    kmax = int(sel_meta_train["kmax"])

    print(f"  num_pairs={num_pairs_train}, kmax={kmax}")
    print(f"  Resuming from pair {start_pair_train} ({num_pairs_train - start_pair_train} remaining)")

    # Build instance model
    inst_cfg = cfg.get("instance_model", cfg.get("cts_model", cfg.get("model_instance")))
    inst_arch = str(inst_cfg.get("arch", inst_cfg.get("name")))
    instance_model = build_model(inst_arch, inst_cfg, data_cfg=data_cfg).to(device)

    inst_ckpt_path = str(em_node.get("instance_ckpt_path", cfg.get("instance_ckpt_path", "")))
    if not Path(inst_ckpt_path).is_absolute():
        inst_ckpt_path = str(PROJECT / inst_ckpt_path)
    load_ckpt_clean(instance_model, inst_ckpt_path, device)

    cts_ds_train = ChunkedCTSDataset(cache_root, data_cfg, "train")

    store_train = MemmapCacheStore(
        cache_root=em_cache_root,
        split="train",
        path_hash=str(sel_meta_train["path_hash"]),
        dataset_hash_key=str(sel_meta_train["dataset_hash_key"]),
    )
    # Open existing files without overwrite
    store_train.create_or_open_instance_pair_indexed(
        num_pairs=num_pairs_train,
        kmax=kmax,
        emb_dim=inst_emb_dim,
        inst_version=inst_version,
        sel_version_used=str(sel_meta_train["sel_version"]),
        cheap_version_used=str(sel_meta_train["cheap_version_used"]),
        overwrite=False,
    )

    resume_instance_split(
        store=store_train,
        em_cache_root=em_cache_root,
        split="train",
        sel_meta=sel_meta_train,
        cts_ds=cts_ds_train,
        instance_model=instance_model,
        kmax=kmax,
        start_pair=start_pair_train,
        num_pairs=num_pairs_train,
        emb_dim=inst_emb_dim,
        pair_chunk_size=256,
        use_amp=True,
        normalize_emb=False,
    )

    # ========================================
    # Build val instance cache
    # ========================================
    print(f"\n{'='*60}")
    print(f"[Resume] Building VAL instance cache")
    print(f"{'='*60}")

    sel_meta_val_path = Path(em_cache_root) / "em_cache" / "val" / "selection" / "meta.json"
    with open(sel_meta_val_path) as f:
        sel_meta_val = json.load(f)
    num_pairs_val = int(sel_meta_val["num_pairs"])
    kmax_val = int(sel_meta_val["kmax"])

    print(f"  num_pairs={num_pairs_val}, kmax={kmax_val}")

    cts_ds_val = ChunkedCTSDataset(cache_root, data_cfg, "val")

    store_val = MemmapCacheStore(
        cache_root=em_cache_root,
        split="val",
        path_hash=str(sel_meta_val["path_hash"]),
        dataset_hash_key=str(sel_meta_val["dataset_hash_key"]),
    )
    store_val.create_or_open_instance_pair_indexed(
        num_pairs=num_pairs_val,
        kmax=kmax_val,
        emb_dim=inst_emb_dim,
        inst_version=inst_version,
        sel_version_used=str(sel_meta_val["sel_version"]),
        cheap_version_used=str(sel_meta_val["cheap_version_used"]),
        overwrite=True,
    )

    resume_instance_split(
        store=store_val,
        em_cache_root=em_cache_root,
        split="val",
        sel_meta=sel_meta_val,
        cts_ds=cts_ds_val,
        instance_model=instance_model,
        kmax=kmax_val,
        start_pair=0,
        num_pairs=num_pairs_val,
        emb_dim=inst_emb_dim,
        pair_chunk_size=256,
        use_amp=True,
        normalize_emb=False,
    )

    print(f"\n{'='*60}")
    print("[resume_k1024] ALL DONE.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
