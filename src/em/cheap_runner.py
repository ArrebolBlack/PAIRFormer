# src/em/cheap_runner.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from src.config.data_config import DataConfig
from src.data.builder import build_dataset_and_loader
from src.data.cache_identity import dataset_identity
from src.data.em_cache import MemmapCacheStore


# -------------------------
# small helpers
# -------------------------

def _strip_prefix(k: str) -> str:
    for pref in ("model.", "module.", "net."):
        if k.startswith(pref):
            return k[len(pref):]
    return k


def load_ckpt_into_model(
    model: torch.nn.Module,
    ckpt_path: Union[str, Path],
    device: torch.device,
    use_ema_shadow: bool = False,
) -> None:
    """
    Inference-only weight loading:
      - load state_dict (support model./module./net. prefixes)
      - optional: apply ckpt["ema_shadow"] as inference weights
    """
    ckpt_path = str(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state_dict, dict):
        raise ValueError(f"[CheapRunner] Unexpected checkpoint format: {ckpt_path}")

    cleaned = {_strip_prefix(k): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[CheapRunner] WARN missing keys: {len(missing)} (first 10): {missing[:10]}")
    if unexpected:
        print(f"[CheapRunner] WARN unexpected keys: {len(unexpected)} (first 10): {unexpected[:10]}")

    if use_ema_shadow and isinstance(ckpt, dict) and isinstance(ckpt.get("ema_shadow", None), dict):
        ema_shadow = ckpt["ema_shadow"]
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in ema_shadow:
                    p.data.copy_(ema_shadow[name].to(p.device, dtype=p.dtype))

    model.to(device)
    model.eval()


def extract_batch(batch: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Based on your window-level pipeline output (cts_collate_fn + ChunkedCTSDataset):
      - inputs: [B, C, L]
      - esa_scores: [B] (optional depending on config, but your script assumes it exists)
      - pos: [B] (optional)
    """
    if "inputs" not in batch:
        raise KeyError(f"[CheapRunner] batch keys mismatch. got={list(batch.keys())}")
    x = batch["inputs"]
    esa = batch.get("esa_scores", None)
    pos = batch.get("pos", None)
    return x, esa, pos


def cheap_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
    esa: Optional[torch.Tensor],
    pos: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return: (logits[B], emb[B,D])

    This matches your build_cheap_cache.py behavior:
      model(x, esa_scores=..., pos=..., return_normalized_emb=True, return_emb_raw=False)
    """
    out = None
    try:
        out = model(
            x,
            esa_scores=esa,
            pos=pos,
            return_normalized_emb=True,
            return_emb_raw=False,
        )
    except TypeError:
        out = None

    # common patterns
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        emb, logit = out[0], out[1]
        return logit, emb

    if isinstance(out, dict):
        logit = out.get("logit", out.get("logits", None))
        emb = out.get("emb", out.get("embedding", None))
        if logit is not None and emb is not None:
            return logit, emb

    raise RuntimeError(
        "[CheapRunner] Cheap model forward didn't return (logit, emb). "
        "Ensure your cheap model supports returning embeddings."
    )


def binary_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    x = logits.view(-1)
    p = torch.sigmoid(x).clamp(1e-6, 1 - 1e-6)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


# -------------------------
# Runner configs
# -------------------------

@dataclass
class CheapCacheBuildConfig:
    splits: List[str] = None  # e.g. ["train","val"] or ["test"]
    overwrite: bool = False
    skip_if_ready: bool = True
    has_entropy: bool = False

    # dataloader
    batch_size: int = 256
    num_workers: int = 8
    pin_memory: bool = True

    # runtime
    amp: bool = False
    non_blocking: bool = True

    # dtype on disk
    dtype_logits: Any = np.float16
    dtype_emb: Any = np.float16

    def __post_init__(self):
        if self.splits is None:
            self.splits = ["train", "val"]


class CheapCacheRunner:
    """
    Cheap step (Stage5 / E-step component):
      - For each split in cfg.splits:
          1) build window-level dataset+loader (shuffle=False)
          2) run cheap model forward over all windows
          3) write logits/emb/(entropy) into MemmapCacheStore (fixed directory)
          4) mark meta.state=ready

    This runner assumes a strictly serialized pipeline (no concurrent readers/writers).
    """

    def __init__(
        self,
        *,
        data_cfg: DataConfig,
        dataset_cache_root: Union[str, Path],
        em_cache_root: Union[str, Path],
        device: Union[str, torch.device] = "cuda",
    ):
        self.data_cfg = data_cfg
        self.dataset_cache_root = str(dataset_cache_root)
        self.em_cache_root = str(em_cache_root)
        self.device = torch.device(device)

    @torch.no_grad()
    def build(
        self,
        *,
        cheap_model: torch.nn.Module,
        cheap_version: str,
        emb_dim: int,
        cfg: CheapCacheBuildConfig,
    ) -> None:
        cheap_model = cheap_model.to(self.device).eval()

        print(f"[CheapRunner] dataset_cache_root={self.dataset_cache_root}")
        print(f"[CheapRunner] em_cache_root={self.em_cache_root}")
        print(f"[CheapRunner] device={self.device} amp={cfg.amp} has_entropy={cfg.has_entropy}")
        print(f"[CheapRunner] splits={cfg.splits} overwrite={cfg.overwrite} skip_if_ready={cfg.skip_if_ready}")
        print(f"[CheapRunner] cheap_version={cheap_version}")
        print(f"[CheapRunner] emb_dim(asserted)={emb_dim}")

        for split in cfg.splits:
            self._build_one_split(
                split=split,
                cheap_model=cheap_model,
                cheap_version=cheap_version,
                emb_dim=emb_dim,
                cfg=cfg,
            )

        print("[CheapRunner] ALL DONE.")

    @torch.no_grad()
    def _build_one_split(
        self,
        *,
        split: str,
        cheap_model: torch.nn.Module,
        cheap_version: str,
        emb_dim: int,
        cfg: CheapCacheBuildConfig,
    ) -> None:
        # 1) window-level dataset + loader (must be shuffle=False)

        # IMPORTANT: must be deterministic order (uid-aligned) when writing memmap by offset.
        # DO NOT set shuffle=True here, otherwise uid->(logit,emb) mapping breaks.
        ds, loader = build_dataset_and_loader(
            data_cfg=self.data_cfg,
            split_idx=str(split),
            cache_data_path=str(Path(self.dataset_cache_root)),
            batch_size=int(cfg.batch_size),
            num_workers=int(cfg.num_workers),
            pin_memory=bool(cfg.pin_memory),
            shuffle=False,
            drop_last=False,
        )
        total_cts = int(len(ds))

        # Identity MUST strictly follow window-level dataset semantics.
        # This binds em_cache to split-local uid space defined by ChunkedCTSDataset.
        _hash_key_data, dataset_hash_key, path_hash = dataset_identity(self.data_cfg, str(split))
  
        store = MemmapCacheStore(
            cache_root=str(self.em_cache_root),
            split=str(split),
            path_hash=str(path_hash),
            dataset_hash_key=str(dataset_hash_key),
        )

        # 2) open/create memmap
        store.create_or_open_cheap(
            total_cts=total_cts,
            emb_dim=int(emb_dim),
            cheap_version=str(cheap_version),
            dtype_logits=cfg.dtype_logits,
            dtype_emb=cfg.dtype_emb,
            has_entropy=bool(cfg.has_entropy),
            overwrite=bool(cfg.overwrite),
        )

        # 3) skip logic
        if (not cfg.overwrite) and store.cheap_meta is not None and store.cheap_meta.state == "ready" and cfg.skip_if_ready:
            if store.cheap_meta.cheap_version != str(cheap_version):
                raise RuntimeError(
                    f"[CheapRunner] Found ready cache but cheap_version mismatch for split={split}:\n"
                    f"  existing={store.cheap_meta.cheap_version}\n"
                    f"  requested={cheap_version}\n"
                    f"Set overwrite=True or use matching cheap_version."
                )
            print(f"[CheapRunner] SKIP split={split} (already ready).")
            return

        # if cache exists but different version and overwrite=False, refuse
        if (not cfg.overwrite) and store.cheap_meta is not None and store.cheap_meta.cheap_version != str(cheap_version):
            raise RuntimeError(
                f"[CheapRunner] Existing cache cheap_version != requested for split={split}:\n"
                f"  existing={store.cheap_meta.cheap_version}\n"
                f"  requested={cheap_version}\n"
                f"Set overwrite=True or use matching cheap_version."
            )

        # 4) sequential write
        offset = 0
        t0 = time.time()

        pbar = tqdm(loader, desc=f"[CheapRunner:{split}]", dynamic_ncols=True)
        for batch in pbar:
            x, esa, pos = extract_batch(batch)

            x = x.to(self.device, non_blocking=cfg.non_blocking)
            esa = esa.to(self.device, non_blocking=cfg.non_blocking) if esa is not None else None
            pos = pos.to(self.device, non_blocking=cfg.non_blocking) if pos is not None else None

            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", enabled=bool(cfg.amp)):
                    logits, emb = cheap_forward(cheap_model, x, esa, pos)
            else:
                logits, emb = cheap_forward(cheap_model, x, esa, pos)

            logits = logits.detach().float().view(-1)
            emb = emb.detach().float().view(-1, int(emb_dim))

            B = int(logits.numel())
            start, end = offset, offset + B

            entropy = None
            if cfg.has_entropy:
                entropy = binary_entropy_from_logits(logits).detach().float().view(-1)

            store.write_cheap_slice(
                start=start,
                end=end,
                logits=logits.to(dtype=torch.float16),
                emb=emb.to(dtype=torch.float16),
                entropy=(entropy.to(dtype=torch.float16) if entropy is not None else None),
            )

            offset = end
            pbar.set_postfix(written=offset, total=total_cts)

        if offset != total_cts:
            raise RuntimeError(f"[CheapRunner] written {offset} != total_cts {total_cts} for split={split}")

        store.flush_cheap()
        store.set_cheap_ready()

        dt = time.time() - t0
        out_dir = Path(self.em_cache_root) / "em_cache" / split / "cheap"
        print(f"[CheapRunner] DONE split={split} total_cts={total_cts} time={dt:.1f}s -> {out_dir}")
