# src/em/instance_cache_builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable, Set
from contextlib import nullcontext

import numpy as np
import torch
from tqdm import tqdm

from src.data.em_cache import MemmapCacheStore
from src.models.extractors import get_embedding_and_logit


@dataclass
class InstanceCacheBuildConfig:
    kmax: int = 512
    use_amp: bool = False
    overwrite: bool = False
    batch_size: int = 256




def _gather_inputs_by_uid(cts_ds, uids: torch.Tensor) -> torch.Tensor:
    """
    返回 float32 [N,C,L]。
    你的 chunk 内字段是 "X"（uint8 one-hot），不是 "inputs"。
    """
    uids = uids.to(dtype=torch.long, device="cpu")

    # Fast path: batch_gather_by_uid
    if hasattr(cts_ds, "batch_gather_by_uid"):
        out = cts_ds.batch_gather_by_uid(uids, fields=("X",))
        x = out.get("X", None)
        if x is None:
            raise RuntimeError("[InstanceCacheBuilder] batch_gather_by_uid returned None for field 'X'.")
        return x.to(dtype=torch.float32)

    # Fallback: slow __getitem__
    xs = []
    for i in uids.tolist():
        x, y, set_idx, esa, pos = cts_ds[int(i)]   # __getitem__ 返回 tuple
        xs.append(x)
    return torch.stack(xs, dim=0).to(dtype=torch.float32)


@torch.no_grad()
def build_instance_cache_from_selection(
    *,
    em_cache_root: str,
    split: str,
    store: MemmapCacheStore,
    cts_ds,
    instance_model: torch.nn.Module,
    inst_version: str,
    inst_emb_dim: int,
    total_cts: int,
    cfg: InstanceCacheBuildConfig,
) -> None:
    """
    只对 selection 里出现的 CTS uid 计算 instance 输出并写入 instance_cache。
    要求：store 已经 create_or_open_selection(..., require_ready=True) 完成。
    """

    # --- 必要的前置校验 ---
    if store.sel_meta is None or getattr(store, "_sel_uids", None) is None:
        raise RuntimeError("[InstanceCacheBuilder] selection cache not opened in store. "
                           "Call store.create_or_open_selection(..., require_ready=True) first.")

    # 1) open/create instance cache（和 selection 版本绑定，便于防止错配）
    store.create_or_open_instance(
        total_cts=int(total_cts),
        emb_dim=int(inst_emb_dim),
        inst_version=str(inst_version),
        sel_version_used=str(store.sel_meta.sel_version),
        cheap_version_used=str(store.sel_meta.cheap_version_used),
        overwrite=bool(cfg.overwrite),
    )

    num_pairs = int(store.sel_meta.num_pairs)
    kmax = int(store.sel_meta.kmax)
    K = min(int(cfg.kmax), kmax)

    # 2) 收集所有选中的 uids（分块 unique，避免一次性爆内存）
    uniq: Set[int] = set()
    chunk = 4096

    for s in tqdm(range(0, num_pairs, chunk), desc=f"[InstCache][Collect:{split}]"):
        e = min(num_pairs, s + chunk)
        u = np.asarray(store._sel_uids[s:e, :K])  # [chunk,K]
        u = u.reshape(-1)
        u = u[u >= 0]
        if u.size == 0:
            continue
        for x in np.unique(u):
            uniq.add(int(x))

    if len(uniq) == 0:
        print(f"[InstCache] no selected uids for split={split}.")
        store.flush_instance()
        store.set_instance_ready()
        return

    uids_all = torch.tensor(sorted(list(uniq)), dtype=torch.long, device="cpu")
    # 关键 sanity check：uids 必须落在 [0, total_cts)
    if int(uids_all.min()) < 0 or int(uids_all.max()) >= int(total_cts):
        raise RuntimeError(
            f"[InstCache] selected uid out of range: min={int(uids_all.min())}, "
            f"max={int(uids_all.max())}, total_cts={int(total_cts)}. "
            f"Selection cache likely built against a different dataset/indexing."
        )

    print(f"[InstCache] split={split} unique_selected_uids={uids_all.numel()} / total_cts={total_cts}")

    # 3) batch 推理写 cache
    instance_model.eval()
    device = next(instance_model.parameters()).device
    bs = int(cfg.batch_size)

    autocast_ctx = (
        torch.cuda.amp.autocast(enabled=True)
        if (cfg.use_amp and device.type == "cuda")
        else nullcontext()
    )

    for i in tqdm(range(0, uids_all.numel(), bs), desc=f"[InstCache][Build:{split}]"):
        u = uids_all[i:i + bs]  # CPU uids
        x = _gather_inputs_by_uid(cts_ds, u)      # CPU float32 [N,C,L]

        # 可选：pin_memory 让 non_blocking 生效（如果你在 CPU->GPU 传输上卡）
        # x = x.pin_memory()

        x = x.to(device, non_blocking=True)

        with autocast_ctx:
            feat, logit = get_embedding_and_logit(instance_model, x)

        feat = feat.detach().to("cpu", non_blocking=False).to(torch.float16)
        logit = logit.detach().view(-1).to("cpu", non_blocking=False).to(torch.float16)

        store.write_instance_by_uids(u, logit=logit, emb=feat)

    store.flush_instance()
    store.set_instance_ready()
    print(f"[InstCache] DONE split={split} -> em_cache/{split}/instance")