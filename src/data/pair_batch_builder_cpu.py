# src/data/pair_batch_builder_cpu.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from src.data.dataset import ChunkedCTSDataset
from src.data.em_cache import MemmapCacheStore


@dataclass
class PairBatchBuilderCPUConfig:
    kmax: int = 512
    pin_memory: bool = True

    # 先保留开关：需要时再打开
    include_pos: bool = True
    include_esa: bool = True


class PairBatchBuilderCPU:
    """
    Stage 4 CPU-only collate_fn / BatchBuilder

    Input:
      batch = list of {"pair_id": int}

    Output (all on CPU):
      {
        "pair_id": [B] long,
        "y_pair" : [B] float32,
        "mask"   : [B,K] bool,
        "sel_uids": [B,K] long (debug/optional),
        "X"      : [B,K,C,L] uint8  (padded) OR None if all-empty
        # optional: "pos", "esa_scores"
      }

    Key design:
      - Open selection memmap lazily INSIDE worker process (safe for num_workers>0).
      - Heavy GPU instance encode is NOT here.
    """

    def __init__(
        self,
        *,
        cts_ds: ChunkedCTSDataset,
        em_cache_root: Union[str, Path],
        split: str,
        cfg: PairBatchBuilderCPUConfig,
        expected_path_hash: Optional[str] = None,
        expected_dataset_hash_key: Optional[str] = None,
    ):
        self.cts_ds = cts_ds
        self.em_cache_root = Path(str(em_cache_root))
        self.split = str(split)
        self.cfg = cfg

        self.expected_path_hash = expected_path_hash
        self.expected_dataset_hash_key = expected_dataset_hash_key
        self._sel_kmax: Optional[int] = None

        # do not open memmap in __init__ (fork-safety)
        self._store: Optional[MemmapCacheStore] = None

    def _lazy_open_selection_store(self) -> None:
        if self._store is not None:
            return

        sel_meta_path = self.em_cache_root / "em_cache" / self.split / "selection" / "meta.json"
        if not sel_meta_path.exists():
            raise FileNotFoundError(f"[PairBatchBuilderCPU] selection meta not found: {sel_meta_path}")

        with open(sel_meta_path, "r") as f:
            sel_meta = json.load(f)

        self._sel_kmax = int(sel_meta["kmax"])

        # state 必须 ready
        if sel_meta.get("state") != "ready":
            raise RuntimeError(f"[PairBatchBuilderCPU] selection cache not ready: state={sel_meta.get('state')}")

        # identity 强校验：如果你提供了 expected，就必须一致
        if self.expected_path_hash is not None and str(sel_meta.get("path_hash")) != str(self.expected_path_hash):
            raise RuntimeError(
                f"[PairBatchBuilderCPU] selection path_hash mismatch: meta={sel_meta.get('path_hash')} "
                f"expected={self.expected_path_hash}"
            )
        if self.expected_dataset_hash_key is not None and str(sel_meta.get("dataset_hash_key")) != str(self.expected_dataset_hash_key):
            raise RuntimeError(
                f"[PairBatchBuilderCPU] selection dataset_hash_key mismatch: meta={sel_meta.get('dataset_hash_key')} "
                f"expected={self.expected_dataset_hash_key}"
            )

        # kmax 强约束：cfg.kmax 不能超过 selection cache 的 kmax
        if int(self.cfg.kmax) > int(self._sel_kmax):
            raise RuntimeError(
                f"[PairBatchBuilderCPU] cfg.kmax={self.cfg.kmax} > selection_cache.kmax={self._sel_kmax}. "
                f"Either lower cfg.kmax or rebuild selection cache with larger kmax."
            )

        # 关键：用 expected identity 初始化 store（没提供 expected 时再退回 meta）
        path_hash = self.expected_path_hash if self.expected_path_hash is not None else str(sel_meta["path_hash"])
        dataset_hash_key = self.expected_dataset_hash_key if self.expected_dataset_hash_key is not None else str(sel_meta["dataset_hash_key"])

        store = MemmapCacheStore(
            cache_root=str(self.em_cache_root),
            split=str(self.split),
            path_hash=str(path_hash),
            dataset_hash_key=str(dataset_hash_key),
        )
        store.create_or_open_selection(
            num_pairs=int(sel_meta["num_pairs"]),
            kmax=int(sel_meta["kmax"]),
            sel_version=str(sel_meta["sel_version"]),
            cheap_version_used=str(sel_meta["cheap_version_used"]),
            overwrite=False,
            require_ready=True,  
        )
        self._store = store


    @torch.no_grad()
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        self._lazy_open_selection_store()
        assert self._store is not None

        assert self._sel_kmax is not None
        K = int(self.cfg.kmax)
        if K > self._sel_kmax:
            raise ValueError(f"[PairBatchBuilderCPU] requested K={K} > selection_cache.kmax={self._sel_kmax}")

        # 0) pair_ids
        pair_ids = torch.tensor([int(b["pair_id"]) for b in batch], dtype=torch.long, device="cpu")
        B = int(pair_ids.numel())
        K = int(self.cfg.kmax)

        # 1) selection: sel_uids[B,K], mask[B,K]
        sel_uids, mask = self._store.read_selection(pair_ids, K=K)
        sel_uids = sel_uids.to(dtype=torch.long, device="cpu")
        mask = mask.to(dtype=torch.bool, device="cpu")
        mask = mask & (sel_uids >= 0)

        # 2) y_pair: read one label per pair via start uid
        start_uids = []
        for pid in pair_ids.tolist():
            s, e = self.cts_ds.get_pair_slice(int(pid))
            start_uids.append(int(s) if e > s else -1)
        start_uids_t = torch.tensor(start_uids, dtype=torch.long, device="cpu")

        y_pair = torch.zeros((B,), dtype=torch.float32, device="cpu")
        valid_start = start_uids_t >= 0


        '''
        这里观察dump_cts_embedding.py得，原来老pipeline的处理是直接跳过这种空pair
        于是修改DynamicPairDataset初始化时过滤空pair
        所以如果还遇到，就直接raise error

        但是这样做不是很符合语义，没扫出CTS就应该判负
        DynamicPairDataset保留2版，这里也保留2版，可以自行调整
        '''
        # # 约定：空 pair 作为合法样本，y_pair=0（y_pair 已经默认是 0）
        # # 同时必须强制该样本没有任何有效 token，避免后续 gather 误读 uid
        # empty = ~valid_start  # 空 pair：没有任何 CTS
        # if empty.any():
        #     sel_uids[empty] = -1
        #     mask[empty] = False  # 注意：你的 mask=True 表示有效位置（因为后面 nonzero(mask)）
        #     # 可选：如果你还想保留提示但不刷屏，可以改成每 N step 打一次
        #     # bad = empty.nonzero(as_tuple=False).view(-1)[:10].tolist()
        #     # print(f"[PairBatchBuilderCPU] Info: empty pairs treated as y_pair=0. idx(first10)={bad}")

        # # 重新确保 mask 和 sel_uids 一致（防御式）
        # mask = mask & (sel_uids >= 0)

        # 检测到空pair（不含CTS）报错
        if not valid_start.all():
            bad = (~valid_start).nonzero(as_tuple=False).view(-1)[:10].tolist()
            raise RuntimeError(f"[PairBatchBuilderCPU] found empty pairs (no CTS). idx(first10)={bad}")
            # print(f"[PairBatchBuilderCPU] Warning: found empty pairs (no CTS). idx(first10)={bad}")


        if valid_start.any():
            meta_y = self.cts_ds.batch_gather_by_uid(start_uids_t[valid_start], fields=("labels",))
            y = meta_y["labels"]
            if y is None:
                raise RuntimeError("[PairBatchBuilderCPU] labels missing in CTS dataset chunks.")
            y_pair[valid_start] = y.view(-1).to(dtype=torch.float32)

        # 3) gather selected CTS fields -> pad to [B,K,...]
        idx2d = torch.nonzero(mask, as_tuple=False)  # [N,2]
        N = int(idx2d.shape[0])

        if N == 0:
            return {
                "pair_id": pair_ids,
                "y_pair": y_pair,
                "mask": mask,
                "sel_uids": sel_uids,
                "X": None,
            }

        flat_uids = sel_uids[idx2d[:, 0], idx2d[:, 1]]  # [N]

        # fields = ["inputs"]  # 改：真实数据字段用 inputs
        fields = ["X"]
        if self.cfg.include_pos:
            fields.append("pos")
        if self.cfg.include_esa:
            fields.append("esa_scores")

        meta = self.cts_ds.batch_gather_by_uid(flat_uids, fields=tuple(fields))

        # x_flat = meta.get("inputs", None)
        x_flat = meta.get("X", None)
        
        if x_flat is None:
            raise RuntimeError("[PairBatchBuilderCPU] missing 'X' in CTS dataset chunks.")

        # x_flat: [N,C,L] uint8
        C = int(x_flat.shape[1])
        L = int(x_flat.shape[2])

        X_pad = torch.zeros((B, K, C, L), dtype=x_flat.dtype, device="cpu")
        b_idx = idx2d[:, 0]
        k_idx = idx2d[:, 1]
        X_pad[b_idx, k_idx, :, :] = x_flat

        out: Dict[str, Any] = {
            "pair_id": pair_ids,
            "y_pair": y_pair,
            "mask": mask,
            "sel_uids": sel_uids,  # cached/hybrid 必需
            "X": X_pad,            # 对外：instance 输入叫 X
        }

        # 可选字段保持不变（供 token 拼接）
        if self.cfg.include_pos:
            pos_flat = meta["pos"].reshape(-1).to(torch.float32)
            pos_pad = torch.zeros((B, K), dtype=torch.float32, device="cpu")
            pos_pad[b_idx, k_idx] = pos_flat
            out["pos"] = pos_pad

        if self.cfg.include_esa:
            esa_flat = meta["esa_scores"].reshape(-1).to(torch.float32)
            esa_pad = torch.zeros((B, K), dtype=torch.float32, device="cpu")
            esa_pad[b_idx, k_idx] = esa_flat
            out["esa_scores"] = esa_pad

        return out


    def invalidate(self) -> None:
        self._store = None
        self._sel_kmax = None