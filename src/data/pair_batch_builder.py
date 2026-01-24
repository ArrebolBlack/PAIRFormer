# src/data/pair_batch_builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from src.data.dataset import ChunkedCTSDataset
from src.data.em_cache import MemmapCacheStore
from src.models.extractors import get_embedding_and_logit


@dataclass
class PairBatchBuilderConfig:
    kmax: int = 512
    device: str = "cuda"
    pin_memory: bool = True

    # 先保留开关：需要时再打开（你说可以先注释掉）
    include_pos: bool = False
    include_esa: bool = False

    # tokens 处理
    normalize_tokens: bool = False  # e.g. True -> L2 normalize emb


class PairBatchBuilder:
    """
    Stage 4 collate_fn / BatchBuilder

    输入：batch = list of {"pair_id": int}
    输出：dict，至少包含 tokens/mask/y_pair/pair_id（都可在 GPU）
    """

    def __init__(
        self,
        *,
        cts_ds: ChunkedCTSDataset,
        store: MemmapCacheStore,
        instance_model: torch.nn.Module,
        cfg: PairBatchBuilderConfig,
    ):
        self.cts_ds = cts_ds
        self.store = store
        self.instance_model = instance_model
        self.cfg = cfg

        self.device = torch.device(cfg.device)

        # 重要：Instance 模型必须在 GPU 上（你已明确要求）
        # 这里不强制 .to(device)，由外部统一放置更安全，但会检查一次。
        if next(self.instance_model.parameters()).device != self.device:
            raise RuntimeError(
                f"[PairBatchBuilder] instance_model device mismatch: "
                f"model on {next(self.instance_model.parameters()).device}, "
                f"but cfg.device={self.device}. Please move model to device before building DataLoader."
            )

    @torch.no_grad()
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # -------------------------
        # 0) pair_ids (CPU)
        # -------------------------
        pair_ids = torch.tensor([int(b["pair_id"]) for b in batch], dtype=torch.long, device="cpu")
        B = int(pair_ids.numel())
        K = int(self.cfg.kmax)

        # -------------------------
        # 1) Read selection (CPU memmap)
        # sel_uids: [B,K] (int32), mask: [B,K] bool (len-based)
        # -------------------------
        sel_uids_cpu, mask_cpu = self.store.read_selection(pair_ids, K=K)
        sel_uids_cpu = sel_uids_cpu.to(dtype=torch.long, device="cpu")
        mask_cpu = mask_cpu.to(dtype=torch.bool, device="cpu")

        # 额外防御：uids==-1 的位置强制无效（理论上 len-mask 已经覆盖）
        mask_cpu = mask_cpu & (sel_uids_cpu >= 0)

        # -------------------------
        # 2) Pair label y_pair (CPU gather one uid per pair)
        # 取每个 pair slice 的 start uid，读 labels
        # -------------------------
        start_uids = []
        for pid in pair_ids.tolist():
            s, e = self.cts_ds.get_pair_slice(int(pid))
            if e <= s:
                start_uids.append(-1)
            else:
                start_uids.append(int(s))
        start_uids_t = torch.tensor(start_uids, dtype=torch.long, device="cpu")

        # 对空 slice 的 pair：label 置 0（极端保护）
        y_pair = torch.zeros((B,), dtype=torch.float32, device="cpu")
        valid_start = start_uids_t >= 0
        if valid_start.any():
            meta_y = self.cts_ds.batch_gather_by_uid(start_uids_t[valid_start], fields=("labels",))
            y = meta_y["labels"]
            if y is None:
                raise RuntimeError("[PairBatchBuilder] labels field missing in CTS dataset chunks.")
            y_pair_valid = y.view(-1).to(dtype=torch.float32)
            y_pair[valid_start] = y_pair_valid

        # -------------------------
        # 3) Gather selected CTS (CPU)
        # 用 mask 的 2D indices 保证后续 scatter 对齐
        # -------------------------
        idx2d = torch.nonzero(mask_cpu, as_tuple=False)  # [N,2] in row-major order
        N = int(idx2d.shape[0])

        # 如果某个 batch 全空：直接返回空 tokens
        if N == 0:
            # tokens 维度需要知道 D，但这里没有 instance 输出，给 0-dim tokens（pair_agg 可据 mask 跳过）
            # 若你的 pair_agg 强制要求固定 D：可以在 cfg 里加 token_dim，并用 zeros(B,K,D)
            out = {
                "pair_id": pair_ids.to(self.device, non_blocking=True),
                "y_pair": y_pair.to(self.device, non_blocking=True),
                "tokens": None,
                "mask": mask_cpu.to(self.device, non_blocking=True),
                "sel_uids": sel_uids_cpu.to(self.device, non_blocking=True),
            }
            return out

        flat_uids = sel_uids_cpu[idx2d[:, 0], idx2d[:, 1]]  # [N] aligned with idx2d order

        fields = ["X"]
        # 先保留接口，默认关闭
        if self.cfg.include_pos:
            fields.append("pos")
        if self.cfg.include_esa:
            fields.append("esa_scores")

        meta = self.cts_ds.batch_gather_by_uid(flat_uids, fields=tuple(fields))

        x_flat = meta["X"]
        if x_flat is None:
            raise RuntimeError("[PairBatchBuilder] X field missing in CTS dataset chunks.")
        # X: uint8 [N,C,L] -> float32
        x_flat = x_flat.to(dtype=torch.float32)

        # 4) Instance encode on GPU -> feat_flat [N,D], logit_flat [N]
        x_gpu = x_flat.to(self.device, non_blocking=True)
        feat_flat, logit_flat = get_embedding_and_logit(self.instance_model, x_gpu)

        if self.cfg.normalize_tokens:
            feat_flat = F.normalize(feat_flat.float(), dim=-1)

        D = int(feat_flat.shape[1])

        # 5) Scatter back to [B,K,...]
        inst_emb = torch.zeros((B, K, D), dtype=feat_flat.dtype, device=self.device)
        inst_logit = torch.zeros((B, K), dtype=logit_flat.dtype, device=self.device)

        mask_gpu = mask_cpu.to(self.device, non_blocking=True)
        b_idx = idx2d[:, 0].to(self.device, non_blocking=True)
        k_idx = idx2d[:, 1].to(self.device, non_blocking=True)

        inst_emb[b_idx, k_idx, :] = feat_flat
        inst_logit[b_idx, k_idx] = logit_flat.view(-1)

        # padding 置零（稳）
        inst_emb = inst_emb * mask_gpu.unsqueeze(-1).to(inst_emb.dtype)
        inst_logit = inst_logit * mask_gpu.to(inst_logit.dtype)

        out = {
            "pair_id": pair_ids.to(self.device, non_blocking=True),
            "y_pair": y_pair.to(self.device, non_blocking=True),
            "mask": mask_gpu,
            "sel_uids": sel_uids_cpu.to(self.device, non_blocking=True),
            "inst_emb": inst_emb,
            "inst_logit": inst_logit,
        }
        return out

