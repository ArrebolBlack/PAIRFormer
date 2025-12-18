# src/data/pair_level_dataset.py
import json
import bisect
import gc
import numpy as np

from typing import Dict, List, Tuple, Sequence, Optional
import os

import torch
from torch.utils.data import Dataset


class PairLevelDataset(Dataset):
    """
    PairLevelDataset
    =================
    使用 dump_cts_embeddings.py 预计算好的 cache，构造 pair 级别的样本。

    统一约定：**只支持分 shard 布局**：

        cache_root/{split}_shardXXXX.pt
        cache_root/{split}_meta.json

    每个 shard 文件内部是：

        data: Dict[int, Dict[str, Any]]
          data[pid] = {
              "embeddings": Tensor [N_i, d_emb],
              "logits":     Tensor [N_i],
              "esa":        Tensor [N_i],
              "pos":        None 或 Tensor [N_i],
              "label":      float (0. 或 1.)
          }

    Dataset 输出格式（__getitem__）:
        tokens : Tensor [M, d_token]   # M <= max_cts_per_pair
        label  : Tensor []             # scalar, float32
        length : int                   # M
        pair_id: int                   # 原始 pair id (set_idx)
        pos_seq: Tensor [M]            # 归一化位置（0~1）
    """

    def __init__(
        self,
        cache_root: str,
        split: str,
        max_cts_per_pair: int = 512,
        selection_mode: str = "topk_logit",
        pos_in_token: bool = True,
        order_mode: str = "original",    # "score" 或 "original"
        random_order_seed: int = 12345,
        random_select_seed: Optional[int] = None,  # 若为 None，则保持 legacy random 行为（非确定性）
    ):
        super().__init__()

        self.cache_root = cache_root
        self.split = split
        self.max_cts_per_pair = int(max_cts_per_pair)
        self.selection_mode = selection_mode
        self.pos_in_token = bool(pos_in_token)
        self.order_mode = order_mode

        self.random_order_seed = int(random_order_seed)
        self.random_select_seed = None if random_select_seed is None else int(random_select_seed)

        # -------- 只支持“分 shard 模式” --------
        meta_path = os.path.join(cache_root, f"{split}_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"[PairLevelDataset] meta file '{meta_path}' 不存在，"
                f"请先使用 dump_cts_embeddings.py 生成分块 cache。"
            )

        self.mode = "chunked"
        with open(meta_path, "r") as f:
            shard_meta = json.load(f)

        # 确保按 shard_idx 顺序排序（如果 meta 里没有 shard_idx 字段，则按 path 排序）
        shard_meta.sort(key=lambda m: m.get("shard_idx", m["path"]))

        self.chunk_files = [os.path.join(cache_root, m["path"]) for m in shard_meta]
        self.chunk_sizes = [int(m["num_pairs"]) for m in shard_meta]
        self.cum_sizes = np.cumsum(self.chunk_sizes).tolist()
        self.total_pairs = self.cum_sizes[-1] if self.cum_sizes else 0

        self.current_chunk_idx = -1
        self.current_chunk = None   # 当前 shard 的 dict[pid] -> entry
        self.current_pair_ids = None  # 当前 shard 的 pid 排好序的 list[int]

        if self.total_pairs == 0:
            raise ValueError(
                f"[PairLevelDataset] Chunked mode: total_pairs=0 for split='{split}' in {cache_root}"
            )

        # 为了推断 emb_dim / token_dim，只读第一个 shard 的一个 pair
        first_chunk = torch.load(self.chunk_files[0], map_location="cpu", weights_only=False)
        if not isinstance(first_chunk, dict) or len(first_chunk) == 0:
            raise ValueError(
                f"[PairLevelDataset] First shard {self.chunk_files[0]} is empty or not dict."
            )
        any_pid = sorted(first_chunk.keys())[0]
        any_entry = first_chunk[any_pid]
        emb = any_entry["embeddings"]
        if emb.dim() != 2:
            raise ValueError(
                f"[PairLevelDataset] embeddings for pair {any_pid} "
                f"expected 2D tensor, got shape {emb.shape}"
            )
        self.emb_dim = emb.shape[-1]

        raw_pos = any_entry.get("pos", None)
        self.has_cached_pos = isinstance(raw_pos, torch.Tensor)

        if self.pos_in_token:
            self.token_dim = self.emb_dim + 3   # emb + logit + esa + pos
        else:
            self.token_dim = self.emb_dim + 2   # emb + logit + esa

        # 释放探测用的 shard
        del first_chunk
        gc.collect()

        print(
            f"[PairLevelDataset] Chunked mode: split='{split}' from root='{cache_root}' "
            f"with {self.total_pairs} pairs across {len(self.chunk_files)} shards. "
            f"Embedding dim = {self.emb_dim}, token dim = {self.token_dim}, "
            f"max_cts_per_pair = {self.max_cts_per_pair}, "
            f"selection_mode = {self.selection_mode}, "
            f"order_mode = {self.order_mode}, "
            f"has_cached_pos = {self.has_cached_pos}, "
            f"pos_in_token = {self.pos_in_token}."
        )

    def __len__(self) -> int:
        return self.total_pairs

    # ------- 内部：top-K / random 选取逻辑 ------- #

    def _select_indices(
        self,
        logits: torch.Tensor,
        esa: torch.Tensor,
        pos_full: Optional[torch.Tensor] = None,
        pair_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        根据 selection_mode，在 [0, N-1] 中选择若干 index，最多 max_cts_per_pair 个。

        logits: [N]
        esa   : [N]

        返回:
            idx_keep: LongTensor [M],  M <= max_cts_per_pair
        """
        logits = logits.view(-1)
        esa = esa.view(-1)
        N = logits.shape[0]
        if N == 0:
            raise ValueError("[PairLevelDataset] Encountered pair with 0 CTS.")

        k = min(N, self.max_cts_per_pair)
        mode = self.selection_mode

        if mode == "topk_logit":
            scores = logits
            _, idx = torch.topk(scores, k=k, largest=True, sorted=True)
            return idx

        if mode == "topk_abs_logit":
            scores = logits.abs()
            _, idx = torch.topk(scores, k=k, largest=True, sorted=True)
            return idx

        if mode == "topk_esa":
            scores = esa
            _, idx = torch.topk(scores, k=k, largest=True, sorted=True)
            return idx

        # NEW: all + truncate = 不过滤，只截断前 k 个（保持 cache 原始顺序）
        if mode == "all_truncate":
            if pos_full is None:
                return torch.arange(k, dtype=torch.long)
            idx_pos = torch.argsort(pos_full)
            return idx_pos[:k].to(torch.long)

        # legacy random: 旧行为为非确定性 randperm
        if mode == "random":
            if self.random_select_seed is None:
                perm = torch.randperm(N)
            else:
                # 仅当显式给出 random_select_seed 时才启用确定性随机（不影响旧默认）
                if pair_id is None:
                    raise ValueError("[PairLevelDataset] random_select_seed is set but pair_id is None.")
                g = torch.Generator()
                g.manual_seed(self.random_select_seed + int(pair_id))
                perm = torch.randperm(N, generator=g)
            return perm[:k]

        raise ValueError(
            f"[PairLevelDataset] Unknown selection_mode='{mode}'. "
            f"支持: topk_logit, topk_abs_logit, topk_esa, all_truncate, random."
        )
    

    def _get_entry_by_index_chunked(self, idx: int) -> Tuple[int, Dict[str, torch.Tensor]]:
        """
        在 chunked 模式下，根据全局 pair 下标 idx 找到对应的 shard & pair entry。
        返回 (pair_id, entry_dict)。
        """
        if idx < 0 or idx >= self.total_pairs:
            raise IndexError("PairLevelDataset index out of range")

        chunk_idx = bisect.bisect_right(self.cum_sizes, idx)

        if chunk_idx != self.current_chunk_idx:
            # 切换 shard
            if self.current_chunk is not None:
                del self.current_chunk
                self.current_chunk = None
                self.current_pair_ids = None
                gc.collect()

            shard_path = self.chunk_files[chunk_idx]
            chunk = torch.load(shard_path, map_location="cpu", weights_only=False)
            if not isinstance(chunk, dict) or len(chunk) == 0:
                raise ValueError(
                    f"[PairLevelDataset] Shard {shard_path} is empty or not dict."
                )
            self.current_chunk = chunk
            self.current_pair_ids = sorted(chunk.keys())
            self.current_chunk_idx = chunk_idx

        # 计算在当前 shard 内的局部 pair index
        local_idx = idx if chunk_idx == 0 else idx - self.cum_sizes[chunk_idx - 1]
        if local_idx < 0 or local_idx >= len(self.current_pair_ids):
            raise IndexError(
                f"[PairLevelDataset] local_idx={local_idx} out of range for shard {self.current_chunk_idx}"
            )

        pair_id = self.current_pair_ids[local_idx]
        entry = self.current_chunk[pair_id]
        return int(pair_id), entry

    # ------- Dataset 主接口 ------- #

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor]:
        """
        返回一个 pair 的 token 序列及其 label。

        返回
        ----
        tokens : [M, d_token]
        label  : Tensor [] (float32)
        length : int
        pair_id: int
        pos_seq: [M]，归一化位置（0~1）
        """
        pair_id, entry = self._get_entry_by_index_chunked(idx)

        emb = entry["embeddings"]          # [N_i, d_emb]
        logits = entry["logits"].view(-1)  # [N_i]
        esa = entry["esa"].view(-1)        # [N_i]
        raw_pos = entry.get("pos", None)   # 可能是 None 或 Tensor
        label_val = float(entry["label"])  # scalar

        N = emb.shape[0]
        if N != logits.shape[0] or N != esa.shape[0]:
            raise ValueError(
                f"[PairLevelDataset] Inconsistent lengths for pair {pair_id}: "
                f"embeddings={N}, logits={logits.shape[0]}, esa={esa.shape[0]}"
            )

        # ---------- 构造 pos 向量 ----------
        if isinstance(raw_pos, torch.Tensor):
            pos_full = raw_pos.view(-1).float()
            if pos_full.shape[0] != N:
                raise ValueError(
                    f"[PairLevelDataset] pos length mismatch for pair {pair_id}: "
                    f"pos={pos_full.shape[0]}, N={N}"
                )
        else:
            if N == 1:
                pos_full = torch.tensor([0.0], dtype=torch.float32)
            else:
                pos_full = torch.linspace(0.0, 1.0, steps=N, dtype=torch.float32)

        # ---------- 选取 M 个 CTS ----------
        # idx_keep = self._select_indices(logits, esa, pair_id=pair_id)  # [M]
        idx_keep = self._select_indices(logits, esa, pos_full=pos_full, pair_id=pair_id)


        if self.order_mode == "original":
            pos_sub = pos_full[idx_keep]              # [M]
            sort_pos, sort_idx = torch.sort(pos_sub)  # 升序排序
            idx_keep = idx_keep[sort_idx]
            pos_sel = sort_pos
        elif self.order_mode == "score":
            pos_sel = pos_full[idx_keep]
        elif self.order_mode == "random":
            # NEW: deterministic shuffle of selected CTS order (sanity check)
            M = int(idx_keep.numel())
            g = torch.Generator()
            g.manual_seed(self.random_order_seed + int(pair_id))
            perm = torch.randperm(M, generator=g)
            idx_keep = idx_keep[perm]
            pos_sel = pos_full[idx_keep]
        elif self.order_mode in ("score_logit", "logit"):
            # NEW: reorder selected CTS by logits (descending) using topk for consistency
            # (does NOT change legacy 'score' behavior; use this mode explicitly in ablations)
            scores_sub = logits[idx_keep]
            local = torch.argsort(scores_sub, descending=True)
            idx_keep = idx_keep[local]
            pos_sel = pos_full[idx_keep]


        else:
            raise ValueError(
                f"[PairLevelDataset] Unknown order_mode='{self.order_mode}'. "
                f"支持: 'score', 'original', 'random', 'score_logit'."
            )


        emb = emb[idx_keep]                          # [M, d_emb]
        logits_sel = logits[idx_keep].unsqueeze(-1)  # [M, 1]
        esa_sel = esa[idx_keep].unsqueeze(-1)        # [M, 1]

        if self.pos_in_token:
            tokens = torch.cat(
                [emb, logits_sel, esa_sel, pos_sel.unsqueeze(-1)], dim=-1
            )   # [M, d_emb+3]
        else:
            tokens = torch.cat(
                [emb, logits_sel, esa_sel], dim=-1
            )   # [M, d_emb+2]

        length = int(tokens.size(0))
        label = torch.tensor(label_val, dtype=torch.float32)

        return tokens, label, length, pair_id, pos_sel


# ----------------------- #
# Collate function
# ----------------------- #

def pair_level_collate_fn(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor]]
):
    """
    输入 batch: list of (tokens, label, length, pair_id, pos_seq)

      - tokens : [M_i, d_token]
      - label  : scalar tensor
      - length : int
      - pair_id: int
      - pos_seq: [M_i]，归一化位置 (0~1)

    输出:
      batch_dict = {
          "tokens":  [B, K, D],
          "mask":    [B, K],
          "labels":  [B],
          "lengths": [B],
          "pair_ids":[B],
          "pos":     [B, K],   # padding 为 0
      }
    """
    tokens_list, labels_list, lengths_list, pair_ids_list, pos_list = zip(*batch)

    B = len(batch)
    max_len = max(int(l) for l in lengths_list)
    token_dim = tokens_list[0].size(-1)

    tokens_padded = torch.zeros(B, max_len, token_dim, dtype=tokens_list[0].dtype)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    pos_padded = torch.zeros(B, max_len, dtype=torch.float32)

    for i, (tokens, pos_seq) in enumerate(zip(tokens_list, pos_list)):
        L = tokens.size(0)
        tokens_padded[i, :L] = tokens
        mask[i, :L] = True

        # pos_seq: [M_i] -> pad 到 [max_len]
        pos_padded[i, :L] = pos_seq

    labels = torch.stack(labels_list).view(-1)  # [B]
    lengths = torch.tensor(lengths_list, dtype=torch.long)  # [B]
    pair_ids = torch.tensor(pair_ids_list, dtype=torch.long)  # [B]

    batch_dict = {
        # === 给 PairTransformerAggregator 用的字段 ===
        "tokens": tokens_padded,      # [B, K, D]
        "mask": mask,                 # [B, K]  True = valid
        "pos": pos_padded,            # [B, K]
        "lengths": lengths,           # [B]
        "pair_ids": pair_ids,         # [B]  仅用于分析 / debug

        # === 给通用 Trainer._unpack_batch 兼容的 alias ===
        "inputs": tokens_padded,      # [B, K, D]  -> x
        "labels": labels,             # [B]        -> y
        # Stage-2 目前不需要 set-level MIL，先不传 set_idx
        # 若以后想在 pair 之上再做一层 bag，可以加上：
        # "set_idx": pair_ids,
    }
    return batch_dict
