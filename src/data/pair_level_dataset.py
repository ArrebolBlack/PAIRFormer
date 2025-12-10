# src/data/pair_level_dataset.py

from typing import Dict, List, Tuple, Sequence
import os

import torch
from torch.utils.data import Dataset


class PairLevelDataset(Dataset):
    """
    PairLevelDataset
    =================
    使用 dump_cts_embeddings.py 预计算好的 cache，构造 pair 级别的样本。

    输入缓存格式（per split）:
        path = os.path.join(cache_root, f"{split}.pt")
        data = torch.load(path, map_location="cpu")

        data: Dict[int, Dict[str, Any]]
          data[pid] = {
              "embeddings": Tensor [N_i, d_emb],
              "logits":     Tensor [N_i],
              "esa":        Tensor [N_i],
              "pos":        None 或 Tensor [N_i],   # v0 为 None，预留
              "label":      float (0. 或 1.)
          }

    Dataset 输出格式（__getitem__）:
        tokens : Tensor [M, d_token]   # M <= max_cts_per_pair
        label  : Tensor []             # scalar, float32
        length : int                   # M
        pair_id: int                   # 原始 pair id (set_idx)
    """

    def __init__(
        self,
        cache_root: str,
        split: str,
        max_cts_per_pair: int = 512,
        selection_mode: str = "topk_logit",
    ):
        """
        参数
        ----
        cache_root : str
            dump_cts_embeddings.py 的 --out-root 目录。
            内部会加载 cache_root/{split}.pt。

        split : str
            "train" / "val" / "test" 等，对应 {split}.pt。

        max_cts_per_pair : int
            每个 pair 最多保留多少个 CTS，剩余的截断。若某个 pair 的 CTS 数少于该值，则保留全部。

        selection_mode : str
            选择逻辑：
              - "topk_logit"     : 按 logit 从大到小选前 K 个（v0 默认）
              - "topk_abs_logit" : 按 |logit| 大小选前 K 个
              - "topk_esa"       : 按 ESA score 从大到小选前 K 个
              - "random"         : 随机选 K 个（可用于 data augmentation / ablation）
        """
        super().__init__()

        self.cache_root = cache_root
        self.split = split
        self.max_cts_per_pair = int(max_cts_per_pair)
        self.selection_mode = selection_mode

        path = os.path.join(cache_root, f"{split}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[PairLevelDataset] Cache file not found: {path}. "
                f"请先用 dump_cts_embeddings.py 生成该 split 的 cache。"
            )

        # 加载整个 split 的 pair-level 数据到 CPU 内存
        data = torch.load(path, map_location="cpu")
        if not isinstance(data, dict) or len(data) == 0:
            raise ValueError(
                f"[PairLevelDataset] Loaded cache from {path}, but content is empty or not dict."
            )

        self.data: Dict[int, Dict[str, torch.Tensor]] = data
        # 为了可复现 & 便于 debug，按 pair_id 排序
        self.pair_ids: List[int] = sorted(self.data.keys())

        # 从任意一个 pair 推断 embedding 维度与 token 维度
        any_pid = self.pair_ids[0]
        any_entry = self.data[any_pid]
        emb = any_entry["embeddings"]
        if emb.dim() != 2:
            raise ValueError(
                f"[PairLevelDataset] embeddings for pair {any_pid} "
                f"expected 2D tensor, got shape {emb.shape}"
            )
        self.emb_dim = emb.shape[-1]
        # d_token = d_emb + 2 (logit, esa)
        self.token_dim = self.emb_dim + 2

        print(
            f"[PairLevelDataset] Loaded split='{split}' from {path} "
            f"with {len(self.pair_ids)} pairs. "
            f"Embedding dim = {self.emb_dim}, token dim = {self.token_dim}, "
            f"max_cts_per_pair = {self.max_cts_per_pair}, "
            f"selection_mode = {self.selection_mode}."
        )

    def __len__(self) -> int:
        return len(self.pair_ids)

    # ------- 内部：top-K / random 选取逻辑 ------- #

    def _select_indices(
        self,
        logits: torch.Tensor,
        esa: torch.Tensor,
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
        elif mode == "topk_abs_logit":
            scores = logits.abs()
        elif mode == "topk_esa":
            scores = esa
        elif mode == "random":
            perm = torch.randperm(N)
            return perm[:k]
        else:
            raise ValueError(
                f"[PairLevelDataset] Unknown selection_mode='{mode}'. "
                f"支持: topk_logit, topk_abs_logit, topk_esa, random."
            )

        # torch.topk 会自动处理 k <= N
        _, idx = torch.topk(scores, k=k, largest=True, sorted=True)
        return idx

    # ------- Dataset 主接口 ------- #

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        返回一个 pair 的 token 序列及其 label。

        返回
        ----
        tokens : Tensor [M, d_token]
            token_i = concat(emb_i, logit_i, esa_i)

        label  : Tensor [] (float32)
            pair 级别的标签（0./1.），统一成 float32，方便 BCEWithLogitsLoss 等。

        length : int
            当前 pair 实际使用的 token 数 M (<= max_cts_per_pair)

        pair_id: int
            原始 pair id（set_idx），便于 debug / 分析。
        """
        pair_id = self.pair_ids[idx]
        entry = self.data[pair_id]

        emb = entry["embeddings"]          # [N_i, d_emb]
        logits = entry["logits"].view(-1)  # [N_i]
        esa = entry["esa"].view(-1)        # [N_i]
        label_val = float(entry["label"])  # scalar

        # 安全检查
        if emb.shape[0] != logits.shape[0] or emb.shape[0] != esa.shape[0]:
            raise ValueError(
                f"[PairLevelDataset] Inconsistent lengths for pair {pair_id}: "
                f"embeddings={emb.shape[0]}, logits={logits.shape[0]}, esa={esa.shape[0]}"
            )

        # 选取 M 个 index
        idx_keep = self._select_indices(logits, esa)
        emb = emb[idx_keep]                          # [M, d_emb]
        logits_sel = logits[idx_keep].unsqueeze(-1)  # [M, 1]
        esa_sel = esa[idx_keep].unsqueeze(-1)        # [M, 1]

        # 拼成 token: [M, d_emb+2]
        tokens = torch.cat([emb, logits_sel, esa_sel], dim=-1)

        length = int(tokens.size(0))
        label = torch.tensor(label_val, dtype=torch.float32)

        return tokens, label, length, pair_id


# ----------------------- #
# Collate function
# ----------------------- #

def pair_level_collate_fn(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor, int, int]]
):
    """
    collate_fn，把若干 pair 的可变长 token 序列 pad 成一个 batch。

    输入 batch: list of (tokens, label, length, pair_id)
      - tokens: [M_i, d_token]
      - label : scalar tensor
      - length: int
      - pair_id: int

    输出:
      batch_dict = {
          "tokens":  FloatTensor [B, K, D],   # K = 当前 batch 内最长的 M_i
          "mask":    BoolTensor  [B, K],      # True 表示有效 token
          "labels":  FloatTensor [B],         # pair-level labels
          "lengths": LongTensor   [B],        # 各 pair 的 token 数
          "pair_ids":LongTensor   [B],        # pair id，便于 debug
      }
    """
    tokens_list, labels_list, lengths_list, pair_ids_list = zip(*batch)

    B = len(batch)
    # 当前 batch 内的最大长度
    max_len = max(int(l) for l in lengths_list)
    token_dim = tokens_list[0].size(-1)

    # 初始化 padding 容器
    tokens_padded = torch.zeros(B, max_len, token_dim, dtype=tokens_list[0].dtype)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, tokens in enumerate(tokens_list):
        L = tokens.size(0)
        tokens_padded[i, :L] = tokens
        mask[i, :L] = True

    labels = torch.stack(labels_list).view(-1)  # [B]
    lengths = torch.tensor(lengths_list, dtype=torch.long)  # [B]
    pair_ids = torch.tensor(pair_ids_list, dtype=torch.long)  # [B]

    batch_dict = {
        "tokens": tokens_padded,  # [B, K, D]
        "mask": mask,             # [B, K]
        "labels": labels,         # [B]
        "lengths": lengths,       # [B]
        "pair_ids": pair_ids,     # [B]
    }
    return batch_dict
