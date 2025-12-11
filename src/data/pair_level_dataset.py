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
        pos_in_token: bool = True,
        order_mode: str = "score",    # "score" 或 "original"
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
        self.pos_in_token = bool(pos_in_token)
        self.order_mode = order_mode  # "score" = 按得分顺序；"original" = 保留原始顺序

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

        # v0：dump_cts_embeddings 里的 "pos" 可能为 None
        raw_pos = any_entry.get("pos", None)
        self.has_cached_pos = isinstance(raw_pos, torch.Tensor)

        # token 维度：
        #   emb_dim
        #   + 2 (logit, esa)
        #   + 1 (pos，如果 pos_in_token=True)
        if self.pos_in_token:
            self.token_dim = self.emb_dim + 3   # emb + logit + esa + pos
        else:
            self.token_dim = self.emb_dim + 2   # emb + logit + esa

        print(
            f"[PairLevelDataset] Loaded split='{split}' from {path} "
            f"with {len(self.pair_ids)} pairs. "
            f"Embedding dim = {self.emb_dim}, token dim = {self.token_dim}, "
            f"max_cts_per_pair = {self.max_cts_per_pair}, "
            f"selection_mode = {self.selection_mode}, "
            f"order_mode = {self.order_mode}, "
            f"has_cached_pos = {self.has_cached_pos}, "
            f"pos_in_token = {self.pos_in_token}."
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor]:
        """
        返回一个 pair 的 token 序列及其 label。

        返回
        ----
        tokens : [M, d_token]
            - 如果 pos_in_token=True : concat(emb, logit, esa, pos)，pos 单独返回
            - 如果 pos_in_token=False: concat(emb, logit, esa)，pos 单独返回

        label  : Tensor [] (float32)
            pair 级别的标签（0./1.），统一成 float32，方便 BCEWithLogitsLoss 等。

        length : int
            当前 pair 实际使用的 token 数 M (<= max_cts_per_pair)

        pair_id: int
            原始 pair id（set_idx），便于 debug / 分析。

        pos_seq: [M]，归一化位置（0~1），用于 direct position encoding / ablation
        """
        pair_id = self.pair_ids[idx]
        entry = self.data[pair_id]

        emb = entry["embeddings"]          # [N_i, d_emb]
        logits = entry["logits"].view(-1)  # [N_i]
        esa = entry["esa"].view(-1)        # [N_i]
        raw_pos = entry.get("pos", None)   # 可能是 None 或 Tensor
        label_val = float(entry["label"])  # scalar

        # 基本安全检查
        N = emb.shape[0]
        if N != logits.shape[0] or N != esa.shape[0]:
            raise ValueError(
                f"[PairLevelDataset] Inconsistent lengths for pair {pair_id}: "
                f"embeddings={N}, logits={logits.shape[0]}, esa={esa.shape[0]}"
            )

        # ---------- 构造 pos 向量 ----------
        if isinstance(raw_pos, torch.Tensor):
            pos_full = raw_pos.view(-1).float()     # 用 dump 时存的真实位置
            if pos_full.shape[0] != N:
                raise ValueError(
                    f"[PairLevelDataset] pos length mismatch for pair {pair_id}: "
                    f"pos={pos_full.shape[0]}, N={N}"
                )
        else:
            # v0：没有显式 pos，用 index 归一化作为“原始顺序位置”
            if N == 1:
                pos_full = torch.tensor([0.0], dtype=torch.float32)
            else:
                pos_full = torch.linspace(0.0, 1.0, steps=N, dtype=torch.float32)

        # ---------- 选取 M 个 CTS ----------
        idx_keep = self._select_indices(logits, esa)   # [M]

        if self.order_mode == "original":
            # 按物理/相对位置排序：pos 越小位置越靠前
            pos_sub = pos_full[idx_keep]              # [M]
            sort_pos, sort_idx = torch.sort(pos_sub)  # 升序排序
            idx_keep = idx_keep[sort_idx]             # 重排 index
            pos_sel = sort_pos                        # [M] 与 idx_keep 对齐
        elif self.order_mode == "score":
            # 保持得分/随机排序
            pos_sel = pos_full[idx_keep]
        else:
            raise ValueError(
                f"[PairLevelDataset] Unknown order_mode='{self.order_mode}'. "
                f"支持: 'score', 'original'."
            )

        # 再据此取 emb/logit/esa
        emb = emb[idx_keep]                          # [M, d_emb]
        logits_sel = logits[idx_keep].unsqueeze(-1)  # [M, 1]
        esa_sel = esa[idx_keep].unsqueeze(-1)        # [M, 1]


        # ---------- 拼 token ----------
        if self.pos_in_token:
            # token: [M, d_emb+3]
            tokens = torch.cat(
                [emb, logits_sel, esa_sel, pos_sel.unsqueeze(-1)], dim=-1
            )
        else:
            # token 不包含 pos，pos 单独返回
            tokens = torch.cat(
                [emb, logits_sel, esa_sel], dim=-1
            )

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

