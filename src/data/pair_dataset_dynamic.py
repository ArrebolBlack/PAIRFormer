# # src/data/pair_dataset_dynamic.py
# from __future__ import annotations

# from typing import Any, Dict, List, Optional, Sequence
# import torch

# from src.data.dataset import ChunkedCTSDataset


# class DynamicPairDataset(torch.utils.data.Dataset):
#     """
#     Dynamic PairDataset (Stage 4)
#     - __getitem__ 只返回 pair_id，保持极轻量
#     - y_pair 等其他信息由 BatchBuilder/collate_fn 批量读取（更快）
#     """

#     def __init__(self, cts_ds: ChunkedCTSDataset):
#         if cts_ds.pair_offsets is None or cts_ds.num_pairs is None:
#             raise RuntimeError(
#                 "[DynamicPairDataset] ChunkedCTSDataset has no PairIndex. "
#                 "Please build pair_index_*.pt first."
#             )
#         self.cts_ds = cts_ds
#         self.num_pairs = int(cts_ds.num_pairs)

#     def __len__(self) -> int:
#         return self.num_pairs

#     def __getitem__(self, pair_id: int) -> Dict[str, Any]:
#         return {"pair_id": int(pair_id)}


# src/data/pair_dataset_dynamic.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import torch

from src.data.dataset import ChunkedCTSDataset


class DynamicPairDataset(torch.utils.data.Dataset):
    """
    Dynamic PairDataset (Stage 4)
    - 只返回 pair_id
    - 关键：可选过滤掉没有任何 CTS 的空 pair（复刻旧代码：根本不会遇到空 pair）
    """

    def __init__(self, cts_ds: ChunkedCTSDataset, *, drop_empty_pairs: bool = True):
        if cts_ds.pair_offsets is None or cts_ds.num_pairs is None:
            raise RuntimeError(
                "[DynamicPairDataset] ChunkedCTSDataset has no PairIndex. "
                "Please build pair_index_*.pt first."
            )
        self.cts_ds = cts_ds

        # pair_offsets: [num_pairs+1]
        offsets = cts_ds.pair_offsets.to(dtype=torch.long, device="cpu")
        if offsets.numel() < 2:
            raise RuntimeError("[DynamicPairDataset] invalid pair_offsets.")

        # counts[i] = offsets[i+1] - offsets[i]
        counts = offsets[1:] - offsets[:-1]  # [num_pairs]

        if drop_empty_pairs:
            # 只保留 count>0 的 pair_id（升序）
            valid = torch.nonzero(counts > 0, as_tuple=False).view(-1)
            self.pair_ids: List[int] = valid.tolist()
        else:
            self.pair_ids = list(range(int(cts_ds.num_pairs)))

        # 可选：日志，确认过滤是否符合预期
        # print(f"[DynamicPairDataset] drop_empty_pairs={drop_empty_pairs} "
        #       f"kept={len(self.pair_ids)}/{int(cts_ds.num_pairs)} pairs")

    def __len__(self) -> int:
        return len(self.pair_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair_id = int(self.pair_ids[idx])
        return {"pair_id": pair_id}
