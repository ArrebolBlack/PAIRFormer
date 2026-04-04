from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch

from src.config.data_config import DataConfig
from src.data.encoding import encode_RNA, extended_seed_alignment, extended_seed_alignment_2, reverse
from src.data.stream_pair_dataset import PairRecord


@dataclass
class CTSCandidate:
    x_u8: torch.Tensor
    esa_score: float
    pos: float


def iter_cts_candidates(
    record: PairRecord,
    data_cfg: DataConfig,
    *,
    esa_min_score: float = 6.0,
) -> Iterator[CTSCandidate]:
    mirna_seq = record.mirna_seq.upper().replace("T", "U")
    mrna_seq = record.mrna_seq.upper().replace("T", "U")
    mrna_rev_seq = reverse(mrna_seq)
    L = len(mrna_seq)

    alignment_mode = getattr(data_cfg, "alignment", "extended_seed_alignment")

    max_pos = max(0, len(mrna_rev_seq) - 40 + 1)
    for pos_idx in range(max_pos):
        win = mrna_rev_seq[pos_idx : pos_idx + 40]

        if alignment_mode == "extended_seed_alignment_2":
            mi_esa, cts_rev_esa, esa_score = extended_seed_alignment_2(mirna_seq, win)
        else:
            mi_esa, cts_rev_esa, esa_score = extended_seed_alignment(mirna_seq, win)

        if esa_score < float(esa_min_score):
            continue
        if mi_esa is None:
            continue

        x_np = encode_RNA(mirna_seq, mi_esa, win, cts_rev_esa, data_cfg.with_esa)
        if x_np is None:
            continue

        if L > 1:
            start = L - 40 - pos_idx
            end = L - 1 - pos_idx
            center = (start + end) / 2.0
            pos_val = center / (L - 1)
        else:
            pos_val = 0.5

        yield CTSCandidate(
            x_u8=torch.from_numpy(x_np).to(torch.uint8),
            esa_score=float(esa_score),
            pos=float(pos_val),
        )
