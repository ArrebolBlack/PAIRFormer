from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional

from src.config.data_config import DataConfig


@dataclass
class PairRecord:
    pair_id: int
    mirna_seq: str
    mrna_seq: str
    label: float
    raw_tokens: list[str]


class StreamPairDataset:
    """
    Raw pair iterator for the scalable pipeline.

    This dataset iterates pair rows directly from the source file and never materializes
    CTS-global caches.
    """

    def __init__(self, data_cfg: DataConfig, split: str, *, max_pairs: Optional[int] = None):
        self.data_cfg = data_cfg
        self.split = str(split)
        self.path = str(data_cfg.get_path(self.split))
        self.split_column = getattr(data_cfg, "split_column", None)
        self.split_map = getattr(data_cfg, "split_map", None)
        self.max_pairs = (None if max_pairs is None else int(max_pairs))

    @staticmethod
    def _resolve_columns(header: list[str]) -> dict[str, int]:
        norm = {h.strip().lower(): i for i, h in enumerate(header)}

        def pick(*names: str, default: Optional[int] = None) -> int:
            for n in names:
                if n in norm:
                    return norm[n]
            if default is None:
                raise KeyError(f"Cannot resolve required column from header. tried={names} header={header}")
            return int(default)

        return {
            "mrna_seq": pick("mrna_seq", "mRNA_seq", default=3),
            "mirna_seq": pick("mirna_seq", "miRNA_seq", default=1),
            "label": pick("label", default=4),
        }

    def iter_records(self) -> Iterator[PairRecord]:
        target_tag: Optional[str] = None
        if self.split_column is not None and self.split_map:
            target_tag = self.split_map.get(self.split, None)

        with open(self.path, "r") as f:
            header_line = next(f, None)
            if header_line is None:
                return
            header = header_line.rstrip("\n").split("\t")
            col = self._resolve_columns(header)
            pair_id = 0
            for raw_line in f:
                if self.max_pairs is not None and pair_id >= self.max_pairs:
                    break
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                toks = line.split("\t")
                if len(toks) < 5:
                    continue

                if target_tag is not None and self.split_column is not None:
                    if len(toks) <= self.split_column or toks[self.split_column] != target_tag:
                        continue

                try:
                    label = float(toks[col["label"]])
                except (TypeError, ValueError):
                    label = 0.0

                yield PairRecord(
                    pair_id=pair_id,
                    mirna_seq=toks[col["mirna_seq"]].upper().replace("T", "U"),
                    mrna_seq=toks[col["mrna_seq"]].upper().replace("T", "U"),
                    label=label,
                    raw_tokens=toks,
                )
                pair_id += 1

    def count_records(self) -> int:
        n = 0
        for _ in self.iter_records():
            n += 1
        return n
