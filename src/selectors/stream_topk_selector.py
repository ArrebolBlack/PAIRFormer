from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from src.selectors.stream_selector import (
    StreamSelector,
    StreamSelectorConfig,
    StreamSelectorInput,
    StreamSelectorOutput,
    empty_selector_output,
)


@dataclass
class StreamTopKSelectorConfig(StreamSelectorConfig):
    name: str = "topk"
    kmax: int = 64
    keep_cheap_logit: bool = True
    keep_cheap_emb: bool = False
    keep_raw_window: bool = True


@dataclass(order=True)
class _HeapItem:
    score: float
    order: int
    x: torch.Tensor = field(compare=False)
    esa: float = field(compare=False)
    pos: float = field(compare=False)
    cheap_logit: float = field(compare=False)
    cheap_emb: Optional[torch.Tensor] = field(compare=False, default=None)


class StreamTopKSelector(StreamSelector):
    """
    True streaming top-K selector.

    This implementation never materializes the full current pair.
    It is the preferred mode when `k1_ratio=1`.
    """

    def __init__(self, cfg: StreamTopKSelectorConfig):
        self.cfg = cfg
        self._heap: List[_HeapItem] = []
        self._counter = 0
        self._pair_id: Optional[int] = None

    def reset(self, *, pair_id: int) -> None:
        self._pair_id = int(pair_id)
        self._heap.clear()
        self._counter = 0

    def add(self, item: StreamSelectorInput) -> None:
        score = float(item.cheap_logit)
        heap_item = _HeapItem(
            score=score,
            order=self._counter,
            x=item.x.detach().cpu(),
            esa=float(item.esa),
            pos=float(item.pos),
            cheap_logit=float(item.cheap_logit),
            cheap_emb=(item.cheap_emb.detach().cpu() if item.cheap_emb is not None else None),
        )
        self._counter += 1

        if len(self._heap) < int(self.cfg.kmax):
            heapq.heappush(self._heap, heap_item)
            return

        if heap_item.score > self._heap[0].score:
            heapq.heapreplace(self._heap, heap_item)

    def finalize(self) -> StreamSelectorOutput:
        if not self._heap:
            return empty_selector_output(self.cfg.kmax)

        items = sorted(self._heap, key=lambda it: it.score, reverse=True)
        X = torch.stack([it.x for it in items], dim=0)
        esa = torch.tensor([it.esa for it in items], dtype=torch.float32)
        pos = torch.tensor([it.pos for it in items], dtype=torch.float32)

        cheap_logit = None
        if self.cfg.keep_cheap_logit:
            cheap_logit = torch.tensor([it.cheap_logit for it in items], dtype=torch.float32)

        cheap_emb = None
        if self.cfg.keep_cheap_emb and items[0].cheap_emb is not None:
            cheap_emb = torch.stack([it.cheap_emb for it in items], dim=0)

        return StreamSelectorOutput(
            X=X,
            esa=esa,
            pos=pos,
            sel_len=len(items),
            cheap_logit=cheap_logit,
            cheap_emb=cheap_emb,
        )
