from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, runtime_checkable

import torch


@dataclass
class StreamSelectorInput:
    x: torch.Tensor
    esa: float
    pos: float
    cheap_logit: float
    cheap_emb: Optional[torch.Tensor] = None


@dataclass
class StreamSelectorOutput:
    X: torch.Tensor  # [K,C,L] uint8/float
    esa: torch.Tensor  # [K]
    pos: torch.Tensor  # [K]
    sel_len: int
    cheap_logit: Optional[torch.Tensor] = None
    cheap_emb: Optional[torch.Tensor] = None


@runtime_checkable
class StreamSelector(Protocol):
    """
    Pair-local selector protocol for the scalable pipeline.

    Two implementation styles are intentionally supported:

    1. True streaming selector:
       - consumes CTS candidates one by one
       - only maintains a small state, e.g. top-K heap

    2. Pair-buffered selector:
       - collects all candidates for the current pair
       - runs a richer selector such as STSelector at finalize()

    In both cases, the selector scope is only the current pair.
    """

    def reset(self, *, pair_id: int) -> None: ...

    def add(self, item: StreamSelectorInput) -> None: ...

    def finalize(self) -> StreamSelectorOutput: ...


@dataclass
class StreamSelectorConfig:
    name: str = "topk"
    kmax: int = 64
    keep_cheap_logit: bool = True
    keep_cheap_emb: bool = False
    keep_raw_window: bool = True


def stack_or_none(xs: List[torch.Tensor]) -> Optional[torch.Tensor]:
    if not xs:
        return None
    return torch.stack(xs, dim=0)


def empty_selector_output(kmax: int) -> StreamSelectorOutput:
    return StreamSelectorOutput(
        X=torch.empty((0, 10, 50), dtype=torch.uint8),
        esa=torch.empty((0,), dtype=torch.float32),
        pos=torch.empty((0,), dtype=torch.float32),
        sel_len=min(0, int(kmax)),
        cheap_logit=None,
        cheap_emb=None,
    )
