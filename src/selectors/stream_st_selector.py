from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from src.selectors.selector_module import SelectorModule
from src.selectors.stream_selector import (
    StreamSelector,
    StreamSelectorConfig,
    StreamSelectorInput,
    StreamSelectorOutput,
    empty_selector_output,
)


@dataclass
class StreamSTSelectorConfig(StreamSelectorConfig):
    name: str = "stselector"
    kmax: int = 64
    keep_cheap_logit: bool = True
    keep_cheap_emb: bool = True
    keep_raw_window: bool = True
    mode: str = "eval"
    epoch: int = 0


class StreamSTSelector(StreamSelector):
    """
    Pair-buffered streamed selector.

    This class is still scalable because it only buffers candidates for the current pair.
    It is intended for `k1_ratio < 1`, including the common `k1_ratio=0.5` STSelector setup.

    The implementation delegates the final selection logic to the existing SelectorModule,
    while removing the dependency on full-dataset cheap cache and CTS-global uid space.
    """

    def __init__(self, cfg: StreamSTSelectorConfig, selector_module: SelectorModule):
        self.cfg = cfg
        self.selector_module = selector_module
        self._pair_id: Optional[int] = None
        self._xs: List[torch.Tensor] = []
        self._esa: List[float] = []
        self._pos: List[float] = []
        self._cheap_logit: List[float] = []
        self._cheap_emb: List[torch.Tensor] = []

    def reset(self, *, pair_id: int) -> None:
        self._pair_id = int(pair_id)
        self._xs.clear()
        self._esa.clear()
        self._pos.clear()
        self._cheap_logit.clear()
        self._cheap_emb.clear()

    def add(self, item: StreamSelectorInput) -> None:
        if item.cheap_emb is None:
            raise ValueError(
                "StreamSTSelector requires cheap_emb for STSelector-style diversity selection."
            )
        self._xs.append(item.x.detach().cpu())
        self._esa.append(float(item.esa))
        self._pos.append(float(item.pos))
        self._cheap_logit.append(float(item.cheap_logit))
        self._cheap_emb.append(item.cheap_emb.detach().cpu())

    def finalize(self) -> StreamSelectorOutput:
        if not self._xs:
            return empty_selector_output(self.cfg.kmax)

        n = len(self._xs)
        # Use pair-local candidate indices; no CTS-global uid is required in the scalable path.
        uids = torch.arange(n, dtype=torch.long)
        pos = torch.tensor(self._pos, dtype=torch.float32)
        cheap_logit = torch.tensor(self._cheap_logit, dtype=torch.float32)
        cheap_emb = torch.stack(self._cheap_emb, dim=0)

        sel_local_uids, sel_len = self.selector_module.forward(
            uids=uids,
            pos=pos,
            cheap_logit=cheap_logit,
            cheap_emb=cheap_emb,
            mode=str(self.cfg.mode),
            epoch=int(self.cfg.epoch),
            pair_id=int(self._pair_id if self._pair_id is not None else 0),
        )

        sel_len_int = int(sel_len.item()) if torch.is_tensor(sel_len) else int(sel_len)
        if sel_len_int <= 0:
            return empty_selector_output(self.cfg.kmax)

        keep = sel_local_uids[:sel_len_int].to(dtype=torch.long)

        X_all = torch.stack(self._xs, dim=0)
        esa_all = torch.tensor(self._esa, dtype=torch.float32)
        pos_all = torch.tensor(self._pos, dtype=torch.float32)
        logit_all = torch.tensor(self._cheap_logit, dtype=torch.float32)

        X = X_all.index_select(0, keep)
        esa = esa_all.index_select(0, keep)
        pos = pos_all.index_select(0, keep)

        cheap_logit_out = None
        if self.cfg.keep_cheap_logit:
            cheap_logit_out = logit_all.index_select(0, keep)

        cheap_emb_out = None
        if self.cfg.keep_cheap_emb:
            cheap_emb_out = cheap_emb.index_select(0, keep)

        return StreamSelectorOutput(
            X=X,
            esa=esa,
            pos=pos,
            sel_len=sel_len_int,
            cheap_logit=cheap_logit_out,
            cheap_emb=cheap_emb_out,
        )
