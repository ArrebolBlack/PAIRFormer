from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from src.data.selected_pair_cache import SelectedPairCacheWriter
from src.data.stream_pair_dataset import PairRecord
from src.models.extractors import get_embedding_and_logit
from src.precompute.online_cts_generator import iter_cts_candidates
from src.selectors.stream_selector import StreamSelector, StreamSelectorInput


@dataclass
class PairStreamBuildConfig:
    cheap_batch_size: int = 4096
    device: str = "cuda"
    esa_min_score: float = 6.0


class PairStreamBuilder:
    """
    Skeleton for the scalable precompute path.

    Intended final behavior:
    - iterate raw pairs
    - generate CTS windows online
    - run cheap model online
    - feed selector incrementally
    - write selected-pair cache

    This class is deliberately additive and does not interact with the old CTS-global cache path.
    """

    def __init__(
        self,
        *,
        data_cfg,
        cheap_model: torch.nn.Module,
        selector: StreamSelector,
        cache_writer: SelectedPairCacheWriter,
        cfg: PairStreamBuildConfig,
    ):
        self.data_cfg = data_cfg
        self.cheap_model = cheap_model
        self.selector = selector
        self.cache_writer = cache_writer
        self.cfg = cfg
        self.device = torch.device(cfg.device)

    def build(self, records: Iterable[PairRecord]) -> None:
        self.cheap_model.to(self.device).eval()
        t0 = time.time()
        num_pairs = 0
        total_candidates = 0
        total_selected = 0

        for record in records:
            pair_t0 = time.time()
            self.selector.reset(pair_id=int(record.pair_id))

            xs_batch: list[torch.Tensor] = []
            esa_batch: list[float] = []
            pos_batch: list[float] = []
            pair_candidates = 0

            for cand in iter_cts_candidates(
                record,
                self.data_cfg,
                esa_min_score=float(self.cfg.esa_min_score),
            ):
                pair_candidates += 1
                xs_batch.append(cand.x_u8)
                esa_batch.append(float(cand.esa_score))
                pos_batch.append(float(cand.pos))

                if len(xs_batch) >= int(self.cfg.cheap_batch_size):
                    self._flush_cheap_batch(xs_batch, esa_batch, pos_batch)
                    xs_batch.clear()
                    esa_batch.clear()
                    pos_batch.clear()

            if xs_batch:
                self._flush_cheap_batch(xs_batch, esa_batch, pos_batch)

            selected = self.selector.finalize()
            self.cache_writer.write_pair(
                int(record.pair_id),
                X=selected.X,
                esa=selected.esa,
                pos=selected.pos,
                label=float(record.label),
                sel_len=int(selected.sel_len),
                cheap_logit=selected.cheap_logit,
                cheap_emb=selected.cheap_emb,
            )
            num_pairs += 1
            total_candidates += pair_candidates
            total_selected += int(selected.sel_len)

            if num_pairs <= 5 or num_pairs % 100 == 0:
                dt_pair = time.time() - pair_t0
                print(
                    f"[PairStreamBuilder] pair_id={record.pair_id} "
                    f"candidates={pair_candidates} selected={int(selected.sel_len)} "
                    f"time={dt_pair:.2f}s"
                )

        self.cache_writer.set_ready()
        total_dt = time.time() - t0
        avg_candidates = total_candidates / max(1, num_pairs)
        avg_selected = total_selected / max(1, num_pairs)
        print(
            "[PairStreamBuilder] DONE "
            f"pairs={num_pairs} total_candidates={total_candidates} total_selected={total_selected} "
            f"avg_candidates={avg_candidates:.2f} avg_selected={avg_selected:.2f} "
            f"elapsed={total_dt:.2f}s"
        )

    def _add_candidate(
        self,
        *,
        x: torch.Tensor,
        esa: float,
        pos: float,
        cheap_logit: float,
        cheap_emb: Optional[torch.Tensor] = None,
    ) -> None:
        self.selector.add(
            StreamSelectorInput(
                x=x,
                esa=float(esa),
                pos=float(pos),
                cheap_logit=float(cheap_logit),
                cheap_emb=cheap_emb,
            )
        )

    @torch.no_grad()
    def _flush_cheap_batch(
        self,
        xs_batch: list[torch.Tensor],
        esa_batch: list[float],
        pos_batch: list[float],
    ) -> None:
        if not xs_batch:
            return

        x = torch.stack(xs_batch, dim=0).to(self.device, non_blocking=True).float()
        esa = torch.tensor(esa_batch, dtype=torch.float32, device=self.device)
        pos = torch.tensor(pos_batch, dtype=torch.float32, device=self.device)

        feat, logit = get_embedding_and_logit(
            self.cheap_model,
            x,
            esa_scores=esa,
            pos=pos,
        )
        feat = feat.detach().cpu()
        logit = logit.detach().cpu()

        for i, x_u8 in enumerate(xs_batch):
            self._add_candidate(
                x=x_u8,
                esa=float(esa_batch[i]),
                pos=float(pos_batch[i]),
                cheap_logit=float(logit[i].item()),
                cheap_emb=feat[i],
            )
