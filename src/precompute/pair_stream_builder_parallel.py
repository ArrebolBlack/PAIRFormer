from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import get_context
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple
import time

import torch

from src.data.selected_pair_cache import SelectedPairCacheWriter
from src.data.stream_pair_dataset import PairRecord
from src.models.extractors import get_embedding_and_logit
from src.precompute.online_cts_generator import iter_cts_candidates
from src.selectors.stream_selector import StreamSelector, StreamSelectorInput


@dataclass
class PairChunkResult:
    pair_ids: List[int]
    labels: List[float]
    xs_list: List[torch.Tensor]
    esa_list: List[torch.Tensor]
    pos_list: List[torch.Tensor]


@dataclass
class PairStreamParallelBuildConfig:
    cheap_batch_size: int = 4096
    device: str = "cuda"
    esa_min_score: float = 6.0
    num_workers: int = 8
    task_pairs: int = 16
    mp_start_method: str = "spawn"


def _chunk_records(records: Iterable[PairRecord], chunk_size: int) -> Iterator[List[PairRecord]]:
    buf: List[PairRecord] = []
    for rec in records:
        buf.append(rec)
        if len(buf) >= int(chunk_size):
            yield buf
            buf = []
    if buf:
        yield buf


def _process_pair_chunk(task: Tuple[Sequence[PairRecord], object, float]) -> PairChunkResult:
    records, data_cfg, esa_min_score = task
    pair_ids: List[int] = []
    labels: List[float] = []
    xs_list: List[torch.Tensor] = []
    esa_list: List[torch.Tensor] = []
    pos_list: List[torch.Tensor] = []

    for record in records:
        xs: List[torch.Tensor] = []
        esa_vals: List[float] = []
        pos_vals: List[float] = []

        for cand in iter_cts_candidates(record, data_cfg, esa_min_score=float(esa_min_score)):
            xs.append(cand.x_u8)
            esa_vals.append(float(cand.esa_score))
            pos_vals.append(float(cand.pos))

        if xs:
            xs_tensor = torch.stack(xs, dim=0).to(torch.uint8)
            esa_tensor = torch.tensor(esa_vals, dtype=torch.float32)
            pos_tensor = torch.tensor(pos_vals, dtype=torch.float32)
        else:
            xs_tensor = torch.empty((0, 10, 50), dtype=torch.uint8)
            esa_tensor = torch.empty((0,), dtype=torch.float32)
            pos_tensor = torch.empty((0,), dtype=torch.float32)

        pair_ids.append(int(record.pair_id))
        labels.append(float(record.label))
        xs_list.append(xs_tensor)
        esa_list.append(esa_tensor)
        pos_list.append(pos_tensor)

    return PairChunkResult(
        pair_ids=pair_ids,
        labels=labels,
        xs_list=xs_list,
        esa_list=esa_list,
        pos_list=pos_list,
    )


class PairStreamBuilderParallel:
    """
    Parallel CPU producer + single-process GPU consumer for scalable selected_raw build.

    CPU workers:
      - generate CTS candidates pair-by-pair
      - return pair-local packed tensors in small chunks

    Main process:
      - runs cheap model on GPU
      - applies selector
      - writes selected_raw cache
    """

    def __init__(
        self,
        *,
        data_cfg,
        cheap_model: torch.nn.Module,
        selector_factory,
        cache_writer: SelectedPairCacheWriter,
        cfg: PairStreamParallelBuildConfig,
    ):
        self.data_cfg = data_cfg
        self.cheap_model = cheap_model
        self.selector_factory = selector_factory
        self.cache_writer = cache_writer
        self.cfg = cfg
        self.device = torch.device(cfg.device)

    def build(self, records: Iterable[PairRecord]) -> None:
        self.cheap_model.to(self.device).eval()
        t0 = time.time()
        num_pairs = 0
        total_candidates = 0
        total_selected = 0

        ctx = get_context(str(self.cfg.mp_start_method))
        task_iter = (
            (chunk, self.data_cfg, float(self.cfg.esa_min_score))
            for chunk in _chunk_records(records, int(self.cfg.task_pairs))
        )

        with ctx.Pool(processes=int(self.cfg.num_workers)) as pool:
            for chunk_result in pool.imap_unordered(_process_pair_chunk, task_iter, chunksize=1):
                for pair_id, label, xs_tensor, esa_tensor, pos_tensor in zip(
                    chunk_result.pair_ids,
                    chunk_result.labels,
                    chunk_result.xs_list,
                    chunk_result.esa_list,
                    chunk_result.pos_list,
                ):
                    pair_t0 = time.time()
                    selected = self._process_one_pair(
                        pair_id=int(pair_id),
                        label=float(label),
                        xs_tensor=xs_tensor,
                        esa_tensor=esa_tensor,
                        pos_tensor=pos_tensor,
                    )
                    num_pairs += 1
                    total_candidates += int(xs_tensor.shape[0])
                    total_selected += int(selected.sel_len)
                    if num_pairs <= 5 or num_pairs % 100 == 0:
                        print(
                            f"[PairStreamBuilderParallel] pair_id={pair_id} "
                            f"candidates={int(xs_tensor.shape[0])} selected={int(selected.sel_len)} "
                            f"time={time.time() - pair_t0:.2f}s"
                        )

        self.cache_writer.set_ready()
        total_dt = time.time() - t0
        avg_candidates = total_candidates / max(1, num_pairs)
        avg_selected = total_selected / max(1, num_pairs)
        print(
            "[PairStreamBuilderParallel] DONE "
            f"pairs={num_pairs} total_candidates={total_candidates} total_selected={total_selected} "
            f"avg_candidates={avg_candidates:.2f} avg_selected={avg_selected:.2f} "
            f"elapsed={total_dt:.2f}s"
        )

    def _process_one_pair(
        self,
        *,
        pair_id: int,
        label: float,
        xs_tensor: torch.Tensor,
        esa_tensor: torch.Tensor,
        pos_tensor: torch.Tensor,
    ):
        selector: StreamSelector = self.selector_factory()
        selector.reset(pair_id=int(pair_id))

        n = int(xs_tensor.shape[0])
        start = 0
        while start < n:
            end = min(n, start + int(self.cfg.cheap_batch_size))
            x = xs_tensor[start:end].to(self.device, non_blocking=True).float()
            esa = esa_tensor[start:end].to(self.device, non_blocking=True)
            pos = pos_tensor[start:end].to(self.device, non_blocking=True)
            feat, logit = get_embedding_and_logit(
                self.cheap_model,
                x,
                esa_scores=esa,
                pos=pos,
            )
            feat = feat.detach().cpu()
            logit = logit.detach().cpu()

            for i in range(end - start):
                selector.add(
                    StreamSelectorInput(
                        x=xs_tensor[start + i],
                        esa=float(esa_tensor[start + i].item()),
                        pos=float(pos_tensor[start + i].item()),
                        cheap_logit=float(logit[i].item()),
                        cheap_emb=feat[i],
                    )
                )
            start = end

        selected = selector.finalize()
        self.cache_writer.write_pair(
            int(pair_id),
            X=selected.X,
            esa=selected.esa,
            pos=selected.pos,
            label=float(label),
            sel_len=int(selected.sel_len),
            cheap_logit=selected.cheap_logit,
            cheap_emb=selected.cheap_emb,
        )
        return selected
