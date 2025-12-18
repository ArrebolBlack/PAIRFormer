# src/utils/efficiency.py
from dataclasses import dataclass, asdict
import os

try:
    import psutil
except Exception:
    psutil = None

import torch


def _bytes_to_gb(x: float) -> float:
    return float(x) / (1024 ** 3)


@dataclass
class EffStats:
    epoch_wall_s: float = 0.0
    data_fetch_s: float = 0.0     # CPU: time waiting for next(loader)
    h2d_s: float = 0.0            # GPU: measured by CUDA events (H2D copies)
    compute_s: float = 0.0        # GPU: measured by CUDA events (fwd/bwd/step)
    peak_vram_gb: float = 0.0
    peak_cpu_rss_gb: float = 0.0

    @property
    def data_overhead_pct(self) -> float:
        if self.epoch_wall_s <= 1e-9:
            return 0.0
        return 100.0 * (self.data_fetch_s + self.h2d_s) / self.epoch_wall_s


class EffMeter:
    """
    EffMeter: 不在每个 batch 强制 cuda synchronize，避免显著拖慢训练速度。
    - data_fetch_s 用 CPU 计时（next(it) 等待）
    - h2d / compute 用 CUDA event 计时（epoch 末尾一次 synchronize 后汇总）
    """

    def __init__(self, device: torch.device, enabled: bool = True):
        self.device = device
        self.enabled = bool(enabled)
        self.stats = EffStats()
        self._proc = psutil.Process(os.getpid()) if psutil is not None else None

        self._h2d_events = []      # list[(start_event, end_event)]
        self._compute_events = []  # list[(start_event, end_event)]

    def reset_epoch(self):
        self.stats = EffStats()
        self._h2d_events.clear()
        self._compute_events.clear()

        if self.enabled and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def record_data_fetch(self, dt_s: float):
        if self.enabled:
            self.stats.data_fetch_s += float(dt_s)

    def record_h2d_events(self, start_ev: torch.cuda.Event, end_ev: torch.cuda.Event):
        if self.enabled and self.device.type == "cuda":
            self._h2d_events.append((start_ev, end_ev))

    def record_compute_events(self, start_ev: torch.cuda.Event, end_ev: torch.cuda.Event):
        if self.enabled and self.device.type == "cuda":
            self._compute_events.append((start_ev, end_ev))

    def update_peak_cpu(self):
        if not self.enabled:
            return
        if self._proc is None:
            return
        rss_gb = _bytes_to_gb(self._proc.memory_info().rss)
        if rss_gb > self.stats.peak_cpu_rss_gb:
            self.stats.peak_cpu_rss_gb = float(rss_gb)

    def finalize_epoch(self, *, epoch_wall_s: float, sync: bool = True):
        if not self.enabled:
            return

        if self.device.type == "cuda":
            if sync:
                torch.cuda.synchronize(self.device)

            # 汇总 event time（ms -> s）
            h2d_ms = 0.0
            for s, e in self._h2d_events:
                h2d_ms += s.elapsed_time(e)
            self.stats.h2d_s = float(h2d_ms / 1000.0)

            comp_ms = 0.0
            for s, e in self._compute_events:
                comp_ms += s.elapsed_time(e)
            self.stats.compute_s = float(comp_ms / 1000.0)

            peak = torch.cuda.max_memory_allocated(self.device)
            self.stats.peak_vram_gb = _bytes_to_gb(peak)

        self.stats.epoch_wall_s = float(epoch_wall_s)

    def to_dict(self):
        d = asdict(self.stats)
        d["data_overhead_pct"] = self.stats.data_overhead_pct
        return d
