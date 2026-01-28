# src/launch/inference_targetnet_like_online.py
from __future__ import annotations

import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

import time
import dataclasses
import json
import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.config.data_config import DataConfig
from src.data.builder import get_or_build_blocks
from src.data.dataset import ChunkedCTSDataset
from src.models.registry import build_model
from src.models.extractors import get_embedding_and_logit

from src.launch.bench_utils import (
    BenchConfig,
    append_records_to_csv,
    compute_n_full_stats,
    ensure_dir,
    get_gpu_name,
    run_benchmark_once,
    select_pair_ids_subset,
    summarize_repeats_to_records,
)

from src.data.cache_identity import dataset_identity


# -----------------------------------------------------------------------------
# Small helpers (mirror BR-MIL/Naive style)
# -----------------------------------------------------------------------------
def _resolve_path(p: Optional[str], orig_cwd: Path) -> Optional[Path]:
    if p is None:
        return None
    pp = Path(os.path.expandvars(os.path.expanduser(str(p))))
    if not pp.is_absolute():
        pp = orig_cwd / pp
    return pp


def _pick_device(cfg: DictConfig) -> torch.device:
    dev_req = str(cfg.get("device", "cuda"))
    if torch.cuda.is_available() and dev_req != "cpu":
        dev_req = "cuda:0" if dev_req == "cuda" else dev_req
        return torch.device(dev_req)
    return torch.device("cpu")


def _strip_prefix_state_dict(sd: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for k, v in sd.items():
        kk = k
        for pref in ("model.", "module.", "net."):
            if kk.startswith(pref):
                kk = kk[len(pref):]
        cleaned[kk] = v
    return cleaned


def _record_to_dict(rec: Any) -> Dict[str, Any]:
    if rec is None:
        return {}
    if isinstance(rec, Mapping):
        return dict(rec)
    if dataclasses.is_dataclass(rec):
        return dataclasses.asdict(rec)
    if hasattr(rec, "to_dict"):
        try:
            d = rec.to_dict()
            if isinstance(d, Mapping):
                return dict(d)
        except Exception:
            pass
    if hasattr(rec, "__dict__"):
        return dict(vars(rec))
    return {"record": str(rec)}


def _dataset_path_hash8(data_cfg: DataConfig, split_idx: str) -> str:
    # MUST match ChunkedCTSDataset hash rule: md5(f"{data_file_path}|{alignment}")[:8]
    data_file_path = str(data_cfg.get_path(str(split_idx)))
    alignment = getattr(data_cfg, "alignment", "extended_seed_alignment")
    hash_key = f"{data_file_path}|{alignment}"
    return hashlib.md5(hash_key.encode("utf-8")).hexdigest()[:8]


def _resolve_required_ckpt_path(p: Optional[str], *, run_dir: Path, orig_cwd: Path) -> Path:
    if p is None:
        raise KeyError(
            "[inference_targetnet_like_online] Missing run.checkpoint. "
            "If you want to load instance weights from EM checkpoint, pass run.checkpoint explicitly. "
            "Otherwise set cfg.instance_ckpt_path for a standalone instance checkpoint."
        )
    raw = Path(os.path.expandvars(os.path.expanduser(str(p))))
    if raw.is_absolute():
        return raw
    cand1 = run_dir / raw
    if cand1.exists():
        return cand1
    return orig_cwd / raw


def _load_single_model_ckpt(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and any(isinstance(v, torch.Tensor) for v in ckpt.values()):
        sd = ckpt
    else:
        sd = ckpt
    if not isinstance(sd, dict):
        raise RuntimeError(f"[inference_targetnet_like_online] ckpt is not a state_dict-like dict: {ckpt_path}")
    model.load_state_dict(_strip_prefix_state_dict(sd), strict=False)
    model.to(device).eval()


def _load_instance_from_em_ckpt(
    ckpt_path: Path,
    *,
    instance_model: torch.nn.Module,
    device: torch.device,
) -> None:
    """
    Load instance_model weights from EM checkpoint (same logic style as BR-MIL/Naive).
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"[inference_targetnet_like_online] checkpoint is not a dict: {ckpt_path}")

    cand_inst = None
    for k in ("instance_state_dict", "inst_state_dict", "instance_model_state_dict",
              "instance_model", "inst_model", "cts_model"):
        if k in ckpt and isinstance(ckpt[k], dict):
            cand_inst = ckpt[k]
            break

    if cand_inst is None:
        # fallback unified
        sd_all = None
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd_all = ckpt["state_dict"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            sd_all = ckpt["model_state_dict"]
        else:
            if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                sd_all = ckpt

        if sd_all is not None:
            sd_all = _strip_prefix_state_dict(sd_all)
            inst_keys = set(instance_model.state_dict().keys())
            inst_sd: Dict[str, Any] = {}
            for k, v in sd_all.items():
                if k in inst_keys:
                    inst_sd[k] = v
            if len(inst_sd) == 0:
                for k, v in sd_all.items():
                    if k.startswith(("instance_model.", "inst_model.", "cts_model.")):
                        kk = k.split(".", 1)[1]
                        if kk in inst_keys:
                            inst_sd[kk] = v
            if len(inst_sd) > 0:
                cand_inst = inst_sd

    if cand_inst is None:
        raise RuntimeError(
            "[inference_targetnet_like_online] Cannot locate instance state_dict in EM ckpt. "
            f"ckpt_path={ckpt_path}"
        )

    cand_inst = _strip_prefix_state_dict(cand_inst)
    miss_i, unexp_i = instance_model.load_state_dict(cand_inst, strict=False)
    if miss_i:
        print(f"[inference_targetnet_like_online] WARN inst missing keys: {len(miss_i)} (first10): {miss_i[:10]}")
    if unexp_i:
        print(f"[inference_targetnet_like_online] WARN inst unexpected keys: {len(unexp_i)} (first10): {unexp_i[:10]}")
    instance_model.to(device).eval()


# -----------------------------------------------------------------------------
# PairIndex adapter (reuse BR-MIL/Naive contract)
# -----------------------------------------------------------------------------
class PairIndexAdapter:
    def __init__(self, pair_index: Any):
        self.pair_index = pair_index
        self._num_pairs = self._infer_num_pairs(pair_index)

    @staticmethod
    def _infer_num_pairs(pi: Any) -> int:
        if hasattr(pi, "pair_offsets") and getattr(pi, "pair_offsets") is not None:
            off = getattr(pi, "pair_offsets")
            return int(len(off) - 1)
        if isinstance(pi, dict) and "pair_offsets" in pi:
            return int(len(pi["pair_offsets"]) - 1)
        for attr in ("num_pairs", "n_pairs", "N_pairs"):
            if hasattr(pi, attr):
                try:
                    return int(getattr(pi, attr))
                except Exception:
                    pass
        raise RuntimeError("[inference_targetnet_like_online] Cannot infer num_pairs from pair_index.")

    def __len__(self) -> int:
        return int(self._num_pairs)

    @property
    def pair_ids(self) -> List[int]:
        return list(range(int(self._num_pairs)))

    def get_pair_slice(self, pair_id: int) -> Tuple[int, int]:
        pi = self.pair_index
        if pair_id < 0 or pair_id >= self._num_pairs:
            raise IndexError(f"[inference_targetnet_like_online] pair_id out of range: {pair_id} / {self._num_pairs}")

        if hasattr(pi, "pair_offsets") and getattr(pi, "pair_offsets") is not None:
            off = getattr(pi, "pair_offsets")
            s = int(off[pair_id].item()) if torch.is_tensor(off) else int(off[pair_id])
            e = int(off[pair_id + 1].item()) if torch.is_tensor(off) else int(off[pair_id + 1])
            return s, e

        if isinstance(pi, dict) and "pair_offsets" in pi:
            off = pi["pair_offsets"]
            s = int(off[pair_id].item()) if torch.is_tensor(off) else int(off[pair_id])
            e = int(off[pair_id + 1].item()) if torch.is_tensor(off) else int(off[pair_id + 1])
            return s, e

        for m in ("get_pair_slice", "get_slice_for_pair"):
            if hasattr(pi, m):
                s, e = getattr(pi, m)(pair_id)
                return int(s), int(e)

        raise RuntimeError("Cannot resolve pair slice from PairIndex; adapt PairIndexAdapter.get_pair_slice().")

    def get_n_full(self, pair_id: int) -> int:
        s, e = self.get_pair_slice(pair_id)
        return int(max(0, e - s))


# -----------------------------------------------------------------------------
# Dataset that only yields pair_ids (exactly like BR-MIL/Naive)
# -----------------------------------------------------------------------------
class PairIdDataset(torch.utils.data.Dataset):
    def __init__(self, pair_ids: Sequence[int]):
        self.pair_ids = [int(x) for x in pair_ids]

    def __len__(self) -> int:
        return int(len(self.pair_ids))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"pair_id": int(self.pair_ids[int(idx)])}


def _collate_pair_ids(items: List[Mapping[str, Any]]) -> Dict[str, Any]:
    pair_ids: List[int] = []
    for it in items:
        if "pair_id" in it:
            pair_ids.append(int(it["pair_id"]))
        else:
            raise RuntimeError("Batch item missing pair_id; adapt _collate_pair_ids.")
    return {"pair_ids": pair_ids}


# -----------------------------------------------------------------------------
# TargetNet-like Online forward (compute-only)
# Expensive scan all n -> max pooling -> output
# -----------------------------------------------------------------------------
@dataclass
class TargetNetOnlineModules:
    instance_model: torch.nn.Module


class TargetNetOnlineForward:
    def __init__(
        self,
        *,
        modules: TargetNetOnlineModules,
        cts_ds: ChunkedCTSDataset,
        pair_index: PairIndexAdapter,
        device: torch.device,
        scan_chunk_size: int = 4096,
        amp: bool = True,
        amp_dtype: torch.dtype = torch.float16,
        profile_sections: bool = False,
        profile_sync_cuda: bool = False,
    ) -> None:
        self.modules = modules
        self.cts_ds = cts_ds
        self.pi = pair_index
        self.device = device

        self.scan_chunk_size = int(scan_chunk_size)
        self.amp = bool(amp and (device.type == "cuda"))
        self.amp_dtype = amp_dtype

        self.profile_sections = bool(profile_sections)
        self.profile_sync_cuda = bool(profile_sync_cuda)
        self._prof_sum: Dict[str, float] = {}
        self._prof_cnt: Dict[str, int] = {}

    def reset_profile(self) -> None:
        self._prof_sum.clear()
        self._prof_cnt.clear()

    def profile_summary_ms(self, reset: bool = False) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k in sorted(self._prof_sum.keys()):
            c = max(1, int(self._prof_cnt.get(k, 0)))
            out[k] = 1000.0 * float(self._prof_sum[k]) / c
        if reset:
            self.reset_profile()
        return out

    from contextlib import contextmanager
    @contextmanager
    def _profile(self, name: str, timer: Any = None):
        if timer is not None and hasattr(timer, "section"):
            try:
                with timer.section(name):
                    yield
                return
            except Exception:
                pass

        if not self.profile_sections:
            yield
            return

        if self.profile_sync_cuda and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.profile_sync_cuda and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            dt = time.perf_counter() - t0
            self._prof_sum[name] = self._prof_sum.get(name, 0.0) + float(dt)
            self._prof_cnt[name] = self._prof_cnt.get(name, 0) + 1

    @staticmethod
    def _normalize_gather_1d(x: Any) -> torch.Tensor:
        t = x if torch.is_tensor(x) else torch.as_tensor(x)
        if t.dim() == 2 and t.shape[-1] == 1:
            t = t.squeeze(-1)
        if t.dim() != 1:
            t = t.reshape(-1)
        return t

    def _batch_gather_all(self, uids_all: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns CPU tensors aligned with uids_all:
          X_flat:  [M,C,L] uint8 CPU
          esa_flat:[M] float32 CPU
          pos_flat:[M] float32 CPU
        """
        uids_all = uids_all.to(dtype=torch.long, device="cpu")

        if hasattr(self.cts_ds, "batch_gather_by_uid"):
            out = self.cts_ds.batch_gather_by_uid(  # type: ignore[attr-defined]
                uids_all,
                fields=("X", "pos", "esa_scores"),
            )
            if not isinstance(out, Mapping):
                raise RuntimeError("[inference_targetnet_like_online] batch_gather_by_uid must return a Mapping")

            X = out["X"]
            pos = out.get("pos", out.get("positions", None))
            esa = out.get("esa_scores", out.get("esa_score", out.get("esa", None)))

            if X is None or pos is None or esa is None:
                raise RuntimeError(
                    "[inference_targetnet_like_online] batch_gather_by_uid missing keys. "
                    f"got_keys={list(out.keys())}"
                )

            X = X.to(dtype=torch.uint8, device="cpu").contiguous()
            pos = self._normalize_gather_1d(pos).to(dtype=torch.float32, device="cpu").contiguous()
            esa = self._normalize_gather_1d(esa).to(dtype=torch.float32, device="cpu").contiguous()
            if X.dim() != 3:
                raise RuntimeError(f"[inference_targetnet_like_online] Expect X_flat [M,C,L], got {tuple(X.shape)}")
            return X, esa, pos

        # fallback slow path
        X_list: List[torch.Tensor] = []
        esa_list: List[float] = []
        pos_list: List[float] = []
        for uid in uids_all.tolist():
            sample = self.cts_ds[int(uid)]
            if isinstance(sample, (tuple, list)):
                x = sample[0]
                esa = float(sample[3]) if len(sample) > 3 else 0.0
                pos = float(sample[4]) if len(sample) > 4 else 0.0
            elif isinstance(sample, Mapping):
                x = sample.get("X", sample.get("inputs", sample.get("x")))
                esa = float(sample.get("esa_scores", sample.get("esa_score", sample.get("esa", 0.0))))
                pos = float(sample.get("pos", 0.0))
            else:
                raise RuntimeError(f"[inference_targetnet_like_online] unsupported sample type: {type(sample)}")

            x = x if torch.is_tensor(x) else torch.as_tensor(x)
            x = x.to(dtype=torch.uint8, device="cpu").contiguous()
            X_list.append(x)
            esa_list.append(float(esa))
            pos_list.append(float(pos))

        X_flat = torch.stack(X_list, dim=0) if X_list else torch.empty((0, 1, 1), dtype=torch.uint8)
        esa_flat = torch.tensor(esa_list, dtype=torch.float32, device="cpu")
        pos_flat = torch.tensor(pos_list, dtype=torch.float32, device="cpu")
        return X_flat, esa_flat, pos_flat

    def __call__(self, batch: Dict[str, Any], timer: Any = None, **_: Any) -> torch.Tensor:
        pair_ids = [int(x) for x in batch["pair_ids"]]
        B = len(pair_ids)
        if B == 0:
            return torch.zeros((), device=self.device)

        # A) build flat uids_all + offsets + pair_idx_all (CPU)
        with self._profile("A_pair_slices", timer):
            starts_uid, lens = [], []
            for pid in pair_ids:
                s_uid, e_uid = self.pi.get_pair_slice(pid)
                starts_uid.append(int(s_uid))
                lens.append(int(max(0, e_uid - s_uid)))

            total = int(sum(lens))
            if total <= 0:
                return torch.zeros((), device=self.device)

            uids_chunks: List[torch.Tensor] = []
            pair_idx_chunks: List[torch.Tensor] = []
            cur = 0
            offsets_n: List[Tuple[int, int]] = []
            for b in range(B):
                ln = lens[b]
                if ln <= 0:
                    offsets_n.append((cur, cur))
                    continue
                s_uid = starts_uid[b]
                u = torch.arange(s_uid, s_uid + ln, dtype=torch.long, device="cpu")
                uids_chunks.append(u)
                pair_idx_chunks.append(torch.full((ln,), b, dtype=torch.long, device="cpu"))
                offsets_n.append((cur, cur + ln))
                cur += ln

            uids_all = torch.cat(uids_chunks, dim=0) if uids_chunks else torch.empty((0,), dtype=torch.long)
            pair_idx_all = torch.cat(pair_idx_chunks, dim=0) if pair_idx_chunks else torch.empty((0,), dtype=torch.long)

            if uids_all.numel() == 0:
                return torch.zeros((), device=self.device)

        # B) one-pass gather X/pos/esa (CPU)
        with self._profile("B_gather_all_cpu", timer):
            X_flat, esa_flat, pos_flat = self._batch_gather_all(uids_all)
            M = int(X_flat.shape[0])
            if M == 0:
                return torch.zeros((), device=self.device)

        # C) expensive scan all (chunked) + incremental max pooling (GPU)
        with self._profile("C_expensive_scan_all", timer):
            pooled = torch.full((B,), -1e9, device=self.device, dtype=torch.float32)

            in_dtype = self.amp_dtype if self.amp else torch.float32
            inst = self.modules.instance_model

            with torch.no_grad():
                for s in range(0, M, self.scan_chunk_size):
                    e = min(M, s + self.scan_chunk_size)

                    X = X_flat[s:e].to(self.device, non_blocking=False).to(dtype=in_dtype)
                    esa = esa_flat[s:e].to(self.device, non_blocking=False)
                    pos = pos_flat[s:e].to(self.device, non_blocking=False)
                    pidx = pair_idx_all[s:e].to(self.device, non_blocking=False)

                    # autocast to match other scripts
                    if self.amp and self.device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                            _, logit = get_embedding_and_logit(inst, X, esa_scores=esa, pos=pos)
                    else:
                        _, logit = get_embedding_and_logit(inst, X, esa_scores=esa, pos=pos)

                    if logit is None:
                        raise RuntimeError("[inference_targetnet_like_online] instance_model produced no logit.")

                    if logit.dim() == 2 and logit.shape[-1] == 1:
                        logit = logit.squeeze(-1)
                    if logit.dim() != 1:
                        logit = logit.reshape(-1)
                    logit = logit.to(dtype=torch.float32)

                    # pooled[b] = max(pooled[b], max_{i in chunk, pair_idx=i} logit[i])
                    pooled.scatter_reduce_(0, pidx, logit, reduce="amax", include_self=True)

            # prevent DCE
            _ = pooled.sum() * 0.0 + pooled.sum()

        # return pair-level logits tensor [B]
        return pooled


# -----------------------------------------------------------------------------
# Main (aligned to BR-MIL/Naive)
# -----------------------------------------------------------------------------
@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = _pick_device(cfg)
    orig_cwd = Path(get_original_cwd())
    run_dir = Path.cwd()

    run_cfg = cfg.run if ("run" in cfg and cfg.run is not None) else {}
    split = str(run_cfg.get("bench_split", "val"))

    # baseline is K-independent, but keep K as anchor for plotting alignment
    K_anchor = int(run_cfg.get("kmax", 0))

    num_pairs_subset = int(run_cfg.get("bench_num_pairs_subset", 1024))
    batch_size = int(run_cfg.get("batch_size", 64))
    num_workers = int(run_cfg.get("num_workers", 0))

    warmup_iters = int(run_cfg.get("bench_warmup_iters", 20))
    measure_iters = int(run_cfg.get("bench_measure_iters", 100))
    repeats = int(run_cfg.get("bench_repeats", 3))
    amp = bool(run_cfg.get("bench_amp", (device.type == "cuda")))
    sync_cuda = bool(run_cfg.get("bench_sync_cuda", True))

    profile_sections = bool(run_cfg.get("bench_profile_sections", False))
    profile_sync_cuda = bool(run_cfg.get("bench_profile_sync_cuda", False))

    scan_chunk_size = int(run_cfg.get("bench_scan_chunk_size", 4096))

    bench_dir = ensure_dir(run_dir / "bench_fig3")

    bench_csv_cfg = run_cfg.get("bench_csv_path", None)
    if bench_csv_cfg is None or str(bench_csv_cfg).strip() in ("", "None", "null"):
        csv_path = bench_dir / "fig3_targetnet_like_online.csv"
    else:
        csv_path = Path(os.path.expandvars(os.path.expanduser(str(bench_csv_cfg))))
        if not csv_path.is_absolute():
            csv_path = run_dir / csv_path

    # cache root for window blocks
    default_cache = cfg.get("paths", {}).get("cache_root", "cache")
    cache_root_cfg = run_cfg.get("cache_path", default_cache)
    cache_root = _resolve_path(str(cache_root_cfg), orig_cwd)
    assert cache_root is not None
    cache_root.mkdir(parents=True, exist_ok=True)

    # seed
    seed = int(cfg.get("seed", 2020))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_cfg = DataConfig.from_omegaconf(cfg.data)

    # ensure blocks exist
    get_or_build_blocks(data_cfg, split, str(cache_root))
    cts_ds = ChunkedCTSDataset(str(cache_root), data_cfg, split)

    # pair_index
    path_hash8 = _dataset_path_hash8(data_cfg, split)
    print(f"[inference_targetnet_like_online] path_hash8={path_hash8}")

    pair_index_raw = None
    if hasattr(cts_ds, "pair_offsets") and getattr(cts_ds, "pair_offsets") is not None:
        pair_index_raw = cts_ds
    else:
        pair_index_path_cfg = run_cfg.get("pair_index_path", None)
        if pair_index_path_cfg is not None:
            pair_index_path = _resolve_path(str(pair_index_path_cfg), orig_cwd)
        else:
            pair_index_path = Path(str(cache_root)) / f"pair_index_{split}_{path_hash8}.pt"
        if pair_index_path is None or (not Path(pair_index_path).exists()):
            existing = sorted(Path(str(cache_root)).glob(f"pair_index_{split}_*.pt"))
            raise FileNotFoundError(
                "[inference_targetnet_like_online] pair_index file not found.\n"
                f"  expected: {pair_index_path}\n"
                f"  existing candidates: {[str(x) for x in existing]}\n"
                "Tip: pass run.pair_index_path=... to override."
            )
        pair_index_raw = torch.load(str(pair_index_path), map_location="cpu", weights_only=False)

    pi = PairIndexAdapter(pair_index_raw)

    # subset pair ids (stable across K runs)
    subset_path = Path(str(run_cfg.get("bench_pair_ids_path", str(bench_dir / f"pair_ids_{split}_{num_pairs_subset}.json"))))
    if not subset_path.is_absolute():
        subset_path = run_dir / subset_path

    # bind identity best-effort
    try:
        ident = dataset_identity(data_cfg, split, str(cache_root))  # type: ignore[call-arg]
        ident_key = json.dumps(ident, sort_keys=True, ensure_ascii=True)
        ident_hash = hashlib.sha1(ident_key.encode("utf-8")).hexdigest()[:10]
        print(f"[inference_targetnet_like_online] dataset_identity_hash={ident_hash}")
    except Exception:
        ident_hash = None

    pair_ids_subset = select_pair_ids_subset(
        pi,
        num_pairs_subset=num_pairs_subset,
        seed=seed,
        out_path=subset_path,
        allow_overwrite=False,
    )

    n_full_list = [pi.get_n_full(pid) for pid in pair_ids_subset]
    avg_n, med_n, p95_n = compute_n_full_stats(n_full_list)

    print(
        f"[inference_targetnet_like_online] split={split} K_anchor={K_anchor} | "
        f"subset_pairs={len(pair_ids_subset)} batch_size={batch_size} num_workers={num_workers} | "
        f"AMP={amp} device={device} GPU={get_gpu_name()} | "
        f"avg_n_full={avg_n:.2f} med_n_full={med_n:.2f} p95_n_full={p95_n:.2f} | "
        f"scan_chunk_size={scan_chunk_size}"
    )

    # dataloader over pair ids only
    ds_pairs = PairIdDataset(pair_ids_subset)
    loader = DataLoader(
        ds_pairs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
        drop_last=False,
        collate_fn=_collate_pair_ids,
    )

    # build instance model
    inst_cfg = cfg.get("instance_model", None) or cfg.get("cts_model", None)
    if inst_cfg is None:
        raise KeyError("[inference_targetnet_like_online] missing instance model config: cfg.instance_model / cfg.cts_model")
    inst_arch = str(inst_cfg.get("arch", inst_cfg.get("name")))
    instance_model = build_model(inst_arch, inst_cfg, data_cfg=data_cfg).to(device).eval()

    # load weights:
    # priority 1) run.checkpoint (EM ckpt) -> extract instance weights (same as BR-MIL/Naive)
    # fallback 2) cfg.instance_ckpt_path (standalone)
    if run_cfg.get("checkpoint", None) is not None:
        em_ckpt = _resolve_required_ckpt_path(run_cfg.get("checkpoint"), run_dir=run_dir, orig_cwd=orig_cwd)
        if not em_ckpt.exists():
            raise FileNotFoundError(f"[inference_targetnet_like_online] EM checkpoint not found: {em_ckpt}")
        _load_instance_from_em_ckpt(em_ckpt, instance_model=instance_model, device=device)
        print(f"[inference_targetnet_like_online] instance loaded from EM ckpt: {em_ckpt}")
    else:
        inst_ckpt_cfg = cfg.get("instance_ckpt_path", None)
        inst_ckpt = _resolve_path(inst_ckpt_cfg, orig_cwd)
        if inst_ckpt is None or (not inst_ckpt.exists()):
            raise FileNotFoundError(
                "[inference_targetnet_like_online] Need either run.checkpoint (EM ckpt) "
                "or cfg.instance_ckpt_path (standalone instance ckpt). "
                f"Got instance_ckpt_path={inst_ckpt}"
            )
        _load_single_model_ckpt(instance_model, inst_ckpt, device)
        print(f"[inference_targetnet_like_online] instance loaded from standalone ckpt: {inst_ckpt}")

    modules = TargetNetOnlineModules(instance_model=instance_model)

    forward_fn = TargetNetOnlineForward(
        modules=modules,
        cts_ds=cts_ds,
        pair_index=pi,
        device=device,
        scan_chunk_size=scan_chunk_size,
        amp=amp,
        amp_dtype=torch.float16,
        profile_sections=profile_sections,
        profile_sync_cuda=profile_sync_cuda,
    )

    bench_cfg = BenchConfig(
        seed=seed,
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
        repeats=repeats,
        amp=amp,
        sync_cuda=sync_cuda,
        device=str(device),
    )

    repeat_stats = []
    profile_repeats: List[Dict[str, float]] = []
    for r in range(int(repeats)):
        if profile_sections:
            forward_fn.reset_profile()

        st = run_benchmark_once(
            loader=loader,
            forward_fn=forward_fn,
            bench_cfg=bench_cfg,
            amp_dtype=torch.float16,
        )
        repeat_stats.append(st)

        if profile_sections:
            prof = forward_fn.profile_summary_ms(reset=True)
            profile_repeats.append(prof)
            print(f"[inference_targetnet_like_online] profile(ms/call) repeat={r}: {prof}")
            print(f"[inference_targetnet_like_online] repeat={r} done | peak_vram_gb={float(st.peak_vram_bytes)/(1024**3):.3f}")

    records = summarize_repeats_to_records(
        pipeline="TargetNet_like_online",
        split=split,
        K=K_anchor,
        repeat_stats=repeat_stats,
        seed=seed,
        num_pairs_subset=num_pairs_subset,
        batch_size=batch_size,
        num_workers=num_workers,
        amp=amp,
        avg_n_full=avg_n,
        median_n_full=med_n,
        p95_n_full=p95_n,
    )

    # add flags + optional profiling fields (keep CSV compatible w/ your plot scripts)
    out_records: List[Dict[str, Any]] = []
    if profile_sections and len(profile_repeats) > 0:
        prof_acc: Dict[str, List[float]] = {}
        for prof in profile_repeats:
            for k, v in prof.items():
                prof_acc.setdefault(str(k), []).append(float(v))
        prof_avg = {k: (sum(vs) / max(1, len(vs))) for k, vs in prof_acc.items()}
    else:
        prof_avg = {}

    for rec in records:
        d = _record_to_dict(rec)
        d["is_k_independent"] = 1
        d["scan_chunk_size"] = int(scan_chunk_size)
        if ident_hash is not None:
            d["dataset_identity_hash"] = str(ident_hash)
        for k, v in prof_avg.items():
            d[f"profile_{k}_ms"] = float(v)
        out_records.append(d)

    append_records_to_csv(csv_path, out_records)
    print(f"[inference_targetnet_like_online] CSV appended: {csv_path}")


if __name__ == "__main__":
    main()
