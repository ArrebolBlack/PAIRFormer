# src/selectors/st_selector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

import math
import heapq
import torch

Mode = Literal["train", "eval"]


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class STSelectorConfig:
    """
    Budgeted Selector (CPU-only)
    ============================

    Final spec implemented here:
      K = min(kmax, n)
      K1 = floor(K1_ratio * K)   (default 0.5K)
      K2 = K - K1

      Step A: TopK(score, K1) -> S1
      Step B: remaining R -> pos bins (B) + per-bin top-m (min-heap) -> C (<= L = B*m)
      Step C: per-bin Axis-SimHash dedup on C, keep <= c per hash key -> C'
      Step D: Balanced Quota allocate K2 over bins based on weights w_b -> S2
      Step E: S = S1 ∪ S2, dedup + score-based fill to size K

    Complexity constraints:
      - One pass over n for bin heaps: O(n log m), m<=64
      - Per bin sort <= m candidates: cheap
      - No cosine / pairwise distance. Embeddings only used for hashing.

    -----------------------
    Hyper-params (tables)
    -----------------------

    1) K1/K2:
      - Default: K1_ratio = 0.5  => K1=floor(0.5K), K2=K-K1

    2) Candidate pool L:
      L = min(max(8*K, 1024), 4096)

    3) Pos bins (B) and per-bin top-m:
      K <= 128      -> L=1024 -> B=32, m=32
      128 < K <=256 -> L=2048 -> B=32, m=64
      256 < K <=512 -> L=4096 -> B=64, m=64
      (Implementation: compute L first, then choose (B,m) by L tier)

    4) Axis-SimHash (emb dim=64):
      L=1024        -> bits b=24, cap c=2
      L=2048/4096   -> bits b=32, cap c=2

      Hash: key[i] packs bits of (emb[i, dims[t]] > 0) over t=0..b-1 into uint32.

    5) Balanced Quota:
      - quota_top_t = 8
      - quota_tau_w = 1.0
      - weight uses top-t quality only:
          w_b = sum_{i in top-t of bin b} exp(z_i / tau_w)
      - z-score normalization is optional but recommended

    """
    # Output budget upper bound (also used as K)
    kmax: int = 512

    # Step A
    k1_ratio: float = 0.5
    score_use_sigmoid: bool = False  # False: rank by logit, True: rank by sigmoid(logit)

    # Optional exploration (train only; default off)
    exploration_sigma: float = 0.0
    base_seed: int = 2020

    # Step B / Pos binning
    pos_bin_eps: float = 1e-6  # guard for pos==1.0
    # Candidate pool rule: L = min(max(8*K, 1024), 4096) (fixed by spec)
    # B/m selection is derived from L tier (fixed table)

    # Step C / Hash dedup
    use_hash_dedup: bool = True
    hash_seed: int = 2020  # seed to sample hash dims once per process
    # b and c are derived from L tier (fixed table)

    # Step D / Balanced quota
    score_norm_z: bool = True
    score_norm_eps: float = 1e-6
    quota_top_t: int = 8
    quota_tau_w: float = 1.0


# -----------------------------------------------------------------------------
# Deterministic exploration helpers
# -----------------------------------------------------------------------------

def _deterministic_seed(base_seed: int, epoch: int, pair_id: int) -> int:
    # stable mix, avoid python hash randomization
    x = (base_seed & 0xFFFFFFFF) ^ ((epoch & 0xFFFFFFFF) * 1000003) ^ ((pair_id & 0xFFFFFFFF) * 9176)
    return int(x & 0x7FFFFFFF)


def _add_gaussian_exploration(score_cpu: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    if sigma <= 0:
        return score_cpu
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    noise = torch.randn(score_cpu.numel(), generator=g, device="cpu", dtype=score_cpu.dtype) * float(sigma)
    return score_cpu + noise


# -----------------------------------------------------------------------------
# Fixed tables from spec (derived by L tier)
# -----------------------------------------------------------------------------

def _compute_L(K: int) -> int:
    # L = min(max(8*K, 1024), 4096)
    return int(min(max(8 * int(K), 1024), 4096))


def _bm_from_L(L: int) -> Tuple[int, int]:
    # choose (B,m) by L tier (fixed)
    if L <= 1024:
        return 32, 32
    if L <= 2048:
        return 32, 64
    return 64, 64


def _hash_params_from_L(L: int) -> Tuple[int, int]:
    # (bits b, cap c) by L tier (fixed)
    if L <= 1024:
        return 24, 2
    return 32, 2


# -----------------------------------------------------------------------------
# Axis-SimHash dims cache (per process)
# -----------------------------------------------------------------------------

_HASH_DIMS_CACHE: Dict[Tuple[int, int], torch.Tensor] = {}
_HASH_DIMS_PRINTED: Dict[Tuple[int, int], bool] = {}


def _get_hash_dims(seed: int, bits: int) -> torch.Tensor:
    """
    Return fixed dims indices of length 'bits' in [0,63], sampled once per process.

    Requirement:
      - sample once with fixed seed, reused for all bags => reproducible
    """
    key = (int(seed), int(bits))
    if key in _HASH_DIMS_CACHE:
        return _HASH_DIMS_CACHE[key]

    # Sample without replacement from 64 dims
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    perm = torch.randperm(64, generator=g, device="cpu")
    dims = perm[:bits].to(dtype=torch.long, device="cpu").contiguous()

    _HASH_DIMS_CACHE[key] = dims
    # Print once per process (very light; replace with logger if you prefer)
    if not _HASH_DIMS_PRINTED.get(key, False):
        print(f"[STSelector] Axis-SimHash dims (seed={seed}, bits={bits}): {dims.tolist()}")
        _HASH_DIMS_PRINTED[key] = True

    return dims


def _axis_simhash_keys(emb_cpu: torch.Tensor, idx: torch.Tensor, bits: int, seed: int) -> torch.Tensor:
    """
    Compute Axis-SimHash keys for selected indices.

    emb_cpu: [n,64] on CPU
    idx    : [m] long on CPU
    return : [m] int64 keys (<= 2^32-1)
    """
    dims = _get_hash_dims(seed=seed, bits=bits)  # [bits]
    # [m, bits] boolean -> int64
    # We only need sign; dtype can be float16/32.
    e = emb_cpu.index_select(0, idx).to(device="cpu")
    bits_bool = (e.index_select(1, dims) > 0)
    bits_i64 = bits_bool.to(dtype=torch.int64)

    # pack bits: key = sum(bits_i64[:,t] << t)
    # Use a precomputed power-of-two vector to avoid Python loops
    pow2 = (torch.ones((bits,), dtype=torch.int64, device="cpu") << torch.arange(bits, dtype=torch.int64, device="cpu"))
    key = (bits_i64 * pow2.view(1, -1)).sum(dim=1)  # [m]
    return key


# -----------------------------------------------------------------------------
# Step B: pos bins + per-bin top-m heaps (O(n log m))
# -----------------------------------------------------------------------------

def _build_candidates_by_pos_bins(
    *,
    score_np,              # numpy float32, shape [n]
    pos_np,                # numpy float32, shape [n]
    excluded_set: set,     # indices in [0..n-1] to exclude (S1)
    B: int,
    m: int,
    pos_eps: float,
) -> List[List[Tuple[float, int]]]:
    """
    One pass over n:
      for each token i not in excluded_set:
        bin = floor(B * pos[i]) clamped to [0, B-1]
        keep top-m by score per bin using min-heap.

    Return:
      heaps[b] : list of (score, idx) of size <= m (min-heap order)
    """
    heaps: List[List[Tuple[float, int]]] = [[] for _ in range(B)]

    # local bindings for speed
    Bm1 = B - 1
    for i in range(pos_np.shape[0]):
        if i in excluded_set:
            continue
        s = float(score_np[i])
        p = float(pos_np[i])

        # clamp position into [0, 1-eps] to avoid overflow at pos=1.0
        if p < 0.0:
            p = 0.0
        elif p >= 1.0:
            p = 1.0 - float(pos_eps)

        b = int(p * B)
        if b > Bm1:
            b = Bm1

        h = heaps[b]
        if len(h) < m:
            heapq.heappush(h, (s, i))
        else:
            # keep only top-m scores => compare with smallest in heap
            if s > h[0][0]:
                heapq.heapreplace(h, (s, i))

    return heaps


# -----------------------------------------------------------------------------
# Step C: per-bin hash dedup (within <= m list)
# -----------------------------------------------------------------------------

def _dedup_per_bin_by_hash(
    *,
    heaps: List[List[Tuple[float, int]]],  # per-bin heaps (score, idx)
    score_cpu: torch.Tensor,               # [n] float32 on CPU
    emb_cpu: Optional[torch.Tensor],       # [n,64] on CPU
    bits: int,
    cap_c: int,
    hash_seed: int,
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    For each bin:
      - sort candidates by score desc
      - compute hash keys (Axis-SimHash)
      - keep at most cap_c per key (scan in score order)

    Return:
      cand_idx_bins[b]   : list of token indices (sorted by score desc, deduped)
      cand_score_bins[b] : list of scores aligned with cand_idx_bins[b]
    """
    B = len(heaps)
    cand_idx_bins: List[List[int]] = [[] for _ in range(B)]
    cand_score_bins: List[List[float]] = [[] for _ in range(B)]

    # If dedup requested but emb missing: fallback to "no dedup" (still stable)
    do_hash = (emb_cpu is not None)

    for b in range(B):
        items = heaps[b]
        if not items:
            continue

        # sort by score desc (items are (score, idx))
        items_sorted = sorted(items, key=lambda x: x[0], reverse=True)

        idx_list = torch.tensor([it[1] for it in items_sorted], dtype=torch.long, device="cpu")
        if not do_hash:
            # no emb => keep as-is
            cand_idx_bins[b] = [int(x) for x in idx_list.tolist()]
            cand_score_bins[b] = [float(score_cpu[int(x)].item()) for x in cand_idx_bins[b]]
            continue

        keys = _axis_simhash_keys(emb_cpu=emb_cpu, idx=idx_list, bits=bits, seed=hash_seed)  # [m] int64

        counts: Dict[int, int] = {}
        kept: List[int] = []
        kept_scores: List[float] = []

        for j in range(idx_list.numel()):
            key = int(keys[j].item())
            cnt = counts.get(key, 0)
            if cnt >= cap_c:
                continue
            counts[key] = cnt + 1
            ii = int(idx_list[j].item())
            kept.append(ii)
            kept_scores.append(float(score_cpu[ii].item()))

        cand_idx_bins[b] = kept
        cand_score_bins[b] = kept_scores

    return cand_idx_bins, cand_score_bins


# -----------------------------------------------------------------------------
# Step D: balanced quota allocation
# -----------------------------------------------------------------------------

def _balanced_quota_select(
    *,
    cand_idx_bins: List[List[int]],       # per-bin indices sorted by score desc
    cand_score_bins: List[List[float]],   # per-bin scores aligned
    K2: int,
    score_norm_z: bool,
    score_norm_eps: float,
    quota_top_t: int,
    quota_tau_w: float,
) -> List[int]:
    """
    Implements the pseudo-code in your spec (Balanced Quota).

    Returns:
      S2 indices (length <= K2), chosen from C' bins.
    """
    if K2 <= 0:
        return []

    B = len(cand_idx_bins)
    a = [len(cand_idx_bins[b]) for b in range(B)]
    B_plus = [b for b in range(B) if a[b] > 0]
    M = len(B_plus)
    if M == 0:
        return []

    # Build a flat list of candidate indices for normalization / fallback fill
    all_idx: List[int] = []
    all_score: List[float] = []
    for b in B_plus:
        all_idx.extend(cand_idx_bins[b])
        all_score.extend(cand_score_bins[b])

    # z-score normalization (optional)
    if score_norm_z and len(all_score) > 1:
        sc = torch.tensor(all_score, dtype=torch.float32, device="cpu")
        mean = float(sc.mean().item())
        std = float(sc.std(unbiased=False).item())
        denom = std + float(score_norm_eps)
        # map each score -> z
        # keep in python list for convenience (sizes <= 4096)
        all_z = [ (s - mean) / denom for s in all_score ]
    else:
        # fallback: use raw score as "z"
        all_z = list(all_score)

    # For fast lookup: idx -> z
    # (Total candidates <= L <= 4096, dict is fine)
    z_map = {int(i): float(z) for i, z in zip(all_idx, all_z)}

    # weights per bin: sum_{top-t} exp(z_i / tau_w)
    tau = float(quota_tau_w)
    t = int(quota_top_t)

    w: Dict[int, float] = {}
    for b in B_plus:
        topk = cand_idx_bins[b][: min(t, a[b])]
        # stable exp; clamp to avoid overflow in extreme logits
        s = 0.0
        for ii in topk:
            z = z_map[int(ii)]
            zz = max(-20.0, min(20.0, z / max(tau, 1e-8)))
            s += math.exp(zz)
        w[b] = float(s)

    # Initialize quotas q_b = 0
    q: Dict[int, int] = {b: 0 for b in B_plus}

    # 3) if K2 >= M -> give everyone 1
    if K2 >= M:
        for b in B_plus:
            q[b] = 1
        R = K2 - M
    else:
        # choose top K2 bins by w_b
        B_sorted = sorted(B_plus, key=lambda bb: w.get(bb, 0.0), reverse=True)
        chosen = B_sorted[:K2]
        for b in chosen:
            q[b] = 1
        R = 0

    # 4) distribute remaining R proportionally by weights
    while R > 0:
        S_cap = [b for b in B_plus if (a[b] - q[b]) > 0]
        if not S_cap:
            break

        total_w = sum(w.get(b, 0.0) for b in S_cap)
        if total_w <= 0:
            # fallback: equal weights
            total_w = float(len(S_cap))
            w_eff = {b: 1.0 for b in S_cap}
        else:
            w_eff = {b: w.get(b, 0.0) for b in S_cap}

        R_before = R

        # proportional floors
        for b in S_cap:
            cap_b = a[b] - q[b]
            add_b = int(math.floor(R * (w_eff[b] / total_w)))
            if add_b <= 0:
                continue
            if add_b > cap_b:
                add_b = cap_b
            q[b] += add_b
            R -= add_b
            if R <= 0:
                break

        if R <= 0:
            break

        # if floors didn't reduce R, allocate leftovers one-by-one by w desc
        if R == R_before:
            S_sorted = sorted(S_cap, key=lambda bb: w_eff[bb], reverse=True)
            for b in S_sorted:
                if R <= 0:
                    break
                if (a[b] - q[b]) > 0:
                    q[b] += 1
                    R -= 1

    # 5) S2 = union over bins of first q_b indices
    S2: List[int] = []
    for b in B_plus:
        qb = q.get(b, 0)
        if qb > 0:
            S2.extend(cand_idx_bins[b][:qb])

    # 6) fill if needed: remaining candidates globally by score
    if len(S2) < K2:
        need = K2 - len(S2)
        S2_set = set(S2)
        # remaining candidates = all candidates - S2
        rem = [(cand_score_bins[b][j], cand_idx_bins[b][j])
               for b in B_plus
               for j in range(a[b])
               if cand_idx_bins[b][j] not in S2_set]
        rem.sort(key=lambda x: x[0], reverse=True)
        for _, ii in rem[:need]:
            S2.append(int(ii))

    # truncate (defensive)
    if len(S2) > K2:
        S2 = S2[:K2]
    return S2


# -----------------------------------------------------------------------------
# Public API: selector_fn
# -----------------------------------------------------------------------------

def selector_fn(
    *,
    uids: torch.Tensor,                 # [n] CTS uid (global id)
    pos: torch.Tensor,                  # [n] in [0,1]
    cheap_logit: torch.Tensor,          # [n]
    cheap_emb: Optional[torch.Tensor],  # [n,64] used ONLY for hash dedup
    cfg: STSelectorConfig,
    mode: Mode,                         # "train"|"eval"
    epoch: int,
    pair_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return:
      sel_uids: int32 tensor [cfg.kmax] padded with -1
      sel_len:  int32 scalar
    """
    # --------------------
    # 0) Validate + normalize device (CPU-only)
    # --------------------
    if uids.dim() != 1 or pos.dim() != 1 or cheap_logit.dim() != 1:
        raise ValueError(
            f"[STSelector] Expect 1D tensors. got uids{tuple(uids.shape)} pos{tuple(pos.shape)} logit{tuple(cheap_logit.shape)}"
        )

    n = int(uids.numel())
    Kmax = int(cfg.kmax)

    out_uids = torch.full((Kmax,), -1, dtype=torch.int32, device="cpu")
    out_len = torch.tensor(0, dtype=torch.int32, device="cpu")

    if n <= 0 or Kmax <= 0:
        return out_uids, out_len

    # CPU copies (cheap; per-bag)
    uids_cpu = uids.to(device="cpu").view(-1)
    pos_cpu = pos.to(device="cpu", dtype=torch.float32).view(-1)
    logit_cpu = cheap_logit.to(device="cpu", dtype=torch.float32).view(-1)

    # Emb on CPU if provided (for hashing)
    emb_cpu: Optional[torch.Tensor] = None
    if cheap_emb is not None:
        emb_cpu = cheap_emb.to(device="cpu")

    # Effective budget K (cannot exceed n)
    K = min(Kmax, n)
    if n <= K:
        # Spec: n<=K => return all (optionally truncated). For stability: sort by score desc.
        score0 = torch.sigmoid(logit_cpu) if cfg.score_use_sigmoid else logit_cpu
        order0 = torch.argsort(score0, descending=True)
        sel_idx = order0[:K]
        sel_u = uids_cpu.index_select(0, sel_idx).to(dtype=torch.int32)
        out_uids[: sel_u.numel()] = sel_u
        return out_uids, torch.tensor(int(sel_u.numel()), dtype=torch.int32, device="cpu")

    # --------------------
    # 1) Score definition (+ optional exploration)
    # --------------------
    score = torch.sigmoid(logit_cpu) if cfg.score_use_sigmoid else logit_cpu
    if mode == "train" and cfg.exploration_sigma > 0:
        seed = _deterministic_seed(cfg.base_seed, epoch, pair_id)
        score = _add_gaussian_exploration(score, cfg.exploration_sigma, seed)

    # --------------------
    # Step A: K1 TopK -> S1
    # --------------------
    K1 = int(math.floor(float(cfg.k1_ratio) * float(K)))
    if K1 < 0:
        K1 = 0
    if K1 > K:
        K1 = K
    K2 = K - K1

    if K1 > 0:
        # topk on CPU
        top1_val, top1_idx = torch.topk(score, k=K1, largest=True, sorted=False)  # indices in [0..n-1]
        # Keep S1 order stable by score desc (optional but makes debugging easier)
        s1_order = torch.argsort(top1_val, descending=True)
        S1_idx = top1_idx.index_select(0, s1_order).to(dtype=torch.long)
        S1_set = set(int(i) for i in S1_idx.tolist())
    else:
        S1_idx = torch.empty((0,), dtype=torch.long, device="cpu")
        S1_set = set()

    if K2 <= 0:
        sel_u = uids_cpu.index_select(0, S1_idx).to(dtype=torch.int32)
        out_uids[: sel_u.numel()] = sel_u
        return out_uids, torch.tensor(int(sel_u.numel()), dtype=torch.int32, device="cpu")

    # --------------------
    # Step B: build candidate pool via pos bins + per-bin top-m heaps
    # --------------------
    L = _compute_L(K)
    B, m = _bm_from_L(L)

    # numpy views for fast python loop
    score_np = score.detach().cpu().numpy()
    pos_np = pos_cpu.detach().cpu().numpy()

    heaps = _build_candidates_by_pos_bins(
        score_np=score_np,
        pos_np=pos_np,
        excluded_set=S1_set,
        B=B,
        m=m,
        pos_eps=float(cfg.pos_bin_eps),
    )

    # --------------------
    # Step C: per-bin Axis-SimHash dedup -> C'
    # --------------------
    if cfg.use_hash_dedup:
        bits, cap_c = _hash_params_from_L(L)
        if emb_cpu is None:
            # Spec says emb exists for hashing; here we keep running but warn loudly.
            # If you prefer hard-fail, replace with: raise ValueError(...)
            print("[STSelector][WARN] use_hash_dedup=True but cheap_emb is None. Skip hash dedup.")
        cand_idx_bins, cand_score_bins = _dedup_per_bin_by_hash(
            heaps=heaps,
            score_cpu=score,
            emb_cpu=emb_cpu,
            bits=bits,
            cap_c=cap_c,
            hash_seed=int(cfg.hash_seed),
        )
    else:
        # no dedup: just sort each bin desc
        cand_idx_bins = [[] for _ in range(B)]
        cand_score_bins = [[] for _ in range(B)]
        for b in range(B):
            if not heaps[b]:
                continue
            items_sorted = sorted(heaps[b], key=lambda x: x[0], reverse=True)
            cand_idx_bins[b] = [int(ii) for _, ii in items_sorted]
            cand_score_bins[b] = [float(score[int(ii)].item()) for ii in cand_idx_bins[b]]

    # --------------------
    # Step D: balanced quota for K2 -> S2 (token indices)
    # --------------------
    S2_idx_list = _balanced_quota_select(
        cand_idx_bins=cand_idx_bins,
        cand_score_bins=cand_score_bins,
        K2=K2,
        score_norm_z=bool(cfg.score_norm_z),
        score_norm_eps=float(cfg.score_norm_eps),
        quota_top_t=int(cfg.quota_top_t),
        quota_tau_w=float(cfg.quota_tau_w),
    )

    # --------------------
    # Step E: merge S = S1 ∪ S2, dedup + score fill to K
    # --------------------
    selected: List[int] = []
    seen = set()

    # S1 first
    for ii in S1_idx.tolist():
        iii = int(ii)
        if iii not in seen:
            seen.add(iii)
            selected.append(iii)
            if len(selected) >= K:
                break

    # then S2
    if len(selected) < K:
        for ii in S2_idx_list:
            iii = int(ii)
            if iii in seen:
                continue
            seen.add(iii)
            selected.append(iii)
            if len(selected) >= K:
                break

    # fill if still short: global remaining by score
    if len(selected) < K:
        need = K - len(selected)
        # pick top from all tokens not in seen
        # n up to 30000; using torch topk with mask is OK for this tail case
        mask = torch.ones((n,), dtype=torch.bool, device="cpu")
        if seen:
            idx_seen = torch.tensor(sorted(list(seen)), dtype=torch.long, device="cpu")
            mask[idx_seen] = False
        remain_idx = torch.nonzero(mask, as_tuple=False).view(-1)
        if remain_idx.numel() > 0:
            remain_score = score.index_select(0, remain_idx)
            k_add = min(int(need), int(remain_idx.numel()))
            _, add_local = torch.topk(remain_score, k=k_add, largest=True, sorted=True)
            add_idx = remain_idx.index_select(0, add_local).tolist()
            for ii in add_idx:
                iii = int(ii)
                if iii in seen:
                    continue
                seen.add(iii)
                selected.append(iii)
                if len(selected) >= K:
                    break

    # Map token indices -> uids and pad to Kmax
    sel_idx_t = torch.tensor(selected[:K], dtype=torch.long, device="cpu")
    sel_uids = uids_cpu.index_select(0, sel_idx_t).to(dtype=torch.int32)  # [K]
    out_uids[: sel_uids.numel()] = sel_uids
    out_len = torch.tensor(int(sel_uids.numel()), dtype=torch.int32, device="cpu")
    return out_uids, out_len
