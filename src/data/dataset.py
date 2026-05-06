# src/data/dataset.py
"""
dataset.py
======================
Provides ChunkedCTSDataset: a lazy-loading Dataset implementation that relies only on meta.json.

Each sample is stored in multiple .pt blocks. The dataset uses metadata to determine:
  * Path to each block
  * Number of samples per block
  * Mapping from global index to specific block and local index

Key features:
- Fast initialization: Only reads meta.json, not all .pt blocks
- Memory-efficient: Loads blocks on-demand, releases old blocks with GC on switch
- Scalable: Block size is configurable (determined by BLOCK_SIZE and encoding size)

Typical usage:
    from src.data.cache import get_or_build_blocks
    from src.data.dataset import ChunkedCTSDataset

    get_or_build_blocks(data_cfg, "train", cache_path)  # Ensure meta.json exists
    ds = ChunkedCTSDataset(cache_path, data_cfg, "train")

    len(ds)             # Total samples
    x, y, set_idx = ds[0]
"""

import gc
import hashlib
import json
import os
from typing import List

import numpy as np
import torch


class ChunkedCTSDataset(torch.utils.data.Dataset):
    """
    Lazy-loading Dataset for large-scale CTS/miRNA data with block-based storage.

    Initialization only requires meta.json, not loading all block files into memory.
    Blocks are loaded on-demand when __getitem__ is called, and old blocks are released
    when switching to a new block.

    Attributes:
        chunk_files (List[str]): Paths to all block files
        chunk_sizes (List[int]): Number of samples per block
        cum_sizes (List[int]): Cumulative sample counts (for binary search to locate block)
        total_size (int): Total number of samples
        current_chunk_idx (int): Index of currently loaded block
        current_chunk (dict or None): Content of current block (dict from torch.load)

    Returns:
        __getitem__ returns a tuple:
            x (torch.Tensor): One-hot features (uint8), shape ~(C, L)
            y (torch.Tensor): Label, shape (1,)
            set_idx (torch.Tensor): Sequential ID within current split, shape (1,), dtype long

        Upstream collate_fn can convert to dict, e.g.:
            {"x": x.float(), "label": y.squeeze(-1), "set_idx": set_idx.squeeze(-1)}
    """

    def __init__(self, cache_data_path: str, data_cfg, split_idx: str):
        """
        Initialize dataset by loading meta.json based on data_cfg and split_idx.

        Args:
            cache_data_path (str): Cache directory path
            data_cfg (DataConfig): Original data configuration (must contain path field)
            split_idx (str): Data split to use (e.g., "train", "val", "test0")
        """
        data_file_path = str(data_cfg.get_path(split_idx))

        # Include alignment in hash key to distinguish caches for different alignments
        alignment = getattr(data_cfg, "alignment", "extended_seed_alignment")
        hash_key = f"{data_file_path}|{alignment}"
        path_hash = hashlib.md5(hash_key.encode("utf-8")).hexdigest()[:8]

        meta_filename = f"cache_{split_idx}_{path_hash}_meta.json"
        meta_filepath = os.path.join(cache_data_path, meta_filename)

        if not os.path.exists(meta_filepath):
            raise FileNotFoundError(
                f"Meta file not found: {meta_filepath}. "
                f"Please call get_or_build_blocks(data_cfg, '{split_idx}', '{cache_data_path}') first to generate cache."
            )

        with open(meta_filepath, "r") as f:
            block_metadata = json.load(f)

        # Ensure order matches build-time order
        block_metadata.sort(key=lambda x: (x["block_idx"], x["shard_idx"]))

        self.chunk_files: List[str] = [m["path"] for m in block_metadata]
        self.chunk_sizes: List[int] = [m["size"] for m in block_metadata]
        self.cum_sizes: List[int] = np.cumsum(self.chunk_sizes).tolist()
        self.total_size: int = self.cum_sizes[-1] if self.cum_sizes else 0

        self.current_chunk_idx: int = -1
        self.current_chunk = None

        print(
            f"Initialized ChunkedCTSDataset(split={split_idx}) "
            f"with {self.total_size} samples across {len(self.chunk_files)} blocks."
        )

        # Stage 1: Load PairIndex (optional but recommended)
        pair_index_name = f"pair_index_{split_idx}_{path_hash}.pt"
        pair_index_path = os.path.join(cache_data_path, pair_index_name)
        self.pair_offsets = None
        self.pair_counts = None
        self.num_pairs = None

        if os.path.exists(pair_index_path):
            obj = torch.load(pair_index_path, map_location="cpu", weights_only=False)
            self.pair_offsets = obj["pair_offsets"].long()
            self.pair_counts = obj.get("pair_counts", None)
            self.num_pairs = int(obj.get("num_pairs", self.pair_offsets.numel() - 1))
            print(f"Loaded PairIndex: {pair_index_path} (num_pairs={self.num_pairs})")
        else:
            print(
                f"[Warn] PairIndex not found: {pair_index_path}. "
                f"Stage-1 APIs get_pair_slice() will be unavailable until built."
            )

    def __len__(self) -> int:
        """Return total number of samples in the dataset."""
        return self.total_size

    def __getitem__(self, idx: int):
        """
        Return a single sample (x, y, set_idx) by global index idx.

        Implementation details:
            1. Use bisect_right on self.cum_sizes to find block_idx for idx
            2. If current block differs from last access, release old block and load new one
               - Load with weights_only=False to avoid future PyTorch default behavior changes
            3. Compute local_idx within current block (local_idx=idx if block 0, else subtract previous cumulative size)
            4. Extract X, labels, set_idxs entries from current block dict

        Args:
            idx (int): Global sample index (0 <= idx < len(self))

        Returns:
            tuple: (x, y, set_idx, esa_score, pos)
                - x (torch.Tensor): One-hot features (uint8)
                - y (torch.Tensor): Label, shape (1,)
                - set_idx (torch.Tensor): Original row number, shape (1,)
                - esa_score (torch.Tensor): ESA score, shape (1,)
                - pos (torch.Tensor): Position, shape (1,)
        """
        if idx < 0 or idx >= self.total_size:
            raise IndexError("Index out of range")

        import bisect

        # Find block index
        chunk_idx = bisect.bisect_right(self.cum_sizes, idx)

        # Switch to corresponding block if needed (lazy loading)
        if self.current_chunk_idx != chunk_idx:
            if self.current_chunk is not None:
                # Explicitly release old block and trigger GC to reduce memory peak
                del self.current_chunk
                gc.collect()

            self.current_chunk = torch.load(
                self.chunk_files[chunk_idx],
                map_location="cpu",
                weights_only=False,  # Loading data, not model weights
            )
            self.current_chunk_idx = chunk_idx

        # Compute local index within current block
        local_idx = idx if chunk_idx == 0 else idx - self.cum_sizes[chunk_idx - 1]

        x = self.current_chunk["X"][local_idx]
        y = self.current_chunk["labels"][local_idx]
        set_idx = self.current_chunk["set_idxs"][local_idx]
        if "esa_scores" in self.current_chunk:
            esa_score = self.current_chunk["esa_scores"][local_idx]
        else:
            esa_score = torch.tensor([0.0], dtype=torch.float32)

        if "pos" in self.current_chunk:
            pos = self.current_chunk["pos"][local_idx]
        else:
            pos = torch.tensor([0.5], dtype=torch.float32)
        return x, y, set_idx, esa_score, pos

    def __repr__(self) -> str:
        """Return readable string representation for debugging."""
        return (
            f"ChunkedCTSDataset(\n"
            f"  total_samples={self.total_size},\n"
            f"  num_blocks={len(self.chunk_files)}\n"
            f")"
        )

    def get_pair_slice(self, pair_id: int):
        """Get CTS uid range [start, end) for a given pair in O(1) time."""
        if self.pair_offsets is None:
            raise RuntimeError(
                "PairIndex not loaded. Please rebuild cache to generate pair_index_*.pt."
            )
        start = int(self.pair_offsets[pair_id].item())
        end = int(self.pair_offsets[pair_id + 1].item())
        return start, end

    def get_pair_num_cts(self, pair_id: int) -> int:
        s, e = self.get_pair_slice(pair_id)
        return e - s

    def get_cts_meta_by_uid(self, uids, fields=("X", "labels", "set_idxs", "pos", "esa_scores")):
        """
        Stage 1 naive implementation: iterate through uids, call __getitem__/load chunk.
        Stage 4 will implement high-performance batch gather (group by chunk).
        """
        out = {k: [] for k in fields}
        for uid in uids:
            idx = int(uid)  # In current design, uid == global idx
            # Directly locate chunk + local_idx and read from current_chunk
            # (calling __getitem__ would lose new fields)
            import bisect

            chunk_idx = bisect.bisect_right(self.cum_sizes, idx)
            if self.current_chunk_idx != chunk_idx:
                if self.current_chunk is not None:
                    del self.current_chunk
                    gc.collect()
                self.current_chunk = torch.load(
                    self.chunk_files[chunk_idx], map_location="cpu", weights_only=False
                )
                self.current_chunk_idx = chunk_idx
            local_idx = idx if chunk_idx == 0 else idx - self.cum_sizes[chunk_idx - 1]
            for k in fields:
                if k in self.current_chunk:
                    out[k].append(self.current_chunk[k][local_idx])
        # Stack (return empty if no data)
        for k in list(out.keys()):
            if len(out[k]) > 0:
                out[k] = torch.stack(out[k], dim=0)
            else:
                out[k] = None
        return out

    def validate_pair_offsets(self, num_checks: int = 20, seed: int = 0):
        """
        Randomly sample pairs and verify:
        1) Slice length == pair_counts
        2) All set_idx values in slice equal pair_id (requires reading a few samples)
        """
        if self.pair_offsets is None:
            raise RuntimeError("PairIndex not loaded.")
        import random

        rng = random.Random(seed)
        P = self.num_pairs
        for _ in range(num_checks):
            pid = rng.randint(0, P - 1)
            s, e = self.get_pair_slice(pid)
            if e < s:
                raise AssertionError(f"Invalid offsets for pid={pid}: {s},{e}")
            # Sample head/tail uids for verification
            probe = [s, min(s + 1, e - 1), max(e - 1, s)]
            probe = [u for u in probe if s <= u < e]
            meta = self.get_cts_meta_by_uid(probe, fields=("set_idxs",))
            set_idxs = meta["set_idxs"].view(-1).tolist()
            for v in set_idxs:
                if int(v) != int(pid):
                    raise AssertionError(
                        f"Pair slice mismatch: pid={pid}, got set_idx={v}, slice=({s},{e})"
                    )
        print(f"[OK] validate_pair_offsets passed with {num_checks} checks.")

    def batch_gather_by_uid(
        self,
        uids,
        fields=("inputs", "labels", "set_idx", "esa_scores", "pos"),
    ):
        """
        Stage 4: High-performance gather by grouping by chunk, loading each chunk once,
        and using tensor indexing to extract multiple entries.
        Convention: cts_uid == global sample idx (ChunkedCTSDataset global index).
        """
        import torch

        if isinstance(uids, (list, tuple)):
            uids = torch.tensor(uids, dtype=torch.long)
        else:
            uids = uids.to(dtype=torch.long)

        if uids.numel() == 0:
            return {k: None for k in fields}

        # cum_sizes_t: [num_chunks], strictly increasing
        if not hasattr(self, "cum_sizes_t"):
            self.cum_sizes_t = torch.tensor(self.cum_sizes, dtype=torch.long)

        # chunk_ids: [N], in [0, num_chunks-1]
        chunk_ids = torch.bucketize(uids, self.cum_sizes_t, right=True)

        # Sort by chunk_id to create contiguous segments
        order = torch.argsort(chunk_ids)
        uids_s = uids[order]
        cids_s = chunk_ids[order]

        # Inverse permutation to restore original order
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), dtype=torch.long)

        out_chunks = {k: [] for k in fields}

        # Iterate segments
        n = uids_s.numel()
        i = 0
        while i < n:
            cid = int(cids_s[i].item())
            j = i + 1
            while j < n and int(cids_s[j].item()) == cid:
                j += 1

            # Load chunk once
            if self.current_chunk_idx != cid:
                if self.current_chunk is not None:
                    del self.current_chunk
                self.current_chunk = torch.load(
                    self.chunk_files[cid], map_location="cpu", weights_only=False
                )
                self.current_chunk_idx = cid

            # Compute local indices
            base = 0 if cid == 0 else self.cum_sizes_t[cid - 1].item()
            local = (uids_s[i:j] - int(base)).to(dtype=torch.long)

            for k in fields:
                if k not in self.current_chunk:
                    # Compatibility: return None if field doesn't exist
                    out_chunks[k].append(None)
                else:
                    out_chunks[k].append(self.current_chunk[k].index_select(0, local))

            i = j

        # Concatenate per field, then reorder back to original uid order
        out = {}
        for k, parts in out_chunks.items():
            if all(p is None for p in parts):
                out[k] = None
                continue
            # Filter None (missing fields)
            parts2 = [p for p in parts if p is not None]
            cat = torch.cat(parts2, dim=0)
            out[k] = cat.index_select(0, inv)

        return out
