from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch

from src.data.selected_pair_cache import SelectedPairCacheReader


class SelectedPairDataset(torch.utils.data.Dataset):
    """
    Pair-level dataset backed by the compact selected-pair cache.

    Memory/parallelism note (critical): the cache files are large mmaps
    (``inst_emb.f16.mmap`` can be tens of GB). With ``mp.set_start_method("spawn")``
    a Dataset is **pickled** to every DataLoader worker, and ``np.memmap`` pickles
    **by value** — so eagerly holding an open reader here would copy the whole cache
    into each worker's RAM and pin the CPU. To avoid that we:

    * never open the reader in ``__init__`` (resolve ``num_pairs`` cheaply from
      ``meta.json`` instead, so ``__len__`` works in the parent without mmapping);
    * open the reader **lazily, inside whichever process first calls __getitem__**
      (each worker mmaps the file itself → the OS page cache is shared, host RAM
      stays ~1× the cache regardless of ``num_workers``, and ``MADV_RANDOM`` set in
      the reader actually takes effect because the worker holds a real mmap);
    * drop the open reader in ``__getstate__`` so spawning a worker pickles ~0 bytes.
    """

    def __init__(self, cache_root: str, split: str, *, cache_type: str = "selected_raw", max_pairs: int | None = None):
        self.cache_root = str(cache_root)
        self.split = str(split)
        self.cache_type = str(cache_type)
        self.max_pairs = None if max_pairs is None else int(max_pairs)
        self._reader: SelectedPairCacheReader | None = None
        # cheap: num_pairs from meta.json, WITHOUT opening the big mmaps
        self._num_pairs = self._read_num_pairs()

    def _read_num_pairs(self) -> int:
        meta_p = Path(self.cache_root) / "selected_pair_cache" / self.split / self.cache_type / "meta.json"
        with open(meta_p, "r") as f:
            return int(json.load(f)["num_pairs"])

    @property
    def reader(self) -> SelectedPairCacheReader:
        """Lazily open the mmap-backed reader in the CURRENT process (worker or main).

        Never crosses the pickle boundary (see __getstate__), so spawn workers each
        open their own reader rather than receiving a by-value copy of the cache.
        """
        if self._reader is None:
            self._reader = SelectedPairCacheReader(self.cache_root, split=self.split, cache_type=self.cache_type)
        return self._reader

    def __len__(self) -> int:
        n = self._num_pairs
        return min(n, self.max_pairs) if self.max_pairs is not None else n

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.reader.read_pair(int(idx))

    def __getstate__(self) -> Dict[str, Any]:
        # Do NOT pickle the open reader (its np.memmaps would serialize by value).
        state = self.__dict__.copy()
        state["_reader"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._reader = None
