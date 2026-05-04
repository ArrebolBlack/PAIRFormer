"""Merge sharded ``selected_inst`` caches into a single monolithic cache.

Supports **incremental** merging and writes output to a **local directory**
(typically ``/dev/shm``) to avoid page-cache thrashing on VepFS.

After all shards are merged to the local directory, the ``--copy-to-vepfs``
flag can be used to copy the result to the final VepFS location using
``shutil.copy2`` (buffered file copy, much safer than numpy mmap writes).

Workflow for K=512 (2 shards)::

    # 1. Build shard 0 → /dev/shm/inst_build_k512_shard0
    # 2. Merge shard 0 → /dev/shm/inst_build_k512_merged (local)
    python -m src.launch.merge_inst_shards \\
        +shard_dirs="[/dev/shm/inst_build_k512_shard0]" \\
        +local_output_dir=/dev/shm/inst_build_k512_merged \\
        +split=train +num_pairs=294029 +kmax=512 +inst_emb_dim=1536 \\
        +has_inst_logit=true +total_shards=2
    # 3. Build shard 1 → /dev/shm/inst_build_k512_shard1
    # 4. Merge shard 1 (append)
    python -m src.launch.merge_inst_shards \\
        +shard_dirs="[/dev/shm/inst_build_k512_shard1]" \\
        +local_output_dir=/dev/shm/inst_build_k512_merged \\
        +split=train +num_pairs=294029 +kmax=512 +inst_emb_dim=1536 \\
        +has_inst_logit=true +total_shards=2
    # 5. Copy merged result to VepFS
    python -m src.launch.merge_inst_shards \\
        +local_output_dir=/dev/shm/inst_build_k512_merged \\
        +copy_to_vepfs=/vepfs/.../cache_mti_full_topk_retrain_r4_v3relbl_k512 \\
        +split=train

The ``inst_emb`` array is created as a sparse mmap on the local filesystem
so it does not need zero-filling.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig


def _create_sparse_mmap(path: Path, *, dtype: str, shape: tuple) -> np.memmap:
    return np.memmap(path, mode="w+", dtype=dtype, shape=shape)


def _create_zeroed_mmap(path: Path, *, dtype: str, shape: tuple) -> np.memmap:
    m = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
    m[:] = 0
    return m


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # --- mode 1: copy merged result to VepFS ---
    copy_to = cfg.get("copy_to_vepfs", None)
    if copy_to is not None:
        _do_copy_to_vepfs(cfg)
        return

    # --- mode 2: merge shards ---
    shard_dirs_cfg = cfg.get("shard_dirs", None)
    if shard_dirs_cfg is None:
        raise ValueError("shard_dirs must be provided")
    if isinstance(shard_dirs_cfg, str):
        shard_dirs = [Path(p.strip()) for p in shard_dirs_cfg.split(",")]
    else:
        shard_dirs = [Path(str(d)) for d in shard_dirs_cfg]

    local_output_dir = cfg.get("local_output_dir", None)
    if local_output_dir is None:
        raise ValueError("local_output_dir must be set (use /dev/shm)")
    cache_dir = Path(str(local_output_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)

    split = str(cfg.get("split", "train"))
    num_pairs = int(cfg.get("num_pairs", 0))
    kmax = int(cfg.get("kmax", 512))
    inst_emb_dim = int(cfg.get("inst_emb_dim", 1536))
    has_inst_logit = bool(cfg.get("has_inst_logit", True))
    total_shards = int(cfg.get("total_shards", len(shard_dirs)))

    if num_pairs <= 0:
        raise ValueError("num_pairs must be > 0")

    # --- validate shards ---
    shard_infos: list[dict] = []
    for d in shard_dirs:
        meta_path = d / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Shard meta.json not found: {meta_path}")
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("state") != "ready":
            raise RuntimeError(f"Shard {d} not ready (state={meta.get('state')})")
        if meta.get("cache_type") != "selected_inst_shard":
            raise ValueError(f"Shard {d} wrong cache_type: {meta.get('cache_type')}")
        shard_infos.append(meta)

    dir_by_start = {int(m["start_idx"]): d for m, d in zip(shard_infos, shard_dirs)}
    shard_infos.sort(key=lambda m: int(m["start_idx"]))
    shard_dirs_sorted = [dir_by_start[int(m["start_idx"])] for m in shard_infos]

    print(
        f"[merge_inst_shards] local_output={cache_dir} split={split} "
        f"num_pairs={num_pairs} kmax={kmax} inst_emb_dim={inst_emb_dim} "
        f"merging {len(shard_dirs)} shard(s), total_shards={total_shards}"
    )

    P, K, D = num_pairs, kmax, inst_emb_dim
    is_first = not (cache_dir / "inst_emb.f16.mmap").exists()

    if is_first:
        print("[merge_inst_shards] Creating output files on local fs...")
        f_label = _create_zeroed_mmap(cache_dir / "label.f32.mmap", dtype=np.float32, shape=(P,))
        f_sel_len = _create_zeroed_mmap(cache_dir / "sel_len.i16.mmap", dtype=np.int16, shape=(P,))
        f_esa = _create_zeroed_mmap(cache_dir / "esa.f16.mmap", dtype=np.float16, shape=(P, K))
        f_pos = _create_zeroed_mmap(cache_dir / "pos.f16.mmap", dtype=np.float16, shape=(P, K))
        f_emb = _create_sparse_mmap(
            cache_dir / "inst_emb.f16.mmap", dtype=np.float16, shape=(P, K, D)
        )
        f_logit: np.memmap | None = None
        if has_inst_logit:
            f_logit = _create_sparse_mmap(
                cache_dir / "inst_logit.f16.mmap", dtype=np.float16, shape=(P, K)
            )
        merged_count = 0
    else:
        print("[merge_inst_shards] Appending to existing output...")
        f_label = np.memmap(cache_dir / "label.f32.mmap", mode="r+", dtype=np.float32, shape=(P,))
        f_sel_len = np.memmap(cache_dir / "sel_len.i16.mmap", mode="r+", dtype=np.int16, shape=(P,))
        f_esa = np.memmap(cache_dir / "esa.f16.mmap", mode="r+", dtype=np.float16, shape=(P, K))
        f_pos = np.memmap(cache_dir / "pos.f16.mmap", mode="r+", dtype=np.float16, shape=(P, K))
        f_emb = np.memmap(
            cache_dir / "inst_emb.f16.mmap", mode="r+", dtype=np.float16, shape=(P, K, D)
        )
        f_logit: np.memmap | None = None
        if has_inst_logit:
            f_logit = np.memmap(
                cache_dir / "inst_logit.f16.mmap", mode="r+", dtype=np.float16, shape=(P, K)
            )
        progress_path = cache_dir / "merge_progress.json"
        if progress_path.exists():
            with open(progress_path) as f:
                merged_count = int(json.load(f).get("merged_count", 0))
        else:
            merged_count = 0

    t0 = time.time()

    for shard_dir, meta in zip(shard_dirs_sorted, shard_infos):
        start = int(meta["start_idx"])
        end = int(meta["end_idx"])
        shard_size = int(meta["shard_size"])
        print(
            f"[merge_inst_shards] Merging rows [{start}, {end}) size={shard_size} from {shard_dir}"
        )

        s_label = np.memmap(
            shard_dir / "label.f32.mmap", mode="r", dtype=np.float32, shape=(shard_size,)
        )
        s_sel_len = np.memmap(
            shard_dir / "sel_len.i16.mmap", mode="r", dtype=np.int16, shape=(shard_size,)
        )
        s_esa = np.memmap(
            shard_dir / "esa.f16.mmap", mode="r", dtype=np.float16, shape=(shard_size, K)
        )
        s_pos = np.memmap(
            shard_dir / "pos.f16.mmap", mode="r", dtype=np.float16, shape=(shard_size, K)
        )
        s_emb = np.memmap(
            shard_dir / "inst_emb.f16.mmap", mode="r", dtype=np.float16, shape=(shard_size, K, D)
        )
        s_logit: np.memmap | None = None
        if has_inst_logit:
            s_logit = np.memmap(
                shard_dir / "inst_logit.f16.mmap", mode="r", dtype=np.float16, shape=(shard_size, K)
            )

        # Small arrays: one-shot
        f_label[start:end] = s_label[:]
        f_sel_len[start:end] = s_sel_len[:]
        f_esa[start:end] = s_esa[:]
        f_pos[start:end] = s_pos[:]

        # Large arrays: chunked copy with flush between chunks
        chunk_rows = 1000
        n_chunks = (shard_size + chunk_rows - 1) // chunk_rows
        for ci in range(n_chunks):
            cs = ci * chunk_rows
            ce = min(cs + chunk_rows, shard_size)
            f_emb[start + cs : start + ce] = s_emb[cs:ce]
            if f_logit is not None and s_logit is not None:
                f_logit[start + cs : start + ce] = s_logit[cs:ce]
            if (ci + 1) % 10 == 0 or ci == n_chunks - 1:
                f_emb.flush()
                if f_logit is not None:
                    f_logit.flush()

        f_label.flush()
        f_sel_len.flush()
        f_esa.flush()
        f_pos.flush()

        merged_count += 1
        print(
            f"[merge_inst_shards] Merged {merged_count}/{total_shards}, elapsed={time.time() - t0:.1f}s"
        )

        del s_label, s_sel_len, s_esa, s_pos, s_emb, s_logit
        shutil.rmtree(str(shard_dir), ignore_errors=True)
        print(f"[merge_inst_shards] Cleaned: {shard_dir}")

    # Write progress
    with open(cache_dir / "merge_progress.json", "w") as f:
        json.dump({"merged_count": merged_count, "total_shards": total_shards}, f)

    all_done = merged_count >= total_shards
    meta_dict = {
        "state": "ready" if all_done else "building",
        "split": split,
        "cache_type": "selected_inst",
        "num_pairs": P,
        "kmax": K,
        "channels": 10,
        "seq_len": 50,
        "has_cheap_logit": False,
        "has_cheap_emb": False,
        "cheap_emb_dim": 64,
        "has_inst_logit": has_inst_logit,
        "has_inst_emb": True,
        "inst_emb_dim": D,
    }
    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta_dict, f, indent=2, sort_keys=True)

    if all_done:
        (cache_dir / "merge_progress.json").unlink(missing_ok=True)
        print(f"[merge_inst_shards] ALL DONE total={num_pairs} elapsed={time.time() - t0:.1f}s")
        print(f"[merge_inst_shards] Next: copy to VepFS with +copy_to_vepfs=...")
    else:
        print(
            f"[merge_inst_shards] Partial: {merged_count}/{total_shards}. Run again for remaining shards."
        )


def _do_copy_to_vepfs(cfg: DictConfig) -> None:
    """Copy merged local files to final VepFS location using buffered I/O."""
    local_dir = Path(str(cfg.get("local_output_dir")))
    vepfs_root = Path(str(cfg.get("copy_to_vepfs")))
    split = str(cfg.get("split", "train"))

    src_dir = local_dir
    dst_dir = vepfs_root / "selected_pair_cache" / split / "selected_inst"
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Validate source
    src_meta_path = src_dir / "meta.json"
    with open(src_meta_path) as f:
        src_meta = json.load(f)
    if src_meta.get("state") != "ready":
        raise RuntimeError(f"Source not ready (state={src_meta.get('state')})")

    print(f"[merge_inst_shards] Copying {src_dir} → {dst_dir}")
    t0 = time.time()

    for name in sorted(src_dir.iterdir()):
        if name.is_dir():
            continue
        fname = name.name
        src_file = src_dir / fname
        dst_file = dst_dir / fname
        size_mb = src_file.stat().st_size / (1024 * 1024)
        print(f"  {fname} ({size_mb:.0f} MB)...", end=" ", flush=True)
        t1 = time.time()
        shutil.copy2(str(src_file), str(dst_file))
        print(f"{time.time() - t1:.1f}s")
        # Delete source after copy to free /dev/shm
        src_file.unlink()

    print(f"[merge_inst_shards] Copy DONE elapsed={time.time() - t0:.1f}s")
    # Clean up empty local dir
    shutil.rmtree(str(local_dir), ignore_errors=True)


if __name__ == "__main__":
    main()
