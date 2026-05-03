"""Optimized Robustness vs. n experiment builder.

Strategy: score all candidates with cheap model ONCE per pair,
then for each n value: random sample -> select K -> instance encode.
Avoids re-running cheap model for every n.

Usage:
    python scripts/build_robustness_caches_fast.py \
        n_values="[64,128,256,512,1024,2048]" \
        K=512 seed=2020 split=test batch_size=4096
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.config.data_config import DataConfig
from src.data.stream_pair_dataset import StreamPairDataset
from src.em.cheap_runner import load_ckpt_into_model
from src.data.selected_pair_cache import SelectedInstPairCacheWriter
from src.models.extractors import get_embedding_and_logit
from src.models.registry import build_model
from src.precompute.pair_stream_builder_parallel import (
    PairChunkResult,
    _chunk_records,
    _process_pair_chunk,
)
from src.selectors.stream_topk_selector import StreamTopKSelector, StreamTopKSelectorConfig
from src.utils import set_seeds


def main():
    parser = argparse.ArgumentParser(description="Build Robustness vs. n caches (optimized)")
    parser.add_argument("--n_values", type=str, default="[64,128,256,512,1024,2048]")
    parser.add_argument("--K", type=int, default=512, help="Budget K (model's kmax)")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=4096, help="Cheap model batch size")
    parser.add_argument("--inst_batch_size", type=int, default=4096, help="Instance model batch size")
    parser.add_argument("--output_dir", type=str, default="cache_robustness_k{n}")
    parser.add_argument("--cheap_ckpt", type=str,
                        default="checkpoints/MTI_CheapCTSNet_shard_v1_compact_r4/best.pt")
    parser.add_argument("--inst_ckpt", type=str,
                        default="checkpoints/MTI_v3_xlarge_resume/best.pt")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--task_pairs", type=int, default=16)
    parser.add_argument("--esa_min_score", type=float, default=6.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_config", type=str,
                        default="configs/experiment/MTI_train_selected_inst.yaml")
    args = parser.parse_args()

    set_seeds(args.seed)

    n_values = json.loads(args.n_values)
    K = args.K
    device = torch.device(args.device)

    print(f"[Robustness Fast] K={K} n_values={n_values} split={args.split} seed={args.seed}")

    # --- Load models ---
    from omegaconf import OmegaConf
    from hydra import initialize_config_dir, compose

    config_dir = str(Path("configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"experiment=MTI_build_selected_inst"])

    data_cfg = DataConfig.from_omegaconf(cfg.data)

    # Cheap model - construct manually since config may not have cheap_model section
    cheap_cfg = OmegaConf.create({
        "arch": "CheapCTSNet_TinyConv",
        "num_channels": [16, 16],
        "num_blocks": [2, 2],
        "pool_size": 2,
        "stem_kernel_size": 5,
        "block_kernel_size": 3,
        "skip_connection": True,
        "dropout": 0.3,
        "in_channels": 10,
        "seq_len": 50,
    })
    cheap_model = build_model("CheapCTSNet_TinyConv", cheap_cfg, data_cfg=data_cfg)
    load_ckpt_into_model(cheap_model, Path(args.cheap_ckpt), device=device, use_ema_shadow=False)
    cheap_model.to(device).eval()
    cheap_emb_dim = cheap_model.linear.out_features if hasattr(cheap_model, 'linear') else 64
    print(f"[Robustness Fast] Cheap model loaded, emb_dim={cheap_emb_dim}")

    # Instance model (X-Large: se_reduction=8, target_output_length=12 → emb_dim=128*12=1536)
    inst_cfg = OmegaConf.create({
        "arch": "TargetNet_Optimized",
        "num_channels": [64, 64, 128, 128],
        "num_blocks": [3, 3, 3, 3],
        "multi_scale": True,
        "se_type": "cbam",
        "se_reduction": 8,
        "target_output_length": 12,
        "use_bn": True,
        "dropout": 0.1,
        "in_channels": 10,
        "seq_len": 50,
    })
    instance_model = build_model("TargetNet_Optimized", inst_cfg, data_cfg=data_cfg)
    load_ckpt_into_model(instance_model, Path(args.inst_ckpt), device=device, use_ema_shadow=False)
    instance_model.to(device).eval()
    inst_emb_dim = instance_model.linear.in_features if hasattr(instance_model, 'linear') else 1536
    print(f"[Robustness Fast] Instance model loaded, emb_dim={inst_emb_dim}")

    # --- Create dataset ---
    pair_ds = StreamPairDataset(data_cfg, split=args.split)
    num_pairs = pair_ds.count_records()
    print(f"[Robustness Fast] {args.split}: {num_pairs} pairs")

    # --- Create output writers (one per n value) ---
    writers = {}
    for n in n_values:
        out_dir = args.output_dir.format(n=n, K=K)
        writer = SelectedInstPairCacheWriter(
            out_dir,
            split=args.split,
            num_pairs=num_pairs,
            kmax=K,
            inst_emb_dim=inst_emb_dim,
            has_inst_logit=True,
        )
        writers[n] = writer
        print(f"[Robustness Fast] n={n} -> {out_dir}")

    # Also write baseline (n=inf, use existing cache or skip)
    baseline_dir = args.output_dir.format(n="inf", K=K)

    # --- Main loop ---
    from multiprocessing import get_context

    t0 = time.time()
    total_candidates = 0
    pair_count = 0

    ctx = get_context("spawn")
    task_iter = (
        (chunk, data_cfg, args.esa_min_score)
        for chunk in _chunk_records(pair_ds.iter_records(), args.task_pairs)
    )

    with ctx.Pool(processes=args.num_workers) as pool:
        from multiprocessing import Queue
        import queue as queue_mod

        for chunk_result in pool.imap_unordered(_process_pair_chunk, task_iter, chunksize=1):
            for pair_id, label, xs_tensor, esa_tensor, pos_tensor in zip(
                chunk_result.pair_ids,
                chunk_result.labels,
                chunk_result.xs_list,
                chunk_result.esa_list,
                chunk_result.pos_list,
            ):
                n_cts = int(xs_tensor.shape[0])
                total_candidates += n_cts

                # Step 1: Run cheap model on ALL candidates (ONCE)
                all_cheap_logit = torch.empty(n_cts, dtype=torch.float32)
                offset = 0
                with torch.no_grad():
                    while offset < n_cts:
                        end = min(n_cts, offset + args.batch_size)
                        x = xs_tensor[offset:end].to(device, non_blocking=True).float()
                        esa = esa_tensor[offset:end].to(device, non_blocking=True)
                        pos = pos_tensor[offset:end].to(device, non_blocking=True)
                        _, logit = get_embedding_and_logit(cheap_model, x, esa_scores=esa, pos=pos)
                        all_cheap_logit[offset:end] = logit.detach().cpu()[: end - offset]
                        offset = end

                # Step 2: For each n value, random sample -> select K -> instance encode
                for n_val in n_values:
                    if n_cts == 0:
                        # Empty pair
                        writers[n_val].write_pair(
                            int(pair_id),
                            inst_emb=torch.zeros(K, inst_emb_dim),
                            inst_logit=torch.zeros(K),
                            esa=torch.zeros(K),
                            pos=torch.zeros(K),
                            label=float(label),
                            sel_len=0,
                        )
                        continue

                    # Random sample n candidates
                    gen = torch.Generator(device="cpu")
                    gen.manual_seed(int(args.seed + pair_id))
                    if n_val < n_cts:
                        sampled_idx = torch.randperm(n_cts, generator=gen)[:n_val]
                    else:
                        sampled_idx = torch.arange(n_cts)

                    # Select top-K from sampled
                    sampled_logits = all_cheap_logit[sampled_idx]
                    k_actual = min(K, len(sampled_idx))
                    _, top_local = torch.topk(sampled_logits, k=k_actual, largest=True)
                    selected_idx = sampled_idx[top_local]

                    # Step 3: Instance encode selected candidates
                    sel_X = xs_tensor[selected_idx]
                    sel_esa = esa_tensor[selected_idx]
                    sel_pos = pos_tensor[selected_idx]

                    inst_emb_out = torch.zeros(K, inst_emb_dim)
                    inst_logit_out = torch.zeros(K)

                    if k_actual > 0:
                        with torch.no_grad():
                            # Batch instance encoding
                            all_inst_emb = []
                            all_inst_logit = []
                            off = 0
                            while off < k_actual:
                                end = min(k_actual, off + args.inst_batch_size)
                                x_batch = sel_X[off:end].to(device, non_blocking=True).float()
                                esa_batch = sel_esa[off:end].to(device, non_blocking=True)
                                pos_batch = sel_pos[off:end].to(device, non_blocking=True)
                                emb, logit = get_embedding_and_logit(
                                    instance_model, x_batch, esa_scores=esa_batch, pos=pos_batch
                                )
                                all_inst_emb.append(emb.detach().cpu())
                                all_inst_logit.append(logit.detach().cpu())
                                off = end

                            inst_emb_sel = torch.cat(all_inst_emb, dim=0)
                            inst_logit_sel = torch.cat(all_inst_logit, dim=0)
                            inst_emb_out[:k_actual] = inst_emb_sel
                            inst_logit_out[:k_actual] = inst_logit_sel

                    # Pad esa/pos to K
                    esa_padded = torch.zeros(K)
                    pos_padded = torch.zeros(K)
                    esa_padded[:k_actual] = sel_esa[:k_actual]
                    pos_padded[:k_actual] = sel_pos[:k_actual]

                    writers[n_val].write_pair(
                        int(pair_id),
                        inst_emb=inst_emb_out,
                        inst_logit=inst_logit_out,
                        esa=esa_padded,
                        pos=pos_padded,
                        label=float(label),
                        sel_len=k_actual,
                    )

                pair_count += 1
                if pair_count <= 5 or pair_count % 500 == 0:
                    elapsed = time.time() - t0
                    speed = pair_count / elapsed if elapsed > 0 else 0
                    print(
                        f"[Robustness Fast] pair={pair_count}/{num_pairs} "
                        f"cts={n_cts} speed={speed:.1f} pairs/s "
                        f"elapsed={elapsed:.0f}s"
                    )

    # Finalize
    for n_val, writer in writers.items():
        writer.set_ready()
        print(f"[Robustness Fast] n={n_val} cache finalized")

    total_time = time.time() - t0
    avg_cts = total_candidates / max(1, pair_count)
    print(
        f"[Robustness Fast] DONE pairs={pair_count} avg_cts={avg_cts:.0f} "
        f"total_time={total_time:.0f}s ({total_time/60:.1f}min)"
    )


if __name__ == "__main__":
    main()
