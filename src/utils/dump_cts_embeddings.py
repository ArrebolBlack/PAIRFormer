#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例用法：

python -m src.utils.dump_cts_embeddings \
  --config configs/experiment/miRAW_TargetNet_Optimized.yaml \
  --data-config configs/experiment/miRAW_pair_agg_baseline.yaml \
  --checkpoint checkpoints/miRAW_TargetNet_Optimized_dp-0.1/checkpoints/last.pt \
  --splits train val test \
  --out-root pair_cache \
  --name miRAW_TargetNet_Optimized_dp-0.1_Test_1-5_Test_0,6-9 \
  --data-train data/miRAW_Test1-5_split-ratio-0.9_Train_Validation.txt \
  --data-val   data/miRAW_Test1-5_split-ratio-0.9_Train_Validation.txt \
  --data-test  data/miRAW_Test_0,6-9.txt \
  --cache-root pair_cache \
  --dump-max-cts-per-pair 512 \
  --dump-score-mode logit \
  --max-pairs-per-shard 512 \
  --device cuda:0

输出目录结构（示例）：
  pair_cache/miRAW_TargetNet_dp-0.5_Test_1-5_Test_0/
    train_shard0000.pt
    train_shard0001.pt
    ...
    val_shard0000.pt
    ...

说明：
- --config       ：CTS 单窗口模型的 experiment yaml（提供 model 配置、with_esa 等）。
- --data-config  ：pair-level aggregator 的数据 yaml（只看 data.* 部分，用于 DataConfig/build_dataset_and_loader）。
- 如果不提供 --data-config，则默认仍然使用 --config 里的 data 作为数据配置。
- data.*.path 默认来自 data-config 中的 data.path.{split}，
  但可以通过 --data-train / --data-val / --data-test 显式覆盖，
  用于指定「这次 dump 对应哪个 pair-level dataset 的 split」。
- --out-root 作为“根目录”，脚本会在其下自动创建一个子目录：
    {run_tag}/{split}.pt
- run_tag 的规则：
    1) 若命令行指定 --name，则 run_tag = name；
    2) 否则，run_tag = {train_dataset_stem}__{model_arch}__{ckpt_stem}
"""

import json
import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from omegaconf import OmegaConf, DictConfig

from src.config.data_config import DataConfig
from src.data.builder import build_dataset_and_loader
from src.models.registry import build_model

# from src.models.TargetNet import TargetNet
# from src.models.TargetNet_Optimized import TargetNet_Optimized
# from src.models.TargetNet_transformer import TargetNetTransformer1D

from src.models.extractors import get_embedding_and_logit


import os
import time
import json
import subprocess
import psutil
import torch


def get_total_rss_bytes(proc: psutil.Process) -> int:
    """Main process + all (recursive) children RSS."""
    total = 0
    try:
        total += proc.memory_info().rss
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0

    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total



# =================================================
#  Argument parsing
# =================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dump per-CTS embeddings & logits grouped by pair (set_idx)."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to CTS single-CTS model experiment yaml (e.g. configs/experiment/miRAW_TargetNet.yaml)",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default=None,
        help=(
            "Path to pair-level data yaml（只用于 DataConfig/build_dataset_and_loader）。\n"
            "若不指定，则默认使用 --config 中的 data 作为数据配置。"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained single-CTS model checkpoint (.ckpt / .pth).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Which splits to dump, e.g. --splits train val.",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        required=True,
        help=(
            "Root dir to save dumps, e.g. cache/cts_cache.\n"
            "脚本会在其下自动创建子目录：{run_tag}/{split}_shardXXXX.pt"
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "可选的 run 名称，用作子目录名，例如：miRAW_TargetNet_v1_pos。\n"
            "若不指定，则自动命名为：{train_dataset_stem}__{model_arch}__{ckpt_stem}"
        ),
    )

    # ---------- 显式指定 pair-level dataset 的路径（覆盖 data-config） ----------
    parser.add_argument(
        "--data-train",
        type=str,
        default=None,
        help="Override data_config.data.path.train，用于指定本次 dump 对应的 train pair-level 数据文件。",
    )
    parser.add_argument(
        "--data-val",
        type=str,
        default=None,
        help="Override data_config.data.path.val，用于指定本次 dump 对应的 val pair-level 数据文件。",
    )
    parser.add_argument(
        "--data-test",
        type=str,
        default=None,
        help="Override data_config.data.path.test，用于指定本次 dump 对应的 test pair-level 数据文件。",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run model on (e.g. cuda:0 or cpu).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10240000,
        help="Batch size for dumping.",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default=None,
        help=(
            "Optional cache root passed to build_dataset_and_loader(cache_data_path=...).\n"
            "若不指定，则不传 cache_data_path，让 builder 自己处理。"
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="num_workers passed to build_dataset_and_loader。",
    )
    parser.add_argument(
        "--pin-memory",
        type=int,
        default=1,
        help="是否开启 pin_memory（1=True, 0=False），传给 build_dataset_and_loader。",
    )
    parser.add_argument(
        "--dump-max-cts-per-pair",
        type=int,
        default=512,
        help=(
            "在 dump 阶段对每个 pair 最多保留多少个 CTS（在线 Top-K 截断）。\n"
            "若 <=0 则不截断，可能导致内存占用过大。"
        ),
    )
    parser.add_argument(
        "--dump-score-mode",
        type=str,
        default="logit",
        choices=["logit", "abs_logit", "esa"],
        help=(
            "Top-K 评分依据：\n"
            "  - logit     : 按 logit 从大到小保留；\n"
            "  - abs_logit : 按 |logit| 从大到小保留；\n"
            "  - esa       : 按 ESA score 从大到小保留。"
        ),
    )
    parser.add_argument(
        "--max-pairs-per-shard",
        type=int,
        default=51200,
        help=(
            "每个 shard 文件最多包含多少个 pair。达到该数量就写一个 {split}_shardXXXX.pt 然后清空缓存。\n"
            "根据内存大小调整（越小越省内存，但磁盘上的 pt 文件越多）。"
        ),
    )
    return parser.parse_args()


# =================================================
#  Core: get embedding + logit for a batch
# =================================================

# def get_embedding_and_logit(model, x: torch.Tensor):
#     """
#     根据当前使用的 CTS 模型类型，统一返回:
#       - feat:  [B, d_emb]  作为 CTS embedding（在最终线性层之前）
#       - logit: [B]         作为该 CTS 的打分
#     """
#     if isinstance(model, TargetNet):
#         z = model.stem(x)
#         z = model.stage1(z)
#         z = model.stage2(z)
#         z = model.dropout(model.relu(z))
#         z = model.avg_pool(z)
#         z = z.reshape(z.size(0), -1)              # [B, d_emb]
#         feat = z
#         logit = model.linear(feat).squeeze(-1)    # [B]
#         return feat, logit

#     if isinstance(model, TargetNet_Optimized):
#         z = model.stem(x)
#         for stage in model.stages:
#             z = stage(z)

#         z = model.se(z)
#         z = model.dropout(model.relu(z))
#         z = model.adaptive_pool(z)
#         z = z.reshape(z.size(0), -1)              # [B, d_emb]
#         feat = z
#         logit = model.linear(feat).squeeze(-1)    # [B]
#         return feat, logit

#     if isinstance(model, TargetNetTransformer1D):
#         z = model.input_proj(x)                   # [B, d_model, L]
#         z = z.transpose(1, 2)                     # [B, L, d_model]
#         h = model.encoder(inputs_embeds=z, attn_mask=None)  # [B, L, d_model]
#         h = h.mean(dim=1)                         # [B, d_model]
#         h = model.post_norm(h)                    # [B, d_model]
#         feat = h
#         logit = model.classifier(h).squeeze(-1)   # [B]
#         return feat, logit

#     raise TypeError(
#         f"Unsupported model type for get_embedding_and_logit: {type(model)}. "
#         f"当前只支持 TargetNet / TargetNet_Optimized / TargetNetTransformer1D。"
#     )




# =================================================
#  Load model + weights
# =================================================

def load_model(model_cfg: DictConfig, ckpt_path: str, device: torch.device):
    if ckpt_path is None:
        raise ValueError("[dump_cts_embeddings] ckpt_path 不能为空，请通过 --checkpoint 指定。")

    ckpt_path = str(Path(ckpt_path))
    print(f"[dump_cts_embeddings] Loading checkpoint from {ckpt_path}")

    data_cfg_for_model = DataConfig.from_omegaconf(model_cfg.data)

    model_name = model_cfg.model.get("arch", model_cfg.model.get("name"))
    model = build_model(model_name, model_cfg.model, data_cfg=data_cfg_for_model)
    model.to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    if not isinstance(ckpt, dict):
        raise TypeError(
            f"[dump_cts_embeddings] Expected checkpoint to be a dict, got {type(ckpt)}. "
            "请确认该权重是由 Trainer.save_checkpoint 保存的。"
        )

    if "state_dict" not in ckpt:
        print(
            "[dump_cts_embeddings] Warning: checkpoint has no 'state_dict' key, "
            "尝试将整个 ckpt 作为 state_dict 加载。"
        )
        state_dict = ckpt
    else:
        state_dict = ckpt["state_dict"]

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("model."):
            new_k = new_k[len("model."):]
        if new_k.startswith("net."):
            new_k = new_k[len("net."):]
        cleaned_state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)

    if missing:
        print(f"[dump_cts_embeddings] Warning: missing keys when loading state_dict: {missing}")
    if unexpected:
        print(f"[dump_cts_embeddings] Warning: unexpected keys in state_dict: {unexpected}")

    model.eval()
    return model


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# =================================================
#  Helper: 在线 Top-K 更新
# =================================================

def _compute_score(score_mode: str, logit: torch.Tensor, esa: torch.Tensor) -> float:
    if score_mode == "logit":
        return float(logit.item())
    elif score_mode == "abs_logit":
        return float(logit.abs().item())
    elif score_mode == "esa":
        return float(esa.item())
    else:
        raise ValueError(f"Unknown dump_score_mode: {score_mode}")


def _new_pair_entry() -> Dict[str, Any]:
    return {
        "embeddings": [],
        "logits": [],
        "esa": [],
        "pos": [],
        "scores": [],
        "min_score": None,
        "min_idx": -1,
    }


def _online_update_entry(
    entry: Dict[str, Any],
    emb_i: torch.Tensor,
    logit_i: torch.Tensor,
    esa_i: torch.Tensor,
    pos_i: Optional[torch.Tensor],
    max_cts_per_pair: int,
    score_mode: str,
):
    """
    对“当前 pair”的 entry 做在线 Top-K 更新。
    """
    if max_cts_per_pair is None or max_cts_per_pair <= 0:
        entry["embeddings"].append(emb_i)
        entry["logits"].append(logit_i)
        entry["esa"].append(esa_i)
        if pos_i is not None:
            entry["pos"].append(pos_i)
        return

    score_val = _compute_score(score_mode, logit_i, esa_i)
    scores = entry["scores"]
    L = len(scores)

    if L < max_cts_per_pair:
        entry["embeddings"].append(emb_i)
        entry["logits"].append(logit_i)
        entry["esa"].append(esa_i)
        if pos_i is not None:
            entry["pos"].append(pos_i)
        scores.append(score_val)

        if L == 0 or entry["min_score"] is None or score_val < entry["min_score"]:
            entry["min_score"] = score_val
            entry["min_idx"] = L
        return

    # 已有 K 个，只有更好的才替换
    if entry["min_score"] is not None and score_val <= entry["min_score"]:
        return

    j = entry["min_idx"]
    entry["embeddings"][j] = emb_i
    entry["logits"][j] = logit_i
    entry["esa"][j] = esa_i
    if pos_i is not None and len(entry["pos"]) == L:
        entry["pos"][j] = pos_i
    scores[j] = score_val

    # 重新扫描找最小
    min_score = scores[0]
    min_idx = 0
    for idx_, s in enumerate(scores):
        if s < min_score:
            min_score = s
            min_idx = idx_
    entry["min_score"] = min_score
    entry["min_idx"] = min_idx


# =================================================
#  Dump a single split (streaming per-pair + shard 写盘)
# =================================================

def dump_split(
    data_cfg_root: DictConfig,
    model: torch.nn.Module,
    split: str,
    out_root: str,
    device: torch.device,
    batch_size: int,
    data_path_overrides: dict,
    cache_root: str,
    num_workers: int,
    pin_memory: bool,
    dump_max_cts_per_pair: int,
    dump_score_mode: str,
    max_pairs_per_shard: int,
):
    """
    流式处理 CTS 样本，按 set_idx 聚合为 pair，并对每个 pair 在线做 Top-K，
    按 shard 写盘，避免内存爆炸。

    输出（按 split）：
      out_root/{split}_shard0000.pt, {split}_shard0001.pt, ...
      out_root/{split}_meta.json

    每个 shard 文件的内容：
      {
        pid0: {
          "embeddings": Tensor [M_i, d_emb],
          "logits":     Tensor [M_i],
          "esa":        Tensor [M_i],
          "pos":        Tensor [M_i] or None,
          "label":      float,
        },
        pid1: { ... },
        ...
      }

    meta.json：
      [
        { "path": "train_shard0000.pt", "num_pairs": 1734 },
        { "path": "train_shard0001.pt", "num_pairs": 1820 },
        ...
      ]
    """
    global peak_rss


    print(f"\n===== Dumping split: {split} =====")

    # ---- 构建 DataConfig ----
    data_cfg = DataConfig.from_omegaconf(data_cfg_root.data)

    override_path = data_path_overrides.get(split, None)
    if override_path is not None:
        if not hasattr(data_cfg, "path") or data_cfg.path is None:
            data_cfg.path = {}
        data_cfg.path[split] = override_path

    if hasattr(data_cfg, "path") and split in getattr(data_cfg, "path", {}):
        data_file = str(data_cfg.path[split])
    else:
        data_file = "<unknown>"
    print(f"[dump_split:{split}] Using data file: {data_file}")

    # ---- Dataset + DataLoader ----
    dataset, loader = build_dataset_and_loader(
        data_cfg=data_cfg,
        split_idx=split,
        cache_data_path=cache_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=False,
    )

    ensure_dir(out_root)

    # 当前 shard 里的 pair -> entry（已经是 Tensor 的 dict）
    shard_pairs: Dict[int, Dict[str, Any]] = {}
    shard_meta = []
    shard_idx = 0
    num_total_pairs = 0

    # streaming 当前正在累积的 pair
    current_pid = None
    current_entry = None   # {"embeddings": [], "logits": [], "esa": [], "pos": [], "scores": [], "min_score": ..., "min_idx": ...}
    current_label = None

    def _new_pair_entry():
        return {
            "embeddings": [],
            "logits": [],
            "esa": [],
            "pos": [],
            "scores": [],
            "min_score": None,
            "min_idx": -1,
        }

    def flush_current_pair():
        nonlocal current_pid, current_entry, current_label
        nonlocal shard_idx, shard_pairs, shard_meta, num_total_pairs

        if current_pid is None or current_entry is None:
            return

        if len(current_entry["embeddings"]) == 0:
            print(f"[WARN] pair {current_pid} has 0 CTS after filtering, skip.")
        else:
            emb = torch.stack(current_entry["embeddings"], dim=0)   # [M_i, d_emb]
            logits = torch.stack(current_entry["logits"], dim=0)    # [M_i]
            esa = torch.stack(current_entry["esa"], dim=0)          # [M_i]

            if len(current_entry["pos"]) > 0:
                pos_tensor = torch.stack(current_entry["pos"], dim=0)   # [M_i]
            else:
                pos_tensor = None

            shard_pairs[current_pid] = {
                "embeddings": emb,
                "logits": logits,
                "esa": esa,
                "pos": pos_tensor,
                "label": current_label,
            }
            num_total_pairs += 1

        # shard 满了就写盘并清空
        if len(shard_pairs) >= max_pairs_per_shard:
            shard_name = f"{split}_shard{shard_idx:04d}.pt"
            out_path = os.path.join(out_root, shard_name)
            torch.save(shard_pairs, out_path)
            shard_meta.append({
                "path": shard_name,
                "num_pairs": len(shard_pairs),
            })
            print(
                f"[dump_split:{split}] Saved shard {shard_idx} with "
                f"{len(shard_pairs)} pairs to {out_path}"
            )
            shard_pairs.clear()
            shard_idx += 1

        current_pid = None
        current_entry = None
        current_label = None

    model.eval()
    last_seen_pid = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if not isinstance(batch, dict):
                raise TypeError(
                    f"[dump_split] Expected batch to be dict from cts_collate_fn, "
                    f"got {type(batch)}. 请确认 build_dataset_and_loader 是否被正确使用。"
                )

            x = batch["inputs"].to(device)          # [B, C, L]
            y = batch["labels"].view(-1)            # [B]
            set_idx = batch["set_idx"].view(-1)     # [B]

            if "esa_scores" in batch:
                esa_score = batch["esa_scores"].view(-1)
            else:
                esa_score = torch.zeros_like(y)

            pos = batch.get("pos", None)
            if pos is not None:
                pos = pos.view(-1)

            feats, logits = get_embedding_and_logit(model, x)

            feats = feats.detach().cpu()
            logits = logits.detach().cpu()
            y = y.detach().cpu()
            set_idx = set_idx.detach().cpu().long()
            esa_score = esa_score.detach().cpu()
            if pos is not None:
                pos = pos.detach().cpu()

            B = feats.size(0)
            for i in range(B):
                pid = int(set_idx[i].item())

                # 断言 set_idx 单调不减，保证 streaming 正确
                if last_seen_pid is not None and pid < last_seen_pid:
                    raise RuntimeError(
                        f"[dump_split:{split}] Detected non-monotonic set_idx: "
                        f"previous pid={last_seen_pid}, current pid={pid}. "
                        f"当前 streaming 写法要求 ChunkedCTSDataset 中 set_idx 单调不减。"
                    )
                last_seen_pid = pid

                emb_i = feats[i]
                logit_i = logits[i]
                esa_i = esa_score[i]
                label_i = float(y[i].item())
                pos_i = pos[i] if pos is not None else None

                # 遇到新 pair：先 flush 旧 pair
                if current_pid is None:
                    current_pid = pid
                    current_entry = _new_pair_entry()
                    current_label = label_i
                elif pid != current_pid:
                    flush_current_pair()
                    current_pid = pid
                    current_entry = _new_pair_entry()
                    current_label = label_i
                else:
                    if current_label is not None and current_label != label_i:
                        print(
                            f"[WARN] pair {pid} has inconsistent labels in streaming: "
                            f"{current_label} vs {label_i}"
                        )

                # 在线 Top-K 更新
                _online_update_entry(
                    entry=current_entry,
                    emb_i=emb_i,
                    logit_i=logit_i,
                    esa_i=esa_i,
                    pos_i=pos_i,
                    max_cts_per_pair=dump_max_cts_per_pair,
                    score_mode=dump_score_mode,
                )

            if (batch_idx + 1) % 50 == 0:
                print(f"  [dump_split:{split}] Processed {(batch_idx + 1) * B} samples...")

            if (batch_idx + 1) % 10 == 0:
                rss = get_total_rss_bytes(proc)
                peak_rss = max(peak_rss, rss)




        # 所有 batch 结束，flush 最后一个 pair
        flush_current_pair()

    # 把最后一个 shard 写盘
    if len(shard_pairs) > 0:
        shard_name = f"{split}_shard{shard_idx:04d}.pt"
        out_path = os.path.join(out_root, shard_name)
        torch.save(shard_pairs, out_path)
        shard_meta.append({
            "path": shard_name,
            "num_pairs": len(shard_pairs),
        })
        print(
            f"[dump_split:{split}] Saved final shard {shard_idx} with "
            f"{len(shard_pairs)} pairs to {out_path}"
        )

    # 写 meta.json
    meta_path = os.path.join(out_root, f"{split}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(shard_meta, f)
    print(f"[dump_split:{split}] Wrote meta file to {meta_path}")

    print(f"[dump_split:{split}] Total pairs dumped: {num_total_pairs}")
    print(f"[dump_split:{split}] Shards written   : {len(shard_meta)}")





# =================================================
#  Main
# =================================================

def main():

    global t0, proc, peak_rss
    t0 = time.perf_counter()
    proc = psutil.Process(os.getpid())
    peak_rss = 0
    
    args = parse_args()

    model_cfg = OmegaConf.load(args.config)

    if args.data_config is not None:
        data_cfg_root = OmegaConf.load(args.data_config)
    else:
        data_cfg_root = model_cfg

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()

    model = load_model(model_cfg, args.checkpoint, device)


    model_arch = getattr(model, "_registry_name", model.__class__.__name__)

    if args.data_train is not None:
        train_path_for_tag = args.data_train
    else:
        try:
            if hasattr(data_cfg_root, "data"):
                data_cfg_tmp = DataConfig.from_omegaconf(data_cfg_root.data)
                if hasattr(data_cfg_tmp, "path") and "train" in getattr(data_cfg_tmp, "path", {}):
                    train_path_for_tag = str(data_cfg_tmp.path["train"])
                else:
                    train_path_for_tag = "<unknown>"
            else:
                train_path_for_tag = "<unknown>"
        except Exception:
            train_path_for_tag = "<unknown>"

    if train_path_for_tag not in ("<unknown>", "", None):
        train_tag = Path(train_path_for_tag).stem
    else:
        train_tag = "no-train"

    ckpt_tag = Path(args.checkpoint).stem

    if args.name is not None:
        run_tag = args.name
    else:
        run_tag = f"{train_tag}__{model_arch}__{ckpt_tag}"

    out_root = os.path.join(args.out_root, run_tag)
    ensure_dir(out_root)

    data_path_overrides = {
        "train": args.data_train,
        "val": args.data_val,
        "test": args.data_test,
    }

    cache_root = args.cache_root
    num_workers = int(args.num_workers)
    pin_memory = bool(args.pin_memory)

    print("\n[dump_cts_embeddings] ========== SUMMARY ==========")
    print(f"  Model config            : {args.config}")
    print(f"  Data  config            : {args.data_config}")
    print(f"  Model arch              : {model_arch}")
    print(f"  Checkpoint              : {args.checkpoint} (tag={ckpt_tag})")
    print(f"  Pair train path         : {train_path_for_tag}")
    print(f"  Run tag                 : {run_tag}")
    print(f"  Out root                : {out_root}")
    print(f"  Splits                  : {args.splits}")
    print(f"  Cache root              : {cache_root}")
    print(f"  num_workers             : {num_workers}")
    print(f"  pin_memory              : {pin_memory}")
    print(f"  dump_max_cts_per_pair   : {args.dump_max_cts_per_pair}")
    print(f"  dump_score_mode         : {args.dump_score_mode}")
    print(f"  max_pairs_per_shard     : {args.max_pairs_per_shard}")
    print(f"  Override paths          :")
    for k, v in data_path_overrides.items():
        print(f"      {k}: {v}")
    print("====================================================\n")

    for split in args.splits:
        dump_split(
            data_cfg_root=data_cfg_root,
            model=model,
            split=split,
            out_root=out_root,
            device=device,
            batch_size=args.batch_size,
            data_path_overrides=data_path_overrides,
            cache_root=cache_root,
            num_workers=num_workers,
            pin_memory=pin_memory,
            dump_max_cts_per_pair=args.dump_max_cts_per_pair,
            dump_score_mode=args.dump_score_mode,
            max_pairs_per_shard=args.max_pairs_per_shard,
        )

    elapsed_s = time.perf_counter() - t0
    peak_cpu_gb = peak_rss / (1024 ** 3)

    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # out_root = 你 dump 输出目录（pair_cache/...）
    du_bytes = int(subprocess.check_output(["du", "-sb", str(out_root)]).split()[0])
    disk_gb = du_bytes / (1024 ** 3)

    summary = {
        "dump_elapsed_min": elapsed_s / 60.0,
        "dump_peak_vram_gb": peak_vram_gb,
        "dump_peak_cpu_rss_gb": peak_cpu_gb,
        "dump_disk_gb": disk_gb,
        "dump_out_root": str(out_root),
    }
    print("[EFF_BREAKDOWN_DUMP]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
