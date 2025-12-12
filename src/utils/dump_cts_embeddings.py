#!/usr/bin/env python

"""
示例用法：

python -m src.utils.dump_cts_embeddings \
  --config configs/experiment/miRAW_TargetNet.yaml \
  --data-config configs/experiment/miRAW_pair_agg_baseline.yaml \
  --checkpoint checkpoints/miRAW_TargetNet_dp-0.5/checkpoints/last.pt \
  --splits train val test \
  --out-root pair_cache \
  --name miRAW_TargetNet_dp-0.5_Test_1-5_Test_0 \
  --data-train data/miRAW_Test1-5_split-ratio-0.9_Train_Validation.txt \
  --data-val   data/miRAW_Test1-5_split-ratio-0.9_Train_Validation.txt \
  --data-test  data/miRAW_Test0.txt \
  --cache-root pair_cache \
  --dump-max-cts-per-pair 512 \
  --dump-score-mode logit
  --device cuda:0

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

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any

import torch
from omegaconf import OmegaConf, DictConfig

from src.config.data_config import DataConfig
from src.data.builder import build_dataset_and_loader
from src.models.registry import build_model

from src.models.TargetNet import TargetNet
from src.models.TargetNet_Optimized import TargetNet_Optimized
from src.models.TargetNet_transformer import TargetNetTransformer1D


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
            "脚本会在其下自动创建子目录：{run_tag}/{split}.pt"
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
        default=256,
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
        default=4,
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
        default=512, # -1 
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
    return parser.parse_args()


# =================================================
#  Core: get embedding + logit for a batch
# =================================================

def get_embedding_and_logit(model, x: torch.Tensor):
    """
    根据当前使用的 CTS 模型类型，统一返回:
      - feat:  [B, d_emb]  作为 CTS embedding（在最终线性层之前）
      - logit: [B]         作为该 CTS 的打分

    支持的模型:
      - TargetNet
      - TargetNet_Optimized（当前新版）
      - TargetNetTransformer1D
    """

    # ---------------- TargetNet ----------------
    if isinstance(model, TargetNet):
        z = model.stem(x)
        z = model.stage1(z)
        z = model.stage2(z)
        z = model.dropout(model.relu(z))
        z = model.avg_pool(z)
        z = z.reshape(z.size(0), -1)              # [B, d_emb]
        feat = z
        logit = model.linear(feat).squeeze(-1)    # [B]
        return feat, logit

    # ---------------- TargetNet_Optimized ----------------
    if isinstance(model, TargetNet_Optimized):
        # 新版 TargetNet_Optimized.forward 拆解
        z = model.stem(x)
        for stage in model.stages:
            z = stage(z)

        z = model.se(z)
        z = model.dropout(model.relu(z))
        z = model.adaptive_pool(z)
        z = z.reshape(z.size(0), -1)              # [B, d_emb]
        feat = z
        logit = model.linear(feat).squeeze(-1)    # [B]
        return feat, logit

    # ---------------- TargetNetTransformer1D ----------------
    if isinstance(model, TargetNetTransformer1D):
        z = model.input_proj(x)                   # [B, d_model, L]
        z = z.transpose(1, 2)                     # [B, L, d_model]
        h = model.encoder(inputs_embeds=z, attn_mask=None)  # [B, L, d_model]
        h = h.mean(dim=1)                         # [B, d_model]
        h = model.post_norm(h)                    # [B, d_model]
        feat = h
        logit = model.classifier(h).squeeze(-1)   # [B]
        return feat, logit

    # ---------------- 其他未支持模型 ----------------
    raise TypeError(
        f"Unsupported model type for get_embedding_and_logit: {type(model)}. "
        f"当前只支持 TargetNet / TargetNet_Optimized / TargetNetTransformer1D。"
    )


# =================================================
#  Load model + weights
# =================================================

def load_model(model_cfg: DictConfig, ckpt_path: str, device: torch.device):
    """
    从 model_cfg 和 checkpoint 路径构建并加载单 CTS 模型。

    对齐 Trainer.save_checkpoint/Trainer.load_checkpoint 的约定：
    - checkpoint 必须是一个 dict，且包含 "state_dict" 键；
    - "state_dict" 的 key 与 model.state_dict() 对齐（若将来加前缀，这里负责剥离）。
    """
    if ckpt_path is None:
        # 在脚本层面，--checkpoint 已经是 required，这里只是防御性检查
        raise ValueError("[dump_cts_embeddings] ckpt_path 不能为空，请通过 --checkpoint 指定。")

    ckpt_path = str(Path(ckpt_path))
    print(f"[dump_cts_embeddings] Loading checkpoint from {ckpt_path}")

    # ---- 1) 构建 data_cfg（仅为构建模型提供必要信息，例如 with_esa / 输入通道数）---- #
    data_cfg_for_model = DataConfig.from_omegaconf(model_cfg.data)

    # ---- 2) 构建模型结构（不带权重）---- #
    model_name = model_cfg.model.get("arch", model_cfg.model.get("name"))
    model = build_model(model_name, model_cfg.model, data_cfg=data_cfg_for_model)
    model.to(device)

    # ---- 3) 读取 checkpoint，并取出 state_dict ---- #
    ckpt = torch.load(ckpt_path, map_location=device)

    if not isinstance(ckpt, dict):
        raise TypeError(
            f"[dump_cts_embeddings] Expected checkpoint to be a dict, got {type(ckpt)}. "
            "请确认该权重是由 Trainer.save_checkpoint 保存的。"
        )

    if "state_dict" not in ckpt:
        # 理论上按照 Trainer.save_checkpoint，一定会有 "state_dict"
        # 这里做个兜底：如果用户自己用 torch.save(model.state_dict()) 存的，也尽量支持。
        print(
            "[dump_cts_embeddings] Warning: checkpoint has no 'state_dict' key, "
            "尝试将整个 ckpt 作为 state_dict 加载。"
        )
        state_dict = ckpt
    else:
        state_dict = ckpt["state_dict"]

    # ---- 4) 处理可能存在的前缀（如 'model.', 'net.'），以适配封装模型的情况 ---- #
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("model."):
            new_k = new_k[len("model."):]
        if new_k.startswith("net."):
            new_k = new_k[len("net."):]
        cleaned_state_dict[new_k] = v

    # ---- 5) 加载权重 ---- #
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
    """
    根据 score_mode 计算当前 CTS 的打分（Python float，用于 Top-K）。
    """
    if score_mode == "logit":
        return float(logit.item())
    elif score_mode == "abs_logit":
        return float(logit.abs().item())
    elif score_mode == "esa":
        return float(esa.item())
    else:
        raise ValueError(f"Unknown dump_score_mode: {score_mode}")


def _online_update_entry(
    entry: Dict[str, Any],
    emb_i: torch.Tensor,
    logit_i: torch.Tensor,
    esa_i: torch.Tensor,
    pos_i: torch.Tensor,
    max_cts_per_pair: int,
    score_mode: str,
):
    """
    对某个 pair 的缓存 entry 做“在线 Top-K 更新”。

    entry 结构：
      {
        "embeddings": [Tensor, ...],
        "logits":     [Tensor, ...],
        "esa":        [Tensor, ...],
        "pos":        [Tensor, ...],
        "label":      float or None,
        "scores":     [float, ...],
        "min_score":  float or None,
        "min_idx":    int,
      }

    max_cts_per_pair > 0 时：
      - 始终保证 len(entry["embeddings"]) <= max_cts_per_pair
      - score_mode 控制评分依据（logit / abs_logit / esa）
    """
    if max_cts_per_pair is None or max_cts_per_pair <= 0:
        # 不做截断，直接累加（可能非常占内存）
        entry["embeddings"].append(emb_i)
        entry["logits"].append(logit_i)
        entry["esa"].append(esa_i)
        if pos_i is not None:
            entry["pos"].append(pos_i)
        # scores/min_* 仅在启用 top-k 时有意义
        return

    score_val = _compute_score(score_mode, logit_i, esa_i)
    scores = entry["scores"]
    L = len(scores)

    if L < max_cts_per_pair:
        # 还没填满 K 个，直接追加
        entry["embeddings"].append(emb_i)
        entry["logits"].append(logit_i)
        entry["esa"].append(esa_i)
        if pos_i is not None:
            entry["pos"].append(pos_i)
        scores.append(score_val)

        # 更新当前最小值
        if L == 0 or score_val < entry["min_score"]:
            entry["min_score"] = score_val
            entry["min_idx"] = L
        return

    # 已经有 K 个，只有当新样本“更好”时才替换现有最差样本
    if score_val <= entry["min_score"]:
        # 新样本不如当前最差，忽略
        return

    # 用新样本替换当前最差位置
    j = entry["min_idx"]
    entry["embeddings"][j] = emb_i
    entry["logits"][j] = logit_i
    entry["esa"][j] = esa_i
    if pos_i is not None:
        if len(entry["pos"]) == L:
            entry["pos"][j] = pos_i
        else:
            # 理论上 pos 和 embeddings 长度应一致，这里只是防御
            pass
    scores[j] = score_val

    # 重新扫描，更新 min_score / min_idx（代价 O(K)，但只在发生替换时调用）
    min_score = scores[0]
    min_idx = 0
    for idx_, s in enumerate(scores):
        if s < min_score:
            min_score = s
            min_idx = idx_
    entry["min_score"] = min_score
    entry["min_idx"] = min_idx


# =================================================
#  Dump a single split
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
):
    """
    对给定 split（train / val / test）：
      - 遍历 ChunkedCTSDataset（CTS 级 sample）
      - 用单 CTS 模型生成 embedding + logit
      - 在线按 set_idx 做 Top-K 聚合到 pair 级别
      - 存成一个 dict[ pair_id ] = {...} 的 .pt 文件

    存储结构（v1）：
      out_pairs[pid] = {
          "embeddings": Tensor [M_i, d_emb],   # CTS embedding（线性层前），M_i <= dump_max_cts_per_pair
          "logits":     Tensor [M_i],          # 单 CTS logit
          "esa":        Tensor [M_i],          # ESA 分数
          "pos":        Tensor [M_i] 或 None,  # 归一化位置（若 Dataset 提供）
          "label":      float (0. / 1.)
      }

    Dataset 约定（v1）：
      - DataLoader 使用 cts_collate_fn，batch 为 dict：
            {
              "inputs": FloatTensor [B, C, L],
              "labels": FloatTensor [B],
              "set_idx": LongTensor  [B],
              "esa_scores": FloatTensor [B] 可选,
              "pos": FloatTensor [B] 可选,
            }
    """
    print(f"\n===== Dumping split: {split} =====")

    # ---- 构建 DataConfig ----
    data_cfg = DataConfig.from_omegaconf(data_cfg_root.data)

    # 如果命令行对该 split 指定了 pair-level 数据路径，则覆盖
    override_path = data_path_overrides.get(split, None)
    if override_path is not None:
        if not hasattr(data_cfg, "path") or data_cfg.path is None:
            data_cfg.path = {}
        data_cfg.path[split] = override_path

    # 打印一下当前 split 用的是哪个数据文件
    if hasattr(data_cfg, "path") and split in getattr(data_cfg, "path", {}):
        data_file = str(data_cfg.path[split])
    else:
        data_file = "<unknown>"
    print(f"[dump_split:{split}] Using data file: {data_file}")

    # ---- 通过 builder 构建 dataset + dataloader（与 train.py 对齐）---- #
    dataset, loader = build_dataset_and_loader(
        data_cfg=data_cfg,
        split_idx=split,
        cache_data_path=cache_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,      # dump 不需要 shuffle
        drop_last=False,
    )

    # pair_id -> 累积内容（在线 Top-K）
    pair_dict = defaultdict(lambda: {
        "embeddings": [],
        "logits": [],
        "esa": [],
        "pos": [],
        "label": None,
        "scores": [],      # 仅用于 Top-K
        "min_score": None,
        "min_idx": -1,
    })

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # 这里假定 loader 一定用了 cts_collate_fn
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

            # ---- 得到 embedding + logit ----
            feats, logits = get_embedding_and_logit(model, x)
            # feats: [B, d_emb], logits: [B]

            feats = feats.detach().cpu()
            logits = logits.detach().cpu()
            y = y.detach().cpu()
            set_idx = set_idx.detach().cpu().long()
            esa_score = esa_score.detach().cpu()
            if pos is not None:
                pos = pos.detach().cpu()

            B = feats.size(0)
            for i in range(B):
                pid = int(set_idx[i].item())       # pair id（你的 set_idx 即 pair 编号）
                emb_i = feats[i]                   # [d_emb]
                logit_i = logits[i]                # scalar tensor
                esa_i = esa_score[i]               # scalar tensor
                label_i = float(y[i].item())       # scalar float
                pos_i = pos[i] if pos is not None else None  # scalar tensor 或 None

                entry = pair_dict[pid]

                # 在线 Top-K 更新（控制每个 pair 内最多保留 K 个 CTS）
                _online_update_entry(
                    entry=entry,
                    emb_i=emb_i,
                    logit_i=logit_i,
                    esa_i=esa_i,
                    pos_i=pos_i,
                    max_cts_per_pair=dump_max_cts_per_pair,
                    score_mode=dump_score_mode,
                )

                # 确保同一个 pair 的 label 一致
                if entry["label"] is None:
                    entry["label"] = label_i
                else:
                    if entry["label"] != label_i:
                        print(
                            f"[WARN] pair {pid} has inconsistent labels: "
                            f"{entry['label']} vs {label_i}"
                        )

            if (batch_idx + 1) % 50 == 0:
                print(f"  [dump_split:{split}] Processed {(batch_idx + 1) * B} samples...")

    # ---- 把 list 转成 tensor，并构建最终 dict ----
    out_pairs: Dict[int, Dict[str, Any]] = {}
    for pid, entry in pair_dict.items():
        if len(entry["embeddings"]) == 0:
            # 正常情况下不应该出现；若某个 pair 完全没有窗口，直接跳过或报错
            print(f"[WARN] pair {pid} has 0 CTS after filtering, skip.")
            continue

        emb = torch.stack(entry["embeddings"], dim=0)   # [M_i, d_emb]
        logits = torch.stack(entry["logits"], dim=0)    # [M_i]
        esa = torch.stack(entry["esa"], dim=0)          # [M_i]

        # 如果 Dataset 提供了 pos，则这里应该有同样长度的 list
        if len(entry["pos"]) > 0:
            pos_tensor = torch.stack(entry["pos"], dim=0)   # [M_i]
        else:
            pos_tensor = None   # 兼容没有 pos 的情况

        out_pairs[pid] = {
            "embeddings": emb,          # [M_i, d_emb]
            "logits": logits,           # [M_i]
            "esa": esa,                 # [M_i]
            "pos": pos_tensor,          # [M_i] or None
            "label": entry["label"],    # scalar float
        }

    ensure_dir(out_root)
    out_path = os.path.join(out_root, f"{split}.pt")
    torch.save(out_pairs, out_path)

    # 简单打印一个示例 shape，方便 sanity check
    any_pid = next(iter(out_pairs))
    print(f"[dump_split:{split}] Example emb shape: {out_pairs[any_pid]['embeddings'].shape}")
    print(
        f"[dump_split:{split}] Example pos type: "
        f"{type(out_pairs[any_pid]['pos'])}, "
        f"shape={None if out_pairs[any_pid]['pos'] is None else out_pairs[any_pid]['pos'].shape}"
    )
    print(f"[dump_split:{split}] Num pairs: {len(out_pairs)}")
    print(f"[dump_split:{split}] Saved {len(out_pairs)} pairs to {out_path}")


# =================================================
#  Main
# =================================================

def main():
    args = parse_args()

    # 1) 读 CTS 模型的 experiment 配置
    model_cfg = OmegaConf.load(args.config)

    # 2) 读 pair-level 数据配置（若未提供，则退回 model_cfg）
    if args.data_config is not None:
        data_cfg_root = OmegaConf.load(args.data_config)
    else:
        data_cfg_root = model_cfg

    # 3) 加载模型（用 CTS experiment cfg）
    device = torch.device(args.device)
    model = load_model(model_cfg, args.checkpoint, device)

    # 4) 推断模型名字（优先 registry 名，其次类名）
    model_arch = getattr(model, "_registry_name", model.__class__.__name__)

    # 5) 为 run_tag 决定一个「数据集代表路径」（优先使用显式的 --data-train）
    if args.data_train is not None:
        train_path_for_tag = args.data_train
    else:
        # 尝试从 data_cfg_root.data.path.train 取
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

    # 6) checkpoint tag
    ckpt_tag = Path(args.checkpoint).stem

    # 7) run_tag（目录名）
    if args.name is not None:
        run_tag = args.name
    else:
        # 默认：pair-dataset 文件名 + 模型名 + ckpt 名
        # 形如：miRAW_Test1-5_split-ratio-0.9_Train_Validation__TargetNet_Optimized__last
        run_tag = f"{train_tag}__{model_arch}__{ckpt_tag}"

    # 8) 最终的 out_root：只包含一个清晰可读的子目录
    out_root = os.path.join(args.out_root, run_tag)
    ensure_dir(out_root)

    # 9) 收集 data path 的 override 信息
    data_path_overrides = {
        "train": args.data_train,
        "val": args.data_val,
        "test": args.data_test,
    }

    # 10) cache / dataloader 参数
    cache_root = args.cache_root  # 可以是 None
    num_workers = int(args.num_workers)
    pin_memory = bool(args.pin_memory)

    print("\n[dump_cts_embeddings] ========== SUMMARY ==========")
    print(f"  Model config        : {args.config}")
    print(f"  Data  config        : {args.data_config}")
    print(f"  Model arch          : {model_arch}")
    print(f"  Checkpoint          : {args.checkpoint} (tag={ckpt_tag})")
    print(f"  Pair train path     : {train_path_for_tag}")
    print(f"  Run tag             : {run_tag}")
    print(f"  Out root            : {out_root}")
    print(f"  Splits              : {args.splits}")
    print(f"  Cache root          : {cache_root}")
    print(f"  num_workers         : {num_workers}")
    print(f"  pin_memory          : {pin_memory}")
    print(f"  dump_max_cts_per_pair: {args.dump_max_cts_per_pair}")
    print(f"  dump_score_mode     : {args.dump_score_mode}")
    print(f"  Override paths      :")
    for k, v in data_path_overrides.items():
        print(f"      {k}: {v}")
    print("====================================================\n")

    # 11) 对每个 split 做 dump
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
        )


if __name__ == "__main__":
    main()
