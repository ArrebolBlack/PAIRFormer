#!/usr/bin/env python

"""
示例用法：

python scripts/dump_cts_embeddings.py \
  --config configs/experiment/miRAW_TargetNet_baseline.yaml \
  --checkpoint checkpoints/miRAW_TargetNet_best.ckpt \
  --splits train val test \
  --out-root cache/cts_cache \
  --device cuda:0

注意：现在 --out-root 作为“根目录”，脚本会在其下自动创建
  arch=.../data=.../ckpt=... 这一套层级，
从而在路径本身就包含“用的是哪个模型 + 哪个数据集 + 哪个 checkpoint”。
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path
import hashlib

import torch
from torch.utils.data import DataLoader
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
        help="Path to experiment yaml (e.g. configs/experiment/miRAW_TargetNet_baseline.yaml)",
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
            "脚本会在其下自动创建子目录："
            "arch=.../data=.../ckpt=.../{split}.pt"
        ),
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
    return parser.parse_args()


# =================================================
#  Core: get embedding + logit for a batch
# =================================================

def get_embedding_and_logit(model, x: torch.Tensor):
    """
    根据当前使用的 CTS 模型类型，统一返回:
      - feat: [B, d_emb]  作为 CTS embedding
      - logit: [B]        作为该 CTS 的打分

    支持的模型:
      - TargetNet
      - TargetNet_Optimized
      - TargetNetTransformer1D
    """

    # ---------------- TargetNet ----------------
    if isinstance(model, TargetNet):
        # 对应 TargetNet.forward 的拆解：
        # x = self.stem(x)
        # x = self.stage1(x)
        # x = self.stage2(x)
        # x = self.dropout(self.relu(x))
        # x = self.avg_pool(x)
        # x = x.reshape(len(x), -1)
        # x = self.linear(x)
        # return x.squeeze(-1)

        z = model.stem(x)
        z = model.stage1(z)
        z = model.stage2(z)
        z = model.dropout(model.relu(z))
        z = model.avg_pool(z)
        z = z.reshape(z.size(0), -1)     # [B, d_emb]
        feat = z
        logit = model.linear(feat).squeeze(-1)  # [B]
        return feat, logit

    # ---------------- TargetNet_Optimized ----------------
    if isinstance(model, TargetNet_Optimized):
        # 对应 TargetNet_Optimized.forward 的拆解：
        # x = self.stem(x)
        # for stage in self.stages:
        #     x = stage(x)
        # x = self.se(x)
        # x = self.dropout(self.relu(x))
        # x = self.adaptive_pool(x)
        # x = x.reshape(x.size(0), -1)
        # x = self.linear(x)
        # return x.squeeze(-1)

        z = model.stem(x)
        for stage in model.stages:
            z = stage(z)
        z = model.se(z)
        z = model.dropout(model.relu(z))
        z = model.adaptive_pool(z)
        z = z.reshape(z.size(0), -1)     # [B, d_emb]
        feat = z
        logit = model.linear(feat).squeeze(-1)  # [B]
        return feat, logit

    # ---------------- TargetNetTransformer1D ----------------
    if isinstance(model, TargetNetTransformer1D):
        # 对应 TargetNetTransformer1D.forward 的拆解：
        # x = self.input_proj(x)          # [B, C, L] -> [B, d_model, L]
        # x = x.transpose(1, 2)           # [B, L, d_model]
        # h = self.encoder(inputs_embeds=x, attn_mask=None)  # [B, L, d_model]
        # h = h.mean(dim=1)               # [B, d_model]
        # h = self.post_norm(h)
        # logits = self.classifier(h).squeeze(-1)

        z = model.input_proj(x)          # [B, d_model, L]
        z = z.transpose(1, 2)            # [B, L, d_model]
        h = model.encoder(inputs_embeds=z, attn_mask=None)  # [B, L, d_model]
        h = h.mean(dim=1)                # [B, d_model]
        h = model.post_norm(h)           # [B, d_model]
        feat = h
        logit = model.classifier(h).squeeze(-1)  # [B]
        return feat, logit

    # ---------------- 其他未支持模型 ----------------
    raise TypeError(
        f"Unsupported model type for get_embedding_and_logit: {type(model)}. "
        f"当前只支持 TargetNet / TargetNet_Optimized / TargetNetTransformer1D。"
    )


# =================================================
#  Load model + weights
# =================================================

def load_model(cfg: DictConfig, ckpt_path: str, device: torch.device):
    """
    从 cfg 和 checkpoint 路径构建并加载单 CTS 模型。

    返回:
        model: 已经 load_state_dict 并且 .eval()、to(device) 的 nn.Module
    """

    # ---- 确定 checkpoint 路径 ----
    if ckpt_path is None:
        ckpt_path = cfg.get("run", {}).get("checkpoint", None)
        if ckpt_path is None:
            raise ValueError(
                "[dump_cts_embeddings] checkpoint 路径未指定："
                "请在命令行传入 --checkpoint，或在 cfg.run.checkpoint 中指定。"
            )

    ckpt_path = str(Path(ckpt_path))
    print(f"[dump_cts_embeddings] Loading checkpoint from {ckpt_path}")

    # ---- 构建 data_cfg ----
    data_cfg = DataConfig.from_omegaconf(cfg.data)

    # ---- 构建模型结构（不带权重）----#
    model_name = cfg.model.get("arch", cfg.model.get("name"))
    model = build_model(model_name, cfg.model, data_cfg=data_cfg)
    model.to(device)

    # ---- 读取 checkpoint ----
    ckpt = torch.load(ckpt_path, map_location=device)

    # 1) 取出 state_dict
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # ---- 处理可能存在的前缀（如 "model.", "net."）----#
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("model."):
            new_k = new_k[len("model."):]
        if new_k.startswith("net."):
            new_k = new_k[len("net."):]
        cleaned_state_dict[new_k] = v

    # ---- 加载权重 ----
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
#  Dump a single split
# =================================================

def dump_split(cfg, model, split: str, out_root: str, device, batch_size: int):
    """
    对给定 split（train / val / test）：
      - 遍历 ChunkedCTSDataset
      - 用单 CTS 模型生成 embedding + logit
      - 按 set_idx 聚合到 pair 级别
      - 存成一个 dict[ pair_id ] = {...} 的 .pt 文件

    存储结构（v0）：
      out_pairs[pid] = {
          "embeddings": Tensor [N_i, d_emb],
          "logits":     Tensor [N_i],
          "esa":        Tensor [N_i],
          "pos":        None,              # 预留字段，方便 v1 加位置
          "label":      float (0. / 1.)
      }
    """
    print(f"\n===== Dumping split: {split} =====")

    # ---- 构建 DataConfig & Dataset ----
    data_cfg = DataConfig.from_omegaconf(cfg.data)

    # 打印一下当前 split 用的是哪个数据文件
    if hasattr(data_cfg, "path") and split in data_cfg.path:
        data_file = str(data_cfg.path[split])
    else:
        data_file = "<unknown>"
    print(f"[dump_split:{split}] Using data file: {data_file}")

    # build_dataset_and_loader 返回 (dataset, loader)，这里只用 dataset
    dataset, _ = build_dataset_and_loader(data_cfg, split=split)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # pair_id -> 累积内容
    pair_dict = defaultdict(lambda: {
        "embeddings": [],
        "logits": [],
        "esa": [],
        "label": None,
    })

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # v0: Dataset 返回 x, y, set_idx, esa_score
            if len(batch) != 4:
                raise ValueError(
                    f"[dump_split] Expect batch of length 4 (x, y, set_idx, esa_score), "
                    f"got len={len(batch)}"
                )

            x, y, set_idx, esa_score = batch

            x = x.to(device)                 # [B, C, L]
            y = y.view(-1)                   # [B]
            set_idx = set_idx.view(-1)       # [B]
            esa_score = esa_score.view(-1)   # [B]

            # ---- 得到 embedding + logit ----
            feats, logits = get_embedding_and_logit(model, x)
            # feats: [B, d_emb], logits: [B]

            feats = feats.detach().cpu()
            logits = logits.detach().cpu()
            y = y.detach().cpu()
            set_idx = set_idx.detach().cpu().long()
            esa_score = esa_score.detach().cpu()

            B = feats.size(0)
            for i in range(B):
                pid = int(set_idx[i].item())     # pair id（你的 set_idx 即 pair 编号）
                emb_i = feats[i]                 # [d_emb]
                logit_i = logits[i]              # scalar
                esa_i = esa_score[i]             # scalar
                label_i = float(y[i].item())     # scalar

                entry = pair_dict[pid]
                entry["embeddings"].append(emb_i)
                entry["logits"].append(logit_i)
                entry["esa"].append(esa_i)

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
    out_pairs = {}
    for pid, entry in pair_dict.items():
        emb = torch.stack(entry["embeddings"], dim=0)   # [N_i, d_emb]
        logits = torch.stack(entry["logits"], dim=0)    # [N_i]
        esa = torch.stack(entry["esa"], dim=0)          # [N_i]

        out_pairs[pid] = {
            "embeddings": emb,          # [N_i, d_emb]
            "logits": logits,           # [N_i]
            "esa": esa,                 # [N_i]
            "pos": None,                # v0 没有位置，预留字段
            "label": entry["label"],    # scalar
        }

    ensure_dir(out_root)
    out_path = os.path.join(out_root, f"{split}.pt")
    torch.save(out_pairs, out_path)

    # 简单打印一个示例 shape，方便 sanity check
    any_pid = next(iter(out_pairs))
    print(f"[dump_split:{split}] Example emb shape: {out_pairs[any_pid]['embeddings'].shape}")
    print(f"[dump_split:{split}] Num pairs: {len(out_pairs)}")
    print(f"[dump_split:{split}] Saved {len(out_pairs)} pairs to {out_path}")


# =================================================
#  Main
# =================================================

def main():
    args = parse_args()

    # 1) 读 Hydra 配置
    cfg = OmegaConf.load(args.config)

    # 2) 加载模型
    device = torch.device(args.device)
    model = load_model(cfg, args.checkpoint, device)

    # 3) 推断模型名字（优先 registry 名，其次类名）
    model_arch = getattr(model, "_registry_name", model.__class__.__name__)

    # 4) 从 DataConfig 中抽取“数据集身份信息”
    data_cfg = DataConfig.from_omegaconf(cfg.data)

    # 尝试用 train 路径作为“数据集代表”
    if hasattr(data_cfg, "path") and "train" in data_cfg.path:
        train_path = str(data_cfg.path["train"])
    else:
        train_path = "<unknown>"

    alignment = getattr(data_cfg, "alignment", "extended_seed_alignment")
    # 复用你在 ChunkedCTSDataset 里的思路：data_path + alignment -> hash
    hash_key = f"{train_path}|{alignment}"
    path_hash = hashlib.md5(hash_key.encode("utf-8")).hexdigest()[:8]

    train_tag = Path(train_path).stem if train_path not in ("", "<unknown>") else "no-train"
    data_tag = f"data={train_tag}_align={alignment}_h{path_hash}"

    # 5) checkpoint tag
    ckpt_tag = Path(args.checkpoint).stem

    # 6) 最终的 out_root：包含模型 / 数据集 / ckpt 信息
    out_root = os.path.join(
        args.out_root,
        f"arch={model_arch}",
        data_tag,
        f"ckpt={ckpt_tag}",
    )
    ensure_dir(out_root)

    print("\n[dump_cts_embeddings] ========== SUMMARY ==========")
    print(f"  Config file : {args.config}")
    print(f"  Model arch  : {model_arch}")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Train path  : {train_path}")
    print(f"  Alignment   : {alignment}")
    print(f"  Data hash   : {path_hash}")
    print(f"  Out root    : {out_root}")
    print(f"  Splits      : {args.splits}")
    print("====================================================\n")

    # 7) 对每个 split 做 dump
    for split in args.splits:
        dump_split(cfg, model, split, out_root, device, args.batch_size)


if __name__ == "__main__":
    main()
