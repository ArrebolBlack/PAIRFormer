# scripts/analyze_cts_per_pair.py

'''
# 只看 train
python scripts/analyze_cts_per_pair.py \
  --config configs/experiment/miRAW_TargetNet_baseline.yaml \
  --split train \
  --cache_root cache   # 和你训练时的 cache_data_path 保持一致

# 看 val

python -m scripts.analyze_cts_per_pair \
  --config configs/experiment/miRAW_TargetNet_baseline.yaml \
  --split val

python -m scripts.analyze_cts_per_pair \
  --config configs/experiment/miRAW_TargetNet_baseline.yaml \
  --split test


python -m scripts.analyze_cts_per_pair \
  --config configs/experiment/miRAW_TargetNet.yaml \
  --split test

'''
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.config.data_config import DataConfig
from src.data.builder import build_dataset_and_loader


def compute_cts_per_pair_stats(dataset):
    """
    给定一个 Dataset（其 __getitem__ 返回 (x, y, set_idx, esa_score)），
    统计每个 pair (set_idx) 拥有多少个 CTS/window。
    """
    counts = Counter()

    for i in range(len(dataset)):
        _, _, set_idx, _, _ = dataset[i]   # set_idx: Tensor([k])
        k = int(set_idx.item())
        counts[k] += 1

    # 转成 np.array 方便算统计量
    cts_array = np.array(list(counts.values()), dtype=np.int64)

    stats = {
        "num_pairs": int(len(cts_array)),
        "num_cts_total": int(cts_array.sum()),
        "cts_per_pair_mean": float(cts_array.mean()),
        "cts_per_pair_std": float(cts_array.std()),
        "cts_per_pair_min": int(cts_array.min()),
        "cts_per_pair_median": float(np.median(cts_array)),
        "cts_per_pair_95th": float(np.percentile(cts_array, 95)),
        "cts_per_pair_max": int(cts_array.max()),
    }

    # 分布：某个 CTS 数量 -> 有多少个 pair
    hist = Counter(cts_array.tolist())

    return stats, hist


def print_stats(name, stats, hist, top_k=20):
    print(f"\n===== {name} CTS-per-pair Stats =====")
    for k, v in stats.items():
        print(f"{k:>22}: {v}")

    print(f"\n#pairs with given #CTS (up to {top_k}, and '>{top_k}'):")
    acc_other = 0
    for cts_num in sorted(hist.keys()):
        if cts_num <= top_k:
            print(f"  {cts_num:2d} CTS  -> {hist[cts_num]} pairs")
        else:
            acc_other += hist[cts_num]
    if acc_other > 0:
        print(f"  >{top_k:2d} CTS -> {acc_other} pairs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="路径到 experiment yaml，例如 configs/experiment/miRAW_TargetNet_baseline.yaml",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="统计哪个 split（默认为 train）",
    )
    parser.add_argument(
        "--cache_root",
        type=str,
        default="cache",
        help="cache_data_path，与训练时保持一致（相对仓库根目录）",
    )
    args = parser.parse_args()

    # 1. 载入 experiment 配置
    cfg = OmegaConf.load(args.config)

    # 2. 构建 DataConfig
    data_cfg = DataConfig.from_omegaconf(cfg.data)

    # 3. 构建 dataset（不在意 loader 的 batch 等参数，只要 dataset 本体）
    cache_root = Path(args.cache_root)
    ds, _ = build_dataset_and_loader(
        data_cfg=data_cfg,
        split_idx=args.split,
        cache_data_path=cache_root,
        batch_size=1,       # 随便写，反正我们只用 ds
        num_workers=4,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
    )

    # 4. 计算 CTS per pair 统计信息
    stats, hist = compute_cts_per_pair_stats(ds)

    # 5. 打印结果
    print_stats(args.split, stats, hist, top_k=20)


if __name__ == "__main__":
    main()
