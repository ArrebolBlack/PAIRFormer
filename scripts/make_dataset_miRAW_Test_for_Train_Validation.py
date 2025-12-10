#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据 miRAW_Test1~miRAW_Test9 生成多组 Train/Val 数据集。

输入：
    data/miRAW_Test1.txt
    ...
    data/miRAW_Test9.txt
    （miRAW_Test0.txt 不使用）

输出目录：
    data/miRAW_Test_for_Train_Validation/

输出文件（共 9 份），示例：
    miRAW_Test1-1_split-ratio=0.9_Train_Validation.txt   # 只用 Test1
    miRAW_Test1-2_split-ratio=0.9_Train_Validation.txt   # 用 Test1+Test2
    ...
    miRAW_Test1-9_split-ratio=0.9_Train_Validation.txt   # 用 Test1..Test9

处理流程：
    - 多个 Test 文件按行 concat
    - shuffle
    - 按 0.9 : 0.1 划分 train / val
    - 新增一列 "split" 标记 "train" 或 "val"
"""

import os
from pathlib import Path

import pandas as pd
from typing import List


# ======== 可根据需要在这里改路径 / 参数 ========

DATA_DIR = Path("data")
INPUT_PATTERN = "miRAW_Test{idx}.txt"

OUTPUT_DIR = DATA_DIR / "miRAW_Test_for_Train_Validation"

TRAIN_RATIO = 0.9
RANDOM_SEED = 42  # 保证可复现


def load_and_concat(files: List[Path]) -> pd.DataFrame:
    """读取多个 txt 并按行拼接（假定为同样的制表符分隔格式）"""
    dfs = []
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"Input file not found: {f}")
        df = pd.read_csv(f, sep="\t")
        dfs.append(df)
    if not dfs:
        raise RuntimeError("No input files provided.")
    return pd.concat(dfs, axis=0, ignore_index=True)


def add_split_and_shuffle(df: pd.DataFrame,
                          train_ratio: float = TRAIN_RATIO,
                          seed: int = RANDOM_SEED) -> pd.DataFrame:
    """shuffle 后按比例划分，并新增 split 列"""
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n_total = len(df)
    n_train = int(n_total * train_ratio)

    splits = ["train"] * n_train + ["val"] * (n_total - n_train)
    df = df.copy()
    df["split"] = splits
    return df


def make_dataset_for_n(max_idx: int):
    """
    使用 Test1..Test{max_idx} 生成一个 Train/Val 数据集，
    并保存到 OUTPUT_DIR/miRAW_Test1-{max_idx}_split-ratio-0.9_Train_Validation.txt
    """
    # 组装输入文件列表
    input_files = [
        DATA_DIR / INPUT_PATTERN.format(idx=i)
        for i in range(1, max_idx + 1)
    ]

    print(f"[INFO] Building dataset with files: {', '.join(str(f) for f in input_files)}")

    df = load_and_concat(input_files)
    df = add_split_and_shuffle(df, train_ratio=TRAIN_RATIO, seed=RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_name = f"miRAW_Test1-{max_idx}_split-ratio-{TRAIN_RATIO}_Train_Validation.txt"
    out_path = OUTPUT_DIR / out_name

    df.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] Saved: {out_path} (rows={len(df)})")


def main():
    # 需要 9 个数据集：使用 Test1..1, Test1..2, ..., Test1..9
    for n in range(1, 10):
        make_dataset_for_n(n)


if __name__ == "__main__":
    main()
