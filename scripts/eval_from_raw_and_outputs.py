#!/usr/bin/env python
"""
离线评估脚本：从多个 Test 原始数据文件 + 对应 *_outputs.txt 合并预测并计算指标。

假设：
- 原始数据文件：Test0.txt, Test1.txt, ... （通过 --data_glob 匹配）
- 输出文件：Test0_outputs.txt, Test1_outputs.txt, ... （通过 --outputs_glob 匹配）
- 原始数据文件中含有标签列（例如 "label"），顺序与 *_outputs.txt 完全一致。

原始数据文件格式示例（tab 分隔，仅示意）：
    miRNA_id    mRNA_id    ...    label
    miR-1       geneA      ...    1
    miR-2       geneB      ...    0
    ...

输出文件格式（tab 分隔）：
    set_idx output
    0       0.8732
    1       0.1123
    ...

用法示例：
python -m scripts.eval_from_raw_and_outputs \
  --data_glob "data/origin_results/data/miRAW_Test*.txt" \
  --outputs_glob "data/origin_results/outputs/test*_outputs.txt" \
  --label_col "label" \
  --output_dir "data/origin_results/results" \
  --do_threshold_sweep
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.evaluator.evaluator import evaluate_predictions


def collect_files(pattern: str):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"没有匹配到任何文件：{pattern}")
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_glob",
        type=str,
        required=True,
        help="原始数据文件的 glob 模式，比如 'data/Test*.txt'",
    )
    parser.add_argument(
        "--outputs_glob",
        type=str,
        required=True,
        help="输出文件的 glob 模式，比如 'results/Test*_outputs.txt'",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="label",
        help="原始数据文件中标签列的列名，默认 'label'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="评估结果输出目录（metrics.json / report.txt / 图像等）",
    )
    parser.add_argument(
        "--from_logits",
        action="store_true",
        help="若指定，表示 *_outputs.txt 中的 output 是 logits；否则视作 [0,1] 概率",
    )
    parser.add_argument(
        "--do_threshold_sweep",
        action="store_true",
        help="是否做阈值扫描并寻找 F1 最优阈值",
    )
    parser.add_argument(
        "--data_sep",
        type=str,
        default="\t",
        help="原始数据文件的分隔符，默认制表符 '\\t'",
    )
    parser.add_argument(
        "--outputs_sep",
        type=str,
        default="\t",
        help="输出文件的分隔符，默认制表符 '\\t'",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) 收集文件并检查数量一致（比如都应该是 10 个）
    data_files = collect_files(args.data_glob)
    output_files = collect_files(args.outputs_glob)

    if len(data_files) != len(output_files):
        raise RuntimeError(
            f"原始数据文件数={len(data_files)} 与 输出文件数={len(output_files)} 不一致，"
            f"请检查 glob 模式。"
        )

    print("[INFO] 原始数据文件：")
    for f in data_files:
        print("  -", f)
    print("[INFO] 输出文件：")
    for f in output_files:
        print("  -", f)

    all_y_true = []
    all_y_pred_raw = []
    
    # 2) 逐对读取 Test_i + Test_i_outputs
    for data_path, out_path in zip(data_files, output_files):
        print(f"[INFO] 处理 pair：{os.path.basename(data_path)}  &  {os.path.basename(out_path)}")

        df_data = pd.read_csv(data_path, sep=args.data_sep)
        if args.label_col not in df_data.columns:
            raise ValueError(
                f"{data_path} 中找不到标签列 '{args.label_col}'，实际列名为：{list(df_data.columns)}"
            )

        # 关键改动：给原始数据显式加 set_idx
        df_data = df_data.reset_index().rename(columns={"index": "set_idx"})
        labels = df_data[["set_idx", args.label_col]].copy()
        labels.columns = ["set_idx", "label"]

        df_out = pd.read_csv(out_path, sep=args.outputs_sep)
        df_out.columns = [c.strip() for c in df_out.columns]
        if "output" not in df_out.columns:
            raise ValueError(f"{out_path} 中找不到 'output' 列，实际列名为：{list(df_out.columns)}")

        # 如果输出文件没有 set_idx，则回退到“行号对齐”
        if "set_idx" not in df_out.columns:
            df_out = df_out.reset_index().rename(columns={"index": "set_idx"})

        preds = df_out[["set_idx", "output"]].copy()

        merged = labels.merge(preds, on="set_idx", how="inner")
        if len(merged) != len(labels):
            raise RuntimeError(
                f"对齐失败：{data_path} 样本数={len(labels)}，但 merge 后={len(merged)}。"
                f"请检查 outputs 是否对应同一个输入文件。"
            )

        all_y_true.append(merged["label"].to_numpy())
        all_y_pred_raw.append(merged["output"].to_numpy())

    # 3) 拼接所有文件的标签和预测
    y_true = np.concatenate(all_y_true, axis=0).astype(int)
    y_pred_raw = np.concatenate(all_y_pred_raw, axis=0).astype(float)

    print(f"[INFO] 合并后的样本总数 = {len(y_true)}")

    # 4) 构造 task_cfg，复用你的 metrics/evaluator 逻辑
    task_cfg = OmegaConf.create(
        {
            "problem_type": "binary_classification",
            "from_logits": bool(args.from_logits),
            "threshold": 0.5,
        }
    )

    # 5) 调用 evaluate_predictions 进行完整评估
    result = evaluate_predictions(
        y_true=y_true,
        y_pred_raw=y_pred_raw,
        task_cfg=task_cfg,
        do_threshold_sweep=args.do_threshold_sweep,
        sweep_num_thresholds=101,
        output_dir=args.output_dir,
        logging_cfg=None,
    )

    # 6) 在控制台打印一份简要指标
    metrics = result["metrics"]
    print("\n===== Basic metrics (threshold = {:.3f}) =====".format(
        metrics.get("threshold", 0.5)
    ))
    scalar_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "roc_auc",
        "pr_auc",
        "fpr",
        "fnr",
        "npv",
        "fdr",
    ]
    for k in scalar_keys:
        if k in metrics:
            v = metrics[k]
            print(f"{k:>12s}: {v:.4f}")

    if "confusion_matrix" in metrics:
        print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
        print(np.array(metrics["confusion_matrix"]))

    if "best_threshold_info" in result:
        b = result["best_threshold_info"]
        print("\n===== Best F1 from threshold sweep =====")
        print("best_threshold = {:.4f}".format(b["threshold"]))
        print("best_f1        = {:.4f}".format(b["f1"]))
        print("precision@best = {:.4f}".format(b["precision"]))
        print("recall@best    = {:.4f}".format(b["recall"]))


if __name__ == "__main__":
    main()
