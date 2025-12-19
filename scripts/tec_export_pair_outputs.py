#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np


def _norm_id(s: pd.Series) -> pd.Series:
    # 去 BOM + 去首尾空格，避免 merge/map 失败
    return s.astype(str).str.replace("\ufeff", "", regex=False).str.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cts_predict_tsv", required=True,
                    help="TEC CTS-level output predict.tsv (from evaluate_transcript_level.py)")
    ap.add_argument("--raw_pair_tsv", required=True,
                    help="Raw pair-level TSV used by your evaluator (order defines set_idx)")
    ap.add_argument("--out_pair_pmax_tsv", required=True,
                    help="Output pair-level TSV (unique pairs) with tec_pmax / tec_score")
    ap.add_argument("--out_outputs_txt", required=True,
                    help="Output *_outputs.txt (with header: set_idx, output)")
    ap.add_argument("--sep_raw", default="\t", help="Delimiter for raw_pair_tsv (default: tab)")
    ap.add_argument("--invert", action="store_true",
                    help="Use score = 1 - tec_pmax for pairs WITH CTS; pairs without CTS stay 0")
    ap.add_argument("--missing_score", type=float, default=0.0,
                    help="Score used when pair missing / no CTS (default: 0.0)")
    args = ap.parse_args()

    # ---------- 0) Read raw ----------
    raw = pd.read_csv(args.raw_pair_tsv, sep=args.sep_raw, dtype=str)
    for col in ["mirna_id", "mrna_id"]:
        if col not in raw.columns:
            raise KeyError(f"Raw file missing column '{col}'. Found: {list(raw.columns)}")
    raw["mirna_id"] = _norm_id(raw["mirna_id"])
    raw["mrna_id"] = _norm_id(raw["mrna_id"])

    raw_unique = raw[["mirna_id", "mrna_id"]].drop_duplicates().reset_index(drop=True)

    # ---------- 1) Read CTS-level predictions ----------
    df = pd.read_csv(args.cts_predict_tsv, sep="\t", dtype=str)
    required = {"query_ids", "target_ids", "predictions"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"CTS predict file missing columns {missing}. Found: {list(df.columns)}")

    df["query_ids"] = _norm_id(df["query_ids"])
    df["target_ids"] = _norm_id(df["target_ids"])
    df["predictions"] = pd.to_numeric(df["predictions"], errors="coerce").fillna(0.0)

    # 识别“是否筛出 CTS”
    # TEC 脚本中：没找到候选时，会写 query_seqs='Not found'，predictions=0
    if "query_seqs" in df.columns:
        has_cts_flag = df["query_seqs"].astype(str).str.strip().ne("Not found")
        has_cts = df.groupby(["query_ids", "target_ids"])["query_seqs"].apply(
            lambda x: (x.astype(str).str.strip() != "Not found").any()
        ).reset_index(name="has_cts")
    else:
        # 若没有该列，则只能认为“出现过该 pair 就算有 CTS”（保守）
        has_cts = df.groupby(["query_ids", "target_ids"], as_index=False).size()
        has_cts = has_cts.rename(columns={"size": "has_cts"})
        has_cts["has_cts"] = True

    # pmax over CTSs
    pmax = df.groupby(["query_ids", "target_ids"], as_index=False)["predictions"].max()
    pair_pred = pmax.merge(has_cts, on=["query_ids", "target_ids"], how="left")
    pair_pred = pair_pred.rename(columns={
        "query_ids": "mirna_id",
        "target_ids": "mrna_id",
        "predictions": "tec_pmax",
    })
    pair_pred["has_cts"] = pair_pred["has_cts"].fillna(False).astype(bool)

    # ---------- 2) Make FULL pair table covering ALL raw unique pairs ----------
    # 左连接：raw unique pairs 全保留；缺失/无CTS -> tec_pmax=missing_score 且 has_cts=False
    pair_full = raw_unique.merge(pair_pred, on=["mirna_id", "mrna_id"], how="left")
    pair_full["has_cts"] = pair_full["has_cts"].fillna(False).astype(bool)
    pair_full["tec_pmax"] = pd.to_numeric(pair_full["tec_pmax"], errors="coerce").fillna(args.missing_score)

    # score 定义：
    # - 若 has_cts=False（没筛出 CTS 或 TEC 缺失） -> 恒为 missing_score（默认 0）
    # - 否则：
    #     invert=False -> score = tec_pmax
    #     invert=True  -> score = 1 - tec_pmax
    if args.invert:
        pair_full["tec_score"] = np.where(
            pair_full["has_cts"].values,
            1.0 - pair_full["tec_pmax"].astype(float).values,
            float(args.missing_score),
        )
    else:
        pair_full["tec_score"] = np.where(
            pair_full["has_cts"].values,
            pair_full["tec_pmax"].astype(float).values,
            float(args.missing_score),
        )

    pair_full.to_csv(args.out_pair_pmax_tsv, sep="\t", index=False)

    # ---------- 3) Build outputs aligned to raw row order ----------
    score_map = dict(zip(
        zip(pair_full["mirna_id"].astype(str), pair_full["mrna_id"].astype(str)),
        pair_full["tec_score"].astype(float),
    ))

    scores = []
    miss_rows = 0
    for mid, tid in zip(raw["mirna_id"].astype(str), raw["mrna_id"].astype(str)):
        s = score_map.get((mid, tid), None)
        if s is None:
            # 理论上不应发生，因为 pair_full 覆盖了 raw_unique
            miss_rows += 1
            s = float(args.missing_score)
        scores.append(float(s))

    out_df = pd.DataFrame({
        "set_idx": np.arange(len(scores), dtype=int),
        "output": np.asarray(scores, dtype=float),
    })

    # 你要求：outputs 文件必须有表头
    out_df.to_csv(args.out_outputs_txt, sep="\t", index=False, header=True)

    # ---------- 4) Summary ----------
    print("[OK] raw rows       =", len(raw))
    print("[OK] raw unique     =", len(raw_unique))
    print("[OK] pair_full rows =", len(pair_full))
    print("[INFO] invert =", args.invert, "| missing_score =", args.missing_score)
    print("[INFO] pairs missing in TEC predict (filled as missing_score):",
          int((~pair_full["has_cts"]).sum()))
    print("[WARN] unexpected missing rows during mapping =", miss_rows)
    print("[INFO] outputs preview:")
    print(out_df.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
