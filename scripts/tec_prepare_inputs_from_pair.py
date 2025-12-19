#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
用法（你的 TSV 没有 split 或 split 全是 None：自动当 test 全量使用）
python scripts/tec_prepare_inputs_from_pair.py \
  --input_tsv /root/autodl-tmp/PAIRFormer/data/miRAW_Test_0,6-9.txt \
  --out_dir  data/tec_inputs/miRAW_Test_0,6-9


如果你确实有 split 列且里面有 test/val/train：

python scripts/tec_prepare_inputs_from_pair.py \
  --input_tsv data/my_pairs.txt \
  --out_dir tec_inputs/my_pairs_test \
  --split_col split \
  --split_value test
'''

import argparse
import os
from pathlib import Path
import pandas as pd

VALID_RNA = set(list("ACGU"))

def clean_seq(s: str) -> str:
    """Normalize sequences to TEC expected RNA alphabet."""
    if s is None:
        return ""
    s = str(s).strip()
    # Some files may contain leading '>' (fasta-like dumps) or gaps '-'
    if s.startswith(">"):
        s = s[1:]
    s = s.replace("-", "").replace(" ", "").replace("\t", "").replace("\r", "").replace("\n", "")
    s = s.upper().replace("T", "U")
    return s

def write_fasta(path: Path, id2seq: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _id in sorted(id2seq.keys()):
            seq = id2seq[_id]
            f.write(f">{_id}\n")
            # wrap at 80 chars
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")

def validate_seq_or_raise(id_: str, seq: str, kind: str):
    bad = sorted(set(seq) - VALID_RNA)
    if bad:
        raise ValueError(
            f"{kind} sequence contains non-ACGU characters for id='{id_}': {bad}. "
            f"First 80 bases: {seq[:80]}"
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_tsv", required=True, help="Your pair TSV (columns: mirna_id mirna_seq mrna_id mrna_seq label [split])")
    ap.add_argument("--out_dir", required=True, help="Output dir to store mirna.fasta, mrna.fasta, query.tsv")
    ap.add_argument("--sep", default="\t", help="Delimiter (default: tab)")
    ap.add_argument("--mirna_id_col", default="mirna_id")
    ap.add_argument("--mirna_seq_col", default="mirna_seq")
    ap.add_argument("--mrna_id_col", default="mrna_id")
    ap.add_argument("--mrna_seq_col", default="mrna_seq")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--split_col", default="split", help="If not present / all None -> treat as all test")
    ap.add_argument("--split_value", default="test", help="Rows with split==this value will be kept (if split col exists & not all null)")
    ap.add_argument("--allow_missing_split", action="store_true", help="If split column missing, do not error (default behavior already keeps all).")
    args = ap.parse_args()

    df = pd.read_csv(args.input_tsv, sep=args.sep, dtype=str)
    required = [args.mirna_id_col, args.mirna_seq_col, args.mrna_id_col, args.mrna_seq_col, args.label_col]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}'. Found columns: {list(df.columns)}")

    # split handling: if split col exists and not all null/None/empty, filter to split_value; else keep all
    if args.split_col in df.columns:
        split_series = df[args.split_col].astype(str)
        nonempty = split_series.notna() & (split_series.str.lower() != "none") & (split_series.str.strip() != "")
        if nonempty.any():
            df = df[split_series.str.strip().str.lower() == args.split_value.lower()].copy()
    else:
        # no split column -> keep all (pure test)
        pass

    if len(df) == 0:
        raise ValueError("No rows left after split filtering. Check --split_col / --split_value.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mirna_id2seq = {}
    mrna_id2seq = {}

    # Build fasta maps with consistency checks
    for _, row in df.iterrows():
        mid = str(row[args.mirna_id_col]).strip()
        mseq = clean_seq(row[args.mirna_seq_col])
        tid = str(row[args.mrna_id_col]).strip()
        tseq = clean_seq(row[args.mrna_seq_col])

        # Validate alphabet strictly (TEC tokenization base_number_dict only supports A/G/C/U)
        validate_seq_or_raise(mid, mseq, "miRNA")
        validate_seq_or_raise(tid, tseq, "mRNA")

        if mid in mirna_id2seq and mirna_id2seq[mid] != mseq:
            raise ValueError(f"miRNA id '{mid}' has inconsistent sequences in file.")
        if tid in mrna_id2seq and mrna_id2seq[tid] != tseq:
            raise ValueError(f"mRNA id '{tid}' has inconsistent sequences in file.")

        mirna_id2seq[mid] = mseq
        mrna_id2seq[tid] = tseq

    # Write fasta
    write_fasta(out_dir / "mirna.fasta", mirna_id2seq)
    write_fasta(out_dir / "mrna.fasta", mrna_id2seq)

    # Write query.tsv for TEC read_ground_truth(): columns are positional, header can be arbitrary
    q = pd.DataFrame({
        "query_ids": df[args.mirna_id_col].astype(str).str.strip(),
        "target_ids": df[args.mrna_id_col].astype(str).str.strip(),
        "labels": df[args.label_col].astype(str).str.strip().astype(int),
    })
    # 1) 严格对齐 TEC 示例命名（虽然叫 csv，但用 \t 分隔）
    q.to_csv(out_dir / "query.csv", sep="\t", index=False, header=False)

    # 2) 可选：同时保留一个带表头的版本，便于你自己排查/对账
    q.to_csv(out_dir / "query_with_header.tsv", sep="\t", index=False, header=True)


    # Optional: keep a normalized copy for traceability
    norm = df.copy()
    norm[args.mirna_seq_col] = norm[args.mirna_seq_col].map(clean_seq)
    norm[args.mrna_seq_col] = norm[args.mrna_seq_col].map(clean_seq)
    norm.to_csv(out_dir / "source_pairs.normalized.tsv", sep="\t", index=False)

    print(f"[OK] Wrote:\n  {out_dir/'mirna.fasta'}\n  {out_dir/'mrna.fasta'}\n  {out_dir/'query.csv'}\n   {out_dir/'query_with_header.tsv'}\n  {out_dir/'source_pairs.normalized.tsv'}")

if __name__ == "__main__":
    main()
