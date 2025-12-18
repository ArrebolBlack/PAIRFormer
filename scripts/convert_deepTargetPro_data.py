#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from typing import Dict, Tuple, List, Optional


def read_fasta_as_dict(path: str) -> Dict[str, str]:
    """
    Parse FASTA into {record_id: sequence}.
    record_id = header first token after '>' (split by whitespace).
    """
    seqs: Dict[str, List[str]] = {}
    cur_id: Optional[str] = None

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                cur_id = line[1:].split()[0]
                if cur_id in seqs:
                    # Duplicate ID: keep the first occurrence (ignore later).
                    cur_id = cur_id
                else:
                    seqs[cur_id] = []
            else:
                if cur_id is None:
                    raise ValueError(f"FASTA format error in {path}: sequence appears before any header.")
                seqs[cur_id].append(line)

    return {k: "".join(v) for k, v in seqs.items()}


def detect_delimiter(sample_line: str) -> str:
    if "\t" in sample_line:
        return "\t"
    if "," in sample_line:
        return ","
    return None  # whitespace fallback


def read_pairs(path: str, has_header: bool, delimiter: Optional[str]) -> Tuple[List[List[str]], bool]:
    """
    Returns (rows, has_split_col)
    row format expected:
      3 cols: mirna_id, mrna_id, label
      4 cols: mirna_id, mrna_id, label, split
    """
    rows: List[List[str]] = []
    has_split = False

    with open(path, "r", encoding="utf-8") as f:
        first_nonempty = None
        for line in f:
            if line.strip():
                first_nonempty = line.strip()
                break

    if first_nonempty is None:
        return [], False

    delim = delimiter if delimiter is not None else detect_delimiter(first_nonempty)

    with open(path, "r", encoding="utf-8") as fin:
        if delim is None:
            it = fin
            if has_header:
                next(it, None)
            for raw in it:
                s = raw.strip()
                if not s:
                    continue
                parts = s.split()
                if len(parts) not in (3, 4):
                    raise ValueError(f"Bad row (need 3 or 4 columns): {raw!r}")
                if len(parts) == 4:
                    has_split = True
                rows.append(parts)
        else:
            reader = csv.reader(fin, delimiter=delim)
            if has_header:
                next(reader, None)
            for parts in reader:
                if not parts or all((c.strip() == "") for c in parts):
                    continue
                parts = [c.strip() for c in parts if c is not None]
                if len(parts) not in (3, 4):
                    raise ValueError(f"Bad row (need 3 or 4 columns): {parts!r}")
                if len(parts) == 4:
                    has_split = True
                rows.append(parts)

    return rows, has_split


def main():
    ap = argparse.ArgumentParser(
        description="Build pair TSV by joining miRNA/mRNA IDs with sequences from FASTA."
    )
    ap.add_argument("--pairs", required=True, help="Input pairs file (CSV/TSV). Columns: mirna_id, mrna_id, label, [split]")
    ap.add_argument("--mirna_fasta", required=True, help="miRNA FASTA path (IDs must match mirna_id)")
    ap.add_argument("--mrna_fasta", required=True, help="mRNA FASTA path (IDs must match mrna_id)")
    ap.add_argument("--out", required=True, help="Output TSV path")
    ap.add_argument("--pairs_has_header", action="store_true", help="Set if pairs file has a header row")
    ap.add_argument("--delimiter", default=None, help="Force delimiter: '\\t' or ',' . If not set, auto-detect.")
    ap.add_argument("--t2u", action="store_true",
                    help="Convert T->U in output sequences (useful if FASTA uses DNA alphabet).")

    # New: limit missing prints to avoid log spam
    ap.add_argument("--max_missing_print", type=int, default=50,
                    help="Max number of missing-ID rows to print (0 = print all). Default: 50")
    args = ap.parse_args()

    mirna_map = read_fasta_as_dict(args.mirna_fasta)
    mrna_map = read_fasta_as_dict(args.mrna_fasta)

    delim = args.delimiter
    if delim == "\\t":
        delim = "\t"

    rows, has_split = read_pairs(args.pairs, has_header=args.pairs_has_header, delimiter=delim)

    out_header = ["mirna_id", "mirna_seq", "mrna_id", "mrna_seq", "label"] + (["split"] if has_split else [])

    total = 0
    written = 0

    missing_mirna = 0
    missing_mrna = 0
    missing_both = 0

    printed_missing = 0
    max_print = args.max_missing_print

    with open(args.out, "w", encoding="utf-8", newline="") as fout:
        fout.write("\t".join(out_header) + "\n")

        for idx, parts in enumerate(rows, start=1):
            total += 1

            if len(parts) == 3:
                mirna_id, mrna_id, label = parts
                split_val = None
            else:
                mirna_id, mrna_id, label, split_val = parts

            mirna_seq = mirna_map.get(mirna_id)
            mrna_seq = mrna_map.get(mrna_id)

            mirna_missing = mirna_seq is None
            mrna_missing = mrna_seq is None

            if mirna_missing or mrna_missing:
                if mirna_missing and mrna_missing:
                    missing_both += 1
                elif mirna_missing:
                    missing_mirna += 1
                else:
                    missing_mrna += 1

                # Print and skip
                if max_print == 0 or printed_missing < max_print:
                    print(
                        f"[MISS] row={idx} mirna_id={mirna_id} mrna_id={mrna_id} "
                        f"label={label} split={split_val if split_val is not None else ''} "
                        f"(mirna_missing={mirna_missing}, mrna_missing={mrna_missing})"
                    )
                    printed_missing += 1
                continue

            if args.t2u:
                mirna_seq = mirna_seq.replace("T", "U").replace("t", "u")
                mrna_seq = mrna_seq.replace("T", "U").replace("t", "u")

            out_row = [mirna_id, mirna_seq, mrna_id, mrna_seq, label]
            if has_split:
                out_row.append(split_val if split_val is not None else "")
            fout.write("\t".join(out_row) + "\n")
            written += 1

    failed = total - written
    print(f"[DONE] total_rows={total} written={written} failed={failed}")
    print(f"[DONE] missing_mirna_only={missing_mirna} missing_mrna_only={missing_mrna} missing_both={missing_both}")
    if max_print != 0 and failed > printed_missing:
        print(f"[DONE] missing rows printed={printed_missing} (use --max_missing_print 0 to print all)")


if __name__ == "__main__":
    main()
