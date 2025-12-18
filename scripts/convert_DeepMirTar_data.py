#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse


HEADER = ["mirna_id", "mirna_seq", "mrna_id", "mrna_seq", "label", "split"]


def parse_line(line: str):
    """
    Expected input per line (whitespace-separated):
        >MIRNA_SEQ    >MRNA_SEQ    LABEL
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return None

    parts = s.split()
    if len(parts) < 3:
        raise ValueError(f"Bad line (need 3 fields): {line!r}")

    mirna_raw, mrna_raw, label = parts[0], parts[1], parts[2]

    if not mirna_raw.startswith(">") or not mrna_raw.startswith(">"):
        raise ValueError(f"Bad line (seq must start with '>'): {line!r}")

    mirna_seq = mirna_raw[1:]
    mrna_seq = mrna_raw[1:]
    return mirna_seq, mrna_seq, label


def main():
    ap = argparse.ArgumentParser(
        description="Convert lines '>miRNA_seq >mRNA_seq label' to TSV with header and auto-increment IDs."
    )
    ap.add_argument("--infile", "-i", required=True, help="Input text file path")
    ap.add_argument("--outfile", "-o", required=True, help="Output TSV file path")
    ap.add_argument("--split_value", default="split", help='Value to write into "split" column (default: split)')
    ap.add_argument("--no_t2u", action="store_true", help="Do NOT convert T->U in sequences")
    ap.add_argument("--id_prefix_mirna", default="mirna_", help="Prefix for mirna_id (default: mirna_)")
    ap.add_argument("--id_prefix_mrna", default="mrna_", help="Prefix for mrna_id (default: mrna_)")
    ap.add_argument("--id_start", type=int, default=1, help="Start index for IDs (default: 1)")
    ap.add_argument("--id_width", type=int, default=6, help="Zero-pad width (default: 6)")
    args = ap.parse_args()

    idx = args.id_start

    with open(args.infile, "r", encoding="utf-8") as fin, open(args.outfile, "w", encoding="utf-8") as fout:
        fout.write("\t".join(HEADER) + "\n")

        for lineno, line in enumerate(fin, start=1):
            parsed = parse_line(line)
            if parsed is None:
                continue

            mirna_seq, mrna_seq, label = parsed

            if not args.no_t2u:
                mirna_seq = mirna_seq.replace("T", "U").replace("t", "u")
                mrna_seq = mrna_seq.replace("T", "U").replace("t", "u")

            mirna_id = f"{args.id_prefix_mirna}{idx:0{args.id_width}d}"
            mrna_id = f"{args.id_prefix_mrna}{idx:0{args.id_width}d}"

            row = [mirna_id, mirna_seq, mrna_id, mrna_seq, label, args.split_value]
            fout.write("\t".join(row) + "\n")
            idx += 1


if __name__ == "__main__":
    main()
