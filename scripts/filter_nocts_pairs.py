"""Filter out pairs that cannot generate any CTS window from the MTI dataset.

Reasons a pair may have zero windows:
1. mRNA sequence is "Sequenceunavailable" (length 19 < window_size 40)
2. mRNA sequence is real but shorter than 40 nt
3. mRNA >= 40 nt but no window passes the ESA >= 6.0 threshold

This script scans all pairs, checks each one against the CTS generation logic,
and writes a cleaned version of the data file.
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.data.encoding import extended_seed_alignment, reverse


WINDOW_SIZE = 40
ESA_MIN_SCORE = 6.0


def has_any_window(mirna_seq: str, mrna_seq: str) -> bool:
    """Return True if at least one CTS window can be generated."""
    mirna_seq = mirna_seq.upper().replace("T", "U")
    mrna_seq = mrna_seq.upper().replace("T", "U")
    mrna_rev_seq = reverse(mrna_seq)
    max_pos = max(0, len(mrna_rev_seq) - WINDOW_SIZE + 1)
    if max_pos == 0:
        return False
    # Only need to check if ANY window passes ESA threshold
    for pos_idx in range(max_pos):
        win = mrna_rev_seq[pos_idx : pos_idx + WINDOW_SIZE]
        _, _, esa_score = extended_seed_alignment(mirna_seq, win)
        if esa_score >= ESA_MIN_SCORE:
            return True
    return False


def main():
    src = Path("data/MTI/MTI_pair_random_split.txt")
    dst = Path("data/MTI/MTI_pair_random_split_filtered.txt")

    if not src.exists():
        print(f"Source file not found: {src}")
        sys.exit(1)

    stats = {"total": 0, "kept": 0, "removed": 0, "removed_by_split": {}}
    reasons = {"unavailable": 0, "short": 0, "no_esa": 0}

    with open(src, "r") as fin, open(dst, "w") as fout:
        header = next(fin)
        fout.write(header)

        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            stats["total"] += 1
            toks = line.split("\t")
            if len(toks) < 5:
                stats["removed"] += 1
                continue

            mrna_seq = toks[2] if len(toks) > 2 else ""
            mirna_seq = toks[4] if len(toks) > 4 else ""
            split_tag = toks[6] if len(toks) > 6 else "unknown"

            mrna_upper = mrna_seq.upper().replace("T", "U")
            mirna_upper = mirna_seq.upper().replace("T", "U")

            # Quick checks before expensive ESA scan
            if "SEQUENCEUNAVAILABLE" in mrna_upper or len(mrna_upper) < WINDOW_SIZE:
                if "SEQUENCEUNAVAILABLE" in mrna_upper:
                    reasons["unavailable"] += 1
                else:
                    reasons["short"] += 1
                stats["removed"] += 1
                stats["removed_by_split"][split_tag] = stats["removed_by_split"].get(split_tag, 0) + 1
                continue

            if not has_any_window(mirna_upper, mrna_upper):
                reasons["no_esa"] += 1
                stats["removed"] += 1
                stats["removed_by_split"][split_tag] = stats["removed_by_split"].get(split_tag, 0) + 1
                continue

            fout.write(line + "\n")
            stats["kept"] += 1

    print(f"Source: {src}")
    print(f"Output: {dst}")
    print(f"Total:  {stats['total']}")
    print(f"Kept:   {stats['kept']}")
    print(f"Removed: {stats['removed']}")
    print(f"  - Sequence unavailable: {reasons['unavailable']}")
    print(f"  - Short mRNA (< {WINDOW_SIZE} nt): {reasons['short']}")
    print(f"  - No ESA >= {ESA_MIN_SCORE}: {reasons['no_esa']}")
    print(f"Removed by split: {stats['removed_by_split']}")


if __name__ == "__main__":
    main()
