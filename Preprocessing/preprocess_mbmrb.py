#!/usr/bin/env python3
"""
Preprocess mBMRB dataset for token classification training.

Label mapping:
  A  →  0  (class 0: no special annotation)
  .  →  1  (class 1: missing data / positive case)
  others →  _  (ignore in loss)
"""
import csv
import os
import argparse
from collections import Counter

LABEL_MAP = {
    "A": "0",
    ".": "1",
}

MAPPING_TABLE = {
    "A": "0  (class 0) no special annotation (default)",
    ".": "1  (class 1) missing data",
}


def process_mbmrb(input_path: str, output_path: str):
    with open(input_path, newline="") as fin:
        reader = csv.DictReader(fin)
        rows = list(reader)

    print(f"[Input]  {input_path}  ({len(rows)} samples)")

    valid, skipped = 0, 0
    stats = Counter()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["sequence", "label"])

        for row in rows:
            seq = row["sequence"].strip().upper()
            raw_label = row["label"].strip()

            if len(seq) != len(raw_label):
                skipped += 1
                continue

            converted = []
            for ch in raw_label:
                mapped = LABEL_MAP.get(ch, "_")
                converted.append(mapped)
                stats[ch] += 1

            writer.writerow([seq, "".join(converted)])
            valid += 1

    print(f"[Output] {output_path}  ({valid} valid, {skipped} skipped)")
    print()
    print("Label distribution (raw → converted):")
    for ch in sorted(stats):
        arrow = LABEL_MAP.get(ch, "_")
        origin = MAPPING_TABLE.get(ch, f"_  (ignore)  raw char '{ch}'")
        print(f"  {ch} -> {arrow:>3s}   x{stats[ch]:>8d}   {origin}")

    total = sum(stats.values())
    cls0 = sum(v for k, v in stats.items() if LABEL_MAP.get(k) == "0")
    cls1 = sum(v for k, v in stats.items() if LABEL_MAP.get(k) == "1")
    ign = total - cls0 - cls1
    print()
    print(f"Summary:")
    print(f"  class 0: {cls0:>8d}  ({cls0/total*100:.1f}%)")
    print(f"  class 1: {cls1:>8d}  ({cls1/total*100:.1f}%)")
    print(f"  ignore:  {ign:>8d}  ({ign/total*100:.1f}%)")
    print(f"  total:   {total:>8d}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess mBMRB CSV for PLM training")
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "Dataset", "mBMRB.csv"
        ),
        help="Path to mBMRB.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("..", "Training", "examples", "mbmrb_processed.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    process_mbmrb(args.input, args.output)


if __name__ == "__main__":
    main()
