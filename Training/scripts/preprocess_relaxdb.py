#!/usr/bin/env python3
"""
Preprocess RelaxDB dataset for token classification training.

Label mapping (from relaxdb_data.csv):
  t, x  →  _  (ignore in loss — no data / not reported)
  p, A, v → 0  (no motion / default)
  ., b, ^ → 1  (motion / exchange detected)

Output: CSV with columns: sequence, label
  label is a string of '0', '1', and '_' (same length as sequence).
"""
import csv
import os
import argparse

LABEL_MAP = {
    't': '_', 'x': '_',
    'p': '0', 'A': '0', 'v': '0',
    '.': '1', 'b': '1', '^': '1',
}

MAPPING_TABLE = {
    't': '_  (ignore)  no data due to disordered terminus',
    'x': '_  (ignore)  no data; R1/R2/NOE not reported',
    'p': '0  (class 0) proline (not evaluated)',
    'A': '0  (class 0) no special annotation (default)',
    'v': '0  (class 0) fast internal motion',
    '.': '1  (class 1) missing data',
    'b': '1  (class 1) mixed fast and slow motion',
    '^': '1  (class 1) chemical exchange detected (Rex)',
}


def process_relaxdb(input_path: str, output_path: str):
    with open(input_path, newline="") as fin:
        reader = csv.DictReader(fin)
        rows = list(reader)

    print(f"[Input]  {input_path}  ({len(rows)} samples)")

    valid, skipped = 0, 0
    stats = {k: 0 for k in LABEL_MAP}

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
                stats[ch] = stats.get(ch, 0) + 1

            writer.writerow([seq, "".join(converted)])
            valid += 1

    print(f"[Output] {output_path}  ({valid} valid, {skipped} skipped)")
    print()
    print("Label distribution (raw → converted):")
    for ch in sorted(stats):
        print(f"  {ch} -> {LABEL_MAP[ch]:>3s}   x{stats[ch]:>6d}   {MAPPING_TABLE[ch]}")
    print()
    total_0 = sum(v for k, v in stats.items() if LABEL_MAP[k] == "0")
    total_1 = sum(v for k, v in stats.items() if LABEL_MAP[k] == "1")
    total_x = sum(v for k, v in stats.items() if LABEL_MAP[k] == "_")
    print(f"Summary:  class 0: {total_0}  |  class 1: {total_1}  |  ignore: {total_x}")
    print(f"          training targets (0/1): {total_0 + total_1}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess RelaxDB CSV for PLM training")
    parser.add_argument("--input", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "..", "Dataset", "relaxdb_data.csv"),
                        help="Path to relaxdb_data.csv")
    parser.add_argument("--output", type=str, default="./examples/relaxdb_processed.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    process_relaxdb(args.input, args.output)


if __name__ == "__main__":
    main()
