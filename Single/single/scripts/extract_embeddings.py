#!/usr/bin/env python
"""
Extract hidden state embeddings from the fine-tuned ESM2-8M model for SAE training.

Usage:
    python scripts/extract_embeddings.py \
        --ckpt_path ../Outputs/my_experiment/checkpoints/epoch_0.26_f1_7904 \
        --sequences_csv ../Dataset/mBMRB.csv \
        --output_dir ../data/embeddings/esm2_8m/layer_6 \
        --layer 6 \
        --batch_size 8
"""

import os
import sys

# Allow running directly from the repository root, e.g.
#   python Single/single/scripts/analyze_sequence.py ...
# without `cd Single` or installing the package (the `single` package lives at
# Single/single/, two levels up from this file).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm

from single.embedders.ft_esm import FineTunedESMEmbedder


def extract_embeddings(
    ckpt_path: Path,
    sequences_csv: Path,
    output_dir: Optional[Path] = None,
    experiment: Optional[str] = None,
    exp_dir: Optional[Path] = None,
    source: Optional[str] = None,
    layer: int = 6,
    batch_size: int = 8,
    max_length: int = 512,
    label_column: Optional[str] = None,
    sequence_column: str = "sequence",
    n_shards: int = 5,
    label_map: str = "mBMRB",
    min_seq_len: int = 0,
    max_seq_len: int = 10_000,
    max_sequences: Optional[int] = None,
):
    from single.paths import resolve_experiment

    # Prefer explicit output_dir (legacy); else route into the experiment dir.
    if output_dir is None:
        exp = resolve_experiment(exp_dir=exp_dir, name=experiment, source=source)
        output_dir = exp.embeddings_dir(layer=layer)
        print(f"Experiment dir: {exp.dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from single.label_maps import get_label_map, resolve_columns
    label_map_spec = get_label_map(label_map)
    # The label map describes the dataset's columns; use them unless the user
    # explicitly overrode them on the command line.
    sequence_column, label_column = resolve_columns(
        label_map_spec, sequence_column, label_column
    )

    embedder = FineTunedESMEmbedder(
        ckpt_path=ckpt_path,
        max_length=max_length,
    )
    print(f"Embedder ready: {embedder.embedding_dim}D, {embedder.n_layers} layers")
    print(f"Extracting layer {layer} (0-indexed, total {embedder.n_layers} layers)")
    print(f"Label map: {label_map}")

    # Shared loader: auto-detect separator, optional length filter (must MATCH
    # build_concept_matrix's filter so embedding/concept shards contain the SAME
    # proteins), and fixed-seed shuffle+shard — all in single.data.
    from single.data import load_sequences_df, shuffled_shards
    df = load_sequences_df(sequences_csv, sequence_column=sequence_column,
                           min_seq_len=min_seq_len, max_seq_len=max_seq_len,
                           max_sequences=max_sequences)
    sequences = df[sequence_column].tolist()
    print(f"Loaded {len(sequences)} sequences")

    has_labels = label_column is not None and label_column in df.columns
    if has_labels:
        df[label_column] = df[label_column].fillna("").astype(str)

    # Same sharding as build_concept_matrix so shards align.
    if n_shards > 1:
        shards = shuffled_shards(df, n_shards)
        for shard_id, shard_df in enumerate(shards):
            shard_seqs = shard_df[sequence_column].tolist()
            shard_dir = output_dir / f"shard_{shard_id}"
            shard_dir.mkdir(parents=True, exist_ok=True)

            if has_labels and label_column in shard_df.columns:
                shard_labels = shard_df[label_column].tolist()
                result = embedder.extract_embeddings_with_labels(
                    shard_seqs, shard_labels, layer=layer, batch_size=batch_size,
                    label_map=label_map_spec,
                )
            else:
                result = embedder.extract_embeddings(
                    shard_seqs, layer=layer, batch_size=batch_size
                )
                result = {"embeddings": result}
            torch.save(result, shard_dir / "embeddings.pt")

            print(f"Shard {shard_id}: {result['embeddings'].shape[0]} tokens → {shard_dir / 'embeddings.pt'}")
    else:
        if has_labels:
            result = embedder.extract_embeddings_with_labels(
                sequences, df[label_column].tolist(), layer=layer, batch_size=batch_size,
                label_map=label_map_spec,
            )
        else:
            result = embedder.extract_embeddings(
                sequences, layer=layer, batch_size=batch_size
            )
            result = {"embeddings": result}
        torch.save(result, output_dir / "embeddings.pt")

        print(f"Saved {result['embeddings'].shape[0]} tokens to {output_dir / 'embeddings.pt'}")

    print("Done!")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract hidden states from fine-tuned ESM2-8M")
    parser.add_argument("--ckpt_path", type=Path, required=True, help="Path to fine-tuned model checkpoint")
    parser.add_argument("--sequences_csv", type=Path, required=True, help="CSV with sequences and labels")
    parser.add_argument("--source", type=str, default=None,
                        help="Data-source id; nests outputs under Outputs/<experiment>/<source> (default: flat)")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Experiment name; creates Outputs/<experiment>_<ts>/")
    parser.add_argument("--exp_dir", type=Path, default=None,
                        help="Reuse an existing experiment directory")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Explicit output dir (overrides experiment routing)")
    parser.add_argument("--layer", type=int, default=6, help="Transformer layer to extract (0=embedding)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--label_column", type=str, default=None, help="Column name for labels")
    parser.add_argument("--sequence_column", type=str, default="sequence")
    parser.add_argument("--n_shards", type=int, default=5, help="Number of shards to split data into")
    parser.add_argument("--label_map", type=str, default="mBMRB",
                        help="Label encoding preset name or path to YAML label-map file")
    parser.add_argument("--min_seq_len", type=int, default=0,
                        help="Drop sequences shorter than this (must match concept build)")
    parser.add_argument("--max_sequences", type=int, default=None,
                        help="Deterministic subset; must match extract_embeddings --max_sequences")
    parser.add_argument("--max_seq_len", type=int, default=10000,
                        help="Drop sequences longer than this (must match concept build)")
    args = parser.parse_args(argv)
    extract_embeddings(**vars(args))


if __name__ == "__main__":
    main()
