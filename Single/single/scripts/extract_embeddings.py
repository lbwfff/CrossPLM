#!/usr/bin/env python
"""
Extract hidden state embeddings from the fine-tuned ESM2-8M model for SAE training.

Usage:
    python scripts/extract_embeddings.py \
        --ckpt_path ../Training/outputs/tasks/just_test_20260813_113758/checkpoints/epoch_0.26_f1_7904 \
        --sequences_csv ../Dataset/mBMRB.csv \
        --output_dir ../data/embeddings/esm2_8m/layer_6 \
        --layer 6 \
        --batch_size 8
"""

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
    output_dir: Path,
    layer: int = 6,
    batch_size: int = 8,
    max_length: int = 512,
    label_column: Optional[str] = None,
    sequence_column: str = "sequence",
    n_shards: int = 5,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embedder = FineTunedESMEmbedder(
        ckpt_path=ckpt_path,
        max_length=max_length,
    )
    print(f"Embedder ready: {embedder.embedding_dim}D, {embedder.n_layers} layers")
    print(f"Extracting layer {layer} (0-indexed, total {embedder.n_layers} layers)")

    df = pd.read_csv(sequences_csv)
    sequences = df[sequence_column].tolist()
    print(f"Loaded {len(sequences)} sequences")

    has_labels = label_column is not None and label_column in df.columns
    if has_labels:
        df[label_column] = df[label_column].fillna("").astype(str)

    if n_shards > 1:
        shards = [df.iloc[i::n_shards] for i in range(n_shards)]
        for shard_id, shard_df in enumerate(shards):
            shard_seqs = shard_df[sequence_column].tolist()
            shard_dir = output_dir / f"shard_{shard_id}"
            shard_dir.mkdir(parents=True, exist_ok=True)

            if has_labels and label_column in shard_df.columns:
                shard_labels = shard_df[label_column].tolist()
                result = embedder.extract_embeddings_with_labels(
                    shard_seqs, shard_labels, layer=layer, batch_size=batch_size
                )
                torch.save(result, shard_dir / "activations.pt")
            else:
                result = embedder.extract_embeddings(
                    shard_seqs, layer=layer, batch_size=batch_size
                )
                torch.save({"embeddings": result}, shard_dir / "activations.pt")

            print(f"Shard {shard_id}: {result['embeddings'].shape[0]} tokens → {shard_dir / 'activations.pt'}")
    else:
        if has_labels:
            result = embedder.extract_embeddings_with_labels(
                sequences, df[label_column].tolist(), layer=layer, batch_size=batch_size
            )
            torch.save(result, output_dir / "activations.pt")
        else:
            result = embedder.extract_embeddings(
                sequences, layer=layer, batch_size=batch_size
            )
            torch.save({"embeddings": result}, output_dir / "activations.pt")

        print(f"Saved {result['embeddings'].shape[0]} tokens to {output_dir / 'activations.pt'}")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract hidden states from fine-tuned ESM2-8M")
    parser.add_argument("--ckpt_path", type=Path, required=True, help="Path to fine-tuned model checkpoint")
    parser.add_argument("--sequences_csv", type=Path, required=True, help="CSV with sequences and labels")
    parser.add_argument("--output_dir", type=Path, default=Path("../Outputs/embeddings/esm2_8m/layer_6"),
                        help="Directory to save embeddings")
    parser.add_argument("--layer", type=int, default=6, help="Transformer layer to extract (0=embedding)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--label_column", type=str, default=None, help="Column name for labels")
    parser.add_argument("--sequence_column", type=str, default="sequence")
    parser.add_argument("--n_shards", type=int, default=5, help="Number of shards to split data into")
    args = parser.parse_args()
    extract_embeddings(**vars(args))
