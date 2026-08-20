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
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm

from single.embedders.ft_esm import FineTunedESMEmbedder


def _slug_for_hub_id(hub_id: str) -> str:
    """Sanitise a Hub ID (e.g. ``facebook/esm2_t6_8M_UR50D``) for a filesystem path."""
    return hub_id.replace("/", "--").replace(":", "--")


def _ensure_pretrained_dir(hub_id: str) -> Path:
    """Return a local ``Outputs/_pretrained/<slug>`` directory for a Hub ID.

    The directory is the single on-disk copy of the raw ``M0`` weights so that
    ``MA`` and ``MB`` (which share the same backbone) do not materialise two
    independent ``M0`` checkpoints.  If the directory does not yet exist the
    model and tokenizer are fetched from the Hub once and saved there.
    """
    repo_root = Path(__file__).resolve().parents[3]
    # Prefer the repo-root Outputs/_pretrained so every experiment shares it
    target = repo_root / "Outputs" / "_pretrained" / _slug_for_hub_id(hub_id)
    if target.exists() and (target / "config.json").exists():
        return target
    # Fallback: allow a Hub ID to be used directly without persisting, but when
    # persisting prefer the central location.
    print(f"[M0] Ensuring central pretrained cache at {target} for {hub_id} ...")
    target.mkdir(parents=True, exist_ok=True)
    # Lazy import so the fetch only happens for base M0
    from transformers import AutoModel, AutoTokenizer

    # Tokenizer first (cheap)
    tok = AutoTokenizer.from_pretrained(hub_id, trust_remote_code=True)
    tok.save_pretrained(str(target))
    # Encoder only is sufficient for SAE0; AutoModel is the minimal artefact
    try:
        mdl = AutoModel.from_pretrained(hub_id, trust_remote_code=True)
    except Exception:
        from transformers import AutoModelForMaskedLM

        mdl = AutoModelForMaskedLM.from_pretrained(hub_id, trust_remote_code=True)
    mdl.save_pretrained(str(target))
    # Minimal provenance for the central copy
    (target / "m0_provenance.json").write_text(
        json.dumps({"hub_id": hub_id, "slug": _slug_for_hub_id(hub_id)}, indent=2)
    )
    print(f"[M0] Saved central pretrained checkpoint to {target}")
    return target


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
    model_type: str = "ft",
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

    # --model_type base: allow a Hub ID (e.g. facebook/esm2_t6_8M_UR50D) and
    # ensure the central Outputs/_pretrained/<slug> single-copy is used so that
    # MA and MB (same backbone) do not materialise two M0 directories.
    resolved_ckpt = ckpt_path
    if str(model_type).lower() == "base":
        # Hub ID (no local dir) → centralise; local path → use as-is
        ckpt_str = str(ckpt_path)
        if not Path(ckpt_str).exists():
            # Heuristic: Hub IDs contain a "/" and are not a local file
            if "/" in ckpt_str:
                resolved_ckpt = _ensure_pretrained_dir(ckpt_str)
            else:
                # Might be a local _pretrained/<slug> already
                pass
        print(f"[Model type] base (M0) — encoder-only, hub={ckpt_path} → resolved={resolved_ckpt}")
    else:
        print(f"[Model type] ft (fine-tuned) — token-classification head, ckpt={ckpt_path}")

    embedder = FineTunedESMEmbedder(
        ckpt_path=resolved_ckpt,
        max_length=max_length,
        model_type=model_type,
    )
    print(f"Embedder ready: {embedder.embedding_dim}D, {embedder.n_layers} layers")
    print(f"Extracting layer {layer} (0-indexed, total {embedder.n_layers} layers)")
    print(f"Label map: {label_map}  model_type: {model_type}")

    # Shared loader: auto-detect separator, optional length filter (must MATCH
    # build_concept_matrix's filter so embedding/concept shards contain the SAME
    # proteins), and fixed-seed shuffle+shard — all in single.data.
    from single.data import (
        load_sequences_df,
        shuffled_shards,
        sequence_hash,
        drop_mismatched_lengths,
        validate_sequence_label_lengths,
    )
    df = load_sequences_df(sequences_csv, sequence_column=sequence_column,
                           min_seq_len=min_seq_len, max_seq_len=max_seq_len,
                           max_sequences=max_sequences)
    n_before = len(df)
    sequences = df[sequence_column].tolist()
    print(f"Loaded {len(sequences)} sequences")

    has_labels = label_column is not None and label_column in df.columns
    n_dropped_mismatched = 0
    if has_labels:
        df[label_column] = df[label_column].fillna("").astype(str)
        df, n_dropped_mismatched = drop_mismatched_lengths(df, sequence_column, label_column)
        if n_dropped_mismatched:
            print(f"[Filter] Dropped {n_dropped_mismatched} row(s) with sequence/label length mismatches "
                  f"(kept {len(df)}/{n_before}, matches Training's silent drop).")
            sequences = df[sequence_column].tolist()
        # Any remaining mismatch is a bug in drop logic; keep the strict check as a safety net
        validate_sequence_label_lengths(df, sequence_column, label_column)

    metadata = {
        "sequence_column": sequence_column,
        "label_column": label_column,
        "max_length": max_length,
        "min_seq_len": min_seq_len,
        "max_seq_len": max_seq_len,
        "max_sequences": max_sequences,
        "n_shards": n_shards,
        "shuffle_seed": 42,
        "residues_per_sequence": None,
        "n_loaded": int(n_before),
        "n_dropped_mismatched": int(n_dropped_mismatched),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

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
            residue_lengths = embedder.residue_lengths(shard_seqs, batch_size=batch_size)
            residue_rows = []
            for (_, row), residue_length in zip(shard_df.iterrows(), residue_lengths):
                seq = str(row[sequence_column])
                seq_hash = sequence_hash(seq)
                for pos, aa in enumerate(seq[:residue_length]):
                    residue = {
                        "sequence_hash": seq_hash,
                        "amino_acid": aa.upper(),
                        "position": pos,
                    }
                    if "Entry" in shard_df.columns:
                        residue["Entry"] = str(row["Entry"])
                    residue_rows.append(residue)
            pd.DataFrame(residue_rows).to_csv(shard_dir / "residues.csv", index=False)

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
        residue_lengths = embedder.residue_lengths(sequences, batch_size=batch_size)
        residue_rows = []
        for (_, row), residue_length in zip(df.iterrows(), residue_lengths):
            seq = str(row[sequence_column])
            seq_hash = sequence_hash(seq)
            for pos, aa in enumerate(seq[:residue_length]):
                residue = {
                    "sequence_hash": seq_hash,
                    "amino_acid": aa.upper(),
                    "position": pos,
                }
                if "Entry" in df.columns:
                    residue["Entry"] = str(row["Entry"])
                residue_rows.append(residue)
        pd.DataFrame(residue_rows).to_csv(output_dir / "residues.csv", index=False)

        print(f"Saved {result['embeddings'].shape[0]} tokens to {output_dir / 'embeddings.pt'}")

    print("Done!")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract hidden states from ESM (fine-tuned or base M0)")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to model checkpoint (local dir, e.g. Outputs/exp/checkpoints/best) "
                             "or Hub ID for base M0 (e.g. facebook/esm2_t6_8M_UR50D / Synthyra/ESM2-8M)")
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
    parser.add_argument("--model_type", type=str, default="ft", choices=["ft", "base"],
                        help="ft = fine-tuned token-classification checkpoint (MA/MB); "
                             "base = pre-trained encoder-only M0 (Hub ID, no head)")
    args = parser.parse_args(argv)
    # argparse gives ckpt_path as str for base Hub IDs; keep that type for hub handling
    kwargs = vars(args)
    # Keep ckpt_path as str when it is a Hub ID so _ensure_pretrained_dir can run
    extract_embeddings(**kwargs)


if __name__ == "__main__":
    main()
