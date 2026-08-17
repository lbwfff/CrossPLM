#!/usr/bin/env python
"""
Evaluate SAE fidelity on a fine-tuned token-classification PLM.

Measures how well a SAE preserves the model's task performance by replacing
layer activations with SAE reconstructions and comparing task loss.

Usage:
    python -m single.scripts.evaluate_fidelity \
        --ckpt_path ../Outputs/my_experiment/checkpoints/best \
        --sequences_csv ../Dataset/mBMRB.csv \
        --sae_dir ../Outputs/.../sae \
        --label_column label --label_map mBMRB \
        --layer 6
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

from single.sae.inference import load_sae
from single.label_maps import get_label_map, resolve_columns
from single.train.fidelity import evaluate_fidelity


def evaluate(
    ckpt_path: Path,
    sequences_csv: Path,
    sae_dir: Optional[Path] = None,
    experiment: Optional[str] = None,
    exp_dir: Optional[Path] = None,
    source: Optional[str] = None,
    output_dir: Optional[Path] = None,
    layer: int = 6,
    label_column: str = "label",
    sequence_column: str = "sequence",
    label_map: Optional[str] = None,
    batch_size: int = 8,
    max_length: int = 512,
    max_sequences: Optional[int] = None,
):
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from single.paths import resolve_experiment

    # --sae_dir and --output_dir default into Outputs/<experiment>/.
    exp = None
    if sae_dir is None or output_dir is None:
        exp = resolve_experiment(exp_dir=exp_dir, name=experiment, source=source)
    if sae_dir is None:
        sae_dir = exp.sae_dir
        print(f"  SAE dir (inferred): {sae_dir}")
    if output_dir is None:
        output_dir = exp.analysis_dir
        print(f"Experiment dir: {exp.dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if label_map is None:
        sidecar = Path(ckpt_path) / "label_map.json"
        if sidecar.exists():
            with open(sidecar) as f:
                saved = json.load(f)
            label_map_spec = {
                "mapping": {str(k): int(v) for k, v in saved["label2id"].items()},
                "ignore": saved.get("ignore", "_"),
                "positive_class": int(saved.get("positive_class", 1)),
                "class_names": {
                    int(k): str(v) for k, v in saved.get("id2label", {}).items()
                },
            }
            print(f"Using label map from checkpoint: {sidecar}")
        else:
            label_map = "mBMRB"
            label_map_spec = get_label_map(label_map)
            print("No checkpoint label map found; falling back to mBMRB")
    else:
        label_map_spec = get_label_map(label_map)
    # The label map describes the dataset's columns; use them unless the user
    # explicitly overrode them on the command line.
    sequence_column, label_column = resolve_columns(
        label_map_spec, sequence_column, label_column
    )


    print("=" * 60)
    print("SAE FIDELITY EVALUATION")
    print("=" * 60)

    # Load model + tokenizer
    print(f"\nLoading fine-tuned model from {ckpt_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path, local_files_only=True, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"
    model = AutoModelForTokenClassification.from_pretrained(
        ckpt_path, local_files_only=True, trust_remote_code=True
    )
    model.to(device)
    model.eval()

    # Load SAE
    print(f"Loading SAE from {sae_dir}...")
    sae = load_sae(sae_dir, device=device)
    print(f"  {sae.__class__.__name__}: {sae.dict_size} features, {sae.activation_dim}D")

    # Load sequences + labels
    print(f"\nLoading data from {sequences_csv}...")
    from single.data import load_sequences_df, validate_sequence_label_lengths
    df = load_sequences_df(sequences_csv, sequence_column=sequence_column,
                           max_sequences=max_sequences)
    df[label_column] = df[label_column].fillna("").astype(str)
    validate_sequence_label_lengths(df, sequence_column, label_column)
    sequences = df[sequence_column].tolist()
    labels = df[label_column].tolist()
    print(f"  {len(sequences)} sequences")

    # Evaluate
    print(f"\nEvaluating fidelity at layer {layer}...")
    results = evaluate_fidelity(
        model=model,
        sae=sae,
        layer_idx=layer,
        sequences=sequences,
        labels=labels,
        tokenizer=tokenizer,
        label_map_spec=label_map_spec,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )

    # Report
    print("\n" + "=" * 60)
    print("FIDELITY RESULTS")
    print("=" * 60)
    print(f"  Layer: {results['layer_idx']}")
    print(f"  Sequences: {results['n_sequences']}")
    print(f"  CE (original activations):   {results['ce_orig']:.4f}")
    print(f"  CE (SAE reconstructions):    {results['ce_sae']:.4f}")
    print(f"  CE (zero ablation):          {results['ce_zero']:.4f}")
    recovered = results.get("loss_recovered_pct")
    recovered_text = "N/A (invalid zero-ablation baseline)" if recovered is None else f"{recovered:.2f}%"
    print(f"  Loss recovered:              {recovered_text}")
    raw = results.get("loss_recovered_raw")
    if results.get("sae_better_than_original") and raw is not None:
        print(f"    (unclipped {raw:.2f}% — SAE reconstruction loss is LOWER than the "
              "original activations; not an actual recovery)")
    elif raw is not None:
        print(f"    (unclipped {raw:.2f}%)")
    print(f"  Reconstruction MSE:          {results['reconstruction_mse']:.6f}")
    print("=" * 60)
    print("  Interpretation:")
    print("    100% = SAE perfectly preserves task info at this layer")
    print("     0%  = SAE is as harmful as zeroing the layer")
    print("    Reconstruction MSE = mean squared error between SAE reconstruction")
    print("      and the original hidden states (sanity check). Lower is better.")
    print("=" * 60)

    # Save
    out_path = output_dir / "fidelity_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate SAE fidelity on a fine-tuned PLM")
    parser.add_argument("--ckpt_path", type=Path, required=True, help="Fine-tuned model checkpoint")
    parser.add_argument("--sequences_csv", type=Path, required=True, help="CSV with sequences+labels")
    parser.add_argument("--sae_dir", type=Path, default=None,
                        help="Trained SAE dir (default: Outputs/<experiment>/sae)")
    parser.add_argument("--source", type=str, default=None,
                        help="Data-source id; nests outputs under Outputs/<experiment>/<source> (default: flat)")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--exp_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--sequence_column", type=str, default="sequence")
    parser.add_argument("--label_map", type=str, default=None,
                        help="Label map preset/YAML (default: checkpoint sidecar, then mBMRB)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_sequences", type=int, default=None,
                        help="Limit number of sequences for a quick test")
    args = parser.parse_args(argv)
    evaluate(**vars(args))


if __name__ == "__main__":
    main()
