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
from single.label_maps import get_label_map
from single.train.fidelity import evaluate_fidelity


def evaluate(
    ckpt_path: Path,
    sequences_csv: Path,
    sae_dir: Optional[Path] = None,
    experiment: Optional[str] = None,
    exp_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    layer: int = 6,
    label_column: str = "label",
    sequence_column: str = "sequence",
    label_map: str = "mBMRB",
    batch_size: int = 8,
    max_length: int = 512,
    max_sequences: Optional[int] = None,
):
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from single.paths import resolve_experiment

    # --sae_dir and --output_dir default into Outputs/<experiment>/.
    exp = None
    if sae_dir is None or output_dir is None:
        exp = resolve_experiment(exp_dir=exp_dir, name=experiment)
    if sae_dir is None:
        sae_dir = exp.sae_dir
        print(f"  SAE dir (inferred): {sae_dir}")
    if output_dir is None:
        output_dir = exp.analysis_dir
        print(f"Experiment dir: {exp.dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_map_spec = get_label_map(label_map)

    print("=" * 60)
    print("SAE FIDELITY EVALUATION")
    print("=" * 60)

    # Load model + tokenizer
    print(f"\nLoading fine-tuned model from {ckpt_path}...")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, local_files_only=True)
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
    with open(sequences_csv, "r") as f:
        first = f.readline()
    sep = "\t" if first.count("\t") > first.count(",") else ","
    df = pd.read_csv(sequences_csv, sep=sep, low_memory=False)
    df[sequence_column] = df[sequence_column].fillna("").astype(str)
    df[label_column] = df[label_column].fillna("").astype(str)
    if max_sequences is not None:
        df = df.head(max_sequences)
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
    print(f"  Loss recovered:              {results['loss_recovered_pct']:.2f}%")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SAE fidelity on a fine-tuned PLM")
    parser.add_argument("--ckpt_path", type=Path, required=True, help="Fine-tuned model checkpoint")
    parser.add_argument("--sequences_csv", type=Path, required=True, help="CSV with sequences+labels")
    parser.add_argument("--sae_dir", type=Path, default=None,
                        help="Trained SAE dir (default: Outputs/<experiment>/sae)")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--exp_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--sequence_column", type=str, default="sequence")
    parser.add_argument("--label_map", type=str, default="mBMRB")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_sequences", type=int, default=None,
                        help="Limit number of sequences for a quick test")
    args = parser.parse_args()
    evaluate(**vars(args))
