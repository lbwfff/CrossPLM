#!/usr/bin/env python
"""
Causal intervention: perturb a single SAE feature and measure prediction changes.

Unlike fidelity (which replaces the WHOLE layer), this zeroes or amplifies ONE
SAE feature at a time and observes whether the fine-tuned model's per-residue
predictions flip. If zeroing "feature #42" changes the model's flexibility
predictions, that feature causally drives those decisions.

Usage:
    python -m single.scripts.evaluate_intervention \
        --ckpt_path ../Outputs/my_experiment/checkpoints/best \
        --sequences_csv ../Dataset/mBMRB.csv \
        --sae_dir ../Outputs/.../sae \
        --feature_idx 234 --mode zero \
        --label_column label --label_map mBMRB \
        --layer 6 --max_sequences 200
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
from single.train.fidelity import evaluate_intervention


def evaluate(
    ckpt_path: Path,
    sequences_csv: Path,
    feature_idx: int,
    mode: str = "zero",
    scale: float = 2.0,
    experiment: Optional[str] = None,
    exp_dir: Optional[Path] = None,
    source: Optional[str] = None,
    output_dir: Optional[Path] = None,
    layer: int = 6,
    label_column: str = "label",
    sequence_column: str = "sequence",
    label_map: str = "mBMRB",
    batch_size: int = 8,
    max_length: int = 512,
    max_sequences: Optional[int] = None,
    sae_dir: Optional[Path] = None,
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
    label_map_spec = get_label_map(label_map)
    # The label map describes the dataset's columns; use them unless the user
    # explicitly overrode them on the command line.
    sequence_column, label_column = resolve_columns(
        label_map_spec, sequence_column, label_column
    )


    print("=" * 60)
    print("SAE CAUSAL INTERVENTION")
    print("=" * 60)

    print(f"\nLoading fine-tuned model from {ckpt_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path, local_files_only=True, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"
    model = AutoModelForTokenClassification.from_pretrained(
        ckpt_path, local_files_only=True, trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    print(f"Loading SAE from {sae_dir}...")
    sae = load_sae(sae_dir, device=device)
    print(f"  {sae.__class__.__name__}: {sae.dict_size} features, {sae.activation_dim}D")
    if feature_idx >= sae.dict_size:
        raise ValueError(f"feature_idx {feature_idx} out of range [0, {sae.dict_size})")

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

    print(f"\nIntervening on feature #{feature_idx} (mode={mode}, scale={scale})...")
    results = evaluate_intervention(
        model=model,
        sae=sae,
        layer_idx=layer,
        sequences=sequences,
        labels=labels,
        tokenizer=tokenizer,
        label_map_spec=label_map_spec,
        feature_idx=feature_idx,
        mode=mode,
        scale=scale,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )

    print("\n" + "=" * 60)
    print("INTERVENTION RESULTS")
    print("=" * 60)
    print(f"  Feature:      #{results['feature_idx']}")
    print(f"  Mode:         {results['mode']} (scale={results['scale']})")
    print(f"  Layer:        {results['layer_idx']}")
    print(f"  Tokens:       {results['n_tokens']:,}  (active: {results['active_tokens']:,}, "
          f"inactive: {results['inactive_tokens']:,})")
    print(f"  Flip rate (all):          {results['flip_rate']*100:.2f}%")
    print(f"  Flip rate (on active):    {results['flip_rate_on_active']*100:.2f}%")
    print(f"  Flip rate (on inactive):  {results['flip_rate_on_inactive']*100:.2f}%  ← control")
    print(f"  Flipped → pos (active):   {results['flip_to_pos']*100:.2f}%")
    print(f"  Flipped → neg (active):   {results['flip_to_neg']*100:.2f}%")
    print("=" * 60)
    print("  Interpretation:")
    print("    'flip_rate_on_active' = predictions changed on tokens where the feature")
    print("    actually fires → the causal effect of this feature.")
    print("    Compare to 'flip_rate_on_inactive' (control). If active >> inactive, the")
    print("    effect is real; if similar, it's noise.")
    print("    Flipped to positive = steering pushes predictions toward the positive class.")
    print("=" * 60)

    out_path = output_dir / f"intervention_feat{feature_idx}_{mode}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Causal intervention on a single SAE feature")
    parser.add_argument("--ckpt_path", type=Path, required=True, help="Fine-tuned model checkpoint")
    parser.add_argument("--sequences_csv", type=Path, required=True, help="CSV with sequences+labels")
    parser.add_argument("--sae_dir", type=Path, default=None,
                        help="Trained SAE dir (default: Outputs/<experiment>/sae)")
    parser.add_argument("--feature_idx", type=int, required=True, help="Feature to perturb")
    parser.add_argument("--mode", type=str, default="zero", choices=["zero", "amplify", "set"],
                        help="How to perturb the feature")
    parser.add_argument("--scale", type=float, default=2.0,
                        help="Multiplier (amplify) or value (set) for the perturbation")
    parser.add_argument("--source", type=str, default=None,
                        help="Data-source id; nests outputs under Outputs/<experiment>/<source> (default: flat)")
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
    args = parser.parse_args(argv)
    evaluate(**vars(args))


if __name__ == "__main__":
    main()
