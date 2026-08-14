#!/usr/bin/env python
"""
Train a Sparse Autoencoder on extracted PLM embeddings.

Usage:
    python scripts/train_sae.py \
        --embeddings_dir ../data/embeddings/esm2_8m/layer_6 \
        --save_dir ../models/sae/layer_6 \
        --activation_dim 320 \
        --dict_size 1280 \
        --steps 10000 \
        --batch_size 512 \
        --l1_penalty 0.06
"""

import argparse
from pathlib import Path
from typing import Optional

from single.configs import SAEConfig, TrainingConfig, DataConfig, CheckpointConfig
from single.train.training_run import SAETrainingRun


def train_sae(
    embeddings_dir: Path,
    save_dir: Optional[Path] = None,
    experiment: Optional[str] = None,
    exp_dir: Optional[Path] = None,
    activation_dim: int = 320,
    dict_size: int = 1280,
    expansion_factor: int = 4,
    lr: float = 2e-4,
    steps: int = 10000,
    batch_size: int = 512,
    l1_penalty: float = 0.06,
    warmup_steps: int = 1000,
    decay_start: int = 8000,
    save_steps: int = 2000,
    seed: int = 42,
    resume_from: Optional[Path] = None,
):
    from single.paths import resolve_experiment

    # Prefer explicit save_dir (legacy); else route into the experiment dir.
    if save_dir is None:
        exp = resolve_experiment(exp_dir=exp_dir, name=experiment)
        save_dir = exp.sae_dir
        print(f"Experiment dir: {exp.dir}")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = DataConfig(
        plm_embd_dir=embeddings_dir,
    )

    sae_cfg = SAEConfig(
        activation_dim=activation_dim,
        dict_size=dict_size,
        expansion_factor=expansion_factor,
    )

    train_cfg = TrainingConfig(
        lr=lr,
        steps=steps,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        decay_start=decay_start,
        l1_penalty=l1_penalty,
        seed=seed,
    )

    checkpoint_cfg = CheckpointConfig(
        save_dir=save_dir,
        save_steps=save_steps,
    )

    print("=" * 60)
    print("SAE Training Configuration")
    print("=" * 60)
    print(f"Embeddings: {embeddings_dir}")
    print(f"Model: {activation_dim}D → {dict_size} features")
    print(f"Steps: {steps}, Batch: {batch_size}, LR: {lr:.1e}")
    print(f"L1 penalty: {l1_penalty}")
    print(f"Save to: {save_dir}")
    print()

    training_run = SAETrainingRun.from_configs(
        data_cfg=data_cfg,
        sae_cfg=sae_cfg,
        train_cfg=train_cfg,
        checkpoint_cfg=checkpoint_cfg,
    )

    if resume_from is not None:
        training_run.resume_from_checkpoint(resume_from)

    training_run.run()

    print(f"\nModel saved to {save_dir / 'ae.pt'}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAE on PLM embeddings")
    parser.add_argument("--embeddings_dir", type=Path, required=True)
    parser.add_argument("--experiment", type=str, default=None,
                        help="Experiment name; creates Outputs/<experiment>_<ts>/")
    parser.add_argument("--exp_dir", type=Path, default=None,
                        help="Reuse an existing experiment directory")
    parser.add_argument("--save_dir", type=Path, default=None,
                        help="Explicit save dir (overrides experiment routing)")
    parser.add_argument("--activation_dim", type=int, default=320)
    parser.add_argument("--dict_size", type=int, default=1280)
    parser.add_argument("--expansion_factor", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--l1_penalty", type=float, default=0.06)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--decay_start", type=int, default=8000)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=Path, default=None)
    args = parser.parse_args()
    train_sae(**{k: v for k, v in vars(args).items() if v is not None})
