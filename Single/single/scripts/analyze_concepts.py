#!/usr/bin/env python
"""
Align SAE features against Swiss-Prot / UniProtKB biological concepts.

Two steps:

1. Build concept matrices from a UniProtKB TSV export:
     python -m single.scripts.analyze_concepts build \
         --annotations_tsv ../Dataset/uniprotkb_swissprot.tsv.gz \
         --concepts_dir ../Outputs/concepts

2. Align a trained SAE's features to those concepts:
     python -m single.scripts.analyze_concepts align \
         --sae_dir ../Outputs/sae/esm2_8m_l6_d640_<ts> \
         --embeddings_dir ../Outputs/embeddings/esm2_8m/layer_6 \
         --concepts_dir ../Outputs/concepts \
         --output_dir ../Outputs/analysis/concepts

For each feature × concept pair, computes F1 / precision / recall / AUROC
across activation thresholds and saves a summary CSV.
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
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import pandas as pd

from single.sae.inference import load_sae
from single.analysis.feature_alignment import align_features_to_concepts


def _discover_shard_ids(path: Path) -> List[int]:
    ids = []
    for child in Path(path).glob("shard_*"):
        match = re.fullmatch(r"shard_(\d+)", child.name)
        if match:
            ids.append(int(match.group(1)))
    ids = sorted(set(ids))
    if not ids:
        raise ValueError(f"No numeric shard directories found in {path}")
    expected = list(range(ids[-1] + 1))
    if ids != expected:
        raise ValueError(f"Shard IDs in {path} are not contiguous: {ids}")
    return ids


def cmd_build(
    annotations_tsv: Path,
    concepts_dir: Optional[Path] = None,
    experiment: Optional[str] = None,
    exp_dir: Optional[Path] = None,
    source: Optional[str] = None,
    n_shards: int = 5,
    min_seq_len: int = 0,
    max_seq_len: int = 10_000,
    max_residues: Optional[int] = 510,
):
    from single.analysis.concepts import build_concept_matrix
    from single.paths import resolve_experiment

    if concepts_dir is None:
        exp = resolve_experiment(exp_dir=exp_dir, name=experiment, source=source)
        concepts_dir = exp.concepts_dir
        print(f"Experiment dir: {exp.dir}")
    build_concept_matrix(
        annotations_tsv=annotations_tsv,
        output_dir=concepts_dir,
        n_shards=n_shards,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
        max_residues=max_residues,
    )


def _load_shard_pair(embeddings_dir: Path, concepts_dir: Path, shard_id: int, device: str):
    """Load (concept_matrix, shard_embeddings) for one shard, aligned by token count."""
    from single.analysis.concepts import load_concept_shards

    concept_matrix, metadata = load_concept_shards(concepts_dir, shard_id)

    shard_emb_path = Path(embeddings_dir) / f"shard_{shard_id}" / "embeddings.pt"
    if not shard_emb_path.exists():
        cands = sorted(Path(embeddings_dir).glob(f"shard_{shard_id}/**/embeddings.pt"))
        if not cands:
            raise FileNotFoundError(
                f"No per-shard embeddings found for shard {shard_id} in {embeddings_dir}. "
                f"Expected {shard_emb_path}. Re-extract embeddings with the same "
                f"--n_shards used for concept building."
            )
        shard_emb_path = cands[0]
    shard_emb = torch.load(shard_emb_path, map_location="cpu", weights_only=True)
    if isinstance(shard_emb, dict):
        shard_emb = shard_emb.get("embeddings", shard_emb)
    shard_embeddings = shard_emb.to(device)

    embedding_residues = Path(embeddings_dir) / f"shard_{shard_id}" / "residues.csv"
    concept_residues = Path(concepts_dir) / f"shard_{shard_id}" / "residues.csv"
    if not embedding_residues.exists() or not concept_residues.exists():
        raise ValueError(
            f"Missing residue identity metadata for shard {shard_id}. "
            "Re-extract embeddings and rebuild concepts; comparing only token "
            "counts is unsafe."
        )
    emb_meta = pd.read_csv(embedding_residues, dtype=str)
    con_meta = pd.read_csv(concept_residues, dtype=str)
    embedding_config_path = Path(embeddings_dir) / "metadata.json"
    concept_config_path = Path(concepts_dir) / "metadata.json"
    if embedding_config_path.exists() and concept_config_path.exists():
        emb_cfg = json.loads(embedding_config_path.read_text())
        con_cfg = json.loads(concept_config_path.read_text())
        config_pairs = {
            "min_seq_len": (emb_cfg.get("min_seq_len"), con_cfg.get("min_seq_len")),
            "max_seq_len": (emb_cfg.get("max_seq_len"), con_cfg.get("max_seq_len")),
            "n_shards": (emb_cfg.get("n_shards"), con_cfg.get("n_shards")),
            "shuffle_seed": (emb_cfg.get("shuffle_seed"), con_cfg.get("shuffle_seed")),
            "max_residues": (
                emb_cfg.get("residues_per_sequence"),
                con_cfg.get("max_residues"),
            ),
        }
        mismatches = {
            key: values for key, values in config_pairs.items()
            if values[0] is not None and values[1] is not None and values[0] != values[1]
        }
        if mismatches:
            raise ValueError(
                f"Embedding/concept preprocessing metadata differs for shard "
                f"{shard_id}: {mismatches}"
            )

    identity_cols = ["sequence_hash", "amino_acid", "position"]
    if "Entry" in emb_meta.columns and "Entry" in con_meta.columns:
        identity_cols.insert(0, "Entry")
    missing_emb = [c for c in identity_cols if c not in emb_meta.columns]
    missing_con = [c for c in identity_cols if c not in con_meta.columns]
    if missing_emb or missing_con:
        raise ValueError(
            f"Shard {shard_id} residue metadata lacks identity columns: "
            f"embeddings={missing_emb}, concepts={missing_con}. Rebuild both artifacts."
        )
    if len(emb_meta) != len(con_meta):
        raise ValueError(
            f"Residue metadata row counts differ for shard {shard_id}: "
            f"embeddings={len(emb_meta)}, concepts={len(con_meta)}"
        )
    emb_identity = emb_meta[identity_cols].reset_index(drop=True)
    con_identity = con_meta[identity_cols].reset_index(drop=True)
    for frame in (emb_identity, con_identity):
        frame["amino_acid"] = frame["amino_acid"].str.upper()
    if not emb_identity.equals(con_identity):
        mismatch = (emb_identity != con_identity).any(axis=1)
        first = int(np.flatnonzero(mismatch.to_numpy())[0])
        raise ValueError(
            f"Residue identity mismatch at shard {shard_id}, row {first}; "
            "embedding and concept artifacts were not produced from the same "
            "ordered proteins/residues."
        )

    if concept_matrix.shape[0] != shard_embeddings.shape[0]:
        raise ValueError(
            f"Concept matrix ({concept_matrix.shape[0]} rows) and embeddings "
            f"({shard_embeddings.shape[0]} tokens) do not match for shard {shard_id}.\n"
            f"  Embeddings come from: {shard_emb_path}\n"
            f"  Concepts come from:   {Path(concepts_dir) / f'shard_{shard_id}'}\n"
            f"  These MUST be the SAME proteins, sharded identically. Common causes:\n"
            f"    - Embeddings extracted from a different CSV than the concept TSV\n"
            f"    - Different --n_shards used for extraction vs concept building\n"
            f"    - Different sequence filtering (min/max_seq_len, --max_residues)\n"
            f"  Re-extract embeddings from the SAME TSV used to build concepts."
        )

    protein_key = "Entry" if "Entry" in metadata.columns else "sequence_hash"
    protein_ids = pd.factorize(metadata[protein_key], sort=False)[0]
    return concept_matrix, shard_embeddings, protein_ids


def _load_combined_shards(
    embeddings_dir: Path, concepts_dir: Path, shard_ids: List[int], device: str
):
    """Load validated shards as one pooled alignment dataset."""
    from scipy import sparse

    matrices = []
    embeddings = []
    domain_offset = 0
    for shard_id in shard_ids:
        matrix, shard_embeddings, protein_ids = _load_shard_pair(
            embeddings_dir, concepts_dir, shard_id, "cpu"
        )
        # Domain instance IDs are generated independently inside each shard.
        # Remap positive values before vstack so identical local IDs from two
        # shards can never be mistaken for the same pooled domain.
        matrix = matrix.copy()
        positive_ids = np.unique(matrix.data[matrix.data > 0])
        if positive_ids.size:
            mapped = domain_offset + np.searchsorted(positive_ids, matrix.data) + 1
            matrix.data = np.where(matrix.data > 0, mapped, 0).astype(np.int64)
            domain_offset += int(positive_ids.size)
        matrices.append(matrix)
        embeddings.append(shard_embeddings)
        if len(matrices) == 1:
            pooled_protein_ids = protein_ids.astype(np.int64)
        else:
            pooled_protein_ids = np.concatenate(
                [pooled_protein_ids, protein_ids + pooled_protein_ids.max() + 1]
            )
    if not matrices:
        raise ValueError("No shards selected for pooled alignment")
    return (
        sparse.vstack(matrices, format="csr"),
        torch.cat(embeddings, dim=0),
        pooled_protein_ids,
    )


def _metrics_to_dataframe(metrics, shard_label):
    rows = []
    for fidx, concept_dict in metrics.items():
        for concept, entry in concept_dict.items():
            row = {
                "feature": fidx,
                "concept": concept,
                "f1": entry["f1"],
                "precision": entry["precision"],
                "recall": entry["recall"],
                "auroc": entry["auroc"],
                "threshold": entry["threshold"],
                "shard": shard_label,
            }
            for key in ("tp", "fp", "fn", "n_tokens", "n_positives"):
                if key in entry:
                    row[key] = entry[key]
            if "f1_per_domain" in entry:
                for key in (
                    "domain_tp", "domain_fp", "domain_fn",
                    "domain_precision", "domain_recall", "domain_f1",
                ):
                    row[key] = entry[key]
                row["f1_per_domain"] = entry["f1_per_domain"]
                row["recall_per_domain"] = entry["recall_per_domain"]
                row["n_domains"] = entry["n_domains"]
            rows.append(row)
    return pd.DataFrame(rows)


def _align_shards(
    sae,
    embeddings_dir: Path,
    concepts_dir: Path,
    concept_names: List[str],
    shard_ids: List[int],
    device: str,
    feature_chunk_size: int,
    batch_size: int,
    compute_auroc: bool,
    compute_domain_f1: bool,
    min_positives: int,
    threshold_percents: Optional[List[float]],
    fixed_thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Run alignment over a set of shards and return all pairs as a DataFrame."""
    all_pairs = []
    for shard_id in shard_ids:
        print(f"  Shard {shard_id}...")
        concept_matrix, shard_embeddings, protein_ids = _load_shard_pair(
            embeddings_dir, concepts_dir, shard_id, device
        )
        print(f"  Embeddings shard_{shard_id}: {shard_embeddings.shape}")

        metrics = align_features_to_concepts(
            sae=sae,
            embeddings=shard_embeddings,
            concept_matrix=concept_matrix,
            concept_names=concept_names,
            feature_chunk_size=feature_chunk_size,
            batch_size=batch_size,
            compute_auroc=compute_auroc,
            compute_domain_f1=compute_domain_f1,
            min_positives=min_positives,
            threshold_percents=threshold_percents,
            fixed_thresholds=fixed_thresholds,
            protein_ids=protein_ids,
        )

        for fidx, concept_dict in metrics.items():
            for concept, entry in concept_dict.items():
                row = {
                    "feature": fidx,
                    "concept": concept,
                    "f1": entry["f1"],
                    "precision": entry["precision"],
                    "recall": entry["recall"],
                    "auroc": entry["auroc"],
                    "threshold": entry["threshold"],
                    "shard": shard_id,
                }
                for key in ("tp", "fp", "fn", "n_tokens", "n_positives"):
                    if key in entry:
                        row[key] = entry[key]
                if compute_domain_f1 and "f1_per_domain" in entry:
                    for key in (
                        "domain_tp", "domain_fp", "domain_fn",
                        "domain_precision", "domain_recall", "domain_f1",
                    ):
                        row[key] = entry[key]
                    row["f1_per_domain"] = entry["f1_per_domain"]
                    row["recall_per_domain"] = entry["recall_per_domain"]
                    row["n_domains"] = entry["n_domains"]
                all_pairs.append(row)
    return pd.DataFrame(all_pairs)


def _resolve_dirs(concepts_dir, output_dir, experiment, exp_dir, source=None):
    from single.paths import resolve_experiment

    if concepts_dir is None or output_dir is None:
        exp = resolve_experiment(exp_dir=exp_dir, name=experiment, source=source)
        print(f"Experiment dir: {exp.dir}")
        if concepts_dir is None:
            concepts_dir = exp.concepts_dir
        if output_dir is None:
            output_dir = exp.analysis_dir
    return Path(concepts_dir), Path(output_dir)


def _resolve_sae_dir(sae_dir, experiment, exp_dir, source=None):
    """--sae_dir defaults into Outputs/<experiment>/sae (shared) but stays overridable."""
    from single.paths import resolve_experiment

    if sae_dir is None:
        exp = resolve_experiment(exp_dir=exp_dir, name=experiment, source=source)
        sae_dir = exp.sae_dir
        print(f"  SAE dir (inferred): {sae_dir}")
    return Path(sae_dir)


def cmd_align(
    embeddings_dir: Path,
    concepts_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    experiment: Optional[str] = None,
    exp_dir: Optional[Path] = None,
    source: Optional[str] = None,
    shard: Optional[int] = None,
    threshold_min_f1: float = 0.0,
    n_top_per_concept: int = 20,
    feature_chunk_size: int = 200,
    batch_size: int = 1024,
    compute_auroc: bool = True,
    compute_domain_f1: bool = True,
    min_positives: int = 10,
    threshold_percents: Optional[List[float]] = None,
    sae_dir: Optional[Path] = None,
):
    from single.analysis.concepts import load_concept_names

    sae_dir = _resolve_sae_dir(sae_dir, experiment, exp_dir, source=source)
    concepts_dir, output_dir = _resolve_dirs(
        concepts_dir, output_dir, experiment, exp_dir, source=source
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("SAE Feature × Concept Alignment")
    print("=" * 60)

    print("\nLoading SAE...")
    sae = load_sae(sae_dir, device=device)
    print(f"  {sae.__class__.__name__}: {sae.dict_size} features, {sae.activation_dim}D")

    concept_names = load_concept_names(concepts_dir)
    if not concept_names:
        raise ValueError(f"No concept columns found in {concepts_dir}/concept_columns.txt")

    print("\nLoading embeddings...")
    shards_to_process = [shard] if shard is not None else _discover_shard_ids(concepts_dir)

    if shard is None:
        pooled_matrix, pooled_embeddings, pooled_protein_ids = _load_combined_shards(
            embeddings_dir, concepts_dir, shards_to_process, device
        )
        pooled_metrics = align_features_to_concepts(
            sae=sae,
            embeddings=pooled_embeddings,
            concept_matrix=pooled_matrix,
            concept_names=concept_names,
            feature_chunk_size=feature_chunk_size,
            batch_size=batch_size,
            compute_auroc=compute_auroc,
            compute_domain_f1=compute_domain_f1,
            min_positives=min_positives,
            threshold_percents=threshold_percents,
            protein_ids=pooled_protein_ids,
            device=device,
        )
        df = _metrics_to_dataframe(pooled_metrics, "pooled")
    else:
        df = _align_shards(
            sae, embeddings_dir, concepts_dir, concept_names, shards_to_process,
            device, feature_chunk_size, batch_size, compute_auroc, compute_domain_f1,
            min_positives, threshold_percents,
        )

    # Save full metrics (latest shard's metrics dict for JSON)
    # Save summary CSV (all pairs, no threshold filtering)
    if not df.empty:
        csv_path = output_dir / "feature_concept_pairs.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved ALL {len(df)} feature-concept pairs to {csv_path}")

        # Print top pairs (filtered by threshold_min_f1 for readability)
        has_domain = "f1_per_domain" in df.columns
        top_df = df[df["f1"] >= threshold_min_f1] if threshold_min_f1 > 0 else df
        print(f"\nTop feature-concept associations (F1 >= {threshold_min_f1}, showing {len(top_df)}):")
        for _, row in top_df.nlargest(30, "f1").iterrows():
            extra = f"  F1dom={row['f1_per_domain']:.3f}" if has_domain else ""
            print(f"  Feature #{row['feature']:<4d} → {row['concept']:<30s} "
                  f"F1={row['f1']:.3f} AUROC={row['auroc']:.3f} "
                  f"P={row['precision']:.3f} R={row['recall']:.3f}{extra}")
    else:
        print("\nNo feature-concept pairs found.")

    print(f"\n{'=' * 60}")
    print("Done!")


def cmd_heldout(
    embeddings_dir: Path,
    concepts_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    experiment: Optional[str] = None,
    exp_dir: Optional[Path] = None,
    source: Optional[str] = None,
    split_mode: str = "half",
    threshold_min_f1: float = 0.0,
    feature_chunk_size: int = 200,
    batch_size: int = 1024,
    compute_auroc: bool = True,
    compute_domain_f1: bool = True,
    min_positives: int = 10,
    threshold_percents: Optional[List[float]] = None,
    heldout_f1_threshold: float = 0.3,
    sae_dir: Optional[Path] = None,
):
    """
    Held-out validation: split concept shards into valid/test, select the top
    feature per concept on the valid split, then evaluate those pairs on the
    held-out test split (unbiased metrics).
    """
    from single.analysis.concepts import load_concept_names
    from single.analysis.heldout import report_heldout

    sae_dir = _resolve_sae_dir(sae_dir, experiment, exp_dir, source=source)
    concepts_dir, output_dir = _resolve_dirs(
        concepts_dir, output_dir, experiment, exp_dir, source=source
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("SAE Feature × Concept Alignment (Held-out)")
    print("=" * 60)

    print("\nLoading SAE...")
    sae = load_sae(sae_dir, device=device)
    print(f"  {sae.__class__.__name__}: {sae.dict_size} features, {sae.activation_dim}D")

    concept_names = load_concept_names(concepts_dir)
    if not concept_names:
        raise ValueError(f"No concept columns found in {concepts_dir}/concept_columns.txt")

    # Split shards into valid / test
    all_shards = _discover_shard_ids(concepts_dir)
    if split_mode == "half":
        mid = len(all_shards) // 2
        valid_shards, test_shards = all_shards[:mid], all_shards[mid:]
    elif split_mode == "alternate":
        valid_shards = all_shards[::2]
        test_shards = all_shards[1::2]
    else:
        raise ValueError(f"Unknown split_mode: {split_mode} (use 'half' or 'alternate')")

    print(f"\nValid shards: {valid_shards}")
    print(f"Test shards:  {test_shards}")

    # Pool the validation split before threshold selection. Selecting a
    # threshold independently per shard and then combining the counts would
    # mix different classifiers and invalidate the pooled F1.
    print("\n--- Valid split alignment (selection) ---")
    valid_matrix, valid_embeddings, valid_protein_ids = _load_combined_shards(
        embeddings_dir, concepts_dir, valid_shards, device
    )
    valid_metrics = align_features_to_concepts(
        sae=sae,
        embeddings=valid_embeddings,
        concept_matrix=valid_matrix,
        concept_names=concept_names,
        feature_chunk_size=feature_chunk_size,
        batch_size=batch_size,
        compute_auroc=compute_auroc,
        compute_domain_f1=compute_domain_f1,
        min_positives=min_positives,
        threshold_percents=threshold_percents,
        protein_ids=valid_protein_ids,
        device=device,
    )
    df_valid = _metrics_to_dataframe(valid_metrics, "validation_pooled")
    valid_csv = output_dir / "heldout_valid_pairs.csv"
    df_valid.to_csv(valid_csv, index=False)
    print(f"  Valid split pairs saved to {valid_csv}")

    # Freeze the validation-selected threshold for every test shard.
    fixed_thresholds = np.full(
        (sae.dict_size, len(concept_names)), np.nan, dtype=np.float32
    )
    concept_indices = {name: i for i, name in enumerate(concept_names)}
    for row in df_valid.itertuples(index=False):
        fixed_thresholds[int(row.feature), concept_indices[row.concept]] = row.threshold

    print("\n--- Test split alignment (evaluation) ---")
    df_test = _align_shards(
        sae, embeddings_dir, concepts_dir, concept_names, test_shards,
        device, feature_chunk_size, batch_size, compute_auroc, compute_domain_f1,
        0, threshold_percents, fixed_thresholds=fixed_thresholds,
    )
    test_csv = output_dir / "heldout_test_pairs.csv"
    df_test.to_csv(test_csv, index=False)
    print(f"  Test split pairs saved to {test_csv}")

    # Held-out report
    report_heldout(valid_csv, test_csv, output_dir, top_threshold=heldout_f1_threshold)

    print(f"\n{'=' * 60}")
    print("Done!")


def main(argv=None):
    parser = argparse.ArgumentParser(description="SAE feature × Swiss-Prot concept alignment")
    sub = parser.add_subparsers(dest="command", required=True)

    # build
    p_build = sub.add_parser("build", help="Build concept matrices from UniProt TSV")
    p_build.add_argument("--annotations_tsv", type=Path, required=True,
                         help="UniProtKB/Swiss-Prot TSV export")
    p_build.add_argument("--source", type=str, default=None,
                        help="Data-source id; nests outputs under Outputs/<experiment>/<source> (default: flat)")
    p_build.add_argument("--experiment", type=str, default=None,
                         help="Experiment name; creates Outputs/<experiment>_<ts>/")
    p_build.add_argument("--exp_dir", type=Path, default=None,
                         help="Reuse an existing experiment directory")
    p_build.add_argument("--concepts_dir", type=Path, default=None,
                         help="Explicit concepts dir (overrides experiment routing)")
    p_build.add_argument("--n_shards", type=int, default=5)
    p_build.add_argument("--min_seq_len", type=int, default=0)
    p_build.add_argument("--max_seq_len", type=int, default=10000)
    p_build.add_argument("--max_residues", type=int, default=510,
                         help="Residues kept per protein (must equal embedder "
                              "max_length - 2, e.g. 510 for max_length=512) so "
                              "concept rows align with embedding tokens")

    # align
    p_align = sub.add_parser("align", help="Align SAE features to concepts")
    p_align.add_argument("--sae_dir", type=Path, default=None,
                         help="Trained SAE dir (default: Outputs/<experiment>/sae)")
    p_align.add_argument("--embeddings_dir", type=Path, required=True)
    p_align.add_argument("--source", type=str, default=None,
                        help="Data-source id; nests outputs under Outputs/<experiment>/<source> (default: flat)")
    p_align.add_argument("--experiment", type=str, default=None,
                         help="Experiment name; creates Outputs/<experiment>_<ts>/")
    p_align.add_argument("--exp_dir", type=Path, default=None,
                         help="Reuse an existing experiment directory")
    p_align.add_argument("--concepts_dir", type=Path, default=None,
                         help="Explicit concepts dir (overrides experiment routing)")
    p_align.add_argument("--output_dir", type=Path, default=None,
                         help="Explicit output dir (overrides experiment routing)")
    p_align.add_argument("--shard", type=int, default=None,
                         help="Process only one concept shard")
    p_align.add_argument("--threshold_min_f1", type=float, default=0.0,
                         help="Only affects console top-pairs display (CSV always contains all pairs)")
    p_align.add_argument("--n_top_per_concept", type=int, default=20)
    p_align.add_argument("--feature_chunk_size", type=int, default=200)
    p_align.add_argument("--batch_size", type=int, default=1024)
    p_align.add_argument("--no_auroc", action="store_true",
                         help="Skip AUROC computation (faster)")
    p_align.add_argument("--no_domain_f1", action="store_true",
                         help="Skip domain-level F1 (faster)")
    p_align.add_argument("--min_positives", type=int, default=10,
                         help="Skip concepts with fewer positive residues")
    p_align.add_argument("--threshold_percents", type=float, nargs="+", default=None,
                         help="InterPLM-style percent-of-max thresholds, e.g. 0 0.15 0.5 0.6 0.8")

    # heldout
    p_held = sub.add_parser("heldout",
                            help="Held-out validation: select on valid split, evaluate on test split")
    p_held.add_argument("--sae_dir", type=Path, default=None,
                        help="Trained SAE dir (default: Outputs/<experiment>/sae)")
    p_held.add_argument("--embeddings_dir", type=Path, required=True)
    p_held.add_argument("--source", type=str, default=None,
                        help="Data-source id; nests outputs under Outputs/<experiment>/<source> (default: flat)")
    p_held.add_argument("--experiment", type=str, default=None,
                        help="Experiment name; creates Outputs/<experiment>_<ts>/")
    p_held.add_argument("--exp_dir", type=Path, default=None,
                        help="Reuse an existing experiment directory")
    p_held.add_argument("--concepts_dir", type=Path, default=None,
                        help="Explicit concepts dir (overrides experiment routing)")
    p_held.add_argument("--output_dir", type=Path, default=None,
                        help="Explicit output dir (overrides experiment routing)")
    p_held.add_argument("--split_mode", type=str, default="half", choices=["half", "alternate"],
                        help="How to split shards into valid/test: 'half' (first half valid) "
                             "or 'alternate' (every other shard)")
    p_held.add_argument("--threshold_min_f1", type=float, default=0.0)
    p_held.add_argument("--feature_chunk_size", type=int, default=200)
    p_held.add_argument("--batch_size", type=int, default=1024)
    p_held.add_argument("--no_auroc", action="store_true",
                        help="Skip AUROC computation (faster)")
    p_held.add_argument("--no_domain_f1", action="store_true",
                        help="Skip domain-level F1 (faster)")
    p_held.add_argument("--min_positives", type=int, default=10)
    p_held.add_argument("--threshold_percents", type=float, nargs="+", default=None,
                        help="InterPLM-style percent-of-max thresholds, e.g. 0 0.15 0.5 0.6 0.8")
    p_held.add_argument("--heldout_f1_threshold", type=float, default=0.3,
                        help="Report held-out pairs with f1_per_domain above this")

    args = parser.parse_args(argv)
    if args.command == "build":
        cmd_build(
            annotations_tsv=args.annotations_tsv,
            concepts_dir=args.concepts_dir,
            experiment=args.experiment,
            exp_dir=args.exp_dir,
            source=args.source,
            n_shards=args.n_shards,
            min_seq_len=args.min_seq_len,
            max_seq_len=args.max_seq_len,
            max_residues=args.max_residues,
        )
    elif args.command == "heldout":
        cmd_heldout(
            embeddings_dir=args.embeddings_dir,
            concepts_dir=args.concepts_dir,
            output_dir=args.output_dir,
            experiment=args.experiment,
            exp_dir=args.exp_dir,
            source=args.source,
            split_mode=args.split_mode,
            threshold_min_f1=args.threshold_min_f1,
            feature_chunk_size=args.feature_chunk_size,
            batch_size=args.batch_size,
            compute_auroc=not args.no_auroc,
            compute_domain_f1=not args.no_domain_f1,
            min_positives=args.min_positives,
            threshold_percents=args.threshold_percents,
            heldout_f1_threshold=args.heldout_f1_threshold,
            sae_dir=args.sae_dir,
        )
    else:
        cmd_align(
            embeddings_dir=args.embeddings_dir,
            concepts_dir=args.concepts_dir,
            output_dir=args.output_dir,
            experiment=args.experiment,
            exp_dir=args.exp_dir,
            source=args.source,
            shard=args.shard,
            threshold_min_f1=args.threshold_min_f1,
            n_top_per_concept=args.n_top_per_concept,
            feature_chunk_size=args.feature_chunk_size,
            batch_size=args.batch_size,
            compute_auroc=not args.no_auroc,
            compute_domain_f1=not args.no_domain_f1,
            min_positives=args.min_positives,
            threshold_percents=args.threshold_percents,
            sae_dir=args.sae_dir,
        )


if __name__ == "__main__":
    main()
