"""
Held-out validation of feature-concept pairings.

Problem: if we pick the "best feature per concept" and report its F1 on the SAME
data used for selection, the metric is optimistically biased (selection bias).
Random noise features can look great just by chance.

Solution (standard in interpretability, used by InterPLM):
  1. Split data into a validation split and a test split.
  2. On the VALIDATION split, pick the top feature per concept (selection).
  3. On the TEST split, evaluate those selected (feature, concept) pairs only.
     These held-out metrics are unbiased estimates of real performance.

This module expects two feature_concept_pairs.csv files (valid + test) and
produces the held-out report.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

# Concepts that are not meaningful biological concepts (e.g. per-amino-acid one-hots)
IGNORE_SUBSTRINGS = ["amino_acid"]


def _filter_meaningful(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["concept"].str.contains("|".join(IGNORE_SUBSTRINGS), case=False, na=False)]


def select_top_feature_per_concept(df_valid: pd.DataFrame) -> pd.DataFrame:
    """
    On the validation set, pick the single best feature per concept.
    Ranked by f1_per_domain (fallback to f1), then dedupe per concept.
    """
    df = _filter_meaningful(df_valid).copy()
    if df.empty:
        return df[["feature", "concept"]]
    # Prefer f1_per_domain if present, else f1
    rank_col = "f1_per_domain" if "f1_per_domain" in df.columns else "f1"
    top = df.sort_values(by=[rank_col, "f1"], ascending=False).drop_duplicates("concept")
    return top[["feature", "concept"]]


def evaluate_heldout(df_test: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate selected (feature, concept) pairs on the held-out test set.

    Uses a LEFT join so that every pair selected on the validation split appears
    in the result. Pairs absent from the test CSV (because align_features_to_concepts
    only emits f1 > 0) are explicitly materialized with zero metrics — otherwise the
    reported held-out average would be upward-biased by silently dropping the failures.

    New alignment CSVs include TP/FP/FN and domain-hit counts. Those are summed
    across shards to compute exact pooled F1/precision/recall. AUROC cannot be
    pooled from shard-level scores alone, so it is reported as a token-weighted
    mean. Older CSVs fall back to a macro mean for compatibility.
    """
    # `shard` identifies the source row and must not be averaged into the
    # result. New alignment CSVs also contain sufficient statistics, allowing
    # exact confusion-matrix aggregation instead of averaging per-shard F1.
    metric_cols = [
        c for c in df_test.columns
        if c not in ("feature", "concept", "shard")
    ]
    merged = pd.merge(
        selected,  # left: every selected pair
        df_test,
        on=["feature", "concept"],
        how="left",
    )
    # Fill missing metrics (test-F1 == 0 pairs) with zeros.
    for col in metric_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    # For new alignment outputs, aggregate the underlying counts. This gives
    # the metric on the union of test shards rather than a mean of shard-level
    # metrics, which can be biased when shard sizes differ.
    if not merged.empty:
        count_cols = {"tp", "fp", "fn"}
        if count_cols.issubset(merged.columns):
            rows = []
            for (feature, concept), group in merged.groupby(
                ["feature", "concept"], sort=False
            ):
                tp = float(group["tp"].sum())
                fp = float(group["fp"].sum())
                fn = float(group["fn"].sum())
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if precision + recall else 0.0
                )
                row = {
                    "feature": feature,
                    "concept": concept,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }

                if "n_tokens" in group:
                    row["n_tokens"] = float(group["n_tokens"].sum())
                if "n_positives" in group:
                    row["n_positives"] = float(group["n_positives"].sum())

                domain_count_cols = {"domain_tp", "domain_fp", "domain_fn"}
                if domain_count_cols.issubset(group.columns):
                    domain_tp = float(group["domain_tp"].sum())
                    domain_fp = float(group["domain_fp"].sum())
                    domain_fn = float(group["domain_fn"].sum())
                    domain_precision = (
                        domain_tp / (domain_tp + domain_fp)
                        if domain_tp + domain_fp else 0.0
                    )
                    domain_recall = (
                        domain_tp / (domain_tp + domain_fn)
                        if domain_tp + domain_fn else 0.0
                    )
                    domain_f1 = (
                        2 * domain_precision * domain_recall
                        / (domain_precision + domain_recall)
                        if domain_precision + domain_recall else 0.0
                    )
                    row["domain_tp"] = domain_tp
                    row["domain_fp"] = domain_fp
                    row["domain_fn"] = domain_fn
                    row["domain_precision"] = domain_precision
                    row["domain_recall"] = domain_recall
                    row["domain_f1"] = domain_f1
                    row["n_domains"] = float(group["n_domains"].sum()) if "n_domains" in group else domain_tp + domain_fn
                    row["recall_per_domain"] = domain_recall
                    row["f1_per_domain"] = domain_f1
                elif {"domain_hits", "n_domains"}.issubset(group.columns):
                    # Legacy CSV compatibility for the old hybrid metric.
                    domain_hits = float(group["domain_hits"].sum())
                    n_domains = float(group["n_domains"].sum())
                    recall_dom = domain_hits / n_domains if n_domains else 0.0
                    f1_dom = (
                        2 * precision * recall_dom / (precision + recall_dom)
                        if precision + recall_dom else 0.0
                    )
                    row["domain_hits"] = domain_hits
                    row["n_domains"] = n_domains
                    row["recall_per_domain"] = recall_dom
                    row["f1_per_domain"] = f1_dom

                # Threshold is not additive; report the token-weighted mean.
                if "threshold" in group:
                    weights = group.get("n_tokens", pd.Series(dtype=float))
                    if weights.sum() > 0:
                        row["threshold"] = float(
                            (group["threshold"] * weights).sum() / weights.sum()
                        )
                    else:
                        row["threshold"] = float(group["threshold"].mean())

                # AUROC cannot be reconstructed from shard-level AUROC values.
                # Use a token-weighted mean and keep the limitation explicit in
                # the function documentation rather than selecting the maximum.
                if "auroc" in group:
                    weights = group.get("n_tokens", pd.Series(dtype=float))
                    if weights.sum() > 0:
                        row["auroc"] = float(
                            (group["auroc"] * weights).sum() / weights.sum()
                        )
                    else:
                        row["auroc"] = float(group["auroc"].mean())

                # Preserve any future numeric columns using a size-weighted
                # mean, without overwriting recomputed metrics above.
                for col in metric_cols:
                    if col in row or col in {"f1", "precision", "recall"}:
                        continue
                    if pd.api.types.is_numeric_dtype(group[col]):
                        row[col] = float(group[col].mean())
                rows.append(row)

            result = pd.DataFrame(rows)
            # Preserve the input schema where possible while retaining the
            # newly computed sufficient statistics.
            ordered = ["feature", "concept"] + [
                c for c in metric_cols if c in result.columns
            ]
            return result[ordered]

        # Backward compatibility for old CSVs without sufficient statistics.
        # This is a macro average across shards, never the optimistic maximum.
        agg = merged.groupby(["feature", "concept"], as_index=False)[metric_cols].mean()
        cols = ["feature", "concept"] + metric_cols
        return agg[cols]
    return merged


def report_heldout(
    valid_csv: Path,
    test_csv: Path,
    output_dir: Optional[Path] = None,
    top_threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Run held-out validation: select on valid, evaluate on test.

    Args:
        valid_csv: feature_concept_pairs.csv from the validation split
        test_csv: feature_concept_pairs.csv from the test split
        output_dir: where to save reports (defaults to test_csv parent)
        top_threshold: only report test pairs whose f1_per_domain is above this

    Returns:
        DataFrame of held-out (feature, concept) pairs with test metrics.
    """
    df_valid = pd.read_csv(valid_csv)
    df_test = pd.read_csv(test_csv)

    selected = select_top_feature_per_concept(df_valid)
    heldout = evaluate_heldout(df_test, selected)

    output_dir = Path(output_dir) if output_dir is not None else Path(test_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save all held-out pairs
    heldout_path = output_dir / "heldout_top_pairings.csv"
    heldout.to_csv(heldout_path, index=False)

    # Save pairs above threshold
    if "f1_per_domain" in heldout.columns:
        above = heldout[heldout["f1_per_domain"] >= top_threshold].sort_values(
            ["f1_per_domain", "f1"], ascending=False
        )
    else:
        above = heldout[heldout["f1"] >= top_threshold].sort_values("f1", ascending=False)
    above_path = output_dir / "heldout_all_top_pairings.csv"
    above.to_csv(above_path, index=False)

    # Summary statistics
    rank_col = "f1_per_domain" if "f1_per_domain" in heldout.columns else "f1"
    print("=" * 60)
    print("HELD-OUT VALIDATION REPORT")
    print("=" * 60)
    print(f"Selection set size (valid): {len(df_valid)} pairs")
    print(f"Test set size: {len(df_test)} pairs")
    print(f"Selected feature-concept pairs: {len(selected)}")
    print(f"Held-out pairs evaluated: {len(heldout)}")
    print(f"Concepts covered: {heldout['concept'].nunique()}")
    print(f"Features associated: {heldout['feature'].nunique()}")
    if not heldout.empty:
        print(f"Average best {rank_col} per concept (held-out): "
              f"{heldout.sort_values(rank_col, ascending=False).drop_duplicates('concept')[rank_col].mean():.3f}")
    print(f"Saved to {heldout_path} and {above_path}")

    return heldout
