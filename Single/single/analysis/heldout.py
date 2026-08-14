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
    Returns the test metrics for those pairs only.
    """
    merged = pd.merge(
        df_test,
        selected,
        on=["feature", "concept"],
        how="inner",
    )
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
