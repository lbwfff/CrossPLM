"""
Centralized output path management for the SAE interpretability pipeline.

All scripts write into a single experiment directory:

    Outputs/<experiment>_<timestamp>/
        embeddings/layer_<N>/shard_<i>/activations.pt
        sae/ae.pt
        concepts/shard_<i>/aa_concepts.npz
        analysis/...

Usage:
    from single.paths import Experiment

    exp = Experiment(name="swissprot")        # -> Outputs/swissprot_20260814_123000/
    exp.embeddings_dir(layer=6)               # Outputs/.../embeddings/layer_6
    exp.sae_dir                               # Outputs/.../sae
    exp.concepts_dir                          # Outputs/.../concepts
    exp.analysis_dir                          # Outputs/.../analysis

Pass the experiment dir (exp.dir) between steps; each script re-resolves
subdirectories from it. If you pass a name without timestamp, we create one.
If you pass an existing dir, we reuse it as-is.
"""

import datetime
import re
from pathlib import Path
from typing import Optional


class Experiment:
    """A single end-to-end run: embeddings → SAE → concepts → analysis."""

    def __init__(
        self,
        name: str,
        root: Path = Path("../Outputs"),
        timestamp: Optional[str] = None,
    ):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)

        # If name already looks like an experiment dir (contains a timestamp),
        # reuse it; otherwise create name_timestamp.
        if re.search(r"\d{8}_\d{6}$", str(name)):
            self.dir = root / name
        else:
            if timestamp is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.dir = root / f"{name}_{timestamp}"
        self.dir.mkdir(parents=True, exist_ok=True)

    # ---- subdirectories ----
    @property
    def embeddings_root(self) -> Path:
        return self.dir / "embeddings"

    def embeddings_dir(self, layer: int = 6) -> Path:
        p = self.embeddings_root / f"layer_{layer}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def sae_dir(self) -> Path:
        p = self.dir / "sae"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def concepts_dir(self) -> Path:
        p = self.dir / "concepts"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def analysis_dir(self) -> Path:
        p = self.dir / "analysis"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def visualizations_dir(self) -> Path:
        p = self.analysis_dir / "visualizations"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ---- convenience accessors ----
    @property
    def sae_model_path(self) -> Path:
        return self.sae_dir / "ae.pt"

    @property
    def concept_columns_path(self) -> Path:
        return self.concepts_dir / "aa_concepts_columns.txt"

    @property
    def concept_pairs_csv(self) -> Path:
        return self.analysis_dir / "feature_concept_pairs.csv"

    @property
    def label_metrics_json(self) -> Path:
        return self.analysis_dir / "feature_label_metrics.json"

    def __str__(self) -> str:
        return str(self.dir)


def resolve_experiment(
    exp_dir: Optional[Path] = None,
    name: Optional[str] = None,
    root: Path = Path("../Outputs"),
) -> Experiment:
    """
    Build an Experiment from either an existing experiment directory or a name.
    Priority: exp_dir (if given) > name (creates new timestamped dir).
    """
    if exp_dir is not None:
        return Experiment(name=str(exp_dir), root=root)
    if name is None:
        raise ValueError("Must provide either exp_dir or experiment name")
    return Experiment(name=name, root=root)
