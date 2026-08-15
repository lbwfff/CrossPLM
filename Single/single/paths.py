"""
Centralized output path management for the SAE interpretability pipeline.

All scripts write into a single experiment directory:

    Outputs/<experiment>/
        embeddings/layer_<N>/shard_<i>/activations.pt
        sae/ae.pt
        concepts/shard_<i>/aa_concepts.npz
        analysis/...

Usage:
    from single.paths import Experiment

    exp = Experiment(name="swissprot")        # -> Outputs/swissprot/
    exp.embeddings_dir(layer=6)               # Outputs/swissprot/embeddings/layer_6
    exp.sae_dir                               # Outputs/swissprot/sae
    exp.concepts_dir                          # Outputs/swissprot/concepts
    exp.analysis_dir                          # Outputs/swissprot/analysis

The experiment name is used verbatim as the directory name (no timestamp), so a
given `--experiment <name>` always routes to the SAME `Outputs/<name>/`. Re-running
a step with the same name reuses the existing directory (e.g. overwriting ae.pt).
Use distinct experiment names for distinct runs. `--exp_dir <path>` uses that path
verbatim.
"""

from pathlib import Path
from typing import Optional


class Experiment:
    """A single end-to-end run: embeddings → SAE → concepts → analysis."""

    def __init__(
        self,
        name: str,
        root: Path = Path("../Outputs"),
    ):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        # Use the experiment name verbatim as the directory name (no timestamp).
        self.dir = root / name
        self.dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_existing_dir(cls, existing_dir: Path) -> "Experiment":
        """Use an already-existing experiment directory verbatim (no root join)."""
        self = cls.__new__(cls)
        self.dir = Path(existing_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        return self

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

    - `name` -> `root/<name>` (no timestamp; same name always maps to the same dir).
    - `exp_dir` -> used verbatim (NOT re-joined under `root`, so passing
      `Outputs/foo` won't create `Outputs/Outputs/foo`).

    Priority: exp_dir (if given) > name.
    """
    if exp_dir is not None:
        return Experiment.from_existing_dir(exp_dir)
    if name is None:
        raise ValueError("Must provide either exp_dir or experiment name")
    return Experiment(name=name, root=root)
