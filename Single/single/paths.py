"""
Centralized output path management for the SAE interpretability pipeline.

All scripts write into a single experiment directory, optionally split by
*data source* (the input dataset, e.g. `mbmrb` vs `swissprot`). The SAE is
shared at the experiment root (one dictionary reused across sources):

    Outputs/<experiment>/
        sae/model.pt                        # ONE shared SAE
        <source>/                           # e.g. mbmrb | swissprot
            embeddings/layer_<N>/shard_<i>/embeddings.pt
            concepts/shard_<i>/concept_matrix.npz
            analysis/...

Usage:
    from single.paths import Experiment

    exp = Experiment(name="demo", source="mbmrb")   # -> Outputs/demo/mbmrb/
    exp.embeddings_dir(layer=6)                      # Outputs/demo/mbmrb/embeddings/layer_6
    exp.sae_dir                                      # Outputs/demo/sae (shared)
    exp.concepts_dir                                 # Outputs/demo/mbmrb/concepts
    exp.analysis_dir                                 # Outputs/demo/mbmrb/analysis

The experiment name is used verbatim as the directory name (no timestamp), so a
given `--experiment <name>` always routes to the SAME `Outputs/<name>/`. Re-running
a step with the same name reuses the existing directory (e.g. overwriting model.pt).
`--source <id>` nests data-specific dirs under `Outputs/<name>/<id>/`; without it
everything lives directly under `Outputs/<name>/` (flat, legacy layout).
`--exp_dir <path>` uses that path verbatim.
"""

from pathlib import Path
from typing import Optional

# Top-level output root shared with the Training module: <repo>/Outputs/.
# Resolved from this file's location so it works regardless of the CWD.
_DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "Outputs"


class Experiment:
    """A single end-to-end run: embeddings → SAE → concepts → analysis.

    `dir` points at the data-source subdir (when `source` is given); `root_dir`
    is the experiment root where the shared SAE and Training artifacts live.
    """

    def __init__(
        self,
        name: str,
        root: Path = _DEFAULT_ROOT,
        source: Optional[str] = None,
    ):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        # Use the experiment name verbatim as the directory name (no timestamp).
        self.root_dir = root / name
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.source = source
        # Data-specific dirs nest under Outputs/<name>/<source>/.
        self.dir = self.root_dir / source if source else self.root_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_existing_dir(cls, existing_dir: Path) -> "Experiment":
        """Use an already-existing experiment directory verbatim (no root join)."""
        self = cls.__new__(cls)
        self.dir = Path(existing_dir)
        self.root_dir = self.dir
        self.source = None
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
        # Shared across data sources: always at the experiment root.
        p = self.root_dir / "sae"
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
        return self.sae_dir / "model.pt"

    @property
    def sae_normalized_path(self) -> Path:
        return self.sae_dir / "model_normalized.pt"

    @property
    def concept_columns_path(self) -> Path:
        return self.concepts_dir / "concept_columns.txt"

    @property
    def concept_pairs_csv(self) -> Path:
        return self.analysis_dir / "feature_concept_pairs.csv"

    @property
    def label_metrics_json(self) -> Path:
        return self.analysis_dir / "feature_label_metrics.json"

    def __str__(self) -> str:
        return str(self.dir)


def _looks_like_source_nested(exp_root: Path) -> bool:
    """Heuristic: does Outputs/<exp>/ already contain source subdirs (e.g.
    mbmrb/embeddings/, swissprot/concepts/)? If so, a flat run would mix layouts."""
    for d in exp_root.iterdir():
        if not d.is_dir() or d.name in ("sae", "checkpoints", "evaluations"):
            continue
        for sub in d.iterdir():
            if sub.is_dir() and sub.name in {"embeddings", "concepts", "analysis"}:
                return True
    return False


def resolve_experiment(
    exp_dir: Optional[Path] = None,
    name: Optional[str] = None,
    root: Path = _DEFAULT_ROOT,
    source: Optional[str] = None,
) -> Experiment:
    """
    Build an Experiment from either an existing experiment directory or a name.

    - `name` -> `root/<name>` (no timestamp); `source` nests under it as
      `root/<name>/<source>/`. Without `source` the flat layout is used.
    - `exp_dir` -> used verbatim (NOT re-joined under `root`, and `source` is
      ignored).

    Priority: exp_dir (if given) > name.
    """
    if exp_dir is not None:
        return Experiment.from_existing_dir(exp_dir)
    if name is None:
        raise ValueError("Must provide either exp_dir or experiment name")
    exp = Experiment(name=name, root=root, source=source)
    if source is None and _looks_like_source_nested(exp.root_dir):
        print(
            "[source] WARNING: no --source given, but Outputs/"
            f"{name}/ already contains source subdirs (e.g. mbmrb/, swissprot/). "
            "This run writes FLAT under Outputs/<exp>/ and may mix with "
            "source-nested outputs. Pass --source <id> to keep datasets separate."
        )
    return exp
