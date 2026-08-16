"""
Configurable label encoding for different datasets.

A "label map" is a single, shared description of a dataset's per-residue labels.
It is used by BOTH the Training module (fine-tuning) and the Single module
(interpretability), so the two always interpret the same CSV the same way:

    - sequence_column / label_column : which columns hold the protein sequence
                                       and the per-residue label string
    - mapping         : raw label character -> integer class id (binary or
                        multi-class; every char not listed is ignored -> -100)
    - ignore          : documented extra characters that are ignored (any
                        unmapped char is ignored regardless)
    - positive_class  : which integer class is "positive" (for reporting)
    - class_names     : int class id -> human-readable name

Unmapped characters become -100 (ignored for training/analysis), matching the
HuggingFace ignore-index convention.

Usage:
    from single.label_maps import get_label_map, encode_label_string

    spec = get_label_map("mBMRB")       # built-in preset
    spec = get_label_map("file.yaml")   # or load from a YAML file
    ids  = encode_label_string(label_str, spec)
"""

from pathlib import Path
from typing import Dict, Optional

import yaml

# Built-in presets. Structure:
#   sequence_column : CSV/TSV column with the protein sequence
#   label_column    : CSV/TSV column with the per-residue label string
#   positive_class  : which integer label is treated as the "positive" class
#   class_names     : int label -> human-readable name (for reporting)
#   mapping         : raw character -> integer label
#   ignore          : extra characters to treat as -100 (optional; any unmapped char is ignored anyway)
LABEL_MAPS: Dict[str, Dict] = {
    "mBMRB": {
        "sequence_column": "sequence",
        "label_column": "label",
        "positive_class": 1,
        "class_names": {0: "rigid", 1: "flexible"},
        "mapping": {"A": 0, ".": 1, "0": 0, "1": 1},
        "ignore": "_",
    },
    # Raw relaxdb chars p/A/v (static) and ./b/^ (mobile); t/x are ignored. The
    # 0/1 entries also accept the preprocessed relaxdb_processed.csv so one preset
    # works for both raw and preprocessed files.
    "relaxdb": {
        "sequence_column": "sequence",
        "label_column": "label",
        "positive_class": 1,
        "class_names": {0: "static", 1: "mobile"},
        "mapping": {"p": 0, "A": 0, "v": 0, "0": 0, ".": 1, "b": 1, "^": 1, "1": 1},
        "ignore": "_tx",
    },
    # Example for 3-class secondary structure. The integer ids are assigned to
    # match the training module's historical build_label_map() (sorted(unique))
    # so that a model trained with an inferred label map evaluates consistently
    # with this preset: C=0, E=1, H=2.
    "ss3": {
        "sequence_column": "sequence",
        "label_column": "label",
        "positive_class": 1,
        "class_names": {0: "coil", 1: "strand", 2: "helix"},
        "mapping": {"C": 0, "E": 1, "H": 2},
        "ignore": "_",
    },
}


def _normalize_spec(spec: Dict) -> Dict:
    """Fill defaults and validate a label-map spec."""
    spec.setdefault("sequence_column", "sequence")
    spec.setdefault("label_column", "label")
    spec.setdefault("positive_class", 1)
    spec.setdefault("class_names", {})
    spec.setdefault("mapping", {})
    spec.setdefault("ignore", "_")
    assert isinstance(spec["mapping"], dict) and spec["mapping"], \
        "label map 'mapping' must be a non-empty dict of {char: class_id}"
    return spec


def get_label_map(name: str) -> Dict:
    """
    Return a normalized label-map spec, either from a built-in preset name
    or from a YAML file path.

    YAML file format:
        sequence_column: sequence
        label_column: label
        positive_class: 1
        class_names:
          0: rigid
          1: flexible
        mapping:
          A: 0
          .: 1
          "0": 0
          "1": 1
        ignore: "_"

    Only `mapping` is required; the rest get sensible defaults.
    """
    if isinstance(name, Path) or (isinstance(name, str) and Path(name).suffix in {".yaml", ".yml"}):
        path = Path(name)
        if not path.exists():
            raise FileNotFoundError(f"Label map file not found: {path}")
        with open(path, "r") as f:
            spec = yaml.safe_load(f)
        return _normalize_spec(spec)

    if name not in LABEL_MAPS:
        raise ValueError(
            f"Unknown label map '{name}'. Available presets: {list(LABEL_MAPS.keys())}. "
            f"Or provide a path to a YAML label-map file."
        )
    return _normalize_spec(LABEL_MAPS[name])


def encode_label_string(label_str: str, spec: Dict) -> list:
    """
    Encode a per-residue label string into a list of integer class ids (-100 for ignore).

    Args:
        label_str: raw label string (e.g. "A..A.A")
        spec: normalized label-map spec from get_label_map()
    """
    mapping = spec["mapping"]
    return [mapping.get(c, -100) for c in label_str]


LABEL_MAP_TEMPLATE = """\
# Label-map template for a per-residue classification dataset.
# Shared by the Training and Single modules (same semantics everywhere).
#
#   sequence_column : CSV/TSV column holding the protein sequence
#   label_column    : CSV/TSV column holding the per-residue label string
#   positive_class  : which integer class is treated as "positive" (for reporting)
#   class_names     : integer class id -> human-readable name
#   mapping         : raw label character -> integer class id
#                     (binary or multi-class). ANY character not listed here is
#                     ignored -> encoded as -100, excluded from train/eval.
#   ignore          : documented extra ignored characters (optional; unmapped
#                     characters are ignored regardless)

sequence_column: sequence
label_column: label

positive_class: 1
class_names:
  0: class_0
  1: class_1

mapping:
  A: 0
  ".": 1
  # add more characters -> class ids here, e.g. C: 0, E: 1, H: 2 for 3-class

ignore: "_"
"""


def generate_template(path) -> Path:
    """Write an empty label-map YAML template and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(LABEL_MAP_TEMPLATE)
    return path


def n_classes(spec: Dict) -> int:
    """Number of distinct integer classes in a label-map spec (max id + 1)."""
    return max(spec["mapping"].values()) + 1 if spec["mapping"] else 0


def resolve_columns(spec: Dict, sequence_column: str = "sequence",
                    label_column: Optional[str] = None):
    """
    Effective CSV column names for a dataset: explicit arguments win, otherwise
    the label-map spec's `sequence_column` / `label_column` (which describe the
    dataset) are used.
    """
    if sequence_column == "sequence":
        sequence_column = spec.get("sequence_column") or sequence_column
    if label_column is None:
        label_column = spec.get("label_column")
    return sequence_column, label_column


def load_labeled_sequences(csv_path, spec: Dict):
    """
    Load sequences + per-residue label strings from a CSV/TSV using the spec's
    column names. Shared by Training and Single so both read datasets identically.

    - Separator is auto-detected (tab if tabs outnumber commas, else comma).
    - Sequences are uppercased; rows whose label length != sequence length are
      dropped. Labels are returned as raw strings (encode with
      encode_label_string / the spec's mapping).

    Returns:
        (sequences: List[str], labels: List[str])
    """
    import csv as _csv

    seq_col = spec["sequence_column"]
    lbl_col = spec["label_column"]
    sequences, labels = [], []
    with open(csv_path, newline="") as f:
        first = f.readline()
    sep = "\t" if first.count("\t") > first.count(",") else ","
    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f, delimiter=sep)
        if seq_col not in reader.fieldnames:
            raise ValueError(f"Sequence column '{seq_col}' not found in {csv_path}")
        if lbl_col not in reader.fieldnames:
            raise ValueError(f"Label column '{lbl_col}' not found in {csv_path}")
        for row in reader:
            seq = str(row[seq_col]).strip().upper()
            lbl = str(row[lbl_col]).strip()
            if len(seq) != len(lbl):
                continue
            sequences.append(seq)
            labels.append(lbl)
    return sequences, labels


def is_positive_label(label: int, spec: Dict) -> bool:
    return label == spec["positive_class"]


def class_name(label: int, spec: Dict) -> str:
    return spec["class_names"].get(label, str(label))
