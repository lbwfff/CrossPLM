"""
Configurable label encoding for different datasets.

Each dataset defines how raw per-residue characters map to integer class labels.
Unmapped characters become -100 (ignore in analysis), matching HuggingFace convention.

Usage:
    from single.label_maps import get_label_map

    spec = get_label_map("mBMRB")       # built-in preset
    spec = get_label_map("file.yaml")   # or load from a YAML file
"""

from pathlib import Path
from typing import Dict, Optional

import yaml

# Built-in presets. Structure:
#   positive_class : which integer label is treated as the "positive" class
#   class_names    : int label -> human-readable name (for reporting)
#   mapping        : raw character -> integer label
#   ignore         : extra characters to treat as -100 (optional; any unmapped char is ignored anyway)
LABEL_MAPS: Dict[str, Dict] = {
    "mBMRB": {
        "positive_class": 1,
        "class_names": {0: "rigid", 1: "flexible"},
        "mapping": {"A": 0, ".": 1, "0": 0, "1": 1},
        "ignore": "_",
    },
    "relaxdb": {
        "positive_class": 1,
        "class_names": {0: "static", 1: "mobile"},
        "mapping": {"p": 0, "A": 0, "v": 0, ".": 1, "b": 1, "^": 1},
        "ignore": "_",
    },
    # Example for 3-class secondary structure. The integer ids are assigned to
    # match the training module's build_label_map() (sorted(unique)) so that a
    # model trained with an inferred label map evaluates consistently with this
    # preset: C=0, E=1, H=2.
    "ss3": {
        "positive_class": 1,
        "class_names": {0: "coil", 1: "strand", 2: "helix"},
        "mapping": {"C": 0, "E": 1, "H": 2},
        "ignore": "_",
    },
}


def _normalize_spec(spec: Dict) -> Dict:
    """Fill defaults and validate a label-map spec."""
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


def is_positive_label(label: int, spec: Dict) -> bool:
    return label == spec["positive_class"]


def class_name(label: int, spec: Dict) -> str:
    return spec["class_names"].get(label, str(label))
