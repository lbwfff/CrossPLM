"""
Convert UniProtKB/Swiss-Prot protein-level annotations into per-residue sparse concept labels.

A "concept" is a biological annotation (e.g. "Domain_kinase", "Helix", "Active site")
expanded to the amino-acid level. The output is a sparse matrix of shape
[n_tokens, n_concepts] where each column is a binary label vector for one concept.

This enables feature-to-biology alignment: for each SAE feature, check how well its
activations predict each concept (F1 / precision / recall).

UniProtKB TSV columns used (Feature table columns):
    - "Active site", "Binding site", "Domain [FT]", "Region", "Motif",
      "Zinc finger", "Modified residue", "Glycosylation", "Transit peptide",
      "Signal peptide", "Compositional bias", "Cofactor"   (categorical, /note=...)
    - "Helix", "Beta strand", "Turn", "Coiled coil", "Lipidation"          (binary)
    - "Disulfide bond"                                                      (interaction)

Format of a single cell (Feature table):
    DOMAIN        12..93; /note="Protein kinase".
    DOMAIN        300..500; /note="Zinc finger".
    or for binary features:
    HELIX         5..20.
    DISULFID      45..45.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

# Feature table column names in UniProtKB TSV
# NOTE on categorization:
# - "Signal peptide" is BINARY: its cells are "SIGNAL 1..17; /evidence=..." with
#   no subcategory field, so parsing it as categorical drops every entry.
# - "Cofactor" is a PROTEIN-level annotation ("COFACTOR: Name=Mg(2+); ...") with
#   no residue position, so it cannot be expanded to residues here at all.
CATEGORICAL_CONCEPTS = [
    "Active site", "Binding site", "Glycosylation",
    "Modified residue", "Transit peptide", "Compositional bias",
    "Domain [FT]", "Region", "Zinc finger", "Motif",
]
BINARY_CONCEPTS = ["Turn", "Helix", "Beta strand", "Coiled coil", "Lipidation",
                   "Signal peptide"]
INTERACTION_CONCEPTS = ["Disulfide bond"]

# TSV column name -> UniProt Feature-table prefix (fixed abbreviations)
# These prefixes appear literally at the start of each annotation entry in the cell.
COLUMN_TO_PREFIX = {
    "Active site": "ACT_SITE",
    "Binding site": "BINDING",
    "Cofactor": "COFACTOR",
    "Glycosylation": "CARBOHYD",
    "Modified residue": "MOD_RES",
    "Transit peptide": "TRANSIT",
    "Compositional bias": "COMPBIAS",
    "Domain [FT]": "DOMAIN",
    "Region": "REGION",
    "Zinc finger": "ZN_FING",
    "Motif": "MOTIF",
    "Signal peptide": "SIGNAL",
    "Turn": "TURN",
    "Helix": "HELIX",
    "Beta strand": "STRAND",
    "Coiled coil": "COILED",
    "Lipidation": "LIPID",
    "Disulfide bond": "DISULFID",
}


def column_to_prefix(col: str) -> str:
    """Return the UniProt Feature-table prefix for a TSV column name."""
    return COLUMN_TO_PREFIX.get(col, col.split(" ")[0].upper())

# Concepts that are naturally amino-acid-level (single residues, not domains)
PER_AA_CONCEPTS = [
    "Active site", "Cofactor", "Glycosylation", "Modified residue", "Disulfide bond",
]


def is_aa_level_concept(concept_name: str) -> bool:
    """Whether a concept is AA-level (single residue) vs domain-level (contiguous region)."""
    return any(aa_c in concept_name for aa_c in PER_AA_CONCEPTS)


def _parse_position(positions: str) -> Optional[List[int]]:
    """Parse a UniProt position string like '12..93' or '45' into 0-based indices."""
    positions = positions.strip().lstrip("<").rstrip(">")
    if ":" in positions or "?" in positions:
        return None
    if ".." in positions:
        start, end = positions.split("..")
        start = int(start.strip("<").strip())
        end = int(end.strip(">").strip())
        return list(range(start - 1, end))
    return [int(positions) - 1]


def _process_binary(column_data, column_name, seq_len):
    """Return list of instance indices (0 = not annotated) for binary features."""
    indices = [0] * seq_len
    if pd.isna(column_data):
        return indices
    entries = column_data.split(f"{column_name} ")[1:]
    for instance, entry in enumerate(entries, start=1):
        positions = entry.split(";")[0]
        idxs = _parse_position(positions)
        if idxs is None:
            continue
        for i in idxs:
            if 0 <= i < seq_len:
                indices[i] = instance
    return indices


def _process_interaction(column_data, column_name, seq_len):
    """Disulfide bonds: mark both residues, with pair id as instance index."""
    indices = [0] * seq_len
    if pd.isna(column_data):
        return indices
    entries = column_data.split(f"{column_name} ")[1:]
    for instance, entry in enumerate(entries, start=1):
        positions = entry.split(";")[0]
        idxs = _parse_position(positions)
        if idxs is None:
            continue
        for i in idxs:
            if 0 <= i < seq_len:
                indices[i] = instance
    return indices


# UniProt feature annotation fields that can carry a subcategory name
_ANNOT_FIELDS = [
    r'/note="([^"]+)"',
    r'/ligand="([^"]+)"',
    r'/description="([^"]+)"',
    r'/cofactor="([^"]+)"',
]


def _process_categorical(column_data, column_name, category_options, seq_len):
    """
    Categorical features with subcategory annotations (/note, /ligand, /description...).
    Returns dict category -> index list.

    Each distinct annotation instance (e.g. each separate Domain) gets a unique
    integer index (1, 2, 3...), so domain-level F1 can count unique domains hit.
    The "any" category increments per instance as well (like InterPLM).
    """
    cat_indices = {cat: [0] * seq_len for cat in category_options}
    current_index = {cat: 1 for cat in category_options}
    if pd.isna(column_data):
        return cat_indices

    entries = column_data.split(f"{column_name} ")[1:]
    for entry in entries:
        # Only look at this annotation's own segment: everything up to the next
        # occurrence of the prefix (or end of cell). Searching the whole chunk
        # can leak a LATER annotation's /note into an earlier entry that lacks one.
        next_prefix = entry.find(f"{column_name} ", 1)
        segment = entry if next_prefix == -1 else entry[:next_prefix]

        # Try each annotation field type
        subcategory = None
        for field_re in _ANNOT_FIELDS:
            match = re.search(field_re, segment)
            if match:
                subcategory = match.group(1).split(";")[0]
                break
        if subcategory is None:
            # No subcategory -> not categorical; skip (binary-style entry).
            continue

        # Normalize subcategory name
        cat = subcategory.lower().replace(" ", "_").replace("-", "_")
        if cat not in category_options:
            continue

        positions = segment.split(";")[0]
        idxs = _parse_position(positions)
        if idxs is None:
            continue
        for i in idxs:
            if 0 <= i < seq_len:
                cat_indices[cat][i] = current_index[cat]
                # "any" counts every instance regardless of subcategory
                if "any" in cat_indices:
                    cat_indices["any"][i] = current_index["any"]

        current_index[cat] += 1
        if "any" in cat_indices:
            current_index["any"] += 1

    return cat_indices


def compute_categorical_options(df: pd.DataFrame) -> Dict[str, set]:
    """
    Compute the complete set of categorical subcategories across ALL proteins.
    Returns a dict mapping column name -> set of subcategory names (incl. "any").
    Must be called on the full dataset so every shard shares the same concept columns.
    """
    categorical_options = {}
    for col in CATEGORICAL_CONCEPTS:
        col_name = column_to_prefix(col)
        options = set()
        if col in df.columns:
            for value in df[col].dropna():
                for entry in str(value).split(f"{col_name} ")[1:]:
                    # Use the SAME annotation fields as _process_categorical
                    # (/note, /ligand, /description, /cofactor) so the options
                    # collected here match what the parser will look for.
                    for field_re in _ANNOT_FIELDS:
                        match = re.search(field_re, entry)
                        if match:
                            sub = match.group(1).split(";")[0]
                            options.add(sub.lower().replace(" ", "_").replace("-", "_"))
                            break
        options = set(options) | {"any"}
        categorical_options[col] = options
    return categorical_options


def expand_annotations_to_residues(
    df: pd.DataFrame,
    categorical_options: Optional[Dict[str, set]] = None,
    max_residues: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Expand protein-level UniProt annotation columns to per-residue concept columns.

    Args:
        df: DataFrame with UniProt annotations (one row per protein)
        categorical_options: optional precomputed options (from
            compute_categorical_options on the full dataset). If None, computed
            from this df (shard-local, may cause column mismatch across shards).
        max_residues: optional cap on residues per protein. MUST match the
            embedder's truncation (max_length - 2) so concept rows align with
            embedding tokens. If None, the full sequence is used (default).

    Returns:
        df: DataFrame with one row per amino acid, columns include:
            Entry, amino_acid, position, and one column per concept
        concept_columns: list of concept column names
    """
    new_columns = {}

    def _per_residue_len(seq: str) -> int:
        """Residues kept per protein, matching the embedder's truncation."""
        return min(len(seq), max_residues) if max_residues is not None else len(seq)

    # Binary concepts
    for col in BINARY_CONCEPTS:
        col_name = column_to_prefix(col)
        concept_cols = []
        for _, row in df.iterrows():
            indices = _process_binary(row[col], col_name, _per_residue_len(str(row["Sequence"])))
            concept_cols.append(indices)
        new_columns[col] = concept_cols

    # Interaction concepts
    for col in INTERACTION_CONCEPTS:
        col_name = column_to_prefix(col)
        concept_cols = []
        for _, row in df.iterrows():
            indices = _process_interaction(row[col], col_name, _per_residue_len(str(row["Sequence"])))
            concept_cols.append(indices)
        new_columns[col] = concept_cols

    # Categorical concepts
    if categorical_options is None:
        categorical_options = compute_categorical_options(df)
    for col, options in categorical_options.items():
        col_name = column_to_prefix(col)
        # Build per-category residue indices
        cat_columns = {f"{col}_{cat}": [] for cat in options}
        for _, row in df.iterrows():
            cat_indices = _process_categorical(row[col], col_name, options,
                                               _per_residue_len(str(row["Sequence"])))
            for cat in options:
                cat_columns[f"{col}_{cat}"].append(cat_indices[cat])
        for concept_name, concept_lists in cat_columns.items():
            new_columns[concept_name] = concept_lists

    # Build residue-level dataframe (truncated to max_residues per protein).
    rows = []
    for idx, row in df.iterrows():
        seq = str(row["Sequence"])
        for pos in range(_per_residue_len(seq)):
            rows.append({"Entry": row["Entry"], "amino_acid": seq[pos], "position": pos})
    result = pd.DataFrame(rows)

    # Add concept columns as flattened per-residue values.
    # IMPORTANT: instance indices are per-protein (1,2,...). To make them globally
    # unique across proteins (so domain counting per concept is correct), we add a
    # per-protein offset. This preserves per-protein instance identity while
    # preventing different proteins' "instance 1" from being conflated.
    n_proteins = len(df)
    # offset per protein: per-protein instance ids are small (<1000), so a stride
    # of 1e6 is ample and keeps values well within int32 range even for 1000s of
    # proteins (max ~1e6 * 2000 = 2e9). Using int64 throughout for safety.
    _OFFSET_STRIDE = 1_000_000
    for concept_name, concept_lists in new_columns.items():
        flattened = []
        for i, concept_list in enumerate(concept_lists):
            offset = (i + 1) * _OFFSET_STRIDE
            flattened.extend(
                (offset + v) if v > 0 else 0
                for v in concept_list
            )
        result[concept_name] = flattened

    concept_columns = [c for c in result.columns if c not in ["Entry", "amino_acid", "position"]]
    return result, concept_columns


def build_concept_matrix(
    annotations_tsv: Path,
    output_dir: Path,
    n_shards: int = 5,
    min_seq_len: int = 30,
    max_seq_len: int = 1022,
    max_residues: Optional[int] = None,
):
    """
    Convert a UniProtKB TSV into sharded per-residue sparse concept matrices.

    Args:
        annotations_tsv: UniProtKB export TSV with 'Entry', 'Sequence', and
                         feature-table columns (Helix, Domain [FT], etc.)
        output_dir: where to write shard_N/aa_concepts.npz + metadata
        n_shards: number of shards to split proteins into
        max_residues: residues kept per protein. MUST equal the embedder's
            truncation (embedder max_length - 2) so concept rows align with
            embedding tokens. If None, full sequences are used (legacy).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(annotations_tsv, sep="\t", low_memory=False)
    df = df[df["Sequence"].notna()]
    df = df[df["Sequence"].apply(lambda s: min_seq_len <= len(str(s)) <= max_seq_len)]

    print(f"Loaded {len(df)} proteins after length filtering")

    # Compute global categorical options so all shards share identical concept columns
    print("Computing global categorical options...")
    global_options = compute_categorical_options(df)

    # Split into shards
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    shard_size = int(np.ceil(len(df) / n_shards))
    shards = [df.iloc[i:i + shard_size].reset_index(drop=True)
              for i in range(0, len(df), shard_size)]

    all_concept_columns = None
    for shard_id, shard_df in enumerate(shards):
        print(f"Processing shard {shard_id}/{n_shards} ({len(shard_df)} proteins)...")
        residue_df, concept_columns = expand_annotations_to_residues(
            shard_df, categorical_options=global_options, max_residues=max_residues,
        )

        # Build sparse matrix directly from the per-concept residue lists using COO,
        # avoiding a dense [n_residues, n_concepts] intermediate (which can be many
        # GB for large shards). Values are domain-instance indices (>0), 0 = absent.
        n_residues = len(residue_df)
        n_concepts = len(concept_columns)
        rows, cols, data = [], [], []
        for j, concept in enumerate(concept_columns):
            vals = residue_df[concept].values
            nz = np.nonzero(vals)[0]
            if nz.size:
                rows.append(nz)
                cols.append(np.full(nz.size, j, dtype=np.int64))
                data.append(vals[nz])
        if rows:
            rows = np.concatenate(rows)
            cols = np.concatenate(cols)
            data = np.concatenate(data)
        else:
            rows = np.empty(0, dtype=np.int64)
            cols = np.empty(0, dtype=np.int64)
            data = np.empty(0, dtype=np.int64)

        sparse_mat = sparse.csr_matrix((data, (rows, cols)),
                                       shape=(n_residues, n_concepts),
                                       dtype=np.int64)
        shard_dir = output_dir / f"shard_{shard_id}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(shard_dir / "aa_concepts.npz", sparse_mat)

        # Save positive domain counts per concept (unique instance indices)
        n_domains = count_domains_per_concept(sparse_mat)
        np.save(shard_dir / "n_domains_per_concept.npy", n_domains)

        # Save residue metadata
        residue_df[["Entry", "amino_acid", "position"]].to_csv(
            shard_dir / "aa_metadata.csv", index=False
        )

        if all_concept_columns is None:
            all_concept_columns = concept_columns
            (output_dir / "aa_concepts_columns.txt").write_text(
                "\n".join(concept_columns)
            )

        print(f"  Shard {shard_id}: {n_residues:,} residues, {n_concepts} concepts")

    print(f"Done. Concept matrices in {output_dir}")


def load_concept_shards(concepts_dir: Path, shard: int):
    """Load concept matrix and metadata for a shard."""
    shard_dir = Path(concepts_dir) / f"shard_{shard}"
    matrix = sparse.load_npz(shard_dir / "aa_concepts.npz")
    metadata = pd.read_csv(shard_dir / "aa_metadata.csv")
    return matrix, metadata


def count_domains_per_concept(concept_matrix) -> np.ndarray:
    """
    Count the number of unique positive domain-instance indices per concept column.
    For binary concepts (e.g. Helix) each contiguous annotation is a distinct
    instance in the raw annotation, so this counts annotation segments.
    Returns an array of shape [n_concepts].
    """
    mat = concept_matrix.tocsr() if sparse.issparse(concept_matrix) else sparse.csr_matrix(concept_matrix)
    counts = []
    for c in range(mat.shape[1]):
        col_data = mat.getcol(c).data
        if col_data.size == 0:
            counts.append(0)
        else:
            counts.append(len(np.unique(col_data[col_data > 0])))
    return np.array(counts, dtype=np.int64)


def load_concept_names(concepts_dir: Path) -> List[str]:
    names_path = Path(concepts_dir) / "aa_concepts_columns.txt"
    if names_path.exists():
        return names_path.read_text().strip().split("\n")
    return []
