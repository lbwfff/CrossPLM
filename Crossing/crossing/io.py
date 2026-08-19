"""I/O helpers for Crossing: residue-aligned embedding loading."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None


def discover_shard_ids(embed_dir: Path) -> List[int]:
    ids: List[int] = []
    for child in Path(embed_dir).glob("shard_*"):
        m = re.fullmatch(r"shard_(\d+)", child.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(set(ids))


def _load_single_shard_embeddings(shard_path: Path) -> torch.Tensor:
    emb_path = shard_path / "embeddings.pt"
    if not emb_path.exists():
        cands = sorted(shard_path.glob("**/embeddings.pt"))
        if not cands:
            raise FileNotFoundError(f"No embeddings.pt under {shard_path}")
        emb_path = cands[0]
    data = torch.load(emb_path, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        if "embeddings" in data:
            emb = data["embeddings"]
        elif "hidden_states" in data:
            emb = data["hidden_states"]
        else:
            emb = next(v for v in data.values() if isinstance(v, torch.Tensor))
    elif isinstance(data, list):
        emb = data[0] if len(data) == 1 else torch.cat(data, dim=0)
    else:
        emb = data
    if not isinstance(emb, torch.Tensor):
        emb = torch.as_tensor(emb)
    return emb.float()


def _load_shard_residues(shard_path: Path) -> Optional["pd.DataFrame"]:
    csv = shard_path / "residues.csv"
    if csv.exists() and pd is not None:
        try:
            return pd.read_csv(csv, dtype=str)
        except Exception:
            return None
    return None


def load_embeddings_with_residues(
    embed_dir: str | Path,
) -> Tuple[torch.Tensor, Optional["pd.DataFrame"]]:
    """Load all shard embeddings concatenated plus optional residue metadata."""
    embed_dir = Path(embed_dir)
    shard_ids = discover_shard_ids(embed_dir)
    if not shard_ids:
        flat_pt = embed_dir / "embeddings.pt"
        if flat_pt.exists():
            data = torch.load(flat_pt, map_location="cpu", weights_only=True)
            if isinstance(data, dict) and "embeddings" in data:
                emb = data["embeddings"].float()
            else:
                emb = data.float() if isinstance(data, torch.Tensor) else torch.as_tensor(data).float()
            flat_csv = Path(embed_dir) / "residues.csv"
            residues = pd.read_csv(flat_csv, dtype=str) if flat_csv.exists() and pd is not None else None
            if residues is not None and len(residues) != emb.shape[0]:
                raise ValueError(f"Flat residues {len(residues)} != tokens {emb.shape[0]}")
            return emb, residues
        # maybe caller passed .pt file directly
        if Path(embed_dir).is_file():
            data = torch.load(Path(embed_dir), map_location="cpu", weights_only=True)
            if isinstance(data, dict) and "embeddings" in data:
                emb = data["embeddings"].float()
            else:
                emb = data.float() if isinstance(data, torch.Tensor) else torch.as_tensor(data).float()
            return emb, None
        raise FileNotFoundError(f"No shard_* directories and no embeddings.pt in {embed_dir}")

    all_embs: List[torch.Tensor] = []
    all_residues: List["pd.DataFrame"] = []
    has_residues = True
    for sid in shard_ids:
        shard_path = embed_dir / f"shard_{sid}"
        emb = _load_single_shard_embeddings(shard_path)
        all_embs.append(emb)
        residues = _load_shard_residues(shard_path)
        if residues is None:
            has_residues = False
        else:
            if len(residues) != emb.shape[0]:
                raise ValueError(
                    f"Shard {sid}: residues.csv has {len(residues)} rows but "
                    f"embeddings.pt has {emb.shape[0]} tokens"
                )
            all_residues.append(residues)

    embeddings = torch.cat(all_embs, dim=0)
    residues: Optional["pd.DataFrame"] = None
    if has_residues and all_residues:
        residues = pd.concat(all_residues, ignore_index=True)
        if len(residues) != embeddings.shape[0]:
            raise ValueError(f"Total residues {len(residues)} != total tokens {embeddings.shape[0]}")
    return embeddings, residues


def align_two_embeddings(
    emb_a: torch.Tensor,
    residues_a: Optional["pd.DataFrame"],
    emb_b: torch.Tensor,
    residues_b: Optional["pd.DataFrame"],
    max_tokens: Optional[int] = None,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Align two embedding sets token-wise via residues.csv when available."""
    info: Dict = {}
    n_a = emb_a.shape[0]
    n_b = emb_b.shape[0]
    info["n_tokens_a_raw"] = int(n_a)
    info["n_tokens_b_raw"] = int(n_b)
    info["aligned_via"] = "none"

    if residues_a is not None and residues_b is not None and pd is not None:
        cols_a = set(residues_a.columns)
        cols_b = set(residues_b.columns)
        key_cols: List[str] = []
        if "Entry" in cols_a and "Entry" in cols_b:
            key_cols.append("Entry")
        for c in ["sequence_hash", "position"]:
            if c in cols_a and c in cols_b:
                if c not in key_cols:
                    key_cols.append(c)
            else:
                key_cols = []
                break
        if key_cols:
            for df in (residues_a, residues_b):
                if "position" in df.columns:
                    df["position"] = df["position"].astype(str)
                if "sequence_hash" in df.columns:
                    df["sequence_hash"] = df["sequence_hash"].astype(str)
                if "Entry" in df.columns:
                    df["Entry"] = df["Entry"].astype(str)
            ra = residues_a.copy()
            rb = residues_b.copy()
            ra["_idx_a"] = np.arange(len(ra))
            rb["_idx_b"] = np.arange(len(rb))
            merged = ra.merge(rb, on=key_cols, how="inner", suffixes=("_a", "_b"))
            if len(merged) == 0:
                raise ValueError(
                    f"Residue alignment found 0 overlapping tokens via keys {key_cols}. "
                    "No proteins in common – for disjoint tasks re-extract embeddings on the same proteins."
                )
            idx_a = merged["_idx_a"].to_numpy(dtype=np.int64)
            idx_b = merged["_idx_b"].to_numpy(dtype=np.int64)
            info["aligned_via"] = f"residues:{'+'.join(key_cols)}"
            info["n_tokens_overlap"] = int(len(merged))
            if len(merged) != n_a or len(merged) != n_b:
                info["note"] = f"Overlap {len(merged)} vs raw {n_a}/{n_b}; using intersection."
            emb_a = emb_a[idx_a]
            emb_b = emb_b[idx_b]
            info["n_tokens_aligned"] = int(len(merged))
        else:
            if n_a != n_b:
                raise ValueError(f"Cannot residue-align (keys missing) and token counts differ: {n_a} vs {n_b}")
            info["n_tokens_aligned"] = int(n_a)
    else:
        if n_a != n_b:
            raise ValueError(
                f"Token counts differ ({n_a} vs {n_b}) and no residues.csv for alignment. "
                "Re-extract embeddings on the same proteins or ensure residues.csv exists."
            )
        info["n_tokens_aligned"] = int(n_a)

    if max_tokens is not None and max_tokens < emb_a.shape[0]:
        rng = np.random.RandomState(seed)
        idx = rng.choice(emb_a.shape[0], size=max_tokens, replace=False)
        idx = np.sort(idx)
        emb_a = emb_a[idx]
        emb_b = emb_b[idx]
        info["subsampled_to"] = int(max_tokens)
        info["subsample_seed"] = int(seed)

    return emb_a, emb_b, info
