from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from single.sae.dictionary import ReLUSAE, TopKSAE, Dictionary
from single.configs import SAEConfig


def load_sae(
    model_dir: Union[str, Path],
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    prefer_normalized: bool = True,
) -> Dictionary:
    """
    Load a trained SAE.

    By default, prefers the NORMALIZED weights (`model_normalized.pt`, which stores
    per-feature max-activation rescale factors) over the raw `model.pt`, so downstream
    alignment/metrics use features on a comparable 0-1 scale. Set
    `prefer_normalized=False` or pass an explicit `model_name` to force a specific
    checkpoint.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = Path(model_dir)

    if model_name is None:
        # Prefer the normalized checkpoint if it exists.
        if prefer_normalized and (model_dir / "model_normalized.pt").exists():
            model_name = "model_normalized.pt"
        else:
            model_name = "model.pt"

    state_dict = torch.load(
        model_dir / model_name, map_location=torch.device(device), weights_only=True
    )

    dict_size, activation_dim = state_dict["encoder.weight"].shape
    if "k" in state_dict:
        k = state_dict["k"].item()
        sae = TopKSAE.from_pretrained(model_dir / model_name, k=k, device=device)
    else:
        sae = ReLUSAE.from_pretrained(model_dir / model_name, device=device)

    if model_name == "model_normalized.pt":
        print(f"  Loaded normalized SAE: {model_dir / model_name}")
    return sae


def get_sae_feats_in_batches(
    sae: Dictionary,
    aa_embds: Union[np.ndarray, torch.Tensor],
    chunk_size: int = 1024,
    feat_list: Optional[List[int]] = None,
    normalize_features: bool = False,
    device: Optional[str] = None,
) -> torch.Tensor:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if feat_list is None:
        feat_list = list(range(sae.dict_size))

    if torch.is_tensor(aa_embds):
        aa_embds = aa_embds.to(device)
    else:
        aa_embds = torch.tensor(aa_embds, device=device)

    all_features = []
    for i in range(0, len(aa_embds), chunk_size):
        chunk = aa_embds[i : i + chunk_size]
        features = sae.encode_feat_subset(
            chunk, feat_list, normalize_features=normalize_features
        )
        all_features.append(features)

    return torch.vstack(all_features)


def split_up_feature_list(total_features: int, max_feature_chunk_size: int = 2560):
    feature_chunk_size = min(max_feature_chunk_size, total_features)
    num_chunks = int(np.ceil(total_features / feature_chunk_size))
    return np.array_split(range(total_features), num_chunks)


def normalize_sae_features(sae: Dictionary, max_per_feat: torch.Tensor) -> Dictionary:
    # Clamp so never-activated features (max == 0) don't cause 0/0 -> NaN/Inf
    # when encode() divides by the rescale factor.
    max_per_feat = torch.clamp(max_per_feat, min=1e-8)
    if hasattr(sae, "activation_rescale_factor"):
        sae.activation_rescale_factor = max_per_feat
    else:
        sae.register_buffer("activation_rescale_factor", max_per_feat)
    return sae
