from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from single.sae.dictionary import ReLUSAE, TopKSAE, Dictionary
from single.configs import SAEConfig


def load_sae(
    model_dir: Union[str, Path],
    model_name: str = "ae.pt",
    device: Optional[str] = None,
) -> Dictionary:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = Path(model_dir)

    state_dict = torch.load(
        model_dir / model_name, map_location=torch.device(device), weights_only=True
    )

    dict_size, activation_dim = state_dict["encoder.weight"].shape
    if "k" in state_dict:
        k = state_dict["k"].item()
        sae = TopKSAE.from_pretrained(model_dir / model_name, k=k, device=device)
    else:
        sae = ReLUSAE.from_pretrained(model_dir / model_name, device=device)

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
    if hasattr(sae, "activation_rescale_factor"):
        sae.activation_rescale_factor = max_per_feat
    else:
        sae.register_buffer("activation_rescale_factor", max_per_feat)
    return sae
