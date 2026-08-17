from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
    cache: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if feat_list is None:
        feat_list = list(range(sae.dict_size))

    # TopK selection is global. Cache only sparse top-k indices/values rather
    # than the dense [n_tokens, dict_size] activation matrix.
    if cache is not None and isinstance(sae, TopKSAE):
        cache_key = (id(sae), id(aa_embds), chunk_size, normalize_features, str(device))
        if cache.get("key") != cache_key:
            cache.clear()
            cache["key"] = cache_key
        if "indices" not in cache:
            source = aa_embds if torch.is_tensor(aa_embds) else torch.as_tensor(aa_embds)
            indices, values = [], []
            k = min(int(sae.k.item()), sae.dict_size)
            for i in range(0, len(source), chunk_size):
                chunk = source[i : i + chunk_size].to(device)
                full = sae.encode(chunk, normalize_features=normalize_features)
                top_values, top_indices = full.topk(k, dim=-1, sorted=False)
                indices.append(top_indices.detach().cpu())
                values.append(top_values.detach().cpu())
            cache["indices"] = torch.cat(indices, dim=0)
            cache["values"] = torch.cat(values, dim=0)
            cache["n_tokens"] = len(source)

        selected = [int(f) for f in feat_list]
        lookup = torch.full((sae.dict_size,), -1, dtype=torch.long, device=device)
        lookup[torch.tensor(selected, dtype=torch.long, device=device)] = torch.arange(
            len(selected), device=device
        )
        outputs = []
        for i in range(0, cache["n_tokens"], chunk_size):
            indices = cache["indices"][i : i + chunk_size].to(device)
            values = cache["values"][i : i + chunk_size].to(device)
            local_indices = lookup[indices]
            valid = local_indices >= 0
            output = torch.zeros(
                indices.shape[0], len(selected), device=device, dtype=values.dtype
            )
            output.scatter_add_(
                1, local_indices.clamp_min(0), values * valid.to(values.dtype)
            )
            outputs.append(output)
        return torch.cat(outputs, dim=0)

    if cache is not None:
        cache_key = (id(sae), id(aa_embds), chunk_size, normalize_features, str(device))
        if cache.get("key") != cache_key:
            cache.clear()
            cache["key"] = cache_key
        source = aa_embds if torch.is_tensor(aa_embds) else torch.as_tensor(aa_embds)
        estimated_bytes = len(source) * sae.dict_size * 4
        max_cache_bytes = int(cache.get("max_cache_bytes", 512 * 1024 * 1024))
        if estimated_bytes <= max_cache_bytes:
            if "features" not in cache:
                all_features = []
                all_feature_ids = list(range(sae.dict_size))
                for i in range(0, len(source), chunk_size):
                    chunk = source[i : i + chunk_size].to(device)
                    all_features.append(
                        sae.encode_feat_subset(
                            chunk, all_feature_ids,
                            normalize_features=normalize_features,
                        ).detach().cpu()
                    )
                cache["features"] = torch.cat(all_features, dim=0)
            return cache["features"][..., feat_list].to(device)

    if not torch.is_tensor(aa_embds):
        aa_embds = torch.as_tensor(aa_embds)

    all_features = []
    for i in range(0, len(aa_embds), chunk_size):
        chunk = aa_embds[i : i + chunk_size].to(device)
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
