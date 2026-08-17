from abc import ABC, abstractmethod
from typing import Optional

import torch as t
import torch.nn as nn


class Dictionary(ABC, nn.Module):
    dict_size: int
    activation_dim: int

    def __init__(self, normalize_to_sqrt_d=False):
        super().__init__()
        # Register as a buffer so it is persisted in state_dict and restored by
        # from_pretrained (a plain attribute would silently reset to False).
        self.register_buffer(
            "normalize_to_sqrt_d", t.as_tensor(bool(normalize_to_sqrt_d))
        )

    def _normalize_input_and_get_norms(self, x):
        if self.normalize_to_sqrt_d:
            original_norms = t.norm(x, dim=-1, keepdim=True)
            original_norms = t.clamp(original_norms, min=1e-8)
            d = x.shape[-1]
            sqrt_d = t.sqrt(t.tensor(d, dtype=x.dtype, device=x.device))
            normalized = t.nn.functional.normalize(x, p=2, dim=-1) * sqrt_d
            return normalized, original_norms
        return x, None

    def _unnormalize_output(self, x_normalized, original_norms):
        if self.normalize_to_sqrt_d:
            d = x_normalized.shape[-1]
            sqrt_d = t.sqrt(t.tensor(d, dtype=x_normalized.dtype, device=x_normalized.device))
            return x_normalized * original_norms / sqrt_d
        return x_normalized

    @abstractmethod
    def encode(self, x, normalize_features: bool = False):
        pass

    @abstractmethod
    def decode(self, f):
        pass

    @abstractmethod
    def encode_feat_subset(self, x, feat_list, normalize_features: bool = False):
        pass

    @classmethod
    def from_pretrained(cls, path: str, device=None):
        raise NotImplementedError


class ReLUSAE(Dictionary):
    """
    ReLU SAE with pre-encoding bias and untied encoder/decoder weights.

    Architecture:
      f = ReLU(W_enc · (x - b_enc) + b_enc)
      x_hat = W_dec · f + b_dec
    """

    def __init__(self, activation_dim, dict_size, normalize_to_sqrt_d=False):
        super().__init__(normalize_to_sqrt_d)
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        self.bias = nn.Parameter(t.zeros(activation_dim))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        nn.init.zeros_(self.encoder.bias)

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        dec_weight = t.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)

        self.register_buffer("activation_rescale_factor", t.ones(dict_size))

    def encode(self, x, normalize_features: bool = False):
        features = nn.ReLU()(self.encoder(x - self.bias))
        if normalize_features:
            features /= self.activation_rescale_factor
        return features

    def decode(self, f):
        return self.decoder(f) + self.bias

    def forward(self, x, output_features=False, unnormalize=False):
        x, original_norms = self._normalize_input_and_get_norms(x)
        f = self.encode(x)
        x_hat = self.decode(f)
        if unnormalize:
            x_hat = self._unnormalize_output(x_hat, original_norms)
        if output_features:
            return x_hat, f
        return x_hat

    @t.no_grad()
    def encode_feat_subset(self, x, feat_list, normalize_features: bool = False):
        encoder_w_subset = self.encoder.weight[feat_list, :]
        encoder_b_subset = self.encoder.bias[feat_list]
        x, _ = self._normalize_input_and_get_norms(x)
        features = t.nn.ReLU()((x - self.bias) @ encoder_w_subset.T + encoder_b_subset)
        if normalize_features:
            features /= self.activation_rescale_factor[feat_list]
        return features

    @classmethod
    def from_pretrained(cls, path, device=None):
        if device is None:
            device = "cuda" if t.cuda.is_available() else "cpu"
        state_dict = t.load(path, map_location=device, weights_only=True)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        normalize_to_sqrt_d = state_dict.get("normalize_to_sqrt_d", t.tensor(False)).item()
        sae = cls(activation_dim, dict_size, normalize_to_sqrt_d=normalize_to_sqrt_d)
        # strict=False: old checkpoints predate the `normalize_to_sqrt_d` buffer,
        # so their state_dict lacks that key; the constructor already set it.
        sae.load_state_dict(state_dict, strict=False)
        return sae.to(device)


class TopKSAE(Dictionary):
    """
    Top-k SAE: only the k largest activations survive per token.
    """

    def __init__(self, activation_dim, dict_size, k, normalize_to_sqrt_d=False):
        super().__init__(normalize_to_sqrt_d)
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.register_buffer("k", t.tensor(k, dtype=t.int))

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        dec_weight = t.randn_like(self.decoder.weight)
        self.decoder.weight = nn.Parameter(dec_weight / dec_weight.norm(dim=0, keepdim=True))

        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.weight.data = self.decoder.weight.T.clone()
        self.encoder.bias.data.zero_()
        self.b_dec = nn.Parameter(t.zeros(activation_dim))

        self.register_buffer("activation_rescale_factor", t.ones(dict_size))

    def encode(self, x, normalize_features: bool = False):
        post_relu = nn.functional.relu(self.encoder(x - self.b_dec))
        topk = post_relu.topk(self.k, sorted=False, dim=-1)
        buffer = t.zeros_like(post_relu)
        encoded = buffer.scatter_(dim=-1, index=topk.indices, src=topk.values)
        if normalize_features:
            encoded /= self.activation_rescale_factor
        return encoded

    def decode(self, f):
        return self.decoder(f) + self.b_dec

    def forward(self, x, output_features=False, unnormalize=False):
        x, original_norms = self._normalize_input_and_get_norms(x)
        f = self.encode(x)
        x_hat = self.decode(f)
        if unnormalize:
            x_hat = self._unnormalize_output(x_hat, original_norms)
        if output_features:
            return x_hat, f
        return x_hat

    @t.no_grad()
    def encode_feat_subset(self, x, feat_list, normalize_features: bool = False):
        # Top-K selection is global — only the top-k features survive per token.
        # A local Top-K over a subset would allow features that would NOT survive
        # the full Top-K to appear as "active", producing spurious activations.
        # Correct approach: full encode, then slice to the requested subset.
        f = self.encode(x, normalize_features=normalize_features)
        return f[..., feat_list]

    @classmethod
    def from_pretrained(cls, path, k=None, device=None):
        if device is None:
            device = "cuda" if t.cuda.is_available() else "cpu"
        state_dict = t.load(path, map_location=device, weights_only=True)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        if k is None:
            k = state_dict["k"].item()
        normalize_to_sqrt_d = state_dict.get("normalize_to_sqrt_d", t.tensor(False)).item()
        sae = cls(activation_dim, dict_size, k, normalize_to_sqrt_d=normalize_to_sqrt_d)
        # strict=False: old checkpoints predate the `normalize_to_sqrt_d` buffer,
        # so their state_dict lacks that key; the constructor already set it.
        sae.load_state_dict(state_dict, strict=False)
        return sae.to(device)
