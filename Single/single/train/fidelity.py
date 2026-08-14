"""
Fidelity evaluation for a fine-tuned token-classification PLM.

Measures how well a SAE preserves the fine-tuned model's task performance.

Core idea (from InterPLM, adapted from MLM to token classification):
  For a target hidden layer L, run the model three ways:
    ce_orig : task loss with the model's ORIGINAL hidden states at layer L
    ce_sae  : task loss when layer-L hidden states are REPLACED by the SAE's
              reconstructions of those hidden states
    ce_zero : task loss when layer-L hidden states are ZEROED (worst case)

  Loss_Recovered = 1 - (ce_sae - ce_orig) / (ce_zero - ce_orig)

  100% means the SAE perfectly preserves task-relevant information at layer L;
  0% means it is as harmful as zeroing the layer.
"""

import torch
import numpy as np
from tqdm import tqdm

from single.sae.dictionary import Dictionary


def _get_injection_point(model, layer_idx: int):
    """
    Return (submodule, attr) whose output corresponds to `hidden_states[layer_idx]`.

    For the FastESM fine-tuned token-classification model:
      hidden_states[0]            = embedding output
      hidden_states[1..5]         = layer[0..4] outputs (NOT identical here — the
                                    custom model interleaves layernorms)
      hidden_states[6]            = emb_layer_norm_after output (identical, verified)

    To be safe, we inject at `esm.encoder.emb_layer_norm_after` which exactly
    matches hidden_states[6]. If a different layer is requested, we raise a
    clear error (extend support per-model as needed).
    """
    # hidden_states[6] == emb_layer_norm_after output (verified numerically)
    n_layers = len(model.esm.encoder.layer)
    if layer_idx == n_layers:
        return model.esm.encoder.emb_layer_norm_after
    raise NotImplementedError(
        f"Fidelity currently supports injecting at the final layer "
        f"(hidden_states[{n_layers}] == emb_layer_norm_after output). "
        f"Requested layer_idx={layer_idx}. Extend _get_injection_point for "
        f"intermediate layers of this custom FastESM model."
    )


@torch.no_grad()
def _model_loss_with_override(
    model,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    injection_submodule,
    override: torch.Tensor | None,
) -> float:
    """Run the model once; if override is not None, replace the injection module output."""
    if override is None:
        out = model(
            input_ids=tokens,
            attention_mask=attention_mask,
            labels=labels,
        )
        return out.loss.item()

    def _hook(module, args, output):
        return override

    handle = injection_submodule.register_forward_hook(_hook)
    try:
        out = model(
            input_ids=tokens,
            attention_mask=attention_mask,
            labels=labels,
        )
    finally:
        handle.remove()
    return out.loss.item()


def _get_hidden_states(model, tokens, attention_mask, layer_idx):
    """Get the model's own hidden_states[layer_idx] for a batch."""
    with torch.no_grad():
        out = model(
            input_ids=tokens,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    return out.hidden_states[layer_idx]


def _extract_sae_reconstructions(sae: Dictionary, hidden: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct the hidden states using the SAE.
    The SAE's forward returns x_hat in the normalized space; we pass
    unnormalize=True so the reconstruction is back in the original space.
    """
    return sae(hidden, output_features=False, unnormalize=True)


def evaluate_fidelity(
    model,
    sae: Dictionary,
    layer_idx: int,
    sequences: list[str],
    labels: list[str],
    tokenizer,
    label_map_spec: dict,
    batch_size: int = 8,
    max_length: int = 512,
    device: str = "cpu",
) -> dict:
    """
    Evaluate SAE fidelity on a set of sequences + per-residue labels.

    Args:
        model: fine-tuned AutoModelForTokenClassification
        sae: trained SAE
        layer_idx: hidden layer the SAE was trained on
        sequences: list of protein sequences
        labels: list of per-residue label strings (raw chars, e.g. mBMRB 'A'/'.')
        tokenizer: matching tokenizer
        label_map_spec: from single.label_maps.get_label_map()
        batch_size: model batch size
        device: torch device

    Returns:
        dict with ce_orig, ce_sae, ce_zero, loss_recovered
    """
    from single.label_maps import encode_label_string

    injection = _get_injection_point(model, layer_idx)
    model.eval()

    ce_orig_list, ce_sae_list, ce_zero_list = [], [], []
    recon_mse_list = []  # sanity check: how close is SAE reconstruction to original

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]

        enc = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        tokens = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        # Build per-residue label tensor (same truncation as embeddings):
        # -100 for special/ignored positions.
        label_ids = torch.full_like(tokens, -100)
        for b, (seq, lab) in enumerate(zip(batch_seqs, batch_labels)):
            seq_len = min(len(seq), max_length - 2)
            enc_ids = encode_label_string(str(lab)[:seq_len], label_map_spec)
            if len(enc_ids) < seq_len:
                enc_ids += [-100] * (seq_len - len(enc_ids))
            label_ids[b, 1 : seq_len + 1] = torch.tensor(enc_ids[:seq_len])

        # 1) original loss
        ce_orig = _model_loss_with_override(
            model, tokens, attn, label_ids, injection, None
        )

        # 2) zero-ablation loss (worst case)
        hidden = _get_hidden_states(model, tokens, attn, layer_idx)
        zeros = torch.zeros_like(hidden)
        ce_zero = _model_loss_with_override(
            model, tokens, attn, label_ids, injection, zeros
        )

        # 3) SAE reconstruction loss + reconstruction MSE sanity check
        recon = _extract_sae_reconstructions(sae, hidden)
        ce_sae = _model_loss_with_override(
            model, tokens, attn, label_ids, injection, recon
        )

        # Sanity check: MSE between SAE recon and original hidden states,
        # averaged over non-padding tokens AND over the embedding dimension.
        # (weighted.sum() is over [B, L, D]; mask.sum() counts tokens, so we
        #  must also divide by D to get a per-element MSE.)
        mask = attn.unsqueeze(-1).float()  # [B, L, 1]
        d_model = hidden.shape[-1]
        mse = (((hidden - recon) ** 2) * mask).sum() / mask.sum() / d_model

        ce_orig_list.append(ce_orig)
        ce_sae_list.append(ce_sae)
        ce_zero_list.append(ce_zero)
        recon_mse_list.append(mse.item())

    ce_orig = float(np.mean(ce_orig_list))
    ce_sae = float(np.mean(ce_sae_list))
    ce_zero = float(np.mean(ce_zero_list))
    recon_mse = float(np.mean(recon_mse_list))

    # Loss recovered, clipped to [0, 100]
    denom = ce_zero - ce_orig
    if np.isclose(denom, 0):
        loss_recovered = 100.0 if np.isclose(ce_sae, ce_orig) else 0.0
    else:
        loss_recovered = float(np.clip(1 - (ce_sae - ce_orig) / denom, 0, 1) * 100)

    return {
        "layer_idx": layer_idx,
        "n_sequences": len(sequences),
        "ce_orig": ce_orig,
        "ce_sae": ce_sae,
        "ce_zero": ce_zero,
        "loss_recovered_pct": loss_recovered,
        "reconstruction_mse": recon_mse,  # sanity check: lower = better reconstruction
    }
