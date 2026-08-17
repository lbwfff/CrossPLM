"""
Fidelity evaluation for a fine-tuned token-classification PLM.

Measures how well a SAE preserves the fine-tuned model's task performance.

Core idea (from InterPLM, adapted from MLM to token classification):
  For a target hidden layer L, run the model three ways:
    ce_orig : task loss with the model's ORIGINAL hidden states at layer L
    ce_sae  : task loss when layer-L hidden states are REPLACED by the SAE's
              reconstructions of those hidden states
    ce_zero : task loss when layer-L hidden states are ZEROED (zero-ablation baseline)

  Loss_Recovered = 1 - (ce_sae - ce_orig) / (ce_zero - ce_orig)

  100% means the SAE perfectly preserves task-relevant information at layer L;
  0% means it is as harmful as zeroing the layer.
"""

# `torch.Tensor | None` (PEP 604) annotations require Python >=3.10; this keeps
# them working on Python 3.9 (as declared by setup.py).
from __future__ import annotations

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
    return_stats: bool = False,
):
    """Run the model and return token-summed loss/count when requested."""
    if override is None:
        out = model(
            input_ids=tokens,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss_sum, count = _token_loss_sum(out.logits, labels)
        return (loss_sum, count) if return_stats else loss_sum / max(count, 1)

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
    loss_sum, count = _token_loss_sum(out.logits, labels)
    return (loss_sum, count) if return_stats else loss_sum / max(count, 1)


def _token_loss_sum(logits: torch.Tensor, labels: torch.Tensor):
    """Return unweighted CE sum and valid-token count."""
    valid = labels != -100
    if not valid.any():
        return 0.0, 0
    losses = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )
    return float(losses[valid.reshape(-1)].sum().item()), int(valid.sum().item())


def _get_residue_positions(enc, attention_mask, tokenizer, sequences, max_length):
    """Return real residue-token positions for each padded sequence."""
    special = enc.get("special_tokens_mask")
    if special is None:
        special_ids = set(tokenizer.all_special_ids)
        special = torch.tensor(
            [[int(token_id in special_ids) for token_id in row]
             for row in enc["input_ids"].tolist()],
            device=attention_mask.device,
        )
    else:
        special = special.to(attention_mask.device)
    positions = []
    for batch_idx, sequence in enumerate(sequences):
        valid = (attention_mask[batch_idx] == 1) & (special[batch_idx] == 0)
        residue_positions = torch.nonzero(valid).flatten()
        expected = min(len(sequence), int(max_length - special[batch_idx].sum().item()))
        if len(residue_positions) != expected:
            raise ValueError(
                f"Tokenizer produced {len(residue_positions)} residue tokens for "
                f"sequence {batch_idx}, expected {expected}; one-token-per-residue "
                "alignment is required."
            )
        positions.append(residue_positions)
    return positions


def _validate_sequence_label_lengths(sequences, labels):
    if len(sequences) != len(labels):
        raise ValueError(
            f"Received {len(sequences)} sequences but {len(labels)} labels"
        )
    for index, (sequence, label) in enumerate(zip(sequences, labels)):
        if len(str(sequence)) != len(str(label)):
            raise ValueError(
                f"Sequence/label length mismatch at index {index}: "
                f"{len(str(sequence))} != {len(str(label))}"
            )


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

    _validate_sequence_label_lengths(sequences, labels)
    injection = _get_injection_point(model, layer_idx)
    model.eval()

    ce_orig_sum = ce_sae_sum = ce_zero_sum = 0.0
    total_valid_tokens = 0
    recon_mse_sum = 0.0
    recon_mse_count = 0

    for i in tqdm(range(0, len(sequences), batch_size), desc="Evaluating fidelity",
                  unit="batch", total=(len(sequences) + batch_size - 1) // batch_size):
        batch_seqs = sequences[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]

        enc = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )
        tokens = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        residue_positions = _get_residue_positions(
            enc, attn, tokenizer, batch_seqs, max_length
        )

        # Build per-residue label tensor (same truncation as embeddings):
        # -100 for special/ignored positions.
        label_ids = torch.full_like(tokens, -100)
        for b, (seq, lab) in enumerate(zip(batch_seqs, batch_labels)):
            seq_len = len(residue_positions[b])
            enc_ids = encode_label_string(str(lab)[:seq_len], label_map_spec)
            label_ids[b, residue_positions[b]] = torch.tensor(
                enc_ids[:seq_len], device=label_ids.device
            )

        # 1) original loss AND hidden states in ONE forward pass (saves a pass)
        with torch.no_grad():
            out = model(
                input_ids=tokens, attention_mask=attn, labels=label_ids,
                output_hidden_states=True,
            )
        ce_orig, n_valid = _token_loss_sum(out.logits, label_ids)
        hidden = out.hidden_states[layer_idx]
        if n_valid == 0:
            continue

        # 2) zero-ablation loss (worst case)
        zeros = torch.zeros_like(hidden)
        ce_zero, zero_count = _model_loss_with_override(
            model, tokens, attn, label_ids, injection, zeros, return_stats=True
        )

        # 3) SAE reconstruction loss + reconstruction MSE sanity check
        recon = _extract_sae_reconstructions(sae, hidden)
        ce_sae, sae_count = _model_loss_with_override(
            model, tokens, attn, label_ids, injection, recon, return_stats=True
        )

        # Sanity check: MSE between SAE recon and original hidden states,
        # averaged over non-padding tokens AND over the embedding dimension.
        # (weighted.sum() is over [B, L, D]; mask.sum() counts tokens, so we
        #  must also divide by D to get a per-element MSE.)
        mask = (label_ids != -100).unsqueeze(-1).float()  # residue labels only
        d_model = hidden.shape[-1]

        if n_valid != zero_count or n_valid != sae_count:
            raise RuntimeError("Fidelity loss valid-token counts changed across interventions")
        ce_orig_sum += ce_orig
        ce_sae_sum += ce_sae
        ce_zero_sum += ce_zero
        total_valid_tokens += n_valid
        recon_mse_sum += float((((hidden - recon) ** 2) * mask).sum().item())
        recon_mse_count += n_valid * d_model

    if total_valid_tokens == 0:
        raise ValueError("Fidelity evaluation produced no valid residue labels")
    ce_orig = ce_orig_sum / total_valid_tokens
    ce_sae = ce_sae_sum / total_valid_tokens
    ce_zero = ce_zero_sum / total_valid_tokens
    recon_mse = recon_mse_sum / max(recon_mse_count, 1)

    # Loss recovered, clipped to [0, 100] for display.
    # Also report the unclipped value: when the SAE reconstruction lowers the loss
    # below the original activations (ce_sae < ce_orig) the raw value is >100%.
    # Clipping it to 100 would make "reconstruction better than original" look like
    # a merely-perfect recovery, hiding that difference.
    denom = ce_zero - ce_orig
    invalid_zero_baseline = ce_zero <= ce_orig or np.isclose(denom, 0)
    sae_better_than_orig = bool(ce_sae < ce_orig)
    if invalid_zero_baseline:
        loss_recovered = None
        loss_recovered_raw = None
    else:
        loss_recovered_raw = float((1 - (ce_sae - ce_orig) / denom) * 100)
        loss_recovered = float(np.clip(loss_recovered_raw, 0, 100))

    return {
        "layer_idx": layer_idx,
        "n_sequences": len(sequences),
        "n_valid_tokens": total_valid_tokens,
        "ce_orig": ce_orig,
        "ce_sae": ce_sae,
        "ce_zero": ce_zero,
        "loss_recovered_pct": loss_recovered,
        "loss_recovered_raw": loss_recovered_raw,
        "sae_better_than_original": sae_better_than_orig,
        "zero_ablation_baseline": True,
        "zero_ablation_is_worst_case": False,
        "invalid_zero_baseline": bool(invalid_zero_baseline),
        "reconstruction_mse": recon_mse,  # sanity check: lower = better reconstruction
    }


# ---------------------------------------------------------------------------
# Causal intervention (feature steering)
#
# Fidelity replaces the WHOLE layer with a reconstruction. Intervention instead
# perturbs a SINGLE SAE feature and measures how the model's predictions change,
# establishing a causal (not just correlational) link between that feature and
# the model's decision.
#
# Pipeline per batch:
#   hidden (layer L) --SAE.encode--> f  (feature vector)
#   modify f[:, feat_idx]  (zero / amplify / set)
#   --SAE.decode--> steered hidden
#   inject steered hidden at layer L --> compare predictions vs original
# ---------------------------------------------------------------------------


def _model_logits_with_override(
    model,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    injection_submodule,
    override: torch.Tensor | None,
) -> torch.Tensor:
    """Run the model once and return logits; optionally override the injected layer."""
    if override is None:
        with torch.no_grad():
            out = model(input_ids=tokens, attention_mask=attention_mask)
        return out.logits

    def _hook(module, args, output):
        return override

    handle = injection_submodule.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            out = model(input_ids=tokens, attention_mask=attention_mask)
    finally:
        handle.remove()
    return out.logits


def evaluate_intervention(
    model,
    sae: Dictionary,
    layer_idx: int,
    sequences: list[str],
    labels: list[str],
    tokenizer,
    label_map_spec: dict,
    feature_idx: int,
    mode: str = "zero",
    scale: float = 2.0,
    batch_size: int = 8,
    max_length: int = 512,
    device: str = "cpu",
) -> dict:
    """
    Causal intervention: perturb one SAE feature and measure prediction changes.

    For each batch:
      - baseline predictions from original hidden states
      - steered predictions after zeroing/amplifying feature `feature_idx`
      - report how often predictions flip, and whether flips are 'correct'
        (toward/away from the ground-truth positive class).

    Args:
        model: fine-tuned AutoModelForTokenClassification
        sae: trained SAE
        layer_idx: hidden layer the SAE was trained on
        sequences, labels: data (labels are raw char strings, label_map_spec encodes them)
        feature_idx: the feature to perturb
        mode: 'zero' | 'amplify' | 'set'
        scale: multiplier/value for amplify/set
        device: torch device

    Returns:
        dict with n_tokens, flip_rate, and flip metrics.
    """
    from single.label_maps import encode_label_string

    _validate_sequence_label_lengths(sequences, labels)
    injection = _get_injection_point(model, layer_idx)
    model.eval()

    total_tokens = 0
    active_tokens = 0      # tokens where the target feature actually fires (>0)
    inactive_tokens = 0    # tokens where the target feature does NOT fire (control)
    n_flips = 0            # tokens whose argmax prediction changed
    n_flips_active = 0     # flips among active tokens
    n_flips_inactive = 0   # flips among inactive tokens (noise/control baseline)
    n_to_pos = 0           # flips that became the positive class (active positions)
    n_to_neg = 0           # flips that became a non-positive class (active positions)
    recon_mse_sum = 0.0
    recon_mse_count = 0
    logit_delta_sum = 0.0
    logit_delta_count = 0
    loss_delta_sum = 0.0
    loss_delta_count = 0
    pos_class = label_map_spec["positive_class"]

    for i in tqdm(range(0, len(sequences), batch_size), desc="Intervention",
                  unit="batch", total=(len(sequences) + batch_size - 1) // batch_size):
        batch_seqs = sequences[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]

        enc = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )
        tokens = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        residue_positions = _get_residue_positions(
            enc, attn, tokenizer, batch_seqs, max_length
        )

        # Compute per-residue label IDs (needed for baseline loss and flip stats)
        label_ids = torch.full_like(tokens, -100)
        for b, (seq, lab) in enumerate(zip(batch_seqs, batch_labels)):
            seq_len = len(residue_positions[b])
            enc_ids = encode_label_string(str(lab)[:seq_len], label_map_spec)
            label_ids[b, residue_positions[b]] = torch.tensor(
                enc_ids[:seq_len], device=label_ids.device
            )

        # Baseline: loss + logits + hidden states in ONE forward pass.
        with torch.no_grad():
            out = model(
                input_ids=tokens, attention_mask=attn, labels=label_ids,
                output_hidden_states=True,
            )
        base_loss_sum, n_valid = _token_loss_sum(out.logits, label_ids)
        base_logits = out.logits
        base_preds = base_logits.argmax(dim=-1)
        hidden = out.hidden_states[layer_idx]
        if n_valid == 0:
            continue

        # SAE encode + delta steering (SAE-only, no model forward).
        hidden_n, norms = sae._normalize_input_and_get_norms(hidden)
        f = sae.encode(hidden_n)  # [B, L, dict_size]
        active_mask = f[..., feature_idx] > 0  # tokens where the feature fires
        f_steered = f.clone()
        if mode == "zero":
            f_steered[..., feature_idx] = 0.0
        elif mode == "amplify":
            f_steered[..., feature_idx] *= scale
        elif mode == "set":
            f_steered[..., feature_idx] = scale
        else:
            raise ValueError(f"Unknown mode: {mode} (use zero/amplify/set)")

        # Pure delta: add only the change from the target feature
        # WITHOUT introducing SAE reconstruction error on other features.
        delta = sae.decode(f_steered) - sae.decode(f)
        steered = hidden + sae._unnormalize_output(delta, norms)

        # One steered forward supplies both logits and the token loss.
        steer_logits = _model_logits_with_override(
            model, tokens, attn, injection, steered
        )
        steer_loss_sum, steer_count = _token_loss_sum(steer_logits, label_ids)
        if steer_count != n_valid:
            raise RuntimeError("Intervention loss valid-token count changed after steering")
        steer_preds = steer_logits.argmax(dim=-1)

        # Reconstruction baseline MSE (sanity check).
        with torch.no_grad():
            # decode(f) is in the SAE's normalized space when
            # normalize_to_sqrt_d=True. Convert it back before comparing it to
            # the original hidden states.
            recon = sae._unnormalize_output(sae.decode(f), norms)
        valid = (attn == 1) & (label_ids != -100)
        active_valid = valid & active_mask
        logit_delta_on_active = 0.0
        if active_valid.any():
            logit_delta = (steer_logits - base_logits)[active_valid].abs()
            logit_delta_sum += float(logit_delta.sum().item())
            logit_delta_count += int(logit_delta.numel())
        residue_mask = valid.unsqueeze(-1).float()
        recon_mse_sum += float((((hidden - recon) ** 2) * residue_mask).sum().item())
        recon_mse_count += int(valid.sum().item()) * hidden.shape[-1]
        loss_delta_sum += steer_loss_sum - base_loss_sum
        loss_delta_count += n_valid

        for b in range(tokens.size(0)):
            mask = valid[b]
            n_tok = int(mask.sum().item())
            if n_tok == 0:
                continue
            total_tokens += n_tok
            flips = (steer_preds[b][mask] != base_preds[b][mask])
            n_flips += int(flips.sum().item())

            # Flip statistics on the subset where the feature actually activates.
            # `flips` is already restricted to `mask`, so restrict active_mask too.
            act = active_mask[b][mask]
            n_act = int(act.sum().item())
            active_tokens += n_act
            inactive_tokens += n_tok - n_act
            if n_act > 0:
                flips_act = flips & act
                n_flips_active += int(flips_act.sum().item())
                steer_act = steer_preds[b][mask]
                n_to_pos += int(((steer_act == pos_class) & flips_act).sum().item())
                n_to_neg += int((flips_act & (steer_act != pos_class)).sum().item())
            if n_tok - n_act > 0:
                n_flips_inactive += int((flips & ~act).sum().item())

    if total_tokens == 0:
        return {
            "feature_idx": feature_idx, "mode": mode, "scale": scale,
            "layer_idx": layer_idx,
            "n_tokens": 0, "active_tokens": 0, "inactive_tokens": 0,
            "flip_rate": 0.0, "flip_rate_on_active": 0.0, "flip_rate_on_inactive": 0.0,
            "flip_to_pos": 0.0, "flip_to_neg": 0.0,
            "recon_mse": 0.0, "logit_delta_on_active": 0.0, "loss_delta": 0.0,
            "zero_ablation_baseline": False,
            "zero_ablation_is_worst_case": False,
            "invalid_zero_baseline": True,
        }

    avg_recon_mse = recon_mse_sum / max(recon_mse_count, 1)
    avg_logit_delta = logit_delta_sum / max(logit_delta_count, 1)
    avg_loss_delta = loss_delta_sum / max(loss_delta_count, 1)
    return {
        "feature_idx": feature_idx,
        "mode": mode,
        "scale": scale,
        "layer_idx": layer_idx,
        "n_tokens": total_tokens,
        "active_tokens": active_tokens,
        "inactive_tokens": inactive_tokens,
        "flip_rate": n_flips / total_tokens,
        "flip_rate_on_active": n_flips_active / max(active_tokens, 1),
        "flip_rate_on_inactive": n_flips_inactive / max(inactive_tokens, 1),
        "flip_to_pos": n_to_pos / max(active_tokens, 1),
        "flip_to_neg": n_to_neg / max(active_tokens, 1),
        "recon_mse": round(avg_recon_mse, 6),
        "logit_delta_on_active": round(avg_logit_delta, 6),
        "loss_delta": round(avg_loss_delta, 6),
    }
