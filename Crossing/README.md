# Crossing Module — Cross-Model Interpretability

> Status: 🚧 **Phase 1 (Alignment) + Phase 2 (Information) implemented; Phase 0 (M0) base extraction now via `Single --model_type base` + central `Outputs/_pretrained/` (single copy per backbone), `crossing baseline` script still planned** — see ROADMAP Phase 0–2. Remaining phases (intervention / crosscoder / causal / biology) still planned.
>
> Back to [project README](../README.md).

The Crossing module extends CrossPLM from single-model SAE interpretability to **comparisons across PLMs** (e.g. task-A vs task-B fine-tuned ESM-2, both native `facebook/esm2_*` and `Synthyra/ESM2-8M` via `trust_remote_code`).  It answers:

- Do independently fine-tuned models develop **shared** internal features?
- Does one model's representation contain **transferable information** about the other task?
- *(Phase 0 baseline — `ROADMAP.md 0.1/0.2`)* Is a feature `pre-existing` in `M0`, `shared emergent`, or `task-specific`?

## What is implemented

| Phase | CLI | Library | Notes |
|---|---|---|---|
| **0 – Baseline (M0)** | `Single` `--model_type base` (no `crossing` CLI yet) | `Single/single/embedders/ft_esm.py` (`model_type base` → `AutoModel`, `Outputs/_pretrained/<slug>` single copy) + `Training` `provenance.json` / `freeze_backbone` | `M0` embeddings are encoder-only (no task head); SAE is trained as for `MA/MB` and evaluated by `L0` (`recon`/`L0`/`dead%`) + `L1` (concept F1), not by task-label fidelity. `crossing baseline` (`base_sae.py` / `feature_origin.py`) still planned. |
| **1 – Feature Alignment** | `crossplm crossing compute_feature_similarity` | `crossing/similarity.py` `crossing/semantic.py` `crossing/heatmap.py` `crossing/controls.py` `crossing/io.py` | Activation similarity (vectorized Pearson/cosine), linear CKA (correct HSIC), MI, semantic similarity (concept-F1 cosine/Jaccard), `S_cross = α S_act + β S_sem`, controls (permutation null, random-pair, residualization), cluster-reordered correspondence heatmap. Residue-aligned via `residues.csv` (same as Single, padding-aware `_residue_positions` fix). Batched SAE encode (no OOM). |
| **2 – Cross-task Information** | `crossplm crossing cross_task_probe` <br> `crossplm crossing classify_features` | `crossing/probe.py` `crossing/classification.py` | Aligned information-transfer probes (same proteins, shared train/val split) with full `2×2` transfer matrix; disjoint fallback flagged. Per-feature `Shared / A-specific / B-specific` via single-feature probes (aligned) or similarity+F1 (disjoint). |

## Quick start

```bash
# Phase 0 – M0 baseline (encoder-only; central cache avoids duplicate M0 for MA/MB sharing the same backbone)
python crossplm.py single extract_embeddings \
  --ckpt_path facebook/esm2_t6_8M_UR50D --model_type base \
  --sequences_csv Training/examples/sample.csv --experiment m0_demo
# → SAE0: python crossplm.py single train_sae --experiment m0_demo
# → L0: check recon/L0/dead% in train_sae log; L1: concepts as for MA/MB

# Phase 1 – activation + optional semantic/controls/heatmap
python crossplm.py crossing compute_feature_similarity \
  --sae_a Outputs/exp_a/sae --sae_b Outputs/exp_b/sae \
  --embeddings_a Outputs/shared/eval/layer_6 --embeddings_b Outputs/shared/eval/layer_6 \
  --experiment my_crossing \
  --method correlation --use_cka \
  --concepts_a Outputs/exp_a/concepts --concepts_b Outputs/exp_b/concepts --semantic_mode cosine --combined \
  --with_controls --with_heatmap

# Phase 2 – information transfer (aligned, recommended: both models encoded the SAME proteins)
python crossplm.py crossing cross_task_probe \
  --sae_a Outputs/exp_a/sae --sae_b Outputs/exp_b/sae \
  --embeddings_a Outputs/shared/eval/layer_6 --embeddings_b Outputs/shared/eval/layer_6 \
  --labels_a Dataset/mBMRB.csv --labels_b Dataset/relaxdb.csv \
  --label_map_a mBMRB --label_map_b relaxdb \
  --mode aligned --experiment my_crossing

python crossplm.py crossing classify_features \
  --sae_a Outputs/exp_a/sae --sae_b Outputs/exp_b/sae \
  --embeddings_a Outputs/shared/eval/layer_6 --embeddings_b Outputs/shared/eval/layer_6 \
  --labels_a Dataset/mBMRB.csv --labels_b Dataset/relaxdb.csv \
  --label_map_a mBMRB --label_map_b relaxdb \
  --mode aligned --experiment my_crossing
# For disjoint protein sets use --mode cross_sim (similarity+F1 classification)
```

Outputs land in `Outputs/<experiment>/crossing/` : `feature_similarity_matrix.npy` (+ `_semantic.npy` / `_combined.npy`), `heatmap.png`, `controls.json`, `cross_task_probe.json`, `feature_classification.json`.

For the full research ladder and remaining phases (intervention / crosscoder / causal circuit / biology+MD), see:

- [ROADMAP.en.md](ROADMAP.en.md) (English)
- [ROADMAP.md](ROADMAP.md) (中文)
