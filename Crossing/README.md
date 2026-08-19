# Crossing Module — Cross-Model Interpretability

> Status: 🚧 **Phase 1 (Alignment) + Phase 2 (Information) implemented** — see ROADMAP Phase 0–2. Remaining phases (intervention / crosscoder / causal / biology) still planned.
>
> Back to [project README](../README.md).

The Crossing module extends CrossPLM from single-model SAE interpretability to **comparisons across PLMs** (e.g. task-A vs task-B fine-tuned ESM-2).  It answers:

- Do independently fine-tuned models develop **shared** internal features?
- Does one model's representation contain **transferable information** about the other task?

## What is implemented

| Phase | CLI | Library | Notes |
|---|---|---|---|
| **1 – Feature Alignment** | `crossplm crossing compute_feature_similarity` | `crossing/similarity.py` `crossing/semantic.py` `crossing/heatmap.py` `crossing/controls.py` `crossing/io.py` | Activation similarity (vectorized Pearson/cosine), linear CKA (correct HSIC), MI, semantic similarity (concept-F1 cosine/Jaccard), `S_cross = α S_act + β S_sem`, controls (permutation null, random-pair, residualization), cluster-reordered correspondence heatmap. Residue-aligned via `residues.csv` (same as Single). Batched SAE encode (no OOM). |
| **2 – Cross-task Information** | `crossplm crossing cross_task_probe` <br> `crossplm crossing classify_features` | `crossing/probe.py` `crossing/classification.py` | Aligned information-transfer probes (same proteins, shared train/val split) with full `2×2` transfer matrix; disjoint fallback flagged. Per-feature `Shared / A-specific / B-specific` via single-feature probes (aligned) or similarity+F1 (disjoint). |

## Quick start

```bash
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
