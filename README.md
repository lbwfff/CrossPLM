# CrossPLM

**Mechanistic Interpretability for Cross-Task Protein Language Models**

Protein language models (PLMs) perform well on diverse biological tasks, but their internal mechanisms are not yet fully understood. This project builds an interpretability framework for cross-task PLMs, consisting of a **fine-tuning toolkit** (Training) and a **SAE-based interpretability module** (Single), with a cross-model module (Crossing) planned.

---

## Overview

```
Dataset/  Preprocessing/   Training/        Single/        Outputs/
 (raw)        →         fine-tune a PLM → SAE analysis → per-experiment results
```

| Directory | Role |
|-----------|------|
| `Dataset/` | Raw datasets (`mBMRB.csv`, `relaxdb_data.csv`) |
| `Preprocessing/` | Dataset-specific label preprocessing scripts |
| `Training/` | PLM fine-tuning framework (init / train / eval CLI) |
| `Single/` | SAE-based interpretability: extract → train SAE → analyze |
| `Outputs/` | All per-experiment outputs (embeddings, SAE, concepts, analysis) |
| `Crossing/` | Planned: cross-model interpretability (not yet implemented) |

```
CrossPLM/
├── Dataset/                     # Raw datasets
│   ├── relaxdb_data.csv
│   └── mBMRB.csv
├── Preprocessing/               # Dataset preprocessing scripts
│   ├── preprocess_relaxdb.py
│   └── preprocess_mbmrb.py
├── Training/                    # PLM training framework
│   ├── crossplm.py              # Unified CLI (init / train / eval)
│   ├── crossplm/                # Python package
│   ├── outputs/                 # Training outputs (checkpoints, logs)
│   └── examples/                # Sample data and configs
├── Single/                      # SAE-based interpretability
│   ├── setup.py
│   └── single/
│       ├── configs.py           # Configuration dataclasses
│       ├── label_maps.py        # Configurable label encoding for datasets
│       ├── paths.py             # Centralized experiment output paths
│       ├── embedders/           # Hidden state extraction from fine-tuned PLMs
│       ├── sae/                 # SAE architectures (ReLUSAE, TopKSAE)
│       ├── train/               # SAE training loop
│       ├── analysis/            # Feature-to-label & feature-to-concept alignment
│       └── scripts/             # CLI scripts
├── Outputs/                     # Per-experiment output trees
│   └── <experiment>/            # embeddings / sae / concepts / analysis
└── README.md
```

---

## Module 1: Training — Fine-tune a PLM

Fine-tunes a HuggingFace protein language model (e.g. ESM-2) on a per-residue
token-classification task (e.g. backbone dynamics: rigid vs flexible).

### 0. Preprocess Data
```bash
cd Preprocessing
python preprocess_mbmrb.py          # → ../Training/examples/mbmrb_processed.csv
python preprocess_relaxdb.py        # → ../Training/examples/relaxdb_processed.csv
```

### 1. Initialize a Task
```bash
cd Training
python crossplm.py init --task_name my_experiment
```
→ Creates `outputs/tasks/my_experiment_<ts>/config.yaml` template.

### 2. Edit Config → Train
Edit the `config.yaml`, then:
```bash
python crossplm.py train --config outputs/tasks/my_experiment_<ts>/config.yaml
```
→ Checkpoints, training curve, and logs are saved inside the task folder.

### 3. Evaluate a Checkpoint
```bash
python crossplm.py eval \
  --checkpoint outputs/tasks/my_experiment_<ts>/checkpoints/epoch_10_f1_7952 \
  --csv ./examples/relaxdb_processed.csv
```
→ Results (metrics.json, confusion matrix, AUPRC curve) are saved inside the task folder.

### Features

| Feature | Description |
|---------|-------------|
| **Two-phase workflow** | init (template) → train, keeping config separate from code |
| **CSV input** | `sequence` + `label` columns, automatic train/eval split |
| **Ignore positions** | `_` labels are excluded from loss |
| **Auto class weights** | `inverse` / `log` / `none` strategies |
| **Multi-class support** | Correct class count for many-to-one label maps (mBMRB, relaxdb, ss3) |
| **Metrics** | Loss + Accuracy + Macro F1 + AUPRC (macro over all classes) |
| **Top-3 checkpoints** | Keeps best 3 by F1, cleans old ones automatically |
| **Training curve** | Auto-generated epoch–F1 plot after training |
| **Eval plots** | Confusion matrix + Precision-Recall curve |

---

## Module 2: Interpretability — SAE Analysis (Single/)

After fine-tuning a PLM, use **Sparse Autoencoders** to discover which hidden-state
features drive the model's predictions, and interpret them against either the task
labels or Swiss-Prot biological concepts.

### Pipeline

```
Fine-tuned PLM checkpoint
         ↓
[1] extract_embeddings.py    → Extract per-residue hidden states from a target layer
         ↓
[2] train_sae.py             → Train a Sparse Autoencoder to learn interpretable features
         ↓
[3] analyze_features.py     → (A) Align features with task labels   (rigid/flexible)
[4] analyze_concepts.py     → (B) Align features with biological concepts (Swiss-Prot)
         ↓
[5] heldout / fidelity /    → (C) Validate the findings (unbiased, faithful,
    evaluate_intervention        and causal)
         ↓
[6] analyze_sequence.py     → (D) Characterize features along the sequence
                               (Cohen's d + motif enrichment)
         ↓
[7] analyze_coactivation.py → (E) Compare pairs of features (co-localized vs disjoint)
         ↓
[8] visualize_features.py   → Plot feature activation patterns on protein sequences
```

### How SAEs Work

SAEs learn a sparse, overcomplete decomposition of PLM hidden states:

```
ESM hidden state (320-dim)  →  SAE encoder  →  sparse feature vector (640-dim)
                                                   ↑ only ~60 features active per token
```

Each feature can then be interpreted by checking **when** it activates:
- Does it activate preferentially on **flexible** residues? → "flexibility detector"
- Does it activate on **rigid** residues? → "rigidity detector"
- Is it unrelated to the task? → noise feature

### Shared Conventions

**Experiment directory.** Every step routes outputs into one directory,
`Outputs/<experiment>/` (the name is used verbatim, no timestamp). Re-running a step
with the same name reuses the directory (e.g. overwrites `ae.pt`). Use distinct
names for distinct runs, or `--exp_dir <existing_dir>` to point at one directly.

```
Outputs/<experiment>/
    embeddings/layer_<N>/shard_<i>/activations.pt
    sae/ae.pt
    concepts/shard_<i>/aa_concepts.npz
    analysis/*.csv|*.json|visualizations/
```

**Configurable label maps.** Label encoding is not hardcoded to mBMRB. Every script
accepts `--label_map <preset>` (from `single/label_maps.py`) or a YAML file:

| Preset | Positive class | Class names | Character mapping |
|--------|---------------|-------------|-------------------|
| `mBMRB` | 1 | rigid / flexible | `A→0`, `.→1`, `0→0`, `1→1` |
| `relaxdb` | 1 | static / mobile | `p/A/v→0`, `./b/^→1` |
| `ss3` | 1 | coil / strand / helix | `C→0`, `E→1`, `H→2` (3-class) |

> `ss3` ids match the training module's `build_label_map` (`sorted(unique)`, i.e.
> `C < E < H`), so a model trained with an inferred map evaluates consistently.

Custom YAML:
```yaml
# my_dataset.yaml
positive_class: 1
class_names: {0: rigid, 1: flexible}
mapping: {A: 0, ".": 1, "0": 0, "1": 1}
ignore: "_"
```
Characters not in `mapping` become `-100` (ignored), following the HuggingFace
ignore-index convention.

### 2.1 Task-Label Alignment

Align each SAE feature with the task labels (e.g. rigid/flexible) to find
"flexibility detectors".

```bash
cd Single

# Extract hidden states from the fine-tuned model (creates Outputs/mb/)
python -m single.scripts.extract_embeddings \
    --ckpt_path ../Training/outputs/tasks/my_task/checkpoints/best \
    --sequences_csv ../Dataset/mBMRB.csv \
    --experiment mb \
    --label_column label --label_map mBMRB

# Train SAE (320-dim → 640 features; embeddings inferred, all shards by default)
python -m single.scripts.train_sae \
    --experiment mb \
    --batch_size 64 --dict_size 640 --steps 20000 --l1_penalty 0.08 \
    --resample_steps 2000

# Align features with task labels
python -m single.scripts.analyze_features \
    --sae_dir ../Outputs/mb/sae \
    --embeddings_dir ../Outputs/mb/embeddings/layer_6 \
    --experiment mb \
    --sequences_csv ../Dataset/mBMRB.csv --label_column label --label_map mBMRB

# Visualize top features on sequences (embeddings_path inferred from experiment)
python -m single.scripts.visualize_features \
    --sae_dir ../Outputs/mb/sae \
    --experiment mb \
    --sequences_csv ../Dataset/mBMRB.csv \
    --feature_indices 234 426 --label_map mBMRB
```

**Tuning tips for `train_sae`:**
- `--l1_penalty` controls sparsity (~0.06–0.1); higher → sparser features (lower
  `l0`) but higher reconstruction loss.
- `--resample_steps N` periodically revives "dead" features, lowering `dead_pct`.
- Target: **`l0` 20–80, `dead_pct` < 30%, `recon_loss` as low as possible**.

**Metrics per feature:** Precision (when it activates, how often is the label
correct?), Recall (fraction of positives caught), F1, AUROC (0.5=random,
1.0=perfect), Activation Gap.

**Example output:**
```
Top features for 'flexible' (label=1):
  Feature #234: F1=0.514, AUROC=0.713, Prec=0.923, Rec=0.356
  Feature #426: F1=0.517, AUROC=0.693, Prec=0.729, Rec=0.401
```

### 2.2 Biological Concept Analysis (Swiss-Prot / UniProtKB)

Align SAE features against **biological concepts** — e.g. "Helix",
"Domain_kinase", "Binding_site_ATP" — to discover what real biology each feature
encodes. Two-step pipeline (`single/scripts/analyze_concepts.py`).

**Step 1: Build per-residue concept matrices from a UniProtKB TSV export.**

The TSV must contain `Entry`, `Sequence`, and Feature-table columns (`Helix`,
`Beta strand`, `Turn`, `Domain [FT]`, `Active site`, `Binding site`, etc.).
Download from UniProt with `reviewed:true` → Export → TSV.

```bash
python -m single.scripts.analyze_concepts build \
    --annotations_tsv ../Dataset/uniprotkb_swissprot.tsv \
    --experiment sp \
    --n_shards 5 \
    --max_residues 510   # must equal embedder max_length - 2 (512 -> 510)
```

> **Alignment requirement:** `--max_residues` (embedder `max_length` − 2) makes
> concept rows cover exactly the residues the embedder keeps, avoiding misalignment
> on long sequences.

**Step 2: Align SAE features to concepts.**

```bash
python -m single.scripts.analyze_concepts align \
    --sae_dir ../Outputs/sp/sae \
    --embeddings_dir ../Outputs/sp/embeddings/layer_6 \
    --experiment sp \
    --threshold_percents 0 0.15 0.5 0.6 0.8
```

For every feature × concept pair, computes F1 / precision / recall / AUROC /
domain-F1 across thresholds, saving `feature_concept_pairs.csv` (all pairs) and
`feature_concept_metrics.json`.

**Example output:**
```
Top feature-concept associations (by F1):
  Feature #42  → Domain_kinase                  F1=0.623 AUROC=0.781 P=0.710 R=0.551
  Feature #107 → Helix                          F1=0.588 AUROC=0.742 P=0.650 R=0.537
```

**`f1` vs `f1_per_domain`:** `f1` counts residues (strict, skewed by domain length);
`f1_per_domain` counts domain instances (a domain is "hit" if the feature activates
anywhere within it). Prefer high `f1_per_domain` for robust concept recognition.

### 2.3 Validation & Fidelity

Two checks that make the feature findings trustworthy.

**Held-out validation** removes selection bias. Reporting the "best feature per
concept" on the *same* data used for selection is optimistically biased — among 640
features some score high purely by chance.

1. Split concept shards into a **valid** and a **test** split.
2. On **valid**, pick the top feature per concept (selection).
3. On **test**, evaluate only the selected pairs (unbiased).

```bash
python -m single.scripts.analyze_concepts heldout \
    --sae_dir ../Outputs/sp/sae \
    --embeddings_dir ../Outputs/sp/embeddings/layer_6 \
    --experiment sp \
    --split_mode half \
    --threshold_percents 0 0.15 0.5 0.6 0.8
```
Outputs `heldout_top_pairings.csv` (selected pairs, test metrics) and
`heldout_all_top_pairings.csv` (those above `--heldout_f1_threshold`).

**Fidelity (loss recovered)** validates that the SAE faithfully represents the
model's task-relevant activations. The target layer's hidden states are replaced
three ways and the model's **task loss** is compared:

```
ce_orig : original activations
ce_sae  : SAE reconstructions injected
ce_zero : layer zeroed (worst case)

Loss_Recovered = 1 - (ce_sae - ce_orig) / (ce_zero - ce_orig)
```
100% = perfectly preserves task info; 0% = as harmful as zeroing the layer.

```bash
python -m single.scripts.evaluate_fidelity \
    --ckpt_path ../Training/outputs/tasks/my_task/checkpoints/best \
    --sequences_csv ../Dataset/mBMRB.csv \
    --sae_dir ../Outputs/sp/sae \
    --experiment sp \
    --layer 6 --label_column label --label_map mBMRB
```
Saves `fidelity_results.json` incl. a `reconstruction_mse` sanity check.
**Note:** injection currently supports the final layer only
(`hidden_states[6]` = `emb_layer_norm_after` output).

**Causal intervention (feature steering)** moves from *correlation* to
*causation*. Whereas Fidelity replaces the whole layer, intervention perturbs a
**single SAE feature** and measures whether the model's per-residue predictions
change — establishing that the feature *causally drives* (not just co-occurs
with) the model's decision.

```bash
python -m single.scripts.evaluate_intervention \
    --ckpt_path ../Training/outputs/tasks/my_task/checkpoints/best \
    --sequences_csv ../Dataset/mBMRB.csv \
    --sae_dir ../Outputs/mb/sae \
    --experiment mb \
    --feature_idx 375 --mode zero \
    --label_column label --label_map mBMRB \
    --layer 6 --max_sequences 200
```

- `--feature_idx N` — the feature to perturb.
- `--mode zero|amplify|set` — set to 0, scale up (`--scale`), or force a value.
- Outputs `intervention_feat<N>_<mode>.json` with **flip-rate** metrics:
  - `flip_rate_on_active` — how often predictions change **on tokens where the
    feature actually fires** (the causal effect).
  - `flip_rate_on_inactive` — the **control baseline** on tokens where the feature
    does NOT fire. If `active >> inactive`, the effect is real; if similar, it's
    noise.

*Note: single-feature effects are typically a few % (each token uses ~60
features, so one feature tips only borderline samples). Compare against the
inactive control rather than reading the absolute value.*

### 2.4 Sequence Analysis (Cohen's d + Motif Enrichment)

Characterizes *what along the sequence* a feature responds to, using only
sequence data (no 3D structures required):

- **Sequential Cohen's d** — are the feature's activated residues **clustered**
  along the sequence (local/motif-like) or **dispersed** (global/periodic)?
  Negative d = clustered, ~0 = random, positive = dispersed.
- **Motif enrichment** — which amino acids are over-represented in a window
  (`--flank`, default 3) around the activated residues — the amino-acid
  "signature" of the feature.

```bash
python -m single.scripts.analyze_sequence \
    --sae_dir ../Outputs/mb/sae \
    --embeddings_dir ../Outputs/mb/embeddings/layer_6 \
    --sequences_csv ../Dataset/mBMRB.csv \
    --experiment mb \
    --feature_indices 375 42 \
    --shard 0
```

Example output (Feature #42, the "flexibility detector"):
```
Sequential Cohen's d: -0.074  → ~random
Top enriched amino acids (log2 fold):
  P: +0.40   S: +0.37   G: +0.32   D: +0.26   C: +0.12
```
P/S/G/D are classic flexible/loop residues — consistent with a flexibility
detector. Saves `sequence_analysis_shard<N>.json` in the experiment's `analysis/`.

### 2.5 Pairwise Feature Co-Activation

Answers whether two features activate on the **same residues**, on residues
**near each other** along the sequence, or on **disjoint** sets — revealing
redundant vs complementary (co-regulatory) features.

```bash
python -m single.scripts.analyze_coactivation \
    --sae_dir ../Outputs/mb/sae \
    --embeddings_dir ../Outputs/mb/embeddings/layer_6 \
    --sequences_csv ../Dataset/mBMRB.csv \
    --experiment mb \
    --feature_a 375 --feature_b 42 \
    --shard 0
```

Key metrics (each compared to the unconditional activation-rate baseline):
- `overlap_ab` / `enrich_ab` — same-residue co-activation; **>1** → B enriched on
  A's residues, **<1** → mutually exclusive at the same position.
- `neighbor_ab` / `neighbor_enrich_ab` — B active within ±`--neighborhood`
  residues of an A-active residue; **>>1** → features co-localize nearby.
- Both directions (A→B and B→A) are reported to detect asymmetry.

Example (Features #375 and #42): same-residue enrichment **0.69×** (mutually
exclusive at identical positions) but neighborhood enrichment **3.7–6.4×**
(strongly co-localized within ±5 residues) → they are **complementary detectors
that alternate along the sequence**, not redundant. Saves
`coactivation_<a>_<b>_shard<N>.json`.

---

## Module 3: Crossing — Planned

Not yet implemented. Planned directions:

- **Single-model**: neuron / attention-head importance, representation probing.
- **Cross-model**: comparative analysis across PLMs (ESM-2, ProtBERT, Ankh),
  task-specific vs task-common representation separation, feature transferability.

**Structural analysis (planned):** the *structure* side of the structure-vs-
sequence scatter — plot `x = sequential Cohen's d` (implemented above) against
`y = structural Cohen's d`. Features far above the diagonal encode spatially
clustered biology (e.g. a catalytic pocket). *Requires PDB/AlphaFold structures.*

---

## Dependencies

```bash
pip install -r requirements.txt
```
Or install the SAE package in development mode (pulls deps from `setup.py`):
```bash
cd Single && pip install -e .
```

---

## Output Structure

Every step writes into one **experiment directory** (named by `--experiment`,
verbatim, no timestamp; reuse with `--exp_dir`):

```
Outputs/
└── <experiment>/
    ├── embeddings/layer_<N>/shard_<i>/activations.pt   # Hidden states
    ├── sae/
    │   ├── ae.pt                                       # SAE weights
    │   └── checkpoint_<step>/
    ├── concepts/
    │   ├── shard_<i>/aa_concepts.npz                   # Per-residue concepts
    │   └── aa_concepts_columns.txt
    └── analysis/
        ├── feature_concept_pairs.csv                   # Concept alignment
        ├── feature_label_metrics.json                  # Task-label alignment
        ├── heldout_*.csv                               # Held-out validation
        ├── fidelity_results.json                       # Fidelity
        ├── intervention_feat<N>_<mode>.json            # Causal intervention
        ├── sequence_analysis_shard<N>.json             # Cohen's d + motif
        ├── coactivation_<a>_<b>_shard<N>.json          # Pairwise co-activation
        └── visualizations/                             # PNG plots
```

`single/paths.py` centralizes these paths. Explicit `--output_dir` / `--save_dir`
/ `--concepts_dir` flags override the routing.

---

## Citation

A manuscript is in preparation. In the meantime, if you use this code, please cite
the repository:

```bibtex
@misc{crossplm,
  author       = {Bingwu, Li and collaborators},
  title        = {CrossPLM: Cross-Task Mechanistic Interpretability for Protein Language Models},
  year         = {2026},
  howpublished = {\url{https://github.com/lbwfff/CrossPLM}},
  note         = {SAE-based interpretability module: {S}parse {A}utoencoder feature extraction, concept alignment, held-out validation, and fidelity evaluation}
}
```

Please also consider citing the method this project builds upon:

```bibtex
@article{simon2025interplm,
  title={InterPLM: discovering interpretable features in protein language models via sparse autoencoders},
  author={Simon, Elana and Zou, James},
  journal={Nature Methods},
  year={2025},
  doi={10.1038/s41592-025-02836-7},
  url={https://www.nature.com/articles/s41592-025-02836-7}
}
```

> **Note:** The `@misc` entry above uses a placeholder author/description. Update
> `author` and `note` once the repository metadata is finalized.
