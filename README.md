# CrossPLM

**Mechanistic Interpretability for Cross-Task Protein Language Models**

Protein language models (PLMs) perform well on diverse biological tasks, but their internal mechanisms are not yet fully understood. This project builds an interpretability framework for cross-task PLMs, consisting of a **fine-tuning toolkit** (Training) and a **SAE-based interpretability module** (Single), with a cross-model module (Crossing) planned.

---

## Overview

```
Dataset/       Training/        Single/        Outputs/
 (raw)  →  fine-tune a PLM → SAE analysis → per-experiment results
```

| Directory | Role |
|-----------|------|
| `Dataset/` | Raw datasets (`mBMRB.csv`, `relaxdb_data.csv`), downloaded annotations (`uniprotkb_swissprot.tsv`), and label-map YAMLs (from `crossplm.py training labelmap`) |
| `Training/` | PLM fine-tuning framework (init / train / eval CLI) |
| `Single/` | SAE-based interpretability: extract → train SAE → analyze |
| `Outputs/` | All per-experiment outputs (embeddings, SAE, concepts, analysis) |
| `Crossing/` | Planned: cross-model interpretability (not yet implemented) |

---

## Unified CLI

All commands go through one entry point, `crossplm.py` at the repository root,
grouped by module. Run from the repository root (no `cd`, no install needed):

```bash
# Training module
python crossplm.py training init --task_name my_experiment
python crossplm.py training train --config Outputs/my_experiment/config.yaml
python crossplm.py training eval --checkpoint ... --csv Dataset/mBMRB.csv
python crossplm.py training labelmap --name my_dataset

# Single module
python crossplm.py single extract_embeddings ...
python crossplm.py single train_sae ...
python crossplm.py single analyze_features ...
python crossplm.py single analyze_concepts build|align|heldout ...
python crossplm.py single analyze_sequence ...
python crossplm.py single analyze_coactivation ...
python crossplm.py single evaluate_fidelity ...
python crossplm.py single evaluate_intervention ...
python crossplm.py single visualize_features ...
```

With `pip install -e .` the same commands become the bare `crossplm` command
(`crossplm training eval ...`, `crossplm single extract_embeddings ...`). The
module-internal CLIs (`Training/training_cli.py`, `Single/single/scripts/*.py`)
still work standalone.

```
CrossPLM/
├── crossplm.py                  # Unified CLI (training | single)
├── setup.py                     # Optional: pip install -e . -> `crossplm` command
├── Dataset/                     # Raw datasets
│   ├── relaxdb_data.csv
│   └── mBMRB.csv
├── Training/                    # PLM training framework
│   ├── training_cli.py          # `crossplm training` implementation
│   ├── training/                # Python package
│   └── examples/                # Sample data and configs
├── Single/                      # SAE-based interpretability
│   └── single/
│       ├── configs.py           # Configuration dataclasses
│       ├── label_maps.py        # Configurable label encoding for datasets
│       ├── paths.py             # Centralized experiment output paths
│       ├── embedders/           # Hidden state extraction from fine-tuned PLMs
│       ├── sae/                 # SAE architectures (ReLUSAE, TopKSAE)
│       ├── train/               # SAE training loop
│       ├── analysis/            # Feature-to-label & feature-to-concept alignment
│       └── scripts/             # CLI scripts (`crossplm single` delegation)
├── Outputs/                     # SHARED experiment root (Training + Single)
│   └── <experiment>/            # config, checkpoints, embeddings, sae, concepts, analysis
└── README.md
```

---

## Module 1: Training — Fine-tune a PLM

Fine-tunes a HuggingFace protein language model (e.g. ESM-2) on a per-residue
token-classification task (e.g. backbone dynamics: rigid vs flexible).

### 0. Data & Label Map

> **Dataset/ is user-supplied and gitignored** — it is not synced to the repo.
> Place your raw data (`mBMRB.csv`, `relaxdb_data.csv`,
> `uniprotkb_swissprot.tsv`, ...) there yourself, and generate label-map
> templates with `python crossplm.py training labelmap --name my_dataset`.

The Training and Single modules share the **same label-map presets**
(`mBMRB`, `relaxdb`, `ss3`) and YAML label-map files. A label map defines the
CSV columns, the character → class mapping, and which characters are ignored —
so you can usually point training directly at a **raw** dataset CSV:

```yaml
# my_dataset.yaml
sequence_column: sequence   # column holding the protein sequence
label_column: label         # column holding the per-residue label string
positive_class: 1
class_names: {0: rigid, 1: flexible}
mapping: {A: 0, ".": 1, "0": 0, "1": 1}   # char -> class id (binary or multi-class)
ignore: "_"                 # chars excluded from training/eval (-100)
```

Characters not in `mapping` are ignored too (encoded as `-100`, excluded from
train/eval). No separate preprocessing is needed — both modules read the raw
dataset CSV and apply the label map on the fly.

**Generate an empty template** into `Dataset/` (shared by both modules):
```bash
python crossplm.py training labelmap --name my_dataset
# → Dataset/my_dataset.yaml
```
Then reference it in a config as `label_map: ../Dataset/my_dataset.yaml` or via
`--label_map ../Dataset/my_dataset.yaml` (relative paths resolve against
`Training/`).

### 1. Initialize a Task
```bash
# run from the repository root (no cd needed)
python crossplm.py training init --task_name my_experiment
```
→ Creates `Outputs/my_experiment/config.yaml` template (verbatim name, no timestamp,
shared with the Single module's `Outputs/<experiment>` root).

### 2. Edit Config → Train
Edit `Outputs/my_experiment/config.yaml`, then:
```bash
python crossplm.py training train --config Outputs/my_experiment/config.yaml
```
→ Checkpoints, training curve, and logs are saved inside the experiment folder.

The config's `label_map:` field takes a preset name or YAML path (default
template uses `mBMRB` and points `csv_data_path` at the raw
`../Dataset/mBMRB.csv`). Relative `csv_data_path` / `label_map` YAML paths in
the **config** resolve against the `Training/` module directory; on the eval
**CLI**, `--label_map` (like `--csv`) resolves against your current directory.
Leave `label_map` empty to infer the mapping from the CSV (legacy).

### 3. Evaluate a Checkpoint
```bash
python crossplm.py training eval \
  --checkpoint Outputs/my_experiment/checkpoints/best \
  --csv Dataset/mBMRB.csv \
  --label_map mBMRB
```
→ Results (metrics.json, confusion matrix, AUPRC curve) are saved to
`Outputs/my_experiment/evaluations/<csv_name>/`.

Training saves three kinds of checkpoints under `Outputs/<name>/checkpoints/`:
`epoch_<N>_f1_<F>` (best-3 by F1, auto-pruned), `best` (stable alias for the
highest-F1 one — use this for evaluation/interpretability), and `final`.

Evaluation reuses the label map persisted with the checkpoint (a `label_map.json`
sidecar written by training, or `config.label2id`). Passing `--label_map <preset|yaml>`
is recommended — it makes the label semantics explicit and also works on checkpoints
trained before the sidecar existed. Any label character not in the mapping is
ignored (`-100`), consistent with training.

### Features

| Feature | Description |
|---------|-------------|
| **Two-phase workflow** | init (template) → train, keeping config separate from code |
| **CSV input** | `sequence` + `label` columns, automatic train/eval split |
| **Ignore positions** | `_` labels are excluded from loss |
| **Auto class weights** | `inverse` / `sqrt` / `log` / `none` strategies |
| **Configurable label maps** | `label_map:` in the config uses Single's presets (`mBMRB`, `relaxdb`, `ss3`) or a YAML file, instead of inferring from the CSV |
| **Multi-class support** | Correct class count for many-to-one label maps (mBMRB, relaxdb, ss3) |
| **Metrics** | Loss + Accuracy + Macro F1 + AUPRC (macro over all classes) |
| **Mixed precision** | `fp16` / `bf16` AMP on CUDA GPUs (auto-disabled on CPU) |
| **Top-3 checkpoints** | Keeps best 3 by F1, cleans old ones automatically; `checkpoints/best` is a stable alias for the highest-F1 checkpoint |
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

**Experiment directory & data source.** Every step routes outputs into one
experiment directory `Outputs/<experiment>/` (the name is used verbatim, no
timestamp), the **same root the Training module uses**. The optional
`--source <id>` flag (the input dataset, e.g. `mbmrb` / `swissprot`) nests
data-specific dirs under `Outputs/<experiment>/<source>/`, so different datasets
can share one experiment without overwriting each other. Without `--source`,
everything lives flat under `Outputs/<experiment>/` (legacy). The **SAE is
shared** at the experiment root and reused across sources. Re-running a step
reuses the directory (e.g. overwrites `model.pt`); `--exp_dir <existing_dir>`
points at a directory verbatim.

```
Outputs/<experiment>/
    sae/model.pt            # ONE shared SAE (reused across sources)
    sae/model_normalized.pt # per-feature max-activation rescale (see below)
    <source>/               # e.g. mbmrb | swissprot  (--source <id>)
        embeddings/layer_<N>/shard_<i>/embeddings.pt
        concepts/shard_<i>/concept_matrix.npz
        analysis/*.csv|*.json|visualizations/
```

**Feature normalization.** `analyze_features` computes each feature's max
activation and saves a normalized copy `sae/model_normalized.pt`. All analysis
scripts load the SAE via `load_sae`, which **auto-prefers `model_normalized.pt`
when present**, putting every feature on a comparable 0–1 scale (so a
`--threshold_percents 0.15` means "activation > 15% of that feature's max").
On the **first** `analyze_features` run — before `model_normalized.pt` exists —
`load_sae` falls back to the raw `model.pt`, so the metrics that run reports are
computed at the **raw** activation scale; the normalized copy is written at the
end of that run, and rerunning then produces the 0–1 normalized values.
Re-running `train_sae` removes any stale `model_normalized.pt` so it is
regenerated for the new model.

**Filter/subset consistency.** Scripts that read the sequences CSV
(`extract_embeddings`, `analyze_features`, `analyze_sequence`,
`analyze_coactivation`, `visualize_features`) accept `--min_seq_len` /
`--max_seq_len` / `--max_sequences`. If you used any of these flags during
`extract_embeddings`, pass the **same values** here too. `extract_embeddings`
writes per-shard residue metadata (`residues.csv`); `analyze_sequence` and
`analyze_coactivation` derive the protein/residue mapping from that metadata
when it is present, and a filter mismatch then **fails loudly** (token-count or
unknown-protein errors) instead of misaligning silently. The other scripts
re-derive the mapping from the CSV with the flags you pass, so matching the
extraction filters is still required there. `--max_sequences N` draws a
deterministic subset (fixed seed) so it's reproducible.

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
sequence_column: sequence   # column holding the protein sequence
label_column: label         # column holding the per-residue label string
positive_class: 1
class_names: {0: rigid, 1: flexible}
mapping: {A: 0, ".": 1, "0": 0, "1": 1}   # char -> class id (binary or multi-class)
ignore: "_"                 # documented ignore chars (any unmapped char is ignored too)
```
Characters not in `mapping` become `-100` (ignored), following the HuggingFace
ignore-index convention.

> The `relaxdb` preset accepts **both** the raw `relaxdb_data.csv` characters
> (`p/A/v→0`, `./b/^→1`, `t/x` ignored) and a `0/1`-relabeled CSV, so one preset
> covers either file.

### 2.1 Task-Label Alignment

Align each SAE feature with the task labels (e.g. rigid/flexible) to find
"flexibility detectors".

```bash
# Extract hidden states from the fine-tuned model (creates Outputs/demo/mbmrb/)
python crossplm.py single extract_embeddings \
    --ckpt_path Outputs/my_task/checkpoints/best \
    --sequences_csv Dataset/mBMRB.csv \
    --experiment demo --source mbmrb \
    --label_column label --label_map mBMRB

# Train SAE (320-dim → 640 features; embeddings inferred, all shards by default)
python crossplm.py single train_sae \
    --experiment demo --source mbmrb \
    --batch_size 64 --dict_size 640 --steps 20000 --l1_penalty 0.08 \
    --resample_steps 2000

`--reconstruction_loss l2` preserves the legacy unsquared L2 objective. Use
`--reconstruction_loss mse` to train with mean squared error instead. The default
remains `l2`; changing the objective can require retuning `--l1_penalty` because
the reconstruction-loss scale and gradients change.

# Align features with task labels
python crossplm.py single analyze_features \
    --embeddings_dir Outputs/demo/mbmrb/embeddings/layer_6 \
    --experiment demo --source mbmrb \
    --sequences_csv Dataset/mBMRB.csv --label_column label --label_map mBMRB

# Visualize top features on sequences (embeddings_path inferred from experiment)
python crossplm.py single visualize_features \
    --experiment demo --source mbmrb \
    --sequences_csv Dataset/mBMRB.csv \
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
encodes. Pipeline: build concept matrices, extract embeddings (reusing the SAE
trained in §2.1), then align features to concepts
(`single/scripts/analyze_concepts.py`).

**Step 1: Build per-residue concept matrices from a UniProtKB TSV export.**

The TSV must contain `Entry`, `Sequence`, and Feature-table columns (`Helix`,
`Beta strand`, `Turn`, `Domain [FT]`, `Active site`, `Binding site`, etc.).
Download from UniProt with `reviewed:true` → Export → TSV.

```bash
python crossplm.py single analyze_concepts build \
    --annotations_tsv Dataset/uniprotkb_swissprot.tsv \
    --experiment demo --source swissprot \
    --n_shards 5 \
    --max_residues 510   # must equal embedder max_length - 2 (512 -> 510)
```

> **Alignment requirement:** the concept shards must contain the **same proteins,
> in the same order, sharded identically** as the embedding shards. That means
> extracting embeddings from the **same TSV** with the same `--n_shards` and the
> same `--min_seq_len` / `--max_seq_len` length filter, using
> `--sequence_column Sequence`. (`analyze_concepts build` defaults to
> `0` / `10000`; the examples below pass `30` / `1022` — use the **same values**
> for extraction.) `--max_residues` (`max_length` − 2) then makes concept rows
> cover exactly the residues the embedder keeps on long sequences.
>
> If the token counts of a concept shard and its embedding shard do not match,
> `analyze_concepts align` now **fails with an error** (instead of silently
> truncating), so a mis-configured run is caught instead of producing garbage.

**Extract embeddings from the SAME TSV** (must match the concept build above):

```bash
python crossplm.py single extract_embeddings \
    --ckpt_path Outputs/my_task/checkpoints/best \
    --sequences_csv Dataset/uniprotkb_swissprot.tsv \
    --sequence_column Sequence \
    --experiment demo --source swissprot \
    --n_shards 5 --min_seq_len 30 --max_seq_len 1022
```

> No `--label_column` here — UniProt has no per-residue task labels; the embedder
> only needs the `Sequence` column.

> **No second SAE needed.** The same SAE trained in §2.1 (on mBMRB) is reused
> here — features are the same dictionary, so concept alignment measures whether
> mBMRB-learned features capture Swiss-Prot biology.

**Step 2: Align SAE features to concepts.**

```bash
python crossplm.py single analyze_concepts align \
    --embeddings_dir Outputs/demo/swissprot/embeddings/layer_6 \
    --experiment demo --source swissprot \
    --threshold_percents 0 0.15 0.5 0.6 0.8
```

For every feature × concept pair, computes F1 / precision / recall / AUROC /
domain-F1 across thresholds, saving `feature_concept_pairs.csv` (all pairs).

**Example output:**
```
Top feature-concept associations (by F1):
  Feature #42  → Domain_kinase                  F1=0.623 AUROC=0.781 P=0.710 R=0.551
  Feature #107 → Helix                          F1=0.588 AUROC=0.742 P=0.650 R=0.537
```

**Residue vs domain metrics:** `f1`/`precision`/`recall` are residue-level metrics.
`domain_precision`, `domain_recall`, and `domain_f1` use one-to-one matching between
contiguous predicted activation segments and annotated domain instances. The legacy
aliases `f1_per_domain` and `recall_per_domain` now point to the corresponding true
domain-level values.

### 2.3 Validation & Fidelity

Two checks that make the feature findings trustworthy.

**Held-out validation** removes selection bias. Reporting the "best feature per
concept" on the *same* data used for selection is optimistically biased — among 640
features some score high purely by chance.

1. Split concept shards into a **valid** and a **test** split.
2. On **valid**, pick the top feature per concept (selection).
3. On **test**, evaluate only the selected pairs (unbiased).

```bash
python crossplm.py single analyze_concepts heldout \
    --embeddings_dir Outputs/demo/swissprot/embeddings/layer_6 \
    --experiment demo --source swissprot \
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
ce_zero : layer zeroed (zero-ablation baseline)

Loss_Recovered = 1 - (ce_sae - ce_orig) / (ce_zero - ce_orig)
```
100% = perfectly preserves task info; 0% = as harmful as the zero-ablation
baseline. If zero ablation does not increase loss, the recovery percentage is
reported as invalid rather than interpreted as a recovery score.

`loss_recovered_pct` is clipped to [0, 100] for display; the raw unclipped value
is also saved as `loss_recovered_raw`. When the SAE reconstruction lowers the
loss *below* the original activations (`sae_better_than_original=true`,
`ce_sae < ce_orig`), the raw value exceeds 100% — that means the reconstruction
denoises the activations rather than merely preserving them, and it should not be
read as a recovery score.

```bash
python crossplm.py single evaluate_fidelity \
    --ckpt_path Outputs/my_task/checkpoints/best \
    --sequences_csv Dataset/mBMRB.csv \
    --experiment demo --source mbmrb \
    --layer 6 --label_column label --label_map mBMRB \
    --max_sequences 200
```
Saves `fidelity_results.json` incl. a `reconstruction_mse` sanity check.
`--max_sequences` limits to a subset for a quick check — drop it to evaluate the
whole dataset.

> **Data consistency:** `--source` keeps different datasets in separate
> subdirs (`Outputs/<experiment>/mbmrb/` vs `.../swissprot/`), so they never
> overwrite each other; the SAE is shared at the experiment root. Fidelity and
> intervention must use the sequences that match the analysis (`--source mbmrb`
> + mBMRB here).

**Note:** injection currently supports the final layer only
(`hidden_states[6]` = `emb_layer_norm_after` output).

**Causal intervention (feature steering)** moves from *correlation* to
*causation*. Whereas Fidelity replaces the whole layer, intervention perturbs a
**single SAE feature** and measures whether the model's per-residue predictions
change — establishing that the feature *causally drives* (not just co-occurs
with) the model's decision.

```bash
python crossplm.py single evaluate_intervention \
    --ckpt_path Outputs/my_task/checkpoints/best \
    --sequences_csv Dataset/mBMRB.csv \
    --experiment demo --source mbmrb \
    --feature_idx 375 --mode zero \
    --label_column label --label_map mBMRB \
    --layer 6 --max_sequences 200
```

`--max_sequences 200` limits to a quick subset — drop it to run the full dataset.

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
  (`--flank`, default 5) around the activated residues — the amino-acid
  "signature" of the feature.
- The positional motif analysis keeps each relative position separate, uses a
  within-protein permutation null, and saves p-values, BH-FDR q-values, and
  `sequence_logo_feature<N>.png`.

```bash
python crossplm.py single analyze_sequence \
    --embeddings_dir Outputs/demo/mbmrb/embeddings/layer_6 \
    --sequences_csv Dataset/mBMRB.csv \
    --experiment demo --source mbmrb \
    --label_map mBMRB \
    --feature_indices 375 42
```

By default all numeric shards are aggregated. Use `--shard 0` to run a quick
single-shard test. The pooled result is saved as `sequence_analysis.json`;
single-shard runs use `sequence_analysis_shard<N>.json`.

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
python crossplm.py single analyze_coactivation \
    --embeddings_dir Outputs/demo/mbmrb/embeddings/layer_6 \
    --sequences_csv Dataset/mBMRB.csv \
    --experiment demo --source mbmrb \
    --label_map mBMRB \
    --feature_a 375 --feature_b 42
```

By default all numeric shards are aggregated from raw token, overlap and
neighborhood counts. Use `--shard 0` for a quick single-shard test.

Key metrics (each compared to the appropriate null):
- `overlap_ab` / `enrich_ab` — same-residue co-activation vs the unconditional
  activation-rate baseline; **>1** → B enriched on A's residues, **<1** →
  mutually exclusive at the same position.
- `neighbor_ab` / `neighbor_enrich_ab` — B active within ±`--neighborhood`
  residues, excluding the same residue, around an A-active residue.
  `neighbor_enrich_ab` is normalized by the **independence null** (the expected
  window-hit probability `1-(1-p)^(2k)` for independent features, boundary-aware
  at protein ends), NOT by the per-token rate — a raw window probability is
  already ~`2k × p` even for independent features. **>>1** → co-localization
  beyond independence. Same-residue overlap is reported separately.
- Both directions (A→B and B→A) are reported to detect asymmetry.

Example (Features #375 and #42): same-residue enrichment **0.67×** (mutually
exclusive at identical positions) but neighborhood enrichment **≈ 0.87–0.89× vs
the independence null** (B is found near A about as often as chance predicts) →
the two are **largely independent**, neither co-localized nor complementary.
The null baselines (`neighbor_ab_null` / `neighbor_ba_null`) and the per-residue
window-size histograms are saved alongside the probabilities.
Saves `coactivation_<a>_<b>_shard<N>.json`.

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

Optional — a **single** install gets both the bare `crossplm` command and the
`single` package (importable from anywhere):
```bash
pip install -e .          # → `crossplm` command + `single` package
```

---

## Output Structure

The Training and Single modules share one **experiment directory**
(`Outputs/<experiment>/`, name used verbatim, no timestamp; reuse with
`--exp_dir`):

```
Outputs/
└── <experiment>/
    ├── config.yaml                                # Training config
    ├── training_curve.png                         # Training curve
    ├── eval_metrics.jsonl                         # Training eval history
    ├── checkpoints/                               # Trained PLM checkpoints
    │   ├── epoch_<N>_f1_<F>/                      # Best-3 by F1 (auto-pruned)
    │   ├── best/                                  # Stable alias: highest-F1
    │   ├── final/                                 # Final checkpoint
    │   └── ...                                    # each contains label_map.json
    ├── evaluations/<csv_name>/                    # PLM eval (metrics.json, plots)
    ├── sae/                                       # ONE shared SAE (reused)
    │   ├── model.pt                               # SAE weights
    │   ├── model_normalized.pt                    # Max-activation rescaled SAE
    │   └── checkpoints/step_<N>/                  # Resumable SAE checkpoints (newest N kept)
    └── <source>/                                  # per --source <id>, e.g. mbmrb/swissprot
        ├── embeddings/layer_<N>/shard_<i>/embeddings.pt   # Hidden states
        ├── concepts/
        │   ├── shard_<i>/concept_matrix.npz       # Per-residue concepts
        │   ├── shard_<i>/residues.csv             # Residue metadata
        │   └── concept_columns.txt
        └── analysis/
            ├── feature_label_metrics.json         # Task-label alignment
            ├── feature_label_correlations.npy     # Point-biserial r per feature
            ├── feature_label_correlation_stats.json # r, p-value, BH q-value/FDR
            ├── activation_profile.npz             # Per-class mean/max activation
            ├── max_activations_per_feature.pt     # Used to build model_normalized.pt
            ├── feature_concept_pairs.csv          # Feature × concept alignment
            ├── heldout_*.csv                      # Held-out validation
            ├── fidelity_results.json              # Fidelity
            ├── intervention_feat<N>_<mode>.json   # Causal intervention
            ├── sequence_analysis.json              # Pooled Cohen's d + motif
            ├── sequence_logo_feature<N>.png        # Positional motif logo
            ├── coactivation_<a>_<b>.json           # Pooled pairwise co-activation
            └── visualizations/                    # PNG plots
```

`single/paths.py` centralizes these paths. `--source <id>` nests the data-specific
dirs under `Outputs/<experiment>/<source>/` (flat when omitted). `--sae_dir` /
`--embeddings_dir` / `--output_dir` / `--save_dir` / `--concepts_dir` default into
the experiment dir and can be overridden explicitly; `--exp_dir <path>` points at
an existing experiment dir verbatim.

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
