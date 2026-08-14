# CrossPLM

**Mechanistic Interpretability for Cross-Task Protein Language Models**

Protein language models (PLMs) perform well on diverse biological tasks, but their internal mechanisms are not yet fully understood. This project builds an interpretability framework for cross-task PLMs, including a fine-tuning toolkit and a SAE-based interpretability module.

---

## Project Structure

```
CrossPLM/
├── Dataset/                     # Raw datasets
│   ├── relaxdb_data.csv
│   └── mBMRB.csv
├── Preprocessing/               # Dataset-specific preprocessing scripts
│   ├── preprocess_relaxdb.py
│   └── preprocess_mbmrb.py
├── Training/                    # PLM training framework
│   ├── crossplm.py              # Unified CLI (init / train / eval)
│   ├── crossplm/                # Python package
│   ├── outputs/                 # Training outputs (checkpoints, logs)
│   └── examples/                # Sample data and configs
├── Single/                      # ★ SAE-based interpretability
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
│   └── <experiment>_<ts>/       # embeddings / sae / concepts / analysis
└── README.md
```

---

## Training: Fine-tune a PLM

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

---

## Interpretability: SAE Feature Analysis

After fine-tuning a PLM, use Sparse Autoencoders to discover which hidden-state features drive the model's predictions.

### Pipeline

```
Fine-tuned PLM checkpoint
         ↓
[1] extract_embeddings.py    → Extract per-residue hidden states from a target layer
         ↓
[2] train_sae.py             → Train a Sparse Autoencoder to learn interpretable features
         ↓
[3] analyze_features.py     → Align each SAE feature with task labels (rigid/flexible)
         ↓
[4] visualize_features.py   → Plot feature activation patterns on protein sequences
```

### How it Works

SAEs learn a sparse, overcomplete decomposition of PLM hidden states:

```
ESM hidden state (320-dim)  →  SAE encoder  →  sparse feature vector (640-dim)
                                                   ↑ only ~18 features active per token
```

Each feature can then be interpreted by checking **when** it activates:
- Does it activate preferentially on **flexible** residues? → "flexibility detector"
- Does it activate on **rigid** residues? → "rigidity detector"
- Is it unrelated to the task? → noise feature

### Usage

All outputs are organized under a single **experiment directory**. Pick an
experiment name (or reuse one with `--exp_dir`) and every step routes its
output into the same tree:

```
Outputs/<experiment>_<timestamp>/
    embeddings/layer_<N>/shard_<i>/activations.pt
    sae/ae.pt
    concepts/shard_<i>/aa_concepts.npz
    analysis/*.csv|*.json|visualizations/
```

```bash
cd Single

# 1. Extract hidden states from fine-tuned model (creates Outputs/mb_<ts>/)
python -m single.scripts.extract_embeddings \
    --ckpt_path ../Training/outputs/tasks/my_task/checkpoints/best \
    --sequences_csv ../Dataset/mBMRB.csv \
    --experiment mb \
    --label_column label --label_map mBMRB

# 2. Train SAE (320-dim → 640 sparse features)
python -m single.scripts.train_sae \
    --embeddings_dir ../Outputs/mb_*/embeddings/layer_6/shard_0 \
    --experiment mb \
    --batch_size 64 --dict_size 640 --steps 10000 --l1_penalty 0.03

# 3. Align features with task labels
python -m single.scripts.analyze_features \
    --sae_dir ../Outputs/mb_*/sae \
    --embeddings_dir ../Outputs/mb_*/embeddings/layer_6 \
    --experiment mb \
    --sequences_csv ../Dataset/mBMRB.csv --label_column label --label_map mBMRB

# 4. Visualize top features on sequences
python -m single.scripts.visualize_features \
    --sae_dir ../Outputs/mb_*/sae \
    --embeddings_path ../Outputs/mb_*/embeddings/layer_6/shard_0/activations.pt \
    --experiment mb --feature_indices 234 426 --label_map mBMRB
```

Note: `--experiment <name>` creates a fresh `Outputs/<name>_<ts>/` on first use.
To continue the same run, pass `--exp_dir <existing_dir>` instead (or reuse the
same `--experiment` name, which reuses the dir if the name already carries a
timestamp). Any `--output_dir`/`--save_dir`/`--concepts_dir` explicitly given
overrides the experiment routing.

### Example Output

```
Top features for 'flexible' (label=1):
  Feature #234: F1=0.514, AUROC=0.713, Prec=0.923, Rec=0.356
  Feature #426: F1=0.517, AUROC=0.693, Prec=0.729, Rec=0.401

High-precision "flexible residue detectors":
  When feature #234 activates → 92% chance residue is truly flexible
  But it only catches ~36% of all flexible residues
```

### Interpreting the Metrics

| Metric | What it means |
|--------|---------------|
| **Precision** | When this feature activates, how often is the label correct? |
| **Recall** | What fraction of positive cases does this feature catch? |
| **F1** | Harmonic mean of precision and recall |
| **AUROC** | Overall discriminative power (0.5=random, 1.0=perfect) |
| **Activation Gap** | Mean(positive activation) − Mean(negative activation) |

### Configurable Label Maps

The pipeline is **not hardcoded to mBMRB**. Label encoding, the positive class, and
class names are configurable via `--label_map` on every script. This lets you analyze
datasets with different label formats without changing any code.

**Built-in presets** (defined in `single/label_maps.py`):

| Preset | Positive class | Class names | Character mapping |
|--------|---------------|-------------|-------------------|
| `mBMRB` | 1 | rigid / flexible | `A→0`, `.→1`, `0→0`, `1→1` |
| `relaxdb` | 1 | static / mobile | `p/A/v→0`, `./b/^→1` |
| `ss3` | 1 | helix / strand / coil | `H→0`, `E→1`, `C→2` (3-class) |

Use a preset by name:
```bash
python -m single.scripts.analyze_features \
    --sae_dir ... --sequences_csv ... --label_column label --label_map relaxdb
```

Or provide your own YAML file for custom datasets:
```yaml
# my_dataset.yaml
positive_class: 1
class_names:
  0: rigid
  1: flexible
mapping:
  A: 0
  ".": 1
  "0": 0
  "1": 1
ignore: "_"
```
```bash
python -m single.scripts.analyze_features \
    --sae_dir ... --sequences_csv ... --label_column label --label_map my_dataset.yaml
```

Any character not in `mapping` becomes `-100` (ignored in analysis), following the
HuggingFace ignore-index convention. Add new presets to `LABEL_MAPS` in
`single/label_maps.py` to reuse them across runs.

### Biological Concept Analysis (Swiss-Prot / UniProtKB)

Beyond task labels, you can align SAE features against **biological concepts** from
Swiss-Prot — e.g. "Helix", "Domain_kinase", "Binding_site_ATP", "Active site" —
to discover what real biology each feature encodes. This is a two-step pipeline
(`single/scripts/analyze_concepts.py`).

**Step 1: Build per-residue concept matrices from a UniProtKB TSV export.**

The TSV must contain the `Entry` and `Sequence` columns plus Feature-table columns
(`Helix`, `Beta strand`, `Turn`, `Domain [FT]`, `Active site`, `Binding site`, etc.).
Download from UniProt with a query like: `reviewed:true` → Export → TSV.

```bash
python -m single.scripts.analyze_concepts build \
    --annotations_tsv ../Dataset/uniprotkb_swissprot.tsv \
    --experiment sp \
    --n_shards 5
```

This expands each protein-level annotation to amino-acid level and saves
`shard_N/aa_concepts.npz` (sparse matrices) + `aa_concepts_columns.txt` (concept names)
into `Outputs/sp_*/concepts/`.

**Step 2: Align SAE features to concepts.**

```bash
python -m single.scripts.analyze_concepts align \
    --sae_dir ../Outputs/sp_*/sae \
    --embeddings_dir ../Outputs/sp_*/embeddings/layer_6 \
    --experiment sp \
    --threshold_percents 0 0.15 0.5 0.6 0.8
```

Concepts and output are routed into `Outputs/sp_*/` automatically. Note that the
**embeddings must be extracted from the same TSV** (same `--n_shards`) so embedding
and concept shards are token-aligned.

For every feature × concept pair, computes F1 / precision / recall / AUROC / domain-F1
across activation thresholds and saves:

- `feature_concept_pairs.csv` — ALL pairs (filter with pandas, e.g. by AUROC)
- `feature_concept_metrics.json` — full per-pair metrics

**Example output:**

```
Top feature-concept associations (by F1):
  Feature #42  → Domain_kinase                  F1=0.623 AUROC=0.781 P=0.710 R=0.551
  Feature #107 → Helix                          F1=0.588 AUROC=0.742 P=0.650 R=0.537
  Feature #3   → Binding_site_ATP               F1=0.411 AUROC=0.694 P=0.502 R=0.348
```

Note: to get exact per-protein alignment between embeddings and concept matrices,
extract embeddings from the **same TSV** with `extract_embeddings.py` (per-shard order
must match). If shard counts differ, the script falls back to best-effort splitting.

---

## Features

### Training Module
| Feature | Description |
|---------|-------------|
| **Two-phase workflow** | init (template) → train, keeping config separate from code |
| **CSV input** | `sequence` + `label` columns, automatic train/eval split |
| **Ignore positions** | `_` labels are excluded from loss |
| **Auto class weights** | `inverse` / `log` / `none` strategies |
| **Metrics** | Loss + Accuracy + Macro F1 + AUPRC |
| **Top-3 checkpoints** | Keeps best 3 by F1, cleans old ones automatically |
| **Training curve** | Auto-generated epoch–F1 plot after training |
| **Eval plots** | Confusion matrix + Precision-Recall curve |

### SAE Interpretability Module
| Feature | Description |
|---------|-------------|
| **Fine-tuned PLM support** | Extracts hidden states from any fine-tuned `AutoModelForTokenClassification` |
| **Configurable labels** | `--label_map` presets or YAML files for any dataset format |
| **ReLU / TopK SAE** | Two standard SAE architectures |
| **L1 sparsity + dead neuron resampling** | Standard training recipe from mechanistic interpretability |
| **Feature-label alignment** | F1, AUROC, correlation, activation gap per feature |
| **Feature-concept alignment** | F1/AUROC per feature vs Swiss-Prot biological concepts (multi-label) |
| **UniProt annotation parsing** | TSV Feature-table annotations → per-residue sparse concept matrices |
| **Per-protein tracking** | Find which proteins maximally activate each feature |
| **Visualization** | Feature activation vs ground truth on sequences |
| **Normalization** | Feature-wise max-activation normalization for meaningful comparisons |

---

## Status

| Module | Status |
|--------|--------|
| **Training** | ✅ Tested and functional |
| **Single (SAE Interpretability)** | ✅ Tested and functional |
| **Crossing** | 🔄 In planning |

### Crossing Module (Planned)

- **Single-model interpretability**: neuron / attention head importance analysis, causal intervention (activation patching), representation probing
- **Cross-model interpretability**: comparative analysis across different PLMs (ESM-2, ProtBERT, Ankh), task-specific vs task-common representation separation, cross-model feature transferability

---

## Dependencies

```bash
# Training
pip install -r Training/requirements.txt

# SAE Interpretability
pip install -r Single/single/requirements.txt
```
Or install the SAE package in development mode:
```bash
cd Single && pip install -e .
```

---

## Output Structure

Every step of the pipeline writes into one **experiment directory** (created on
first use via `--experiment <name>`; reuse with `--exp_dir`):

```
Outputs/
└── <experiment>_<timestamp>/
    ├── embeddings/
    │   └── layer_<N>/                # Per-residue hidden states
    │       ├── shard_0/activations.pt
    │       ├── shard_1/activations.pt
    │       └── ...
    ├── sae/
    │   ├── ae.pt                      # SAE weights
    │   └── checkpoint_<step>/
    ├── concepts/
    │   ├── shard_<i>/aa_concepts.npz # Per-residue concept matrices
    │   └── aa_concepts_columns.txt
    └── analysis/
        ├── feature_concept_pairs.csv  # Feature × concept alignment (all pairs)
        ├── feature_concept_metrics.json
        ├── feature_label_metrics.json # Task-label alignment
        ├── feature_label_correlations.npy
        ├── activation_profile.npz
        └── visualizations/            # PNG plots
```

`single/paths.py` centralizes these paths; scripts resolve subdirectories from the
experiment dir automatically. Explicit `--output_dir` / `--save_dir` /
`--concepts_dir` flags override the routing when you need custom locations.
