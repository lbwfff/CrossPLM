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
├── Single/                      # ★ NEW: SAE-based interpretability
│   ├── setup.py
│   └── single/
│       ├── configs.py           # Configuration dataclasses
│       ├── embedders/           # Hidden state extraction from fine-tuned PLMs
│       ├── sae/                 # SAE architectures (ReLUSAE, TopKSAE)
│       ├── train/               # SAE training loop
│       ├── analysis/            # Feature-to-label alignment
│       └── scripts/             # CLI scripts
├── Outputs/                     # All SAE outputs (auto-organized)
│   ├── embeddings/              # Extracted PLM hidden states
│   ├── sae/                     # Trained SAE models
│   └── analysis/               # Feature analysis results
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

```bash
cd Single

# 1. Extract hidden states from fine-tuned model
python -m single.scripts.extract_embeddings \
    --ckpt_path ../Training/outputs/tasks/my_task/checkpoints/best \
    --sequences_csv ../Dataset/mBMRB.csv \
    --label_column label

# 2. Train SAE (320-dim → 640 sparse features)
python -m single.scripts.train_sae \
    --embeddings_dir ../Outputs/embeddings/esm2_8m/layer_6/shard_0 \
    --batch_size 64 --dict_size 640 --steps 10000 --l1_penalty 0.03

# 3. Align features with task labels
python -m single.scripts.analyze_features \
    --sae_dir ../Outputs/sae/esm2_8m_l6_d640_<ts> \
    --embeddings_dir ../Outputs/embeddings/esm2_8m/layer_6 \
    --sequences_csv ../Dataset/mBMRB.csv --label_column label

# 4. Visualize top features on sequences
python -m single.scripts.visualize_features \
    --sae_dir ../Outputs/sae/esm2_8m_l6_d640_<ts> \
    --embeddings_path ../Outputs/embeddings/esm2_8m/layer_6/shard_0/activations.pt \
    --output_dir ../Outputs/analysis/vis --feature_indices 234 426
```

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
| **Activation Gap** | Mean(flexible activation) − Mean(rigid activation) |

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
| **ReLU / TopK SAE** | Two standard SAE architectures |
| **L1 sparsity + dead neuron resampling** | Standard training recipe from mechanistic interpretability |
| **Feature-label alignment** | F1, AUROC, correlation, activation gap per feature |
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

All SAE outputs are auto-organized under `Outputs/`:

```
Outputs/
├── embeddings/
│   └── esm2_8m/layer_6/           # Per-residue hidden states
│       ├── shard_0/activations.pt
│       ├── shard_1/activations.pt
│       └── ...
├── sae/
│   └── esm2_8m_l6_d640_<ts>/      # Trained SAE (timestamped)
│       ├── ae.pt                   # Model weights
│       └── checkpoint_2500/
└── analysis/
    └── <ts>/                       # Analysis results (timestamped)
        ├── feature_label_metrics.json
        ├── feature_label_correlations.npy
        ├── activation_profile.npz
        └── visualizations/
```
