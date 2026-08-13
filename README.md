# CrossPLM

**Mechanistic Interpretability for Cross-Task Protein Language Models**

Protein language models (PLMs) perform well on diverse biological tasks, but their internal mechanisms are not yet fully understood. This project builds an interpretability framework for cross-task PLMs, including a fine-tuning toolkit and an interpretability research module.

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
│   │   ├── config/              # Configuration management
│   │   ├── data/                # Data loading
│   │   ├── models/              # Model wrappers
│   │   ├── trainers/            # Training loop
│   │   └── utils/               # Utilities
│   ├── outputs/                 # ← All runtime outputs
│   │   └── tasks/               #   Task folders (config, checkpoints, eval results)
│   └── examples/                # Sample data and configs
└── README.md
```

---

## Usage

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
→ Results (metrics.json, confusion matrix, AUPRC curve) are saved inside the
  task folder under `eval_on_<csv_name>/`.

---

## Features

| Feature | Description |
|---|---|
| **Two-phase workflow** | init (template) → train, keeping config separate from code |
| **CSV input** | `sequence` + `label` columns, automatic train/eval split |
| **Ignore positions** | `_` labels are excluded from loss |
| **Auto class weights** | `inverse` / `log` / `none` strategies |
| **Metrics** | Loss + Accuracy + Macro F1 + AUPRC |
| **Top-3 checkpoints** | Keeps best 3 by F1, cleans old ones automatically |
| **Training curve** | Auto-generated epoch–F1 plot after training |
| **Eval plots** | Confusion matrix + Precision-Recall curve |

---

## Baseline (mBMRB, ESM2-35M)

| Metric | Value |
|---|---|
| Accuracy | ~0.92 |
| Macro F1 | ~0.80 |
| Class distribution | class 0: 933K / class 1: 131K (7:1) |

---

## Dependencies

```bash
pip install -r Training/requirements.txt
```

---

## Status

| Module | Status |
|---|---|
| **Training** | ✅ Tested and functional |
| **Crossing** | 🔄 In planning |

## Crossing Module (Planned)

- **Single-model interpretability**: neuron / attention head importance analysis, causal intervention (activation patching), representation probing
- **Cross-model interpretability**: comparative analysis across different PLMs (ESM-2, ProtBERT, Ankh), task-specific vs task-common representation separation, cross-model feature transferability
