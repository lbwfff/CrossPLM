# Training Module

Fine-tunes a HuggingFace protein language model (e.g. ESM-2) on a per-residue
token-classification task — for example backbone dynamics (rigid vs flexible),
secondary structure (coil/strand/helix), or any per-residue label.

> Back to [project README](../README.md).

---

## Workflow

```
init  →  edit config.yaml  →  train  →  eval
```

All commands run from the **repository root** via the unified CLI:

```bash
python crossplm.py training init      --task_name my_experiment
python crossplm.py training train     --config Outputs/my_experiment/config.yaml
python crossplm.py training eval      --checkpoint ... --csv Dataset/mBMRB.csv
python crossplm.py training labelmap  --name my_dataset
```

With `pip install -e .` these become `crossplm training init ...`, etc.

---

## 0. Data & Label Map

> **`Dataset/` is user-supplied and gitignored** — it is not synced to the repo.
> Place your raw data (`mBMRB.csv`, `relaxdb_data.csv`,
> `uniprotkb_swissprot.tsv`, ...) there yourself.

Training and Single share the **same label-map presets** (`mBMRB`, `relaxdb`,
`ss3`) and YAML label-map files. A label map defines the CSV columns, the
character → class mapping, and which characters are ignored — so you can point
training directly at a **raw** dataset CSV with no preprocessing:

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
train/eval).

**Generate an empty template** into `Dataset/`:

```bash
python crossplm.py training labelmap --name my_dataset
# → Dataset/my_dataset.yaml
```

Then reference it in a config as `label_map: ../Dataset/my_dataset.yaml` or via
`--label_map ../Dataset/my_dataset.yaml` on the CLI.

| Preset | Positive class | Class names | Character mapping |
|--------|---------------|-------------|-------------------|
| `mBMRB` | 1 | rigid / flexible | `A→0`, `.→1`, `0→0`, `1→1` |
| `relaxdb` | 1 | static / mobile | `p/A/v→0`, `./b/^→1` (`t/x` ignored) |
| `ss3` | 1 | coil / strand / helix | `C→0`, `E→1`, `H→2` (3-class) |

> `ss3` ids match the training module's `build_label_map` (`sorted(unique)`,
> i.e. `C < E < H`), so a model trained with an inferred map evaluates
> consistently. The `relaxdb` preset accepts both the raw `relaxdb_data.csv`
> characters and a `0/1`-relabeled CSV.

### Path resolution

Relative `csv_data_path` / `label_map` YAML paths in the **config** resolve
against the `Training/` module directory (so `../Dataset/mBMRB.csv` reaches the
repo's `Dataset/` no matter where you run the command from). On the eval
**CLI**, `--label_map` (like `--csv`) resolves against your current directory.

---

## 1. Initialize a Task

```bash
python crossplm.py training init --task_name my_experiment
```

Creates `Outputs/my_experiment/config.yaml` (verbatim name, no timestamp,
shared with the Single module's `Outputs/<experiment>` root).

## 2. Edit Config → Train

Edit `Outputs/my_experiment/config.yaml`, then:

```bash
python crossplm.py training train --config Outputs/my_experiment/config.yaml
```

Checkpoints, training curve, logs and a `provenance.json` (dataset `sha256`, split, backbone, freeze settings and full config snapshot) plus `config_snapshot.yaml` are saved inside the experiment folder. The training also writes `Outputs/<task>/config_snapshot.yaml` so the run is reproducible even if `Outputs/<task>/config.yaml` was edited after `init`.

The config's `label_map:` field takes a preset name or YAML path. Leave it
empty to infer the mapping from the CSV (e.g. `Training/examples/sample.csv` uses `label_map: ""` with `H/E` labels inferred as `E:0, H:1`).

### Config fields

| Field | Default | Description |
|-------|---------|-------------|
| `task_name` | `my_plm_task` | Identifier (for logging only) |
| `model_name` | `esm2_t6_8M` | Name used when saving the model |
| `backbone_model_id` | `facebook/esm2_t6_8M_UR50D` | HuggingFace backbone model ID — native `facebook/esm2_*` and `Synthyra/ESM2-8M` (FastPLMs) both supported (`trust_remote_code=True` is handled automatically) |
| `csv_data_path` | `../Dataset/mBMRB.csv` | Training CSV (relative to `Training/`) |
| `sequence_column` / `label_column` | `sequence` / `label` | CSV column names (overridden by `label_map`) |
| `train_ratio` | `0.9` | Train/eval split fraction |
| `task_type` | `token_classification` | Only `token_classification` implemented (`mlm` reserved) |
| `max_seq_length` | `512` | Max sequence length (truncate longer) |
| `per_device_train_batch_size` / `per_device_eval_batch_size` | `8` / `8` | Batch sizes |
| `gradient_accumulation_steps` | `1` | Accumulate grads over N batches |
| `learning_rate` | `2.0e-5` | Learning rate |
| `weight_decay` | `0.01` | Weight decay |
| `num_train_epochs` | `3` | Number of epochs |
| `max_steps` | `-1` | `-1` = determined by epochs |
| `logging_steps` / `eval_steps` / `save_steps` | `10` / `500` / `1000` | Step intervals |
| `save_total_limit` | `3` | Keep newest N periodic + best-F1 checkpoints |
| `class_weight_method` | `inverse` | `none` / `inverse` / `sqrt` / `log` |
| `seed` | `42` | Random seed |
| `dataloader_num_workers` | `2` | DataLoader workers |
| `fp16` / `bf16` | `false` / `false` | Mixed-precision AMP (requires CUDA; `bf16` prefers Ampere+) |
| `freeze_backbone` | `false` | If `true`, freeze all encoder layers and train only the classifier head (Phase 0 control) |
| `freeze_layers` | `0` | Freeze bottom N encoder layers (`0`=none; ignored when `freeze_backbone=true`) |
| `label_map` | `mBMRB` | Preset name or YAML path (shared with Single) |

## 3. Evaluate a Checkpoint

```bash
python crossplm.py training eval \
  --checkpoint Outputs/my_experiment/checkpoints/best \
  --csv Dataset/mBMRB.csv \
  --label_map mBMRB
```

Results (`metrics.json`, confusion matrix, AUPRC curve) are saved to
`Outputs/my_experiment/evaluations/<csv_name>/`.

### Checkpoints

Training saves three kinds of checkpoints under `Outputs/<name>/checkpoints/`:

| Type | Description |
|------|-------------|
| `epoch_<N>_f1_<F>/` | Best-3 by F1 (auto-pruned) |
| `best/` | Stable alias for the highest-F1 checkpoint — use this for eval/interpretability |
| `final/` | Final checkpoint |

Each checkpoint contains a `label_map.json` sidecar with the training-time
mapping. Evaluation reuses it automatically; passing `--label_map` explicitly is
recommended for older checkpoints.

### Eval CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | — | Path to checkpoint (required) |
| `--csv` | — | Evaluation CSV (required) |
| `--label_map` | — | Preset or YAML (recommended; overrides checkpoint sidecar) |
| `--batch_size` | `8` | Eval batch size |
| `--max_seq_length` | `512` | Max sequence length |
| `--sequence_column` / `--label_column` | `sequence` / `label` | CSV columns |
| `--output` | `<exp>/evaluations/<csv_name>/` | Output directory |

---

## Features

| Feature | Description |
|---------|-------------|
| **Two-phase workflow** | init (template) → train, keeping config separate from code |
| **CSV input** | `sequence` + `label` columns, automatic train/eval split |
| **Ignore positions** | Unmapped characters excluded from loss (`-100`) |
| **Auto class weights** | `inverse` / `sqrt` / `log` / `none` strategies |
| **Configurable label maps** | Presets or YAML, shared with Single |
| **Multi-class support** | Correct class count for many-to-one label maps |
| **Backbone compatibility** | Native `facebook/esm2_*` and `Synthyra/ESM2-*` (FastPLMs, `trust_remote_code=True`) |
| **Freeze control** | `freeze_backbone` / `freeze_layers` for Phase 0 pre-existing vs emergent controls |
| **Provenance** | `provenance.json` + `config_snapshot.yaml` capture dataset hash, split and config for reproducibility |
| **Metrics** | Loss + Accuracy + Macro F1 + AUPRC (macro over all classes) |
| **Mixed precision** | `fp16` / `bf16` AMP on CUDA GPUs (auto-disabled on CPU) |
| **Top-3 checkpoints** | Keeps best 3 by F1, cleans old ones; `best` is a stable alias |
| **Training curve** | Auto-generated epoch–F1 plot after training |
| **Eval plots** | Confusion matrix + Precision-Recall curve |

---

## Layout

```
Training/
├── training_cli.py     # `crossplm training` implementation
├── training/           # Python package
│   ├── config/         # TrainingConfig dataclass
│   ├── data/           # Dataset, CSV loading, label maps
│   ├── model/          # PLMModel (HuggingFace backbone + classifier head)
│   └── trainer/        # Training loop
└── examples/           # Sample data and configs
```
