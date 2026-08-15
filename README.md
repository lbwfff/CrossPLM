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
│   └── <experiment>/            # embeddings / sae / concepts / analysis
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
Outputs/<experiment>/
    embeddings/layer_<N>/shard_<i>/activations.pt
    sae/ae.pt
    concepts/shard_<i>/aa_concepts.npz
    analysis/*.csv|*.json|visualizations/
```

```bash
cd Single

# 1. Extract hidden states from fine-tuned model (creates Outputs/mb/)
python -m single.scripts.extract_embeddings \
    --ckpt_path ../Training/outputs/tasks/my_task/checkpoints/best \
    --sequences_csv ../Dataset/mBMRB.csv \
    --experiment mb \
    --label_column label --label_map mBMRB

# 2. Train SAE (320-dim → 640 sparse features; embeddings_dir inferred from experiment,
#    uses ALL shards by default. --shard N restricts to one shard.)
python -m single.scripts.train_sae \
    --experiment mb \
    --batch_size 64 --dict_size 640 --steps 20000 --l1_penalty 0.08 \
    --resample_steps 2000

# 3. Align features with task labels
python -m single.scripts.analyze_features \
    --sae_dir ../Outputs/mb/sae \
    --embeddings_dir ../Outputs/mb/embeddings/layer_6 \
    --experiment mb \
    --sequences_csv ../Dataset/mBMRB.csv --label_column label --label_map mBMRB

# 4. Visualize top features on sequences (embeddings_path inferred from experiment)
python -m single.scripts.visualize_features \
    --sae_dir ../Outputs/mb/sae \
    --experiment mb \
    --sequences_csv ../Dataset/mBMRB.csv \
    --feature_indices 234 426 --label_map mBMRB
```

`visualize_features` 默认从实验目录推断嵌入文件
`Outputs/<exp>/embeddings/layer_<N>/shard_<S>/activations.pt`（`--layer` 默认 6，`--shard` 默认 0）。
如需指定其它 shard，加 `--shard 1`；也可用 `--embeddings_path <file>` 显式覆盖。
它会按 `--experiment` 重新对 CSV 做与提取时相同的打乱+分片（`sample(frac=1, random_state=42)`），
确保展示的蛋白与其嵌入、标签严格对齐，并输出 PNG 到 `Outputs/<exp>/analysis/visualizations/`。

`train_sae` 同样默认从实验目录推断嵌入（`Outputs/<exp>/embeddings/layer_<N>`，含全部 shard），
可用 `--embeddings_dir` 覆盖；`--shard N` 只训练单个 shard。常用调参提示：
- `--l1_penalty` 控制稀疏度（约 0.06–0.1），值越大特征越稀疏（l0 越低）但重构损失越高
- `--resample_steps N` 周期性复活"死特征"（从不激活的特征），可降低 `dead_pct`
- 理想目标：**l0 在 20–80、dead_pct < 30%、recon_loss 尽量低**

Note: `--experiment <name>` routes all outputs into `Outputs/<name>/` — the name
is used **verbatim (no timestamp)**, so every step of the same experiment shares
one directory. Re-running a step with the same name reuses that directory (e.g.
overwrites `ae.pt`). Use **distinct experiment names** for distinct runs. Pass
`--exp_dir <existing_dir>` to point at an existing experiment directory directly.
Any `--output_dir`/`--save_dir`/`--concepts_dir` explicitly given overrides the
experiment routing.

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
| `ss3` | 1 | coil / strand / helix | `C→0`, `E→1`, `H→2` (3-class) |

> The `ss3` ids follow the training module's `build_label_map` (`sorted(unique)`,
> i.e. `C < E < H`), so a model trained with an inferred label map evaluates
> consistently with this preset.

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
    --n_shards 5 \
    --max_residues 510   # must equal embedder max_length - 2 (default 512 -> 510)
```

This expands each protein-level annotation to amino-acid level and saves
`shard_N/aa_concepts.npz` (sparse matrices) + `aa_concepts_columns.txt` (concept names)
into `Outputs/sp/concepts/`.

> **Alignment requirement:** pass `--max_residues` (embedder `max_length` − 2, e.g.
> 510 for the default 512) so concept rows cover exactly the same residues the
> embedder keeps. Sequences longer than this are truncated identically on both
> sides; without it, concept matrices use the full-length sequence and can misalign
> with truncated embeddings.

**Step 2: Align SAE features to concepts.**

```bash
python -m single.scripts.analyze_concepts align \
    --sae_dir ../Outputs/sp/sae \
    --embeddings_dir ../Outputs/sp/embeddings/layer_6 \
    --experiment sp \
    --threshold_percents 0 0.15 0.5 0.6 0.8
```

Concepts and output are routed into `Outputs/sp/` automatically. Note that the
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

### Held-out Validation

Reporting the "best feature per concept" with its F1 on the *same* data used for
selection is optimistically biased — among 640 features, some will score high on a
concept purely by chance (selection bias). Held-out validation separates selection
from evaluation:

1. Split concept shards into a **valid** split and a **test** split (non-overlapping).
2. On the **valid** split, pick the top feature per concept (the *selection* step).
3. On the **test** split, evaluate **only the selected pairs** — these metrics are
   unbiased estimates of real performance on unseen data.

```bash
python -m single.scripts.analyze_concepts heldout \
    --sae_dir ../Outputs/sp/sae \
    --embeddings_dir ../Outputs/sp/embeddings/layer_6 \
    --experiment sp \
    --split_mode half \
    --threshold_percents 0 0.15 0.5 0.6 0.8
```

Output files in the experiment's `analysis/` (see the data-flow below):

```
heldout_valid_pairs.csv  (all pairs on valid split)
heldout_test_pairs.csv   (all pairs on test split)
        │                        │
        │  select top feature    │  keep only selected pairs
        │  per concept ─────────►│
        ▼                        ▼
                          heldout_top_pairings.csv   (selected pairs, test metrics)
                                │  filter f1_per_domain ≥ threshold
                                ▼
                          heldout_all_top_pairings.csv  (robust findings)
```

- `heldout_valid_pairs.csv` / `heldout_test_pairs.csv` — the full feature×concept
  tables for each split (raw "mother" tables).
- `heldout_top_pairings.csv` — one selected pair per concept, **metrics computed on
  the test split only**. A large drop vs. the valid-split value reveals selection bias.
- `heldout_all_top_pairings.csv` — the selected pairs that stay strong on the test
  split (`f1_per_domain ≥ --heldout_f1_threshold`). **This is the most trustworthy
  set of feature-concept findings.**

### Fidelity (Loss Recovered)

Fidelity validates that the SAE is a *faithful* representation of the fine-tuned
model's task-relevant activations — the foundation that makes feature analysis
meaningful. The target layer's hidden states are replaced three ways and the model's
**task loss** (token classification) is compared:

```
ce_orig : original activations
ce_sae  : SAE reconstructions injected
ce_zero : layer zeroed (worst case)

Loss_Recovered = 1 - (ce_sae - ce_orig) / (ce_zero - ce_orig)
```

- **100%** — the SAE perfectly preserves the information the model uses for its task.
- **0%** — the SAE is as harmful as zeroing the layer (feature analysis would be moot).

```bash
python -m single.scripts.evaluate_fidelity \
    --ckpt_path ../Training/outputs/tasks/my_task/checkpoints/best \
    --sequences_csv ../Dataset/mBMRB.csv \
    --sae_dir ../Outputs/sp/sae \
    --experiment sp \
    --layer 6 --label_column label --label_map mBMRB
```

Saves `fidelity_results.json` in the experiment's `analysis/`, including a
`reconstruction_mse` sanity check (per-element MSE between SAE reconstruction and
the original hidden states over non-padding residues; low ≈ the reconstruction is
genuinely close). **Note:** injection currently supports the final layer
(`hidden_states[6]` = `emb_layer_norm_after` output); intermediate layers need
extended hooking.

### f1 vs f1_per_domain

- **`f1`** counts **residues**: requires every residue position to be predicted
  correctly — strict, and skewed by domain length.
- **`f1_per_domain`** counts **domain instances**: a domain is "hit" if the feature
  activates anywhere within it — measures whether the feature *recognizes the
  concept*, not its positional precision.

InterPLM uses `f1_per_domain` as the primary concept-association metric. In held-out
reports, prefer pairs with **high `f1_per_domain`** (robust recognition) rather than
raw `f1`.

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
| **Domain-level F1** | `f1_per_domain` counts structural-domain instances hit, not just residues |
| **Percentile thresholds** | InterPLM-style percent-of-max activation thresholds |
| **Held-out validation** | Select on valid split, evaluate on held-out test split (unbiased) |
| **Fidelity (loss recovered)** | Task loss preserved when SAE reconstructions replace layer activations |
| **Per-protein tracking** | Find which proteins maximally activate each feature |
| **Visualization** | Feature activation vs ground truth on sequences |
| **Normalization** | Feature-wise max-activation normalization for meaningful comparisons |

---

## Planned Features

> These functions are **not yet implemented** — placeholders for future work, in
> priority order. They would go in `single/analysis/` alongside the existing tools.

### Structure-vs-Sequence Scatter (planned)

**Purpose:** distinguish whether a feature encodes a *3D structural* property or a
*local sequence motif*. For each feature, compute two effect sizes:

- `sequential Cohen's d` — how far (in sequence) the activated residues are from the
  inactivated ones → whether the feature responds to a linear amino-acid pattern.
- `structural Cohen's d` — whether the activated residues cluster in 3D space
  (AlphaFold / PDB structures) → whether the feature encodes a spatial property.

Plot `x = sequential`, `y = structural`. Features far above the diagonal encode
spatially-clustered biology (e.g. a catalytic pocket), the most interesting
interpretability finding; features near the diagonal are just sequence motifs.

- **Dependencies:** per-protein AlphaFold/PDB structures (`Dataset/uniprot/`),
  residue-level structural alignment.
- **Suggested interface:** `single/scripts/analyze_structure_vs_sequence.py`.

### Causal Intervention (planned)

**Purpose:** move from *correlation* to *causation*. All current analysis shows that
a feature *co-occurs* with a concept/label. Intervention directly perturbs a feature
(e.g. zero or amplify Feature #375's activation) and observes whether the fine-tuned
model's prediction actually changes.

- If zeroing "acidic-region detector" Feature #375 flips the model's flexibility
  prediction on acidic regions → the feature *causally drives* the model's decision.
- Builds on Fidelity's activation-injection mechanism
  (`single/train/fidelity.py`), extended from "whole-layer replacement" to
  "per-feature perturbation" (feature steering / activation patching).

- **Suggested interface:** `single/scripts/steer_features.py` or a
  `single/analysis/intervention.py` module.

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

A single root `requirements.txt` covers both the Training and SAE modules:

```bash
pip install -r requirements.txt
```
Or install the SAE package in development mode (pulls its dependencies from
`setup.py`):
```bash
cd Single && pip install -e .
```

---

## Output Structure

Every step of the pipeline writes into one **experiment directory** (named by
`--experiment <name>`, verbatim with no timestamp; reuse with `--exp_dir`):

```
Outputs/
└── <experiment>/
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

---

## Citation

A manuscript is in preparation. In the meantime, if you use this code in your
research, please cite the repository:

```bibtex
@misc{crossplm,
  author       = {Bingwu, Li and collaborators},
  title        = {CrossPLM: Cross-Task Mechanistic Interpretability for Protein Language Models},
  year         = {2026},
  howpublished = {\url{https://github.com/lbwfff/CrossPLM}},
  note         = {SAE-based interpretability module: {S}parse {A}utoencoder feature extraction, concept alignment, held-out validation, and fidelity evaluation}
}
```

Or cite it as:

> CrossPLM: Cross-Task Mechanistic Interpretability for Protein Language Models.
> GitHub repository, https://github.com/lbwfff/CrossPLM.

Please also consider citing the method this project builds upon (the SAE-based
feature-extraction and concept-alignment approach):

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
