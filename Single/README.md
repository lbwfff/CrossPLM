# Single Module — SAE Interpretability

After fine-tuning a PLM (see the [Training module](../Training/README.md)) — or directly from the pre-trained `M0` (`facebook/esm2_t6_8M_UR50D` / `Synthyra/ESM2-8M`, `trust_remote_code=True` handled) for the `Phase 0` baseline — use **Sparse Autoencoders (SAEs)** to discover which hidden-state features drive the model's predictions, and interpret them against either the task labels or Swiss-Prot biological concepts.

> **M0 baseline** (`Crossing/ROADMAP.md: Phase 0.1/0.2`): extract with `--model_type base --ckpt_path <hub-id>` (e.g. `facebook/esm2_t6_8M_UR50D`). The weights are cached once in `Outputs/_pretrained/<hub-slug>/` so `MA`/`MB` sharing the same backbone do not duplicate `M0`. See `Shared Conventions` below.

> Back to [project README](../README.md).

---

## Pipeline

```
Fine-tuned PLM checkpoint
         ↓
[1] extract_embeddings     → Extract per-residue hidden states from a target layer
         ↓
[2] train_sae              → Train a Sparse Autoencoder to learn interpretable features
         ↓
[3] analyze_features       → (A) Align features with task labels   (rigid/flexible)
[4] analyze_concepts       → (B) Align features with biological concepts (Swiss-Prot)
         ↓
[5] heldout / fidelity /   → (C) Validate the findings (unbiased, faithful, causal)
    evaluate_intervention
         ↓
[6] analyze_sequence       → (D) Characterize features along the sequence
                             (Cohen's d + motif enrichment)
         ↓
[7] analyze_coactivation   → (E) Compare pairs of features (co-localized vs disjoint)
         ↓
[8] visualize_features     → Plot feature activation patterns on protein sequences
```

All commands go through `python crossplm.py single <command> ...` from the repo
root.

---

## How SAEs Work

SAEs learn a sparse, overcomplete decomposition of PLM hidden states:

```
ESM hidden state (320-dim)  →  SAE encoder  →  sparse feature vector (640-dim)
                                                   ↑ only ~60 features active per token
```

Each feature can then be interpreted by checking **when** it activates:
- Does it activate preferentially on **flexible** residues? → "flexibility detector"
- Does it activate on **rigid** residues? → "rigidity detector"
- Is it unrelated to the task? → noise feature

---

## Shared Conventions

### Experiment directory & data source

Every step routes outputs into one experiment directory
`Outputs/<experiment>/` (the name is used verbatim, no timestamp) — the
**same root the Training module uses**. The optional `--source <id>` flag (the
input dataset, e.g. `mbmrb` / `swissprot`) nests data-specific dirs under
`Outputs/<experiment>/<source>/`, so different datasets can share one experiment
without overwriting each other. Without `--source`, everything lives flat under
`Outputs/<experiment>/` (legacy). The **SAE is shared** at the experiment root
and reused across sources.

```
Outputs/
├── _pretrained/<hub-slug>/   # ONE central copy of the raw M0 weights per backbone
│   ├── config.json / model.safetensors / tokenizer_config.json / vocab.txt
│   └── m0_provenance.json
└── <experiment>/
    ├── sae/model.pt            # ONE shared SAE (reused across sources)
    ├── sae/model_normalized.pt # per-feature max-activation rescale (see below)
    └── <source>/               # e.g. mbmrb | swissprot  (--source <id>)
        ├── embeddings/layer_<N>/shard_<i>/embeddings.pt
        ├── concepts/shard_<i>/concept_matrix.npz
        └── analysis/*.csv|*.json|visualizations/
```

`single/paths.py` centralizes `Outputs/<experiment>/` paths; `--sae_dir` / `--embeddings_dir` / `--output_dir` / `--concepts_dir` default into the experiment dir and can be overridden; `--exp_dir <path>` points at an existing dir verbatim. `--model_type base` (M0) vs `ft` (fine-tuned) controls whether `ckpt_path` is a Hub ID or a local `checkpoints/best`.

### Feature normalization

`analyze_features` computes each feature's max activation and saves a
normalized copy `sae/model_normalized.pt`. All analysis scripts load the SAE via
`load_sae`, which **auto-prefers `model_normalized.pt` when present**, putting
every feature on a comparable 0–1 scale (so `--threshold_percents 0.15` means
"activation > 15% of that feature's max"). On the **first** `analyze_features`
run — before `model_normalized.pt` exists — `load_sae` falls back to the raw
`model.pt`. Re-running `train_sae` removes any stale `model_normalized.pt`.

### Filter / subset consistency

Scripts that read the sequences CSV (`extract_embeddings`, `analyze_features`,
`analyze_sequence`, `analyze_coactivation`, `visualize_features`) accept
`--min_seq_len` / `--max_seq_len` / `--max_sequences`. If you used any of these
during `extract_embeddings`, pass the **same values** here too — otherwise
tokens misalign and the script **fails loudly** instead of producing garbage.
`--max_sequences N` draws a deterministic subset (fixed seed, reproducible).

Rows where `len(sequence) != len(label)` are **dropped** (with a `[Filter]` count) to match `Training`'s silent drop (`mBMRB.csv` `9786→9554`); the count is written to `embeddings/layer_N/metadata.json` (`n_dropped_mismatched`).

### Label maps

Label encoding is not hardcoded to mBMRB. Every script accepts
`--label_map <preset>` (from `single/label_maps.py`) or a YAML file. See the
[Training module README](../Training/README.md#0-data--label-map) for the
preset table and YAML format.

---

## 1. Task-Label Alignment

Align each SAE feature with the task labels (e.g. rigid/flexible) to find
"flexibility detectors". This step requires a fine-tuned checkpoint (`--model_type ft`, the default); for the `M0` baseline skip task alignment and go to concepts (`§2`), evaluated by `L0` (recon/`L0`/`dead_pct` from `train_sae`) + `L1` (concept F1).

```bash
# Extract hidden states from the fine-tuned model (creates Outputs/demo/mbmrb/)
# For M0 use: --ckpt_path facebook/esm2_t6_8M_UR50D --model_type base
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

`--reconstruction_loss l2` (default) uses the legacy unsquared L2 objective;
`mse` switches to mean squared error (may require retuning `--l1_penalty`).

**Tuning tips for `train_sae`:**
- `--l1_penalty` controls sparsity (~0.06–0.1); higher → sparser features (lower `l0`) but higher reconstruction loss.
- `--resample_steps N` periodically revives "dead" features, lowering `dead_pct`.
- Target: **`l0` 20–80, `dead_pct` < 30%, `recon_loss` as low as possible**.

**Metrics per feature:** Precision, Recall, F1, AUROC (0.5=random, 1.0=perfect),
Activation Gap.

**Example output:**
```
Top features for 'flexible' (label=1):
  Feature #234: F1=0.514, AUROC=0.713, Prec=0.923, Rec=0.356
  Feature #426: F1=0.517, AUROC=0.693, Prec=0.729, Rec=0.401
```

---

## 2. Biological Concept Analysis (Swiss-Prot / UniProtKB)

Align SAE features against **biological concepts** — e.g. "Helix",
"Domain_kinase", "Binding_site_ATP" — to discover what real biology each
feature encodes.

### Step 1: Build per-residue concept matrices from a UniProtKB TSV export

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

> **Alignment requirement:** concept shards must contain the **same proteins,
> in the same order, sharded identically** as the embedding shards. Extract
> embeddings from the **same TSV** with the same `--n_shards` and
> `--min_seq_len` / `--max_seq_len`, using `--sequence_column Sequence`.
> `--max_residues` (`max_length` − 2) makes concept rows cover exactly the
> residues the embedder keeps. If token counts mismatch, `align` **fails with
> an error** instead of silently truncating.

### Step 2: Extract embeddings from the SAME TSV

```bash
python crossplm.py single extract_embeddings \
    --ckpt_path Outputs/my_task/checkpoints/best \
    --sequences_csv Dataset/uniprotkb_swissprot.tsv \
    --sequence_column Sequence \
    --experiment demo --source swissprot \
    --n_shards 5 --min_seq_len 30 --max_seq_len 1022
```

> No `--label_column` here — UniProt has no per-residue task labels.
> **No second SAE needed** — the SAE trained above (on mBMRB) is reused, so
> concept alignment measures whether mBMRB-learned features capture Swiss-Prot
> biology.

### Step 3: Align SAE features to concepts

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
  Feature #42  → Domain_kinase    F1=0.623 AUROC=0.781 P=0.710 R=0.551
  Feature #107 → Helix            F1=0.588 AUROC=0.742 P=0.650 R=0.537
```

**Residue vs domain metrics:** `f1`/`precision`/`recall` are residue-level.
`domain_precision`/`domain_recall`/`domain_f1` use one-to-one matching between
contiguous predicted activation segments and annotated domain instances.

---

## 3. Validation & Fidelity

Two checks that make the feature findings trustworthy.

### Held-out validation

Removes selection bias — reporting the "best feature per concept" on the *same*
data used for selection is optimistically biased.

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

### Fidelity (loss recovered)

Validates that the SAE faithfully represents the model's task-relevant
activations. The target layer's hidden states are replaced three ways and the
model's **task loss** is compared:

```
ce_orig : original activations
ce_sae  : SAE reconstructions injected
ce_zero : layer zeroed (zero-ablation baseline)

Loss_Recovered = 1 - (ce_sae - ce_orig) / (ce_zero - ce_orig)
```

100% = perfectly preserves task info; 0% = as harmful as zero-ablation. If zero
ablation does not increase loss, recovery is reported as invalid. When the SAE
reconstruction lowers loss *below* original (`sae_better_than_original=true`),
the raw value exceeds 100% (denoising, not just preserving).

```bash
python crossplm.py single evaluate_fidelity \
    --ckpt_path Outputs/my_task/checkpoints/best \
    --sequences_csv Dataset/mBMRB.csv \
    --experiment demo --source mbmrb \
    --layer 6 --label_column label --label_map mBMRB \
    --max_sequences 200
```

Saves `fidelity_results.json` incl. a `reconstruction_mse` sanity check.
`--max_sequences` limits to a subset for a quick check — drop it for the full
dataset.

> **Note:** injection currently supports the final layer only
> (`hidden_states[6]` = `emb_layer_norm_after` output).

### Causal intervention (feature steering)

Moves from *correlation* to *causation*. Whereas Fidelity replaces the whole
layer, intervention perturbs a **single SAE feature** and measures whether the
model's per-residue predictions change.

```bash
python crossplm.py single evaluate_intervention \
    --ckpt_path Outputs/my_task/checkpoints/best \
    --sequences_csv Dataset/mBMRB.csv \
    --experiment demo --source mbmrb \
    --feature_idx 375 --mode zero \
    --label_column label --label_map mBMRB \
    --layer 6 --max_sequences 200
```

- `--feature_idx N` — the feature to perturb.
- `--mode zero|amplify|set` — set to 0, scale up (`--scale`), or force a value.
- Outputs `intervention_feat<N>_<mode>.json` with **flip-rate** metrics:
  - `flip_rate_on_active` — how often predictions change **on tokens where the feature fires** (causal effect).
  - `flip_rate_on_inactive` — **control baseline** on tokens where the feature does NOT fire. If `active >> inactive`, the effect is real; if similar, it's noise.

*Note: single-feature effects are typically a few % (each token uses ~60
features, so one feature tips only borderline samples). Compare against the
inactive control rather than reading the absolute value.*

---

## 4. Sequence Analysis (Cohen's d + Motif Enrichment)

Characterizes *what along the sequence* a feature responds to, using only
sequence data (no 3D structures required):

- **Sequential Cohen's d** — are the feature's activated residues **clustered** along the sequence (local/motif-like) or **dispersed** (global/periodic)? Negative d = clustered, ~0 = random, positive = dispersed.
- **Motif enrichment** — which amino acids are over-represented in a window (`--flank`, default 5) around the activated residues — the amino-acid "signature" of the feature.
- Positional motif analysis keeps each relative position separate, uses a within-protein permutation null, and saves p-values, BH-FDR q-values, and `sequence_logo_feature<N>.png`.

```bash
python crossplm.py single analyze_sequence \
    --embeddings_dir Outputs/demo/mbmrb/embeddings/layer_6 \
    --sequences_csv Dataset/mBMRB.csv \
    --experiment demo --source mbmrb \
    --label_map mBMRB \
    --feature_indices 375 42
```

By default all numeric shards are aggregated. Use `--shard 0` for a quick
single-shard test. The pooled result is saved as `sequence_analysis.json`;
single-shard runs use `sequence_analysis_shard<N>.json`.

**Example output** (Feature #42, the "flexibility detector"):
```
Sequential Cohen's d: -0.074  → ~random
Top enriched amino acids (log2 fold):
  P: +0.40   S: +0.37   G: +0.32   D: +0.26   C: +0.12
```
P/S/G/D are classic flexible/loop residues — consistent with a flexibility
detector.

---

## 5. Pairwise Feature Co-Activation

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

Key metrics (each compared to the appropriate null):
- `overlap_ab` / `enrich_ab` — same-residue co-activation vs the unconditional activation-rate baseline; **>1** → B enriched on A's residues, **<1** → mutually exclusive at the same position.
- `neighbor_ab` / `neighbor_enrich_ab` — B active within ±`--neighborhood` residues (excluding the same residue) around an A-active residue. `neighbor_enrich_ab` is normalized by the **independence null** (expected window-hit probability `1-(1-p)^(2k)` for independent features, boundary-aware at protein ends), NOT by the per-token rate. **>>1** → co-localization beyond independence.
- Both directions (A→B and B→A) are reported to detect asymmetry.

**Example** (Features #375 and #42): same-residue enrichment **0.67×**
(mutually exclusive at identical positions) but neighborhood enrichment
**≈ 0.87–0.89× vs the independence null** → largely independent. Saves
`coactivation_<a>_<b>_shard<N>.json`.

---

## 6. Visualize Features

Plot feature activation patterns on protein sequences.

```bash
python crossplm.py single visualize_features \
    --experiment demo --source mbmrb \
    --sequences_csv Dataset/mBMRB.csv \
    --feature_indices 234 426 --label_map mBMRB
# Single protein (exact sequence, shard auto-corrected):
python crossplm.py single visualize_features \
    --experiment demo --source mbmrb \
    --sequences_csv Dataset/mBMRB.csv \
    --feature_indices 234 --filter_sequence GSIPCLLSPWSEWSDCSVTCGKGMRTRQRMLKSLAELGDCNEDLEQAEKCMLPECP
```

`--embeddings_path` and `--output_dir` are inferred from `--experiment` /
`--layer` / `--shard`. Leave `--feature_indices` empty to auto-select the top
`--n_features` (default 10) features. `--filter_sequence` visualizes one exact
protein (case-insensitive, `strip+upper`); the shard is auto-corrected and the
embeddings fallback `source-nested → flat` (`Outputs/<exp>/embeddings/...`) is handled automatically.

---

## Output Structure

```
Outputs/
├── _pretrained/<hub-slug>/            # ONE central M0 per backbone (e.g. facebook--esm2_t6_8M_UR50D)
│   ├── config.json / model.safetensors / tokenizer_config.json / vocab.txt
│   └── m0_provenance.json
└── <experiment>/
    ├── config.yaml / config_snapshot.yaml / provenance.json  # Training
    ├── training_curve.png                         # Training curve
    ├── eval_metrics.jsonl                         # Training eval history
    ├── checkpoints/                               # Trained PLM checkpoints (MA/MB)
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
            ├── feature_label_metrics.json         # Task-label alignment (ft only; M0 uses L0+L1)
            ├── feature_label_correlations.npy     # Point-biserial r per feature
            ├── feature_label_correlation_stats.json # r, p-value, BH q-value/FDR
            ├── activation_profile.npz             # Per-class mean/max activation
            ├── max_activations_per_feature.pt     # Used to build model_normalized.pt
            ├── feature_concept_pairs.csv          # Feature × concept alignment (M0 L1)
            ├── heldout_*.csv                      # Held-out validation
            ├── fidelity_results.json              # Fidelity (ft only, final layer currently)
            ├── intervention_feat<N>_<mode>.json   # Causal intervention (ft only)
            ├── sequence_analysis.json             # Pooled Cohen's d + motif
            ├── sequence_logo_feature<N>.png       # Positional motif logo
            ├── coactivation_<a>_<b>.json          # Pooled pairwise co-activation
            └── visualizations/                    # PNG plots
```

M0 is evaluated by `L0` (recon/`L0`/`dead%` from `train_sae`) + `L1` (concept alignment), not by task-label fidelity.

---

## Layout

```
Single/single/
├── configs.py           # SAEConfig / TrainingConfig / DataConfig dataclasses
├── label_maps.py        # Configurable label encoding (shared with Training)
├── paths.py             # Centralized experiment output paths
├── data.py              # Shared CSV loading, sharding, sequence hashing
├── embedders/           # Hidden state extraction from fine-tuned PLMs
├── sae/                 # SAE architectures (ReLUSAE, TopKSAE) + inference
├── train/               # SAE training loop
├── analysis/            # Feature-to-label & feature-to-concept alignment
└── scripts/             # CLI scripts (`crossplm single` delegation)
    ├── extract_embeddings.py
    ├── train_sae.py
    ├── analyze_features.py
    ├── analyze_concepts.py
    ├── analyze_sequence.py
    ├── analyze_coactivation.py
    ├── evaluate_fidelity.py
    ├── evaluate_intervention.py
    └── visualize_features.py
```
