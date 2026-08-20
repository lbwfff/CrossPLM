# CrossPLM GUI

Interactive Streamlit-based command builder for CrossPLM. Build CLI commands without typing verbose parameters.

## Installation

```bash
pip install streamlit pyyaml
```

## Usage

From the repository root:

```bash
streamlit run gui/app.py
```

The web interface will open in your browser (default: http://localhost:8501).

## Features

### 📝 Label Map Generator
- Fill in dataset configuration (columns, class mappings, ignore characters)
- Optional CSV Data Path stored in the YAML (auto-filled by Training/Single)
- Number of Classes changes dynamically — class input rows auto add/remove
- Auto-saves to `Dataset/<name>.yaml` + download button
- Validates class IDs are contiguous from 0 and positive class is valid

### 🏋️ Training Module (Unified)
- Fill in experiment info once; label map auto-supplies columns & CSV path
- Supports both native `facebook/esm2_t6_8M_UR50D` and `Synthyra/ESM2-8M` (FastPLMs) backbones
- Advanced options: gradient accumulation, weight decay, class weights, dataloader workers, FP16/BF16, and Phase 0 `freeze_backbone` / `freeze_layers` controls
- Auto-saves config to `Outputs/<task>/config.yaml` (plus `provenance.json` / `config_snapshot.yaml` for reproducibility)
- Generates init / train / eval commands (+ optional eval with its own label map & batch size)
- One-click pipeline script download

### 🔬 Single Module (SAE Pipeline)
- **Pipeline-oriented**: fill shared settings once (experiment/source/layer/label map/ckpt/csv)
- `--model_type` toggle: `ft` for fine-tuned `MA/MB` (`checkpoints/best`) vs `base` for `M0` (`facebook`/`Synthyra` Hub ID, central `Outputs/_pretrained/` cache)
- Tick the steps you want (extract → train SAE → analyze → concepts → fidelity → visualize)
- Each step only shows its own parameters; paths like `embeddings_dir` are auto-derived with flat fallback (`source` nested → flat)
- `visualize` supports `--filter_sequence` (single exact protein, shard auto-corrected; default `max_proteins=3`)
- Warns when a selected step is missing a required value
- One-click pipeline script download

### 🔀 Crossing Module
- **Feature Alignment** (Phase 1): `compute_feature_similarity` — activation/cosine, CKA, MI, semantic `S_cross`, controls, heatmap
- **Cross-task Information** (Phase 2): `cross_task_probe` (2×2 transfer matrix) + `classify_features` (Shared / A-specific / B-specific)
- Quick-fill from existing `Outputs/<exp>` for SAE/embeddings/concepts; auto-inferred paths shown inline

## Structure

```
gui/
├── app.py          # Main Streamlit application (all modules)
├── requirements.txt # GUI dependencies
└── README.md       # This file
```

## Design Principles

- **Independent Component**: Does not modify any project code
- **External Module**: Lives in `gui/` directory, separate from core code
- **Command Generation**: Generates commands for manual execution
- **No Execution**: Does not execute commands directly (safety first)

## User Flow

```
1. Generate Label Map
   └── Fill config → Generate → Auto-save to Dataset/<name>.yaml

2. Training Module
   ├── Fill experiment config (label map auto-fills columns & CSV)
   ├── Advanced: grad accum / weight decay / class weights / FP16/BF16 / freeze (Phase 0)
   ├── Config auto-saved to Outputs/<task>/config.yaml (+ provenance.json)
   └── Get init → train → eval commands

3. Single Module
   ├── Fill shared pipeline settings once (model_type base for M0 → Outputs/_pretrained/ vs ft for MA/MB)
   ├── Tick desired steps
   └── Get the whole set of SAE analysis commands

4. Crossing Module
   ├── Quick-fill A/B from existing Outputs/<exp> (SAE/embeddings/concepts)
   ├── Pick Phase 1 (similarity) / Phase 2 (probe + classification)
   └── Get cross-model commands
```

## Tips

1. **Generate Label Map First**: Create the YAML template before training
2. **Add `csv_data_path` to a label map** to auto-fill the CSV in Training/Single
3. **Number of Classes**: Change the number and class rows update instantly
4. **Pipeline Script**: Download the full pipeline script for automation
5. **Run from the repo root**: `streamlit run gui/app.py` so relative paths resolve correctly