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
- Tick the steps you want (extract → train SAE → analyze → concepts → fidelity → visualize)
- Each step only shows its own parameters; paths like `embeddings_dir` are auto-derived
- Warns when a selected step is missing a required value
- One-click pipeline script download

### 🔀 Crossing Module
- Planned for future implementation

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
   ├── Config auto-saved to Outputs/<task>/config.yaml
   └── Get init → train → eval commands

3. Single Module
   ├── Fill shared pipeline settings once
   ├── Tick desired steps
   └── Get the whole set of SAE analysis commands
```

## Tips

1. **Generate Label Map First**: Create the YAML template before training
2. **Add `csv_data_path` to a label map** to auto-fill the CSV in Training/Single
3. **Number of Classes**: Change the number and class rows update instantly
4. **Pipeline Script**: Download the full pipeline script for automation
5. **Run from the repo root**: `streamlit run gui/app.py` so relative paths resolve correctly