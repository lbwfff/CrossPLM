# CrossPLM

**Mechanistic Interpretability for Cross-Task Protein Language Models**

Protein language models (PLMs) perform well on diverse biological tasks, but
their internal mechanisms are not yet fully understood. CrossPLM is an
interpretability framework that fine-tunes PLMs on per-residue tasks and then
uses **Sparse Autoencoders (SAEs)** to discover human-interpretable features
inside the model — answering *what biology each feature detects* and *whether it
causally drives the model's predictions*.

---

## At a Glance

```
Dataset/   ──►   Training/   ──►   Single/   ──►   Outputs/
 (raw)           fine-tune a PLM    SAE analysis    per-experiment results
                      │                  │
                      └───── share ──────┘
                        Outputs/<exp>/  (one experiment dir)
```

| Module | What it does | Status |
|--------|--------------|--------|
| **[Training](Training/README.md)** | Fine-tune a PLM (e.g. ESM-2) on a per-residue classification task | ✅ Ready |
| **[Single](Single/README.md)** | SAE-based interpretability: extract features, align to labels/concepts, validate | ✅ Ready |
| **[Crossing](Crossing/README.md)** | Cross-model interpretability (compare features across PLMs) | 🚧 Planned |
| **[GUI](gui/README.md)** | Optional Streamlit web UI that builds commands visually | 🚧 In progress |

---

## Quick Start

```bash
pip install -r requirements.txt        # core dependencies
pip install -e .                        # optional: bare `crossplm` command + `single` package
```

All commands go through one entry point at the repo root — no `cd` needed:

```bash
# 1. Fine-tune a PLM on your dataset
python crossplm.py training init   --task_name my_experiment
python crossplm.py training train  --config Outputs/my_experiment/config.yaml

# 2. Analyze the fine-tuned model with SAEs
python crossplm.py single extract_embeddings  --experiment demo ...
python crossplm.py single train_sae           --experiment demo ...
python crossplm.py single analyze_features    --experiment demo ...
```

Prefer a visual interface? Launch the GUI and fill in forms instead of typing flags:

```bash
pip install streamlit pyyaml
streamlit run gui/app.py
```

---

## How It Works

**Training** fine-tunes a HuggingFace PLM (e.g. ESM-2) on a per-residue
token-classification task — for example predicting whether each residue is
*rigid* or *flexible*. A shared **label map** defines how raw dataset
characters map to class IDs, so Training and Single always interpret a dataset
identically.

**Single** then opens up the fine-tuned model with **Sparse Autoencoders**.
An SAE decomposes the PLM's dense hidden states into a sparse, overcomplete set
of features — each of which can be interpreted by checking *when it activates*:

- Does it fire on **flexible** residues? → a "flexibility detector"
- Does it fire on **helices** or **kinase domains**? → captures real biology
- Is it causally necessary? → perturb it and watch predictions change

The pipeline goes: **extract embeddings → train SAE → align features to task
labels / biological concepts → validate (held-out, fidelity, causal
intervention) → visualize.**

See each module's README for command-level details:

- **[Training/README.md](Training/README.md)** — init/train/eval, config fields, label maps, checkpoints
- **[Single/README.md](Single/README.md)** — the full SAE pipeline, parameters, output layout, validation
- **[Crossing/README.md](Crossing/README.md)** — cross-model roadmap
- **[gui/README.md](gui/README.md)** — the Streamlit command builder

---

## Repository Layout

```
CrossPLM/
├── crossplm.py            # Unified CLI entry point (training | single | crossing)
├── setup.py               # Optional: pip install -e . -> `crossplm` command
├── requirements.txt
├── Dataset/               # Raw datasets + label-map YAMLs (user-supplied, gitignored)
├── Training/              # PLM fine-tuning framework   → README.md
├── Single/                # SAE interpretability module  → README.md
├── Crossing/              # Cross-model analysis (planned) → README.md
├── Outputs/               # Shared experiment outputs (config, checkpoints, SAE, analysis)
└── gui/                   # Optional Streamlit web UI     → README.md
```

`Outputs/<experiment>/` is the **shared root** for both Training and Single:
Training writes the config and checkpoints there, Single writes embeddings /
SAE / analysis there. One experiment name = one directory (verbatim, no
timestamp).

---

## Dependencies

```bash
pip install -r requirements.txt
```

Core: `torch`, `transformers`, `pandas`, `numpy`, `scipy`, `scikit-learn`,
`matplotlib`, `PyYAML`, `tqdm`.

---

## Citation

A manuscript is in preparation. In the meantime, if you use this code, please
cite the repository:

```bibtex
@misc{crossplm,
  author       = {Bingwu, Li and collaborators},
  title        = {CrossPLM: Cross-Task Mechanistic Interpretability for Protein Language Models},
  year         = {2026},
  howpublished = {\url{https://github.com/lbwfff/CrossPLM}}
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
