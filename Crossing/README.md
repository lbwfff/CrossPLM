# Crossing Module — Cross-Model Interpretability

> Status: 🚧 **Planned** — not yet implemented.
>
> Back to [project README](../README.md).

The Crossing module extends CrossPLM's interpretability from a single model to
**comparisons across protein language models**. When PLMs are independently
fine-tuned for different biological tasks, do they develop shared,
task-generalizable internal features — and can these shared features be causally
traced across task-specific models?

## Planned Directions

- **Single-model**: neuron / attention-head importance, representation probing.
- **Cross-model**: comparative analysis across PLMs (ESM-2, ProtBERT, Ankh),
  task-specific vs task-common representation separation, feature
  transferability.
- **Structural analysis**: plot `x = sequential Cohen's d` (implemented in the
  [Single module](../Single/README.md#4-sequence-analysis-cohens-d--motif-enrichment))
  against `y = structural Cohen's d`. Features far above the diagonal encode
  spatially clustered biology (e.g. a catalytic pocket). *Requires PDB/AlphaFold
  structures.*

## Roadmap

See the detailed roadmap for the full plan, formalized research questions, and
proposed methods:

- **[ROADMAP.en.md](ROADMAP.en.md)** (English)
- **[ROADMAP.md](ROADMAP.md)** (中文)
