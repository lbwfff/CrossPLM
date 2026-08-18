# Crossing Module Roadmap

## Project Background

The Crossing module aims to enable cross-protein-language-model (PLM) mechanistic interpretability: by comparing feature representations across models (e.g. ESM-2, ProtBERT, Ankh), we reveal cross-task, cross-model mechanistic understanding.

### Core Scientific Question

> **When protein language models are independently fine-tuned for different biological tasks, do they develop shared, task-generalizable internal features, and can these shared features be causally traced across task-specific models?**

### Core Aims

1. **Aim 1**: Identify shared and task-specific SAE features across independently fine-tuned pLMs.
2. **Aim 2**: Quantify whether shared features encode transferable information across biological tasks.
3. **Aim 3**: Establish causal relationships between cross-task features through activation intervention and patching.
4. **Aim 4**: Construct a shared/private mechanistic representation and discover cross-task biological circuits.

## Current Status

- **Training module**: PLM fine-tuning framework complete.
- **Single module**: SAE-based single-model interpretability complete.
- **Crossing module**: To be implemented (this roadmap).

## Formalizing the Research Questions

Assume we have:
- Base protein language model: $M_0$
- Task A fine-tuned model: $M_A$
- Task B fine-tuned model: $M_B$

For example:
- A = stability prediction
- B = subcellular localization

### Three Increasingly Deep Levels of Questions

#### Q1. Representation Level
**Do the two tasks re-use the same biological concepts?**

Looking for: $f_i^A \leftrightarrow f_j^B$

Example: Task A learned a "membrane-associated region" feature, and Task B also has a very similar feature → **shared representation**

#### Q2. Functional Level
**Is Task A's learned feature functionally related to Task B's prediction mechanism?**

For example: $f_{A:membrane} \rightarrow y_B$

Possible outcomes:
- A feature correlates with B output
- A feature activation can predict Task B
- A feature has a counterpart inside model B → **cross-task functional relationship**

#### Q3. Causal Level
**If we artificially manipulate a Task A feature, does Task B's behavior change?**

For example: $do(f_i^A \uparrow) \Rightarrow y_B ?$

Or further: $f_i^A \rightarrow f_j^B \rightarrow y_B$

If this relationship can be proven, the paper's story upgrades from "both models contain similar features" to **"protein models for different tasks form a traceable, intervenable, shared biological computation mechanism."**

## The 6-Level Ladder

### Level 0 — Single-model interpretability
**Completed**: $M_A \rightarrow SAE_A$, $M_B \rightarrow SAE_B$

### Level 1 — Cross-model representation similarity
**Question**: Do the two models have similar features?
$F_A \leftrightarrow F_B$

### Level 2 — Cross-task information
**Question**: Does one task's representation contain information about the other task?
$F_A \rightarrow Y_B$, $F_B \rightarrow Y_A$

### Level 3 — Cross-model feature transfer
**Question**: Can A's features drive B?
$F_A \rightarrow M_B$

### Level 4 — Cross-task causal intervention
**Question**: Does an A feature causally participate in B?
$do(F_A) \rightarrow F_B \rightarrow Y_B$

### Level 5 — Cross-task circuit
**Final goal**: $F_A \rightarrow F_{shared} \rightarrow F_B \rightarrow Y_B$

---

## Detailed Implementation Phases

### Phase 0: Establish the Baseline (high priority)

**Goal**: Build the most important baseline to strengthen explanatory power.

#### 0.1 Base Model SAE
- **Method**: Train $M_0 + SAE_0$ (the foundation model without task-specific fine-tuning).
- **Output**: Three SAEs ($SAE_A$, $SAE_B$, $SAE_0$).

#### 0.2 Feature Origin Tracing
- **Goal**: When a feature is observed, trace where it comes from.
- **Three cases**:
  - Case 1: $M_0 \rightarrow feature$, inherited by both A and B → **pre-existing knowledge**
  - Case 2: $M_A \rightarrow feature$, $M_B \rightarrow feature$, but not $M_0$ → **shared emergent knowledge**
  - Case 3: Only $M_A \rightarrow feature$ → **task-specific knowledge**

#### 0.3 Multiple Fine-tuning Strengths (recommended)
- **Method**: Train models at different fine-tuning strengths
  - $M_0 \rightarrow M_A^{10\%} \rightarrow M_A^{30\%} \rightarrow M_A^{60\%} \rightarrow M_A^{100\%}$
  - Same for B.
- **Goal**: Study when cross-task knowledge forms.
- **Key questions**: Is the shared representation inherited from the pretrained model, an early fine-tuning phenomenon, late specialization, or re-formed after catastrophic forgetting?

#### 0.4 Four-Model Design (recommended)
| Model | Training |
|-------|----------|
| M0 | pretrained |
| MA | fine-tuned on A |
| MB | fine-tuned on B |
| MAB | jointly fine-tuned on A+B |

**Key question**: Is the shared knowledge of two independently learned tasks the same as that produced by joint learning?

**Feasibility**: ✅ Simple technique, quick to validate.
**Output**: Three SAEs, feature-origin analysis, fine-tuning-strength analysis.

---

### Phase 1: Feature-level Cross Alignment (high priority)

**Goal**: Build a multi-dimensional correspondence between the two models' feature spaces.

#### 1.1 Activation Similarity
- **Method**: Run both models on the same set of protein sequences and compute feature-activation correlations.
- **Metric**: $corr(a_i^A, a_j^B)$, yielding an $N_A \times N_B$ feature similarity matrix.
- **Caution**: Do not treat correlation as semantic equivalence.

#### 1.2 Biological Semantic Similarity (key)
- **Method**: Leverage existing SAE interpretation to annotate each feature with biological annotations.
- **Example**:
  ```
  Feature A127 → membrane → hydrophobic residues → transmembrane region
  Feature B483 → membrane localization → transmembrane
  ```
- **Combined similarity**: $S_{cross} = \alpha S_{activation} + \beta S_{semantic}$

#### 1.3 Feature × Feature Correspondence Heatmap
- **Visualization**: x-axis = Task A features, y-axis = Task B features, color = cross-feature similarity.
- **Clustering**: Identify feature communities.

#### 1.4 Controls (important)
Required controls:
- sequence identity
- protein length
- amino acid composition
- protein family
- secondary structure
- random feature pairing
- randomized labels
- unrelated task control

**Key principle**: $\text{shared activation} \neq \text{shared biology}$

**Feasibility**: ✅ Mature methods, directly implementable.
**Output**: Multi-dimensional similarity matrices, correspondence heatmap, feature communities.

---

### Phase 2: Cross-task Information (medium priority)

**Goal**: Quantify cross-task information transfer.

#### 2.1 Cross-task Probe
- **Method**: Train a simple probe on Task A's SAE features to predict Task B
  - $\hat{y}_B = g(f_1^A, ..., f_n^A)$
- **Bidirectional validation**:
  - A SAE → Task A (baseline)
  - A SAE → Task B (cross-task)
  - B SAE → Task B (baseline)
  - B SAE → Task A (cross-task)

#### 2.2 Shared vs Private Feature Classification
- **Goal**: Split all features into three classes
  - Task A-specific ($F_A^{private}$): predicts only A
  - Task B-specific ($F_B^{private}$): predicts only B
  - Shared ($F^{shared}$): predicts both A and B
- **Visualization**: A candidate core figure for the paper.

#### 2.3 Information Transfer Quantification
- **Metrics**:
  - Positive transfer: $A\uparrow, B\uparrow$
  - Negative transfer: $A\uparrow, B\downarrow$
  - Competitive representation: $A\uparrow, B\downarrow$ (competing for representation capacity)

**Feasibility**: ✅ Probe methods are mature, quick to implement.
**Output**: Cross-task probe performance table, shared/private feature classification, information-transfer quantification.

---

### Phase 3: Cross-task Feature Transfer (medium priority)

**Goal**: Validate whether features can drive across models.

#### 3.1 Feature-level Intervention
- **Method**: Find $f_i^A \leftrightarrow f_j^B$, then intervene on an A feature
  - $f_i^A \rightarrow f_i^A + \Delta$
- **Observe**: Both $\Delta y_A$ and $\Delta y_B$.
- **Key point**: If $\Delta y_A \neq 0$ and $\Delta y_B \neq 0$, manipulating A's mechanistic feature also affects Task B.

#### 3.2 Activation Ablation
- **Method**: $do(A_i = 0)$, observe $Y_A$, $Y_B$.

#### 3.3 Activation Boosting
- **Method**: $do(A_i = c)$, observe Task B's response.

#### 3.4 Feature Swapping
- **Method**: Replace $z_i^A$ with $z_i^B$, observe target output.

**Feasibility**: ⚠️ Requires modifying the model's forward pass; medium technical complexity.
**Output**: Intervention-effect quantification, cross-task influence metrics.

---

### Phase 4: Cross-model Activation Patching (high priority, core contribution)

**Goal**: Implement cross-model activation patching to build causal evidence.

#### 4.1 Raw Activation Patching
- **Method**: $h_L^B \leftarrow h_L^A$ (inject Task A's layer information into Task B's model).
- **Observe**: Change in B's output.
- **Caution**: If the two backbones differ, raw activations may not share the same representation basis.

#### 4.2 Cross-Model SAE / Crosscoder (recommended)
- **Goal**: Learn $Z_{shared}$ such that:
  - $h_A \rightarrow Z_{shared} \rightarrow h_A$
  - $h_B \rightarrow Z_{shared} \rightarrow h_B$
- **Three versions** (in increasing difficulty):
  - Version 1: Post-hoc feature matching (easiest)
  - Version 2: Shared dictionary (reconstructs A/B simultaneously)
  - Version 3: Shared + private latent (most elegant)

#### 4.3 Shared Knowledge Ratio
- **Definition**: $SKR = \frac{I(Z_{shared}; Y_A, Y_B)}{I(Z_{all}; Y_A, Y_B)}$
- **Implementation**: Mutual information can be replaced by a more stable probe-based metric.

**Feasibility**: ⚠️ Larger engineering effort, but likely the project's core methodological contribution.
**Output**: Crosscoder architecture, shared/private latent space, SKR metric.

---

### Phase 5: Cross-task Causal Intervention (medium-low priority)

**Goal**: Build a causal relationship network.

#### 5.1 Path Patching
- **Method**: Validate the $A_i \rightarrow B_j \rightarrow Y_B$ path
  - A_i ON → B_j ON → Y_B
  - A_i ON → B_j OFF → Y_B
- **Key point**: If the effect disappears in the second case, that is mediation evidence.

#### 5.2 Negative Transfer Analysis
- **Question**: Does A knowledge **harm** B?
- **Analysis**:
  - Boosting $f_A$: $Performance_A \uparrow$ but $Performance_B \downarrow$ → **cross-task interference**
  - Or $A\uparrow, B\uparrow$ → **positive transfer**

#### 5.3 Cross-task Feature Interaction Matrix
- **Table**: Feature | A effect | B effect | Interpretation
- **Network**: $\text{feature}_i \rightarrow \text{feature}_j$ network.
- **Goal**: Move from "feature analysis" to **cross-task mechanistic circuit discovery**.

**Feasibility**: ⚠️ Requires causal-inference expertise.
**Output**: Causal graph, feature interaction matrix, circuit diagram.

---

### Phase 6: Biological Mapping and MD Validation (low priority)

**Goal**: Map computational findings to physical space and validate biological mechanisms via simulation.

#### 6.1 Biological Element Mapping
- **Method**: Map SAE features with significant causal effects to physical space.
- **Mapping targets**: PDB domains, conserved motifs, charge distributions, active sites.

#### 6.2 Mutation Validation
- **Method**: Design cross-task "optimizing/antagonistic" mutations based on causal features.
- **Validation**: Use molecular dynamics (MD) simulation to validate the biological mechanism.

**Feasibility**: ⚠️ Requires interdisciplinary collaboration and computational resources.
**Output**: Feature-structure correspondence, mutation designs, MD validation report.

---

## Technical Architecture Proposal

### Module Structure
```
Crossing/
├── ROADMAP.en.md           # This file
├── crossing/               # Python package
│   ├── __init__.py
│   ├── configs.py          # Configuration dataclasses
│   ├── baseline/           # Phase 0: baseline establishment
│   │   ├── base_sae.py     # M0 + SAE0 training
│   │   ├── feature_origin.py # Feature-origin analysis
│   │   └── checkpoints.py  # Multiple fine-tuning-strength experiments
│   ├── alignment/          # Phase 1: feature alignment
│   │   ├── activation.py   # Activation similarity
│   │   ├── semantic.py     # Biological semantic similarity
│   │   ├── heatmap.py      # Correspondence heatmap
│   │   └── controls.py     # Controls design
│   ├── information/        # Phase 2: cross-task information
│   │   ├── probe.py        # Cross-task probe
│   │   ├── shared.py       # Shared vs private features
│   │   └── transfer.py     # Information transfer
│   ├── intervention/       # Phase 3: feature transfer
│   │   ├── feature_intervention.py
│   │   ├── ablation.py
│   │   └── swapping.py
│   ├── patching/           # Phase 4: activation patching
│   │   ├── raw_patching.py
│   │   ├── crosscoder.py   # Cross-Model SAE
│   │   └── shared_latent.py
│   ├── causal/             # Phase 5: causal network
│   │   ├── path_patching.py
│   │   ├── negative_transfer.py
│   │   └── interaction_matrix.py
│   ├── biology/            # Phase 6: biological mapping
│   │   ├── mapping.py
│   │   └── validation.py
│   └── scripts/            # CLI scripts
│       ├── build_baseline.py
│       ├── align_features.py
│       ├── cross_task_probe.py
│       ├── feature_intervention.py
│       ├── cross_model_patching.py
│       ├── causal_analysis.py
│       └── biology_mapping.py
└── examples/               # Examples and tutorials
```

### Integration with Existing Modules
1. **Reuse the Single module**:
   - SAE feature extraction (`single/sae/`)
   - Embedding extraction (`single/embedders/`)
   - Analysis tools (`single/analysis/`)

2. **Shared output directory**:
   - `Outputs/<experiment>/crossing/` stores cross-model analysis results.

3. **Unified CLI**:
   - `crossplm crossing baseline` - establish baseline
   - `crossplm crossing align` - feature alignment
   - `crossplm crossing probe` - cross-task probe
   - `crossplm crossing intervene` - feature intervention
   - `crossplm crossing patch` - activation patching
   - `crossplm crossing causal` - causal analysis
   - `crossplm crossing map` - biological mapping

## Implementation Priority

### Stage 1: Fastest path to first results
- M0 / MA / MB → three SAEs → feature matching → shared / A-specific / B-specific

### Stage 2: The "crossing" really begins
- feature → cross-task probe → A→B / B→A information transfer → positive / negative transfer

### Stage 3: Causal evidence
- cross-feature intervention → activation patching → feature ablation / steering

### Stage 4: Core methodological contribution
- shared SAE / crosscoder → shared + private latent space → cross-task causal circuit

## Success Criteria

### Phase 0
- [ ] Successfully train M0 + SAE0
- [ ] Complete feature-origin tracing analysis
- [ ] Complete multiple fine-tuning-strength experiments

### Phase 1
- [ ] Implement multi-dimensional feature similarity
- [ ] Generate feature correspondence heatmap
- [ ] Complete controls design and validation

### Phase 2
- [ ] Implement cross-task probe
- [ ] Complete shared/private feature classification
- [ ] Quantify information transfer

### Phase 3
- [ ] Implement feature-level intervention
- [ ] Validate cross-task influence

### Phase 4
- [ ] Implement Cross-Model SAE / Crosscoder
- [ ] Build shared + private latent space
- [ ] Compute Shared Knowledge Ratio

### Phase 5
- [ ] Implement path patching
- [ ] Identify negative-transfer cases
- [ ] Build feature interaction matrix

### Phase 6
- [ ] Map features to physical space
- [ ] Design and validate cross-task mutations
- [ ] Provide MD validation report

---

## References

1. Adams et al. "From Mechanistic Interpretability to Mechanistic Biology: Training, Evaluating, and Interpreting Sparse Autoencoders on Protein Language Models" (PMLR 2025)
2. Learn Mechanistic Interpretability - Crosscoders
3. "Sparse autoencoders uncover biologically interpretable features in protein language model representations" (PMC)

---

*Last updated: August 2026*