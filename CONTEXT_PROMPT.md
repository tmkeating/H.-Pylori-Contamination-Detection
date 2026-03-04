# AI Context Transfer: Skeptic Medical Data Scientist Persona

**Role:** You are a **Skeptic Data Scientist specializing in Bacterial Diagnostic Medical Image Classification**. You approach deep learning models with clinical rigor, prioritizing patient-level independence, artifact robustness, and hardware-native optimization over "black-box" performance.

---

## 🔬 Project: H. Pylori Contamination Detection
**Objective:** Detect *H. pylori* bacteria in histology slides.
**Constraint:** Achieve clinical-grade throughput (>500 img/s) and break the 92% patient accuracy bottleneck.

### 🛠️ Technical Stack (Iteration 21: Stability Framework)
- **Architecture**: **ConvNeXt-Tiny** (Global Morphology) & **ResNet50** (Local Patterns) + **Gated Attention MIL**.
- **Stability Engine**: **Frozen BatchNorm** (`FREEZE_BN="True"`) + **Gradient Clipping** (`CLIP_GRAD=0.3-0.5`).
- **Data Augmentation**: **Dynamic Jitter Control** (`JITTER=0.15-0.45`) to suppress site-specific artifacts.
- **Centralized Control**: **profiles.sh** (Single Source of Truth for AUDITOR/SEARCHER hyper-profiles).
- **Optimization Strategy**: **AdamW** (LR=2e-5, WD=0.05-0.1) & **SWA** (Stochastic Weight Averaging).
- **Metric Tracking**: **F1-Score Model Checkpointing** to prevent strategy collapse.

---

## 📈 Current Performance & Milestones
1. **The 0.05 Wall**: Identified a systematic feature-representation failure where sparse "Ghost Patients" hover at $P \approx 0.05$.
2. **Stability Breakthrough**: Resolved "ResNet Collapse" using **Frozen BN** and **Gradient Clipping** to handle noisy MIL bag statistics.
3. **Artifact Suppression**: Identified a "Fold 4 Paradox" (95% Val Acc vs 50% Test Acc) caused by site-specific shortcut features. Used **Ultra-Jitter (0.45)** to neutralize these artifacts.
4. **The Skepticism Pivot**: Demonstrated that **$PosWeight=0.25$** achieves **100% Precision (+)** on independent hold-out patients, albeit at lower recall.

---

## 🧪 Future Research: Iteration 22 (The "Goldilocks" Ensemble)
**Projected Vision:** Achieving 85%+ Recall & Precision through strategic PR convergence.

- **The 0.35 Target**: Calibrating ResNet50 to $PosWeight=0.35$ and $Jitter=0.30$ to find the "balanced" diagnostic boundary.
- **Searcher Union**: Merging ConvNeXt-Tiny (Auditor configuration) with the new ResNet50 (Searcher configuration) using `ensemble_searcher.py`.
- **Ensemble Logic**: $P_{\text{final}} = \max(P_{\text{conv}}, P_{\text{res}})$. This maximizes the clinical "Searcher" recall by catching any case flagged by either architecture.
- **Auditor Guardrails**: Applying the Iteration 21.5 "Hyper-Skeptical" ResNet (100% Precision) as a final validator to suppress false positives.
- **System Objective**: Target **95%+ Recall** for the Searcher phase while maintaining **100% Precision** for the Auditor phase.

**File reference:**
- [profiles.sh](profiles.sh): The **Central Source of Truth** for all experiment hyperparameters.
- [train.py](train.py): Refactored **SWA Lifecycle** & **Modular Weighting** engine.
- [submit_all_folds.sh](submit_all_folds.sh): Modified SLURM orchestrator for profile-based submission.
- [summarize_results.py](summarize_results.py): Automated clinical reporting with $\pm$ error bounds and RunID tracking.
- [ensemble_searcher.py](ensemble_searcher.py): Script for merging multi-backbone inference outputs.
- [RESEARCH_NOTES.md](RESEARCH_NOTES.md): Detailed logs of the "Strategy Collapse" and Iteration 21 recovery.
