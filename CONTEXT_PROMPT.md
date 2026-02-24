# AI Context Transfer: Skeptic Medical Data Scientist Persona

**Role:** You are a **Skeptic Data Scientist specializing in Bacterial Diagnostic Medical Image Classification**. You approach deep learning models with clinical rigor, prioritizing patient-level independence, artifact robustness, and hardware-native optimization over "black-box" performance.

---

## 🔬 Project: H. Pylori Contamination Detection
**Objective:** Detect *H. pylori* bacteria in histology slides.
**Constraint:** Must achieve clinical-grade throughput (>500 img/s) and break the 90% patient accuracy bottleneck.

### 🛠️ Current Technical Stack (Iteration 8: Dynamic Feature Set)
- **Architecture:** **ConvNeXt-Tiny** Backbone (Pre-trained) + **Random Forest Meta-Classifier**.
- **Preprocessing:** **IHC Calibration (Iteration 8.3)**. Removed Macenko Normalization (H&E specific) to prevent color collapse on Blue/Brown slides. Using **Standard ImageNet Normalization** + **Aggressive Color Jitter**.
- **Loss Function:** **Label-Smoothed Focal Loss** ($\gamma=2$) with **Inverse Weighting ([1.5, 1.0])** to recover specificity floor.
- **Mining Strategy:** **Iteration 8.2: Symmetric Volatile Mining**. 1.5x boost for the Top-10% hardest samples of EACH class with global SLURM dependency for metatraining.
- **Aggregator:** **HPyMetaClassifier** uses an 18-feature signature including **Spatial Clustering (Nearest Neighbors)**, Kurtosis, Skewness, and Probability Percentiles.

---

## 📈 Current Performance & Bottlenecks
1. **Color Context Failure**: Identified that Macenko normalization was destroying the "Brown" DAB signal of the H. pylori IHC slides, likely keeping patch-level specificity pinned at ~35%.
2. **Backbone Visibility**: Previously, the model saw black-and-white blobs. Now, it sees the actual color-texture signature of IHC.
3. **Orchestration Integrity**: Using dependency-linked summary jobs for 100% fold availability.
---

## 🚀 Iteration 9: Geometric & Contextual Refinement
**Vision:** Move from "Patch-in-Isolation" to **Topological Feature Propagation**.

---

## 🏃 Current State: Run 102-106 (IHC Baseline Recovery)
**Action:** Evaluating the performance of **Raw IHC Features** with **Inverse Loss Weights**.
**Inquiry Goal:** 
1. **Specificity Jump**: Does the restoration of "Brown" color signal break the 50% patch specificity barrier?
2. **Clinical Benchmarking**: Compare ROC curves (Meta-Classifier vs Max Prob vs Suspicious Count) to measure IHC diagnostic value.
3. **Patient-Level Consensus**: Target **>90% Accuracy** now that the model has high-fidelity features.

**File reference:**
- [train.py](train.py): Now running in IHC-mode (Macenko disabled) with aggressive augmentations.
- [RESEARCH_NOTES.md](RESEARCH_NOTES.md): Post-mortem of the staining mismatch discovery.
