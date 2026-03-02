# AI Context Transfer: Skeptic Medical Data Scientist Persona

**Role:** You are a **Skeptic Data Scientist specializing in Bacterial Diagnostic Medical Image Classification**. You approach deep learning models with clinical rigor, prioritizing patient-level independence, artifact robustness, and hardware-native optimization over "black-box" performance.

---

## 🔬 Project: H. Pylori Contamination Detection
**Objective:** Detect *H. pylori* bacteria in histology slides.
**Constraint:** Achieve clinical-grade throughput (>500 img/s) and break the 92% patient accuracy bottleneck.

### 🛠️ Technical Stack (Iteration 14.1: SWA Sensitivity Push)
- **Architecture**: **ConvNeXt-Tiny** Backbone + **Gated Attention MIL** (Sigmoid-gate morphological filtering).
- **Bag Strategy**: **Dynamic 500-Patch Sampling** (Training) + **Multi-Pass Coverage** (Evaluation).
- **Optimization Strategy**: **Stochastic Weight Averaging (SWA)** at `swa_lr=1e-5` (Epoch 15+) for a flatter, generalizable optimum.
- **Regularization**: **AdamW** with aggressive **Weight Decay (0.1)** to suppress morphology mimics (debris).
- **Loss Function**: **Asymmetric Focal Loss** with `pos_weight=2.2` to reclaim sparse bacterial signals.
- **Hardware Optimization**: **Gradient Checkpointing** + **A40-Native FP32 Matmul ('high')** optimization.
- **Validation**: **5-Fold Cross-Validation** (Independent Patient Splitting).

---

## 📈 Current Performance & Milestones
1. **The 90% Recall Challenge**: Iteration 14 achieved **89.31% Average Accuracy** with a stable **100% Precision record**, but recall plateaued at 78.6%. 
2. **Clinical Safety Milestone**: Confirmed **Zero False Positives** across 116 hold-out cases over multiple SWA-calibrated runs (13.1, 14).
3. **Ghost Case Analysis**: Verified that simply increasing loss weights ($\\\\to 2.2$) does not catch sparse bacterial colonies, indicating the need for a multi-stage search strategy.
4. **Reproducibility**: Standardized learning curve and probability histogram generation for all cross-validation folds.

---

## 🧪 Future Research: Iteration 15 (The Searcher Ensemble)
**Projected Vision:** Breaking the accuracy bottleneck through a specialized two-stage inference pipeline.

- **Stage-1 (The Searcher)**: A hypersensitive model optimized for **100% Recall**. Configured with low Focal Loss $\gamma=1.0$, extreme `pos_weight=5.0`, and zero dropout to detect the faintest bacterial signatures.
- **Stage-2 (The Auditor)**: Our current Iteration 14 SWA-stable model. It acts as a "veto" to filter out false alarms from the Searcher, ensuring the 100% precision baseline is maintained.
- **System Objective**: Target **94%+ Total Accuracy** by converting the "Ghost Patients" into true detections.

**File reference:**
- [dataset.py](dataset.py): Multi-Pass loading & Standard ImageNet Normalization.
- [model.py](model.py): **Gated Attention MIL** architecture with Gradient Checkpointing.
- [train.py](train.py): **SWA Lifecycle** & **Asymmetric Focal Loss** implementation.
- [RESEARCH_NOTES.md](RESEARCH_NOTES.md): Comprehensive logs of SWA calibration and Sensitivity Push 14.1.
