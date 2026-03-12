# H. Pylori Contamination Detection - Session Context

## 🛡️ Persona: The Skeptical Data Scientist
**Philosophy**: Prioritize clinical safety and diagnostic rigor over raw accuracy.
- **Clinical-Grade Specificity**: Operate under the "Auditor" mindset where False Positives are unacceptable. Every metric must be cross-validated by the Auditor or Grad-CAM.
- **Data Cynicism**: Be critical of high-performance metrics (e.g., 100% Recall) unless the precision is also stable. Avoid generic praise; focus on finding "shortcut learning" or artifact overfitting.
- **Backbone Skepticism**: Ensure that the model is "looking" at bacteria, not tissue folds or staining noise.

## Current Status: Iteration 25.1 (100% RECALL ACHIEVED)
**Task**: Successfully reached the goal of capturing every infected patient (116/116 in hold-out) using a Hybrid Surgical Consensus strategy.

### 🛡️ Model Architecture (HPyNet / Attention-MIL)
- **Backbone**: ConvNeXt-Tiny (Frozen Batch Norm to prevent noise).
- **Pooling**: Attention-MIL with **Entropy Regularization** (`loss - 0.001 * entropy`) to force focus on multiple patches and prevent "Delta Collapse."
- **Inference**: 16-way Contrast-Boosted TTA (1.1x contrast) and 50% Overlapping Sliding Window (250-patch stride).

### 🧪 Training Configuration (SEARCHER Profile)
- **Profile**: `set_profile_SEARCHER` (Iteration 25.0)
- **Optimizer**: AdamW (WD=0.05), **ReduceLROnPlateau** scheduler for stability.
- **Loss**: FocalLoss (Gamma=3.0, PosWeight=5.0) with **0.0 Label Smoothing** (restored to 0.0 per user preference).
- **Ensemble Logic**: Majority Vote (3/5 models) at **0.40 threshold** OR Safety Sensitivity Override at **0.20 threshold**.

### 📊 Performance History
- **Iteration 21 (Auditor)**: 41% Mean Recall, 100% Precision (+).
- **Iteration 24.8**: Hit a wall at 97.2% Recall; identified "Ghost Patient" B22-81_1.
- **Iteration 24.9**: "Delta Collapse" phase (100% Train Acc, failed Generalization).
- **Iteration 25.1**: **100% RECALL (+)**, 53.8% Precision (+). All 116 hold-out patients detected.

### 📂 Key Files
- [dataset.py](dataset.py): Multi-phase sampling (Guaranteed Positive Patches).
- [model.py](model.py): HPyNet with Attention-MIL and gated noise filtering.
- [train.py](train.py): Top-3 Mixed MIL inference with 16-way TTA.
- [profiles.sh](profiles.sh): Central hyperparameter source for Searcher/Auditor profiles.
- [ensemble_voting.py](ensemble_voting.py): Multi-logic consensus tool (Majority + Safety Override).
