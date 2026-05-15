# H. Pylori Contamination Detection - Session Context

## 🛡️ Persona: The Skeptical Data Scientist
**Philosophy**: Prioritize clinical safety and diagnostic rigor over raw accuracy.
- **Clinical-Grade Specificity**: Operate under the "Auditor" mindset where False Positives are unacceptable. Every metric must be cross-validated by the Auditor or Grad-CAM.
- **Data Cynicism**: Be critical of high-performance metrics (e.g., 100% Recall) unless the precision is also stable. Avoid generic praise; focus on finding "shortcut learning" or artifact overfitting.
- **Backbone Skepticism**: Ensure that the model is "looking" at bacteria, not tissue folds or staining noise.

### 🛡️ Model Architecture (HPyNet / Attention-MIL)
- **Backbone**: ConvNeXt-Tiny (Frozen Batch Norm to prevent noise).
- **Pooling**: Attention-MIL with **Entropy Regularization** (`loss - 0.001 * entropy`) to force focus on multiple patches and prevent "Delta Collapse."
- **Inference**: High-Resolution Rescue Strategy & Interpretability.
  - **Standard**: 16-way Contrast-Boosted TTA (1.1x contrast) and 50% Overlapping Sliding Window (250-patch stride).
  - **Rescue**: Dense Rescue Stride (128 pixels) for the "Unreachable Six" patients to bridge the 95% accuracy gap.
  - **Integrity**: Global MD5 deduplication audit prior to metric reporting to prevent data leakage.

### 🧪 Training Configuration (SEARCHER / AUDITOR / HYBRID ENSEMBLE)
- **Searcher**: High Recall (100% target), 5.0 PosWeight, 3.0 Gamma.
- **Auditor**: High Precision (94%+), 7.5 PosWeight, 1.0 Gamma.
- **Hybrid Ensemble** ⭐ **(CURRENT BEST)**: Intelligently combines:
  - **Ensemble Voting**: Majority vote (3/5) for high-confidence decisions
  - **Meta-Classifier**: Random Forest (LOO-CV) for precision
  - **Smart Blending**: Different decision zones based on prediction confidence
  - **Result**: **92.11% Accuracy, 100% Precision, 100% Specificity** (zero false positives)

### 📊 Performance History
- **Iteration 25.1**: 100% RECALL (+) at 53.8% Precision.
- **Ensemble 1.0**: 94.2% Precision at 86% Recall.
- **Iteration 26.0 (Golden Consensus)**: 94.74% Accuracy and 98.25% Recall.
- **Iteration 26.1 (Hybrid Ensemble)** ⭐ **NEW**: **92.11% Accuracy | 100% Precision | 100% Specificity | 91.43% F1 Score**
  - Combines best-of-breed from ensemble voting and meta-classifier
  - **Zero false positives** for clinical safety (no unnecessary treatments)
  - All negative patients correctly identified (100% specificity)
  - Maintains 84.21% sensitivity with perfect precision trade-off

### 📂 Key Files
- [dataset.py](dataset.py): Multi-phase sampling (Guaranteed Positive Patches) with Live Integrity Checks.
- [model.py](model.py): HPyNet with Attention-MIL and gated noise filtering.
- [train.py](train.py): Top-3 Mixed MIL inference with 16-way TTA.
- [profiles.sh](profiles.sh): Central hyperparameter source for Searcher/Auditor profiles.
- [ensemble_voting.py](ensemble_voting.py): **Hybrid Ensemble Fusion** - combines three methods (Ensemble Voting, Meta-Classifier, Hybrid) with intelligent decision zones. Primary output: `hybrid_ensemble_*.csv` (⭐ RECOMMENDED)
- [meta_classifier.py](meta_classifier.py): Random Forest meta-classifier with Leave-One-Out Cross-Validation for fusion.
- [generate_visuals.py](generate_visuals.py): Robust clinical-grade visual reporting using Matplotlib and normalized image-net stats.
- [global_duplicates_check.py](global_duplicates_check.py): Cross-folder 8KB MD5 deduplication data integrity scanner.


