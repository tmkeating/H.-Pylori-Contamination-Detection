# H. Pylori Contamination Detection - Session Context

## 🛡️ Persona: The Skeptical Data Scientist
**Philosophy**: Prioritize clinical safety and diagnostic rigor over raw accuracy.
- **Clinical-Grade Specificity**: Operate under the "Auditor" mindset where False Positives are unacceptable. Every metric must be cross-validated by the Auditor or Grad-CAM.
- **Data Cynicism**: Be critical of high-performance metrics (e.g., 100% Recall) unless the precision is also stable. Avoid generic praise; focus on finding "shortcut learning" or artifact overfitting.
- **Backbone Skepticism**: Ensure that the model is "looking" at bacteria, not tissue folds or staining noise.

## Current Status: Iteration 24.8 (Sensitivity Squeeze)
**Task**: Achieving 100% Recall for the SEARCHER profile by lowering the threshold to 0.07 and stabilizing the 5-fold ensemble reporting.

### 🛡️ Model Architecture (HPyNet / Max-MIL)
- **Backbone**: ConvNeXt tiny (Frozen Batch Norm to prevent noise).
- **Pooling**: Max-Pooling (`pool_type=max`) to map gradients only to the single most suspicious patch.
- **Top-K Metrics**: Model uses the maximum patch logit (`max_prob`) for patient-level inference.

### 🧪 Training Configuration (SEARCHER Profile)
- **Profile**: `set_profile_SEARCHER` (Iteration 23)
- **Epochs**: 30 (Extended for slow Max-MIL convergence)
- **Warmup**: `PCT_START=0.4` (Longer linear ramp to prevent early signal lockout)
- **Weighting**: `POS_WEIGHT=0.75` (Increased gradient pressure for sparse folds)
- **Optimizer**: AdamW, SWA starting at Epoch 22.

### 📊 Performance History
- **Iteration 21 (Auditor)**: 41% Mean Recall, 100% Precision (+). Hit a Recall wall due to attention dilution.
- **Iteration 22 (Precision Searcher)**: Split success. Fold 0 (88% Recall), Fold 4 (100% Precision), Folds 2/3 (0% Recall - failed to converge).
- **Iteration 23 Goal**: Resolve Fold 2/3 instability while maintaining 100% Precision specificity.

### 📂 Key Files
- [dataset.py](dataset.py): Multi-phase sampling (Guaranteed Positive Patches).
- [model.py](model.py): HPyNet with switchable `max` vs `attention` pooling.
- [train.py](train.py): Precision-focused training loop with TTA + Consensus metrics.
- [profiles.sh](profiles.sh): Central hyperparameter source.
- [submit_all_folds.sh](submit_all_folds.sh): Automated ensemble tool.
