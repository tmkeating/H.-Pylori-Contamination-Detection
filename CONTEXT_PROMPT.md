# H. Pylori Contamination Detection - Session Context

## 🛡️ Persona: The Skeptical Data Scientist
**Philosophy**: Prioritize clinical safety and diagnostic rigor over raw accuracy.
- **Clinical-Grade Specificity**: Operate under the "Auditor" mindset where False Positives are unacceptable. Every metric must be cross-validated by the Auditor or Grad-CAM.
- **Data Cynicism**: Be critical of high-performance metrics (e.g., 100% Recall) unless the precision is also stable. Avoid generic praise; focus on finding "shortcut learning" or artifact overfitting.
- **Backbone Skepticism**: Ensure that the model is "looking" at bacteria, not tissue folds or staining noise.
- **Empirical Validation**: Trust only empirical results; theoretical expectations (e.g., "larger model = better performance") must be validated experimentally.

---

## 🛡️ Model Architecture (HPyNet / Attention-MIL)

### Backbone Architecture ⭐ EMPIRICALLY OPTIMIZED
- **Primary**: **ConvNeXt-Tiny** [RECOMMENDED] (28M params, 768-dim features)
  - Empirically superior: 92.11% accuracy, 100% precision, 0 false positives
  - Most efficient: 56% fewer parameters than ConvNeXt-Small, 33% faster training/inference
  - Optimal clinical trade-off: Perfect precision + perfect specificity + 84.21% sensitivity
- **Alternative**: ConvNeXt-Small (50M params, 768-dim) - NOT RECOMMENDED
  - Empirically underperforms despite +80% more parameters: 87.72% accuracy, 92.16% precision, 4 false positives
  - Indicates overfitting or domain mismatch with histology data
- **Legacy**: ResNet50 (25M params, 2048-dim) - stable but lower performance

### MIL Strategy (Multiple Instance Learning)
- **Attention Head**: Gated Attention mechanism with entropy regularization (`loss - 0.001 * entropy`)
  - V-pathway (tanh): Non-linear morphological pattern recognition
  - U-pathway (sigmoid): Noise filtering gate to suppress stain artifacts
  - Temperature scaling: Tunable focus on suspicious patches
- **Aggregation**: Top-3 Mixed MIL averaging (balances sensitivity with noise resilience)
- **Purpose**: Identify sparse bacteria in noisy tissue backgrounds at patient level

### Inference Pipeline
- **16-way Contrast-Boosted TTA**: 8 spatial transforms + 1.1x contrast jitter to enhance faint IHC signals
- **Standard Sliding Window**: 250-pixel stride with 50% overlap for comprehensive tissue coverage
- **Rescue Dense Pass**: 128-pixel stride for sparse bacteria in gap regions
- **Data Integrity**: Global MD5 deduplication audit (8KB byte-level check) prior to metric reporting

---

## 🧪 Training & Fusion Configuration

### Cross-Validation Design
- **K-Fold**: 5-fold cross-validation
- **Stratification**: Patient-level split (114 patients: 57 positive, 57 negative)
- **Training Patches**: ~3.6M per fold, validated on 228 patient-level cases
- **Imbalance Handling**: Focal Loss with configurable pos_weight/neg_weight/gamma

### Three Fusion Methods Evaluated (Ensemble Voting)
All three methods generated in `ensemble_voting.py` for comparative analysis:

1. **Ensemble Voting** (Baseline)
   - Majority vote: 3/5 models agree at 0.40 threshold
   - Fallback safety override: max>0.39 AND mean>0.28
   - Accuracy: 86.84%, Precision: 85.00%, Recall: 89.47%

2. **Meta-Classifier** (Precision-focused)
   - Random Forest with Leave-One-Out Cross-Validation (100 trees, max_depth=3)
   - Learns optimal blending from training fold predictions
   - Accuracy: 91.23%, Precision: 97.96%, Recall: 84.21%

3. **Hybrid Ensemble** ⭐ **(RECOMMENDED - PRODUCTION)**
   - Intelligent confidence-zone blending:
     - **High confidence (>0.95)**: Use ensemble voting (fastest)
     - **Uncertainty zone (0.35-0.55)**: Use meta-classifier (precise)
     - **Medium confidence (0.55-0.95)**: Blend 60% ensemble + 40% meta (balanced)
   - **Result**: **92.11% Accuracy | 100% Precision | 100% Specificity | 91.43% F1 Score**
   - **Clinical Impact**: Zero false positives, all negative patients correctly identified

### Output Files Per Run Set (e.g., runs 414-418)
Each method generates comprehensive outputs:
- **Results CSV**: PatientID, Actual, Predicted, Probabilities
- **Summary CSV**: 17+ performance metrics (accuracy, precision, recall, F1, MCC, kappa, confusion matrix)
- **Bootstrap CI CSV**: 1000-iteration resampling with 95% confidence intervals (Point Estimate, CI_Lower, CI_Upper, CI_Margin)
- **Bootstrap CI PNG**: Error bar visualization for clinical metrics (Recall, Precision, Accuracy, F1, Sensitivity, Specificity, Balanced_Accuracy, PPV, MCC)
- **ROC/PR Curves PNG**: Receiver Operating Characteristic and Precision-Recall curves
- **Threshold Analysis PNG**: Performance across decision threshold range

---

## 📊 Empirical Performance Results

### Hybrid Ensemble Final Performance (ConvNeXt-Tiny, Runs 414-418)
| Metric | Value |
|--------|-------|
| **Accuracy** | 92.11% |
| **Precision** | 100.00% |
| **Recall** | 84.21% |
| **Specificity** | 100.00% |
| **F1 Score** | 91.43% |
| **Sensitivity** | 84.21% |
| **Balanced Accuracy** | 92.11% |
| **False Positives** | 0 |
| **False Negatives** | 9 |
| **True Positives** | 48/57 |
| **True Negatives** | 57/57 |
| **Matthews Correlation Coefficient** | 0.8528 |
| **Cohen's Kappa** | 0.8421 |

### Architecture Comparison: ConvNeXt-Tiny vs ConvNeXt-Small
**5-Fold CV Hybrid Ensemble Method**

| Metric | ConvNeXt-Tiny ⭐ | ConvNeXt-Small | Difference |
|--------|-----------------|----------------|-----------|
| Accuracy | **92.11%** | 87.72% | +4.39% |
| Precision | **100.00%** | 92.16% | +7.84% |
| Recall | 84.21% | 82.46% | +1.75% |
| Specificity | **100.00%** | 92.98% | +7.02% |
| F1 Score | **91.43%** | 87.04% | +4.39% |
| False Positives | **0** | 4 | -4 FP |
| False Negatives | 9 | 10 | -1 FN |
| MCC | **0.8528** | 0.7586 | +0.0942 |
| Kappa | **0.8421** | 0.7544 | +0.0877 |
| Parameters | 28M | 50M | -22M (56% fewer) |
| Training Time | ~1x | ~1.5x | -33% faster |

**Key Finding**: ConvNeXt-Tiny superior despite smaller size; ConvNeXt-Small likely overfits with only 114 patient samples.

---

## � Dataset Architecture

### HelicoDataSet (IHC H. Pylori Stain)
- **Location**: `/export/hhome/tkeating/8117180/`
- **Structure**: Patient-hierarchical with stratified 5-fold CV
- **Total Patches**: ~400K+ across patient cohort
- **Classes**: Positive (H. pylori) / Negative (no bacteria)
- **Training**: `train.py` with patient-level cross-validation
- **Integrity**: Global MD5 deduplication via `global_duplicates_check.py`

### DeepHP Dataset (H&E Histology)
- **Location**: `/export/hhome/tkeating/8117177/`
- **Structure**: Flat class directories (Positive/, Negative/) - NOT patient-hierarchical
- **Total Patches**: 394,926 patches (111,005 Positive + 283,921 Negative)
- **Class Imbalance**: 2.56:1 (Negative:Positive ratio)
- **Sync**: Pre-synced to `/tmp/tkeating_deephp_data/` for training
- **Purpose**: ConvNeXt-Tiny backbone pre-training on diverse histology patterns
- **Training**: `train_deepHP_patches.py` with 5-fold stratified CV on patches
- **Integrity**: Global MD5 deduplication via `check_global_duplicates_deepHP.py` (verified: 0 duplicates, 394,926/394,926 patches present)

---

## 📂 Core Pipeline Files

### Primary Scripts (HelicoDataSet IHC Workflow)
- **[train.py](train.py)**: k-fold cross-validation training engine with ConvNeXt-Tiny default, SWA BN Recalibration, Grad-CAM visualization
- **[ensemble_voting.py](ensemble_voting.py)**: THREE FUSION METHODS with comprehensive outputs
  - Generates `ensemble_voting_summary_*.csv`, `meta_classifier_summary_*.csv`, `hybrid_ensemble_summary_*.csv` (⭐ PRIMARY)
  - All include results CSV, summary CSV, bootstrap CI CSV, ROC/PR PNG, threshold analysis PNG, bootstrap CI PNG
  - Hybrid Ensemble output recommended for production deployment
- **[run_h_pylori.sh](run_h_pylori.sh)**: SLURM submission wrapper for 5-fold training with automatic run numbering
- **[profiles.sh](profiles.sh)**: Centralized hyperparameters (learning rates, pos_weight, gamma, data paths)

### DeepHP Backbone Pre-Training Scripts
- **[train_deepHP_patches.py](train_deepHP_patches.py)**: Patch-level 5-fold stratified CV for ConvNeXt-Tiny backbone pre-training on 394,926 H&E patches
  - Auto-increments run IDs from `results/` folder state
  - Focal Loss with configurable pos_weight for 2.56:1 imbalance
  - Generates model weights, evaluation reports, confusion matrices, learning curves, ROC/PR curves
- **[audit_png_count_deepHP.py](audit_png_count_deepHP.py)**: PNG patch counting and sync verification for DeepHP dataset
  - Outputs `deephp_audit_report.csv` with class distribution
  - Verifies all 394,926 patches synced to scratch (`/tmp/tkeating_deephp_data/`)
- **[check_global_duplicates_deepHP.py](check_global_duplicates_deepHP.py)**: Full-file MD5 byte-level duplicate detection across DeepHP dataset
  - Outputs: `deephp_image_inventory.csv`, `deephp_image_duplicates.csv`, `deephp_class_distribution.csv`, `deephp_patch_duplicate_audit.csv`, `suggested_deephp_blacklist.json`
  - Verified: 0 duplicates across 394,926 patches (tested Job 113406)
- **[submit_duplicates_check_deepHP.sh](submit_duplicates_check_deepHP.sh)**: SLURM orchestration for duplicate audit (4 CPUs, 2 hours, 32GB memory)

### Model & Data
- **[model.py](model.py)**: HPyNet architecture with Gated Attention MIL head and ConvNeXt-Tiny backbone (default)
  - Supports ConvNeXt-Tiny (28M), ConvNeXt-Small (50M), ResNet50 (25M)
  - Frozen batch norm to prevent noise artifacts
- **[dataset.py](dataset.py)**: Multi-phase data loader with 16-way TTA, live integrity checks, guaranteed positive patches

### Visualization & Analysis
- **[generate_visuals.py](generate_visuals.py)**: Clinical-grade visualization (learning curves, ROC/PR, Grad-CAM heatmaps, confusion matrices)
- **[global_duplicates_check.py](global_duplicates_check.py)**: 8KB MD5 byte-level deduplication scanner for data integrity audit

### Utilities & Reports
- **[summarize_results.py](summarize_results.py)**: Aggregates cross-fold results and generates comprehensive reports
- **[rescue_inference.py](rescue_inference.py)**: Dense-window rescue inference for sparse bacteria detection
- **[CONTEXT_PROMPT.md](CONTEXT_PROMPT.md)**: This file - session context and architecture documentation
- **[FINAL_REPORT.md](FINAL_REPORT.md)**: Executive summary of pipeline performance and clinical implications

---

## 🔍 Statistical Rigor

### Cross-Validation Strategy
- **Patient-Level Split**: All patches from same patient go to same fold (prevents data leakage)
- **Stratification**: Equal positive/negative distribution across folds
- **Reproducibility**: Fixed random seed for deterministic fold assignment

### Uncertainty Quantification
- **Bootstrap Confidence Intervals**: 1000 iterations with replacement for 95% CI computation
- **Wilson Score Intervals**: Binomial proportion confidence for edge cases
- **Per-Metric CI**: Recall, Precision, Accuracy, F1, Sensitivity, Specificity, Balanced Accuracy, PPV, MCC all with CI bounds

### Validation Methods
- **Grad-CAM Audits**: Visual verification that attention weights focus on bacteria
- **Global Deduplication**: MD5 hash audit to prevent spurious high-performance from duplicated test data
- **False Positive Analysis**: Manual review of all mis-predicted cases for systematic errors

---

## 🚀 Execution Workflow

1. **Data Integrity Audit** (`global_duplicates_check.py`): MD5 deduplication verification before training starts
2. **Training** (`run_h_pylori.sh`): 5-fold CV with automatic SLURM job tracking and run numbering
3. **Visualization** (During/After `train.py`): Grad-CAM, confusion matrices, learning curves generated during training and saved post-fold
4. **Fusion** (`ensemble_voting.py`): Generate 3 methods × 6 output types = 18 files per fold
5. **Reporting** (`summarize_results.py`): Aggregate cross-fold metrics and bootstrap CIs

---

## 💡 Design Rationale

- **Hybrid Ensemble**: Balances ensemble speed (high confidence) with meta-classifier precision (uncertainty)
- **ConvNeXt-Tiny**: Empirically optimal; larger models don't generalize to 114-patient dataset
- **Bootstrap CIs**: Quantify uncertainty inherent in small clinical dataset
- **Patient-Level CV**: Prevents data leakage that inflates metrics in histology datasets
- **Zero FP Target**: Clinical deployment prioritizes no false positives over raw accuracy
- **Top-3 MIL**: Robust to sparse bacteria by aggregating signal from multiple patches


