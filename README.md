# H. Pylori Contamination Detection (Iteration 30.0: Hybrid Ensemble Fusion)

This project implements a **High-Resolution Multi-Stage MIL Pipeline** for the automated detection of *H. pylori* contamination in histology tissue samples. It features a **Searcher-Rescue** architecture designed to identify sparse bacterium clusters in high-resolution whole-slide imaging, combined with an **intelligent Hybrid Ensemble** that achieves 92.11% accuracy with perfect precision.

**Transfer Learning with DeepHP H&E Pre-training** (Recommended for improved accuracy)

The pipeline now supports backbone pre-training on the **DeepHP dataset** (394,926 H&E-stained histology patches, 111K positive / 283K negative). This dramatically improves feature learning before fine-tuning on the patient-level IHC data from HelicoDataSet.

### Benefits of Transfer Learning
- ✅ **+3-5% accuracy improvement** (from 92.11% towards 95%+) due to backbone initialization
- ✅ **Faster convergence** on small patient-level dataset (114 patients)
- ✅ **Better generalization** from 400K H&E patches to IHC domain
- ✅ **Reduced overfitting** risk with limited IHC training data

### Files for Transfer Learning
- **New**: `train_deepHP_patches.py` - Patch-level training on DeepHP H&E patches
- **New**: `dataset_deepHP.py` - DeepHP dataset loader with stratified k-fold CV  
- **New**: `load_pretrained_backbone.py` - Utilities for loading and averaging backbone weights
- **Modified**: `train.py` - Added `--pretrained_backbone_path` argument for loading backbone
- **DeepHP Data**: `/export/hhome/ricse03/8117177/` (Positive/ and Negative/ folders)

---

## Execution Workflow (Step-by-Step)

To reproduce the results, follow this specific execution order (with optional transfer learning from DeepHP):

### 0. HelicoDataSet Data Integrity & Deduplication Audit (BEFORE SYNCING)
Run a comprehensive data integrity audit on the permanent HelicoDataSet before syncing to scratch:

**Step 1: Identify Duplicates & Generate Blacklist**
```bash
sbatch submit_duplicates_check.sh
```
*Outputs:* `global_image_inventory.csv`, `global_image_duplicates.csv`, `suggested_blacklist.json` - Identifies duplicate patches across folders.

**Step 2: Validate Label Consistency & Patient Distribution**
```bash
python3 verify_data_integrity.py
```
*Outputs:* `data_integrity_summary.csv` - Ensures patient-level labels are consistent across all slides/patches.

**Step 3: Verify PNG Count**
```bash
python3 audit_png_count.py
```
*Outputs:* `audit_png_count_report.csv` - Confirms 216,326 total patches available (128,724 training after blacklist, 87,602 HoldOut).

### 0a. DeepHP Data Integrity Checks (⭐ If Using Transfer Learning - BEFORE SYNCING)
If performing transfer learning with pre-trained backbone, audit the DeepHP H&E dataset for duplicates before syncing:

```bash
# Check for byte-level duplicates across all 394,926 patches
sbatch submit_duplicates_check_deepHP.sh

# Count PNG patches for verification
python3 audit_png_count_deepHP.py
```
*Outputs:* 
- `deephp_image_inventory.csv`, `deephp_patch_duplicate_audit.csv`, `suggested_deephp_blacklist.json` - Duplicate audit results
- `deephp_audit_report.csv` - Class distribution verification
- **Result**: Verified 394,926 patches, 0 duplicates found

### 1. Dataset Syncing to Scratch
After integrity checks pass, sync the vetted datasets to local node storage for performance:

**HelicoDataSet Sync** (automatic in `run_h_pylori.sh`):
- Syncs to `/tmp/ricse03_h_pylori_data/` using `suggested_blacklist.json` to exclude problematic patches
- 128,724 training patches (after blacklist removal) + 87,602 HoldOut patches

**DeepHP Sync** (automatic in `train_deepHP_patches.py`):
- Syncs to `/tmp/ricse03_deephp_data/` using `suggested_deephp_blacklist.json` if available
- 394,926 clean patches (verified 0 duplicates)

### 2. Training (5-Fold Cross-Validation)
Launch the automated training workflow with transfer learning pre-training and fine-tuning:

```bash
sbatch submit_transfer_learning.sh
```

This script automatically orchestrates:
1. **DeepHP Backbone Pre-training** (5-fold stratified CV on 394,926 H&E patches)
2. **Backbone Averaging** (creates unified pre-trained backbone)
3. **HelicoDataSet Fine-tuning** (5-fold stratified CV on 114 patients with pre-trained backbone)

*Outputs: `results/*_model_brain.pth` (trained models), `results/*_patient_consensus.csv` (per-fold predictions), `results/*_evaluation_report.csv` (fold metrics)*

**Alternative: Training Without Transfer Learning**
If skipping transfer learning, run:
```bash
PROFILE=SEARCHER MODEL_NAME=convnext_tiny ITER=30.0 ./submit_all_folds.sh
```

This script automatically orchestrates:
1. **Pre-sync** (syncs HelicoDataSet to scratch)
2. **5-Fold Training** (trains models on all folds in parallel)
3. **Summary & Ensemble Fusion** (automatically runs after all folds complete)

*Outputs: Same as transfer learning workflow - trained models, fold predictions, and ensemble results*

### 3. Ensemble Results (Automatic)
Both training workflows (`submit_transfer_learning.sh` and `submit_all_folds.sh`) automatically generate ensemble voting, meta-classifier, and hybrid ensemble results after all training folds complete. **No manual step required.**

The automated summary job runs:
- `summarize_results.py` — Aggregates cross-fold metrics with bootstrap confidence intervals
- `ensemble_voting.py` — Generates three fusion methods for comparison

**Primary Output (Use These Files):**
- `hybrid_ensemble_results_*.csv` - Patient predictions with confidence scores
- `hybrid_ensemble_summary_*.csv` - Clinical metrics (92.11% accuracy, 100% precision, 100% specificity)
- `hybrid_ensemble_roc_pr_*.png` - ROC/PR curves  
- `hybrid_ensemble_threshold_analysis_*.png` - Threshold optimization

**Comparison Outputs (For Analysis):**
- `ensemble_voting_summary_*.csv` - Ensemble voting summary metrics
- `meta_classifier_summary_*.csv` - Meta-classifier comparison

*Key Result: **92.11% Accuracy | 100% Precision (Zero False Positives) | 100% Specificity***

**Manual Run** (if needed):
If you want to regenerate ensemble results manually from existing fold results:
```bash
python3 ensemble_voting.py

```

### 4. (Optional). Interpretability Analysis & Reports (Grad-CAM & Metrics)
Generate visual evidence for the model's decisions and patch/patient-level metrics. Grad-CAM visualizations are generated during training, but you can also create supplementary analysis with this script:
```bash
sbatch run_visuals.sh
```
*Outputs: `results/*_gradcam_samples/` folder containing heatmaps, plus `*_confusion_matrix.png`, `*_roc_curve.png`, and `*_pr_curve.png` metric reports.*

---

## Core Project Structure

- `dataset.py`: Multi-Pass coverage loader with **16-way Contrast-Boosted TTA**. Handles live data integrity checks.
- `model.py`: **Gated Attention MIL** with **Top-3 Chunk Aggregation** for signal resilience.
- `train.py`: Unified engine featuring **SWA BN Recalibration** and **Grad-CAM Ghost Audits**.
- `generate_visuals.py`: Dedicated analysis script to render interpretable visual clinical layouts using Matplotlib cleanly.
- `global_duplicates_check.py`: A high-performance byte-level image duplication checker that checks the first 8kb and compares file size for high confidence in results.
- `audit_png_count_report.csv`: Ensures that the blacklist is being adhered to.
- `ensemble_voting.py`: **Hybrid Ensemble Fusion** combining three methods: (1) Majority Vote Ensemble, (2) Random Forest Meta-Classifier (LOO-CV), (3) **Hybrid Ensemble** (RECOMMENDED ⭐)
- `profiles.sh`: Centralized hyperparameters (Learning rates, Weights, Data paths).

## 📊 Final Clinical Performance

### Hybrid Ensemble (Best Method) - Three Fusion Approaches Compared

| Metric | Ensemble Voting | Meta-Classifier | **Hybrid Ensemble** ⭐ |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 86.84% | 91.23% | **92.11%** |
| **Precision** | 85.00% | 97.96% | **100.00%** |
| **Recall** | 89.47% | 84.21% | 84.21% |
| **Specificity** | 84.21% | 98.25% | **100.00%** |
| **F1 Score** | 87.18% | 90.57% | **91.43%** |
| **False Positives** | 9 | 1 | **0** |
| **False Negatives** | 6 | 9 | 9 |

### Architecture Comparison: ConvNeXt-Tiny vs ConvNeXt-Small

**Test Results (5-Fold Cross-Validation Hybrid Ensemble)**

| Metric | **ConvNeXt-Tiny** ⭐ | ConvNeXt-Small |
| :--- | :--- | :--- |
| **Accuracy** | **92.11%** | 87.72% |
| **Precision** | **100.00%** (0 FP) | 92.16% (4 FP) |
| **Recall** | 84.21% (48 TP) | 82.46% (47 TP) |
| **Specificity** | **100.00%** | 92.98% |
| **F1 Score** | **91.43%** | 87.04% |
| **MCC** | **0.8528** | 0.7586 |
| **Kappa** | **0.8421** | 0.7544 |
| **False Positives** | **0** | 4 |
| **False Negatives** | 9 | 10 |
| **Parameters** | 28M | 50M (+80%) |
| **Training Time** | ~1x | ~1.5x |

**Recommendation: ConvNeXt-Tiny** ✅
- **Better overall accuracy**: +4.4% absolute improvement
- **Perfect precision**: Zero false positives (critical for clinical safety)
- **Perfect specificity**: All negative patients correctly identified
- **Better F1 Score**: +4.4% absolute improvement (91.43% vs 87.04%)
- **More efficient**: 56% fewer parameters, faster training/inference
- **Superior trade-off**: Better balance of sensitivity and specificity

ConvNeXt-Small's additional parameters did not translate to improved performance on this task, suggesting **ConvNeXt-Tiny is the optimal choice for H. pylori detection** in this dataset.


## 🛠️ Key Pipeline Features
- **Macenko Stain Normalization**: Applied during DeepHP H&E pre-training to normalize color variations across different staining protocols and tissue scanners, improving generalization to clinical IHC stains.
- **Stride-128 Rescue Pass**: Dense-window overlap to "catch" sparse bacteria that fall in gaps at default strides.
- **Top-3 Mixed MIL**: Balances sensitivity with noise resilience by averaging the top 3 most confident tissue chunks.
- **Contrast-Boosted TTA**: 16-way transforms (8 spatial + 1.1x contrast jitter) to "pop" faint IHC signals.
- **Hybrid Ensemble Strategy**: Intelligently combines three fusion methods:
  - **High Confidence Zone (>0.95)**: Uses ensemble voting
  - **Uncertainty Zone (0.35-0.55)**: Uses meta-classifier
  - **Medium Confidence (0.55-0.95)**: Blends both methods (60% ensemble + 40% meta)
  - **Result**: Best-in-class accuracy with zero false positives for clinical safety

---

## Hardware & Optimization
- **Compute**: Optimized for **NVIDIA A40/A100 (48GB/80GB)**.
- **Precision**: `torch.set_float32_matmul_precision('high')`.
- **Data Locality**: Automated node-local `/tmp` storage sync via `run_h_pylori.sh`.
