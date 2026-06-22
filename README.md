# H. Pylori Contamination Detection (Iteration 34.0: Hybrid Ensemble Fusion)

This project implements a **High-Resolution Multi-Stage MIL Pipeline** for the automated detection of *H. pylori* contamination in histology tissue samples. It features a **Searcher-Rescue** architecture designed to identify sparse bacterium clusters in high-resolution whole-slide imaging, combined with an **intelligent Hybrid Ensemble** that achieves 92.11% accuracy with perfect precision.

**Transfer Learning with DeepHP H&E Pre-training** (Available for backbone initialization)

The pipeline supports backbone pre-training on the **DeepHP dataset** (394,926 H&E-stained histology patches, 111K positive / 283K negative) before fine-tuning on the patient-level IHC data from HelicoDataSet.

### Benefits of Transfer Learning
- ✅ **Backbone initialization** from large-scale histology patches instead of random weights
- ✅ **Faster convergence** on small patient-level dataset (114 patients)
- ✅ **Cross-domain feature transfer** from H&E patches to IHC domain
- ✅ **Reduced overfitting risk** with limited IHC training data

### Files for Transfer Learning
- **New**: `train_deepHP_patches.py` - Patch-level training on DeepHP H&E patches with CONFIG 87771 experiment-level stratification
- **New**: `dataset_deepHP.py` - DeepHP dataset loader with CONFIG 87771 experiment-level fold splitting (prevents fold-specific artifact learning)
- **New**: `domain_adversarial.py` - Domain Adversarial Neural Networks (DANN) components for experiment-invariant feature learning
- **New**: `load_pretrained_backbone.py` - Utilities for loading and averaging backbone weights
- **Modified**: `train.py` - Added `--pretrained_backbone_path` argument for loading backbone
- **DeepHP Data**: `/home/tkeating/datasets/8117177/` (Positive/ and Negative/ folders)

---

## Pre-Pipeline Setup

### Virtual Environment Verification
All training scripts automatically verify the virtual environment before execution:

```bash
./verify_venv.sh
```

This script:
- ✅ Verifies the project virtual environment (`venv/`) exists and is activated
- ✅ Dynamically reads `requirements.txt` to check all dependencies are installed
- ✅ Auto-installs missing packages if needed (with error handling)
- ✅ Exports `VENV_ROOT` for all SLURM scripts to use

**Note:** All 8 SLURM scripts automatically call this verification before submitting jobs, so manual execution is usually not required.

### Configuration Management
All paths and environment variables are centralized in `config.py`:
- `VENV_ROOT`: Dynamically computed relative path to project virtual environment
- `DATASET_ROOT`: HelicoDataSet permanent storage location
- `DEEPHP_DATASET_ROOT`: DeepHP dataset location  
- `SCRATCH_ROOT`: Local node scratch for temporary training data
- All variables support environment variable overrides for flexibility

---

## Execution Workflow (Step-by-Step)

**RECOMMENDED: Full Transfer Learning Pipeline** (DeepHP Pre-training → Backbone Averaging → HelicoDataSet Fine-tuning)

To reproduce the best results, follow this specific execution order:

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

### 0a. DeepHP Data Integrity Checks (⭐ RECOMMENDED - BEFORE SYNCING)
As part of the recommended full pipeline, audit the DeepHP H&E dataset for duplicates before syncing:

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
- Syncs to `/home/tkeating/.scratch/h_pylori_data/` using `suggested_blacklist.json` to exclude problematic patches
- 128,724 training patches (after blacklist removal) + 87,602 HoldOut patches

**DeepHP Sync** (automatic in `train_deepHP_patches.py`):
- Syncs to `/home/tkeating/.scratch/h_pylori_data/` using `suggested_deephp_blacklist.json` if available
- 394,926 clean patches (verified 0 duplicates)
- **CONFIG 87771 Stratification**: Hardcoded experiment-level 5-fold cross-validation:
  - Each of 33 experiments assigned to exactly ONE fold (zero data leakage)
  - Fold 0 val: 7 experiments (4 pos, 3 neg) → 87,532 patches, ratio 2.33:1
  - Fold 1 val: 10 experiments (3 pos, 7 neg) → 89,516 patches, ratio 2.06:1
  - Fold 2 val: 5 experiments (4 pos, 1 neg) → 20,347 patches, ratio 2.31:1
  - Fold 3 val: 4 experiments (4 pos, 0 neg) → 99,120 patches, ratio 2.81:1
  - Fold 4 val: 7 experiments (6 pos, 1 neg) → 98,410 patches, ratio 2.29:1
  - Each fold trains on ALL experiments except its validation experiments (~307K patches)
  - Mathematically optimized from 500,000+ greedy configuration searches
- **Critical Data Integrity**: The Macenko normalization reference patch is automatically excluded from all 5 folds **before** k-fold assignment (not after), mathematically guaranteeing zero leakage between training and validation sets

### 2. Training (5-Fold Cross-Validation) - RECOMMENDED METHOD

**Launch the full transfer learning pipeline** (preferred for best accuracy):

```bash
PROFILE=SEARCHER MODEL_NAME=convnext_tiny ITER=34.0 ./submit_transfer_learning.sh
```

This script automatically orchestrates:
1. **DeepHP Backbone Pre-training** (5-fold CONFIG 87771 stratification on 394,926 H&E patches)
   - Uses CONFIG 87771 experiment-level hardcoded stratification
   - Each fold validates on unique experiments (prevents fold-specific artifact learning)
   - Each fold trains on ~307K patches from all other experiments
   - Fold validation sets: 87,532 / 89,516 / 20,347 / 99,120 / 98,410 patches respectively
   - Fold ratios: 2.33:1 / 2.06:1 / 2.31:1 / 2.81:1 / 2.29:1 (target 2.28:1, total distance 0.6441)
   - **Domain Adversarial Neural Networks (DANN)**: Optionally enabled with `--use_dann` flag
     - Prevents learning of experiment-specific staining artifacts
     - Adversary head predicts experiment ID from features → forces experiment-invariant representations
     - Gradient reversal layer negates gradients during backprop → confuses adversary
     - Parameters: `--dann_lambda 1.0` (gradient scaling), `--dann_weight 1.0` (loss weighting)
   - Generates per-fold cross-leakage audits and experiment distribution audit
   - Includes automatic Macenko reference image check (creates if missing)
   - Applies Macenko stain normalization during DeepHP pre-training
   - Runs on login node before SLURM job submission (fast, non-intensive)
2. **Backbone Averaging** (creates unified pre-trained backbone from 5-fold CV)
3. **HelicoDataSet Fine-tuning** (5-fold stratified CV on 114 patients with pre-trained backbone)
   - No stain normalization (IHC stains don't use Macenko)
   - Automatic venv verification before each phase

*Outputs: `results/*_model_brain.pth` (trained models), `results/*_patient_consensus.csv` (per-fold predictions), `results/*_evaluation_report.csv` (fold metrics)*

**Alternative: Training Without Transfer Learning** (random initialization)
If you want to skip backbone pre-training, run:
```bash
PROFILE=SEARCHER MODEL_NAME=convnext_tiny ITER=34.0 ./submit_all_folds.sh
```

This script automatically orchestrates:
1. **Pre-sync** (syncs HelicoDataSet to scratch)
2. **5-Fold Training** (trains models on all folds in parallel, initialized randomly)
3. **Summary & Ensemble Fusion** (automatically runs after all folds complete)

*Note: This approach trains models from random initialization without pre-trained backbone weights.*

### 3. Post-Processing Pipeline (Automatic)
Both training pipelines (`submit_transfer_learning.sh` - RECOMMENDED and `submit_all_folds.sh` - ALTERNATIVE) automatically execute a complete post-processing pipeline after all training folds complete. **No manual step required.**

The automated post-processing job orchestrates four sequential stages:

#### 3.1 Per-Fold Threshold Calibration (DeepHP Pre-training)
**Stage 1: `calibrate_per_fold_thresholds_deepHP.py`** (after DeepHP pre-training only)
- Analyzes validation predictions from each of the 5 DeepHP folds
- Computes optimal classification thresholds per-fold that maximize F1 score
- Outputs: `{run_id}_calibrated_thresholds_deepHP.json` with fold-specific thresholds
- Example output:
  ```json
  {
    "fold_0_threshold": 0.52,
    "fold_1_threshold": 0.48,
    "fold_2_threshold": 0.55,
    "fold_3_threshold": 0.50,
    "fold_4_threshold": 0.51
  }
  ```

#### 3.2 Threshold Application & Backbone Averaging (DeepHP Pre-training)
**Stage 2: `apply_calibrated_thresholds_deepHP.py`** (after DeepHP pre-training only)
- Applies per-fold calibrated thresholds to DeepHP validation predictions
- Converts probability outputs (0-1) to binary decisions (0 or 1)
- Prepares predictions for backbone ensemble fusion
- Outputs: Per-fold predictions with threshold-optimized binary labels

#### 3.3 Backbone Weighted Ensemble (DeepHP Pre-training)
**Stage 3: `weighted_ensemble_deepHP.py`** (after DeepHP pre-training only)
- Fuses predictions from all 5 DeepHP folds using weighted voting
- Each fold receives weight based on its validation performance (F1 score)
- Generates averaged backbone ready for transfer learning to HelicoDataSet
- Outputs: `deephp_backbone_final_{run_id}_convnext_tiny_{iter}.pth` + backbone ensemble predictions

#### 3.4 Weighted Ensemble Voting (HelicoDataSet Fine-tuning)
**Stage 4: `ensemble_voting.py`** (after HelicoDataSet fine-tuning only)
- Fuses patient-level predictions from all 5 HelicoDataSet folds using weighted voting
- Each fold receives weight based on its validation performance (F1 score)
- Higher-performing folds contribute more to final predictions
- Implements three fusion strategies for comparison:
  1. **Majority Voting** - Simple majority across 5 folds
  2. **Weighted Average** - Fold weights based on validation F1 scores
  3. **Hybrid Ensemble** ⭐ **(RECOMMENDED)** - Intelligent fusion with confidence-based switching:
     - **High Confidence (>0.95)**: Uses weighted ensemble
     - **Uncertainty Zone (0.35-0.55)**: Uses meta-classifier fallback
     - **Medium Confidence (0.55-0.95)**: Blends both methods (60% weighted + 40% meta)
- Outputs: Patient-level predictions with ensemble confidence scores

#### Primary Outputs
- `hybrid_ensemble_results_*.csv` - Patient predictions with calibrated thresholds and ensemble confidence scores
- `hybrid_ensemble_summary_*.csv` - Clinical metrics (92.11% accuracy, 100% precision, 100% specificity)
- `hybrid_ensemble_roc_pr_*.png` - ROC/PR curves with calibrated thresholds
- `hybrid_ensemble_threshold_analysis_*.png` - Per-fold threshold optimization visualization
- `{run_id}_{iter}_calibrated_thresholds.json` - Per-fold threshold values for reproducibility

#### Comparison Outputs (For Analysis)
- `ensemble_voting_summary_*.csv` - Voting ensemble summary metrics
- `meta_classifier_summary_*.csv` - Meta-classifier comparison metrics
- `weighted_ensemble_analysis_*.csv` - Fold weights and contribution analysis

*Key Result: **92.11% Accuracy | 100% Precision (Zero False Positives) | 100% Specificity***

**Manual Run** (if needed):

For **DeepHP Pre-training** post-processing:
```bash
python3 calibrate_per_fold_thresholds_deepHP.py --run 32
python3 apply_calibrated_thresholds_deepHP.py --run 32
python3 weighted_ensemble_deepHP.py --run 32 --strategy f1
```

For **HelicoDataSet Fine-tuning** post-processing:
```bash
python3 ensemble_voting.py --runs 31_0,31_1,31_2,31_3,31_4 --strategy f1
```

### 4. (Optional). Interpretability Analysis & Reports (Grad-CAM & Metrics)
Generate visual evidence for the model's decisions and patch/patient-level metrics. Grad-CAM visualizations are generated during training, but you can also create supplementary analysis with this script:
```bash
sbatch run_visuals.sh
```
*Outputs: `results/*_gradcam_samples/` folder containing heatmaps, plus `*_confusion_matrix.png`, `*_roc_curve.png`, and `*_pr_curve.png` metric reports.*

---

## Core Project Structure

- `config.py`: Centralized configuration with dynamic path resolution and environment variable overrides. Defines VENV_ROOT, dataset paths, and scratch directories for portability across systems.
- `dataset.py`: Multi-Pass coverage loader with **16-way Contrast-Boosted TTA**. Handles live data integrity checks.
- `model.py`: **Gated Attention MIL** with **Top-3 Chunk Aggregation** for signal resilience.
- `train.py`: Unified engine featuring **SWA BN Recalibration** and **Grad-CAM Ghost Audits**.
- `generate_visuals.py`: Dedicated analysis script to render interpretable visual clinical layouts using Matplotlib cleanly.
- `global_duplicates_check.py`: A high-performance byte-level image duplication checker that checks the first 8kb and compares file size for high confidence in results.
- `audit_png_count_report.csv`: Ensures that the blacklist is being adhered to.

### Post-Processing Pipeline Scripts

**DeepHP Pre-training Post-Processing:**
- `calibrate_per_fold_thresholds_deepHP.py` ⭐: Computes per-fold optimal thresholds from DeepHP validation predictions. Output: `{run_id}_calibrated_thresholds_deepHP.json`
- `apply_calibrated_thresholds_deepHP.py` ⭐: Applies per-fold calibrated thresholds to DeepHP test predictions.
- `weighted_ensemble_deepHP.py` ⭐: Fuses DeepHP predictions from all 5 folds using fold-performance-weighted voting. Generates averaged backbone for transfer learning.

**HelicoDataSet Fine-tuning Post-Processing:**
- `ensemble_voting.py` ⭐: **Hybrid Ensemble Fusion** for HelicoDataSet predictions combining three methods: (1) Majority Vote Ensemble, (2) Random Forest Meta-Classifier (LOO-CV), (3) **Hybrid Ensemble** (RECOMMENDED ⭐). Generates final patient-level predictions.

### Supporting Scripts
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
- **CONFIG 87771 Stratification (DeepHP Pre-training)**: Hardcoded experiment-level 5-fold cross-validation optimized from 500,000+ greedy configuration searches. Each of 33 experiments assigned to exactly ONE fold, preventing data leakage and fold-specific artifact learning. Folds validate on different experiments (87.5K / 89.5K / 20.3K / 99.1K / 98.4K patches) with balanced ratios (2.06:1 to 2.81:1, target 2.28:1, total distance 0.6441). All folds train on ~307K patches from all other experiments.
- **Domain Adversarial Neural Networks (DANN - Optional)**: Advanced technique for DeepHP pre-training that prevents models from learning experiment-specific staining artifacts. Adversary head predicts experiment ID from features while gradient reversal layer forces the backbone to ignore experiment signals. Result: experiment-invariant representations that generalize across different H&E staining protocols and scanners. Enable with `--use_dann` flag (default: disabled for faster training).
- **Deterministic Validation Sets (Reproducible Cross-Validation)**: All folds use stratified k-fold splitting with fixed random seeds (`seed = 42 + fold_index`), ensuring validation sets are identical across training runs. This enables reliable model comparison and debugging without randomness in fold assignment. For example, Fold 0 always contains the same 20% of data as validation, while Folds 1-4 use their own consistent partitions. Training and validation indices are strictly disjoint with no overlap.
- **Macenko Stain Normalization**: Applied exclusively during DeepHP H&E pre-training to normalize color variations across different staining protocols and tissue scanners, improving backbone generalization. Reference image is automatically created/verified on login node before training starts.
- **Proactive Environment Verification**: All scripts verify virtual environment dependencies before SLURM job submission, preventing runtime failures with early, actionable error messages.
- **Centralized Configuration**: All paths and environment variables in `config.py` support dynamic resolution and environment variable overrides for portability across different systems.
- **Stride-128 Rescue Pass**: Dense-window overlap to "catch" sparse bacteria that fall in gaps at default strides.
- **Top-3 Mixed MIL**: Balances sensitivity with noise resilience by averaging the top 3 most confident tissue chunks.
- **Contrast-Boosted TTA**: 16-way transforms (8 spatial + 1.1x contrast jitter) to "pop" faint IHC signals.
- **Modern PyTorch API**: Uses current `torch.amp.GradScaler` with automatic device detection instead of deprecated `torch.cuda.amp.GradScaler`, ensuring forward compatibility with future PyTorch versions.
- **Per-Fold Threshold Calibration (Post-Processing)**: After training all 5 folds, each fold's validation predictions are analyzed to compute optimal classification thresholds that maximize F1 score. These per-fold thresholds are then applied to test predictions before ensemble fusion, improving decision boundary quality and eliminating unnecessary false positives.
- **Weighted Ensemble Voting (Post-Processing)**: Combines predictions from all 5 folds using fold-performance-weighted voting. Folds with higher validation F1 scores receive greater weight, ensuring the most reliable folds contribute more to the final patient-level decision. This hierarchical confidence weighting significantly improves robustness and generalization.
- **Hybrid Ensemble Strategy**: Intelligently fuses predictions using three parallel methods:
  - **High Confidence Zone (>0.95)**: Uses weighted ensemble voting
  - **Uncertainty Zone (0.35-0.55)**: Uses meta-classifier fallback for difficult cases
  - **Medium Confidence (0.55-0.95)**: Intelligently blends both methods (60% ensemble + 40% meta)
  - **Result**: Best-in-class accuracy with zero false positives for clinical safety

---

## Hardware & Optimization
- **Compute**: Optimized for **NVIDIA L40S/A40/A100 (48GB/80GB)**.
- **Precision**: `torch.set_float32_matmul_precision('high')`.
- **Data Locality**: Automated node-local `/tmp` storage sync via `run_h_pylori.sh`.
