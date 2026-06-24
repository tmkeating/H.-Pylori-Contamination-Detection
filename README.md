# H. Pylori Contamination Detection (Iteration 34.0: Hybrid Ensemble Fusion)

This project implements a **High-Resolution Multi-Stage MIL Pipeline** for the automated detection of *H. pylori* contamination in histology tissue samples. It features architecture designed to identify sparse bacterium clusters in high-resolution whole-slide imaging, combined with an **intelligent Hybrid Ensemble**.

**Transfer Learning with DeepHP H&E Pre-training** (Available for backbone initialization)

The pipeline supports backbone pre-training on the **DeepHP dataset** (33 experiments, 394,926 H&E-stained histology patches, 120,374 positive / 274,551 negative, ratio 1:2.28) before fine-tuning on the patient-level IHC data from HelicoDataSet (268 patients total: 154 training / 114 holdout; 216,326 IHC-stained histology patches: 128,724 training [62,800 pos / 65,924 neg, ratio 1:1.05] + 87,602 holdout [40,642 pos / 46,960 neg, ratio 1:1.16])

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
  - Fold 0 val: 7 experiments (4 pos, 3 neg) → 87,532 patches, ratio 1:2.33
  - Fold 1 val: 10 experiments (3 pos, 7 neg) → 89,516 patches, ratio 1:2.06
  - Fold 2 val: 5 experiments (4 pos, 1 neg) → 20,347 patches, ratio 1:2.31
  - Fold 3 val: 4 experiments (4 pos, 0 neg) → 99,120 patches, ratio 1:2.81
  - Fold 4 val: 7 experiments (6 pos, 1 neg) → 98,410 patches, ratio 1:2.29
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
   - Fold ratios: 1:2.33 / 1:2.06 / 1:2.31 / 1:2.81 / 1:2.29 (target 1:2.28, total distance 0.6441)
   - **Domain Adversarial Neural Networks (DANN)**: Optionally enabled with `--use_dann` flag
     - Prevents learning of experiment-specific staining artifacts
     - Adversary head predicts experiment ID from features → forces experiment-invariant representations
     - Gradient reversal layer negates gradients during backprop → confuses adversary
     - Parameters: `--dann_lambda 1.0` (gradient scaling), `--dann_weight 1.0` (loss weighting)
   - **Per-Fold Training Weights (Optional)**: Support for fold-specific class weighting during training
     - By default, all folds use the same `--pos_weight` parameter (e.g., 1.5)
     - For fine-tuning, specify comma-separated per-fold weights: `--pos_weight 1.5,1.8,1.6,1.4,1.7`
     - Fold 0 trains with pos_weight=1.5, Fold 1 with pos_weight=1.8, etc.
     - Use case: Compensate for per-fold class imbalance or fold-specific convergence issues
     - Automatically distributed by `submit_train_deepHP.sh` to each SLURM job
     - Example: Folds with more negative patches can use higher pos_weight to balance gradient magnitudes
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

### 2.5 Advanced Configuration Options (Optional)

Both training pipelines support additional environment variables for fine-tuning DeepHP pre-training behavior:

**Per-Fold Positive Class Weights** (for class imbalance compensation):
```bash
# Use same pos_weight for all folds (default: 1.5)
PROFILE=SEARCHER MODEL_NAME=convnext_tiny ITER=34.0 POS_WEIGHT=2.0 ./submit_transfer_learning.sh

# Use fold-specific pos_weights (comma-separated: fold0,fold1,fold2,fold3,fold4)
PROFILE=SEARCHER MODEL_NAME=convnext_tiny ITER=34.0 POS_WEIGHT=1.5,1.8,1.6,1.4,1.7 ./submit_transfer_learning.sh
```

**Focal Loss Configuration** (for hard example mining):
```bash
# Enable Focal Loss with gamma parameter (controls focus on hard examples)
PROFILE=SEARCHER MODEL_NAME=convnext_tiny ITER=34.0 USE_FOCAL_LOSS=True GAMMA=3.0 ./submit_transfer_learning.sh
```

**Domain Adversarial Neural Networks** (for experiment-invariant features):
```bash
# Enable DANN with gradient scaling and loss weighting
PROFILE=SEARCHER MODEL_NAME=convnext_tiny ITER=34.0 USE_DANN=True DANN_LAMBDA=1.0 DANN_WEIGHT=0.5 ./submit_transfer_learning.sh
```

All parameters are automatically propagated to all 5 SLURM training jobs and per-fold threshold calibration.

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
**Stage 3: `ensemble_voting_deepHP.py`** (after DeepHP pre-training only)

**Strategy: F1-Score Weighted Backbone Fusion**

After training all 5 folds on DeepHP H&E patches, the backbone weights are combined using a weighted ensemble approach based on per-fold validation F1 scores:

1. **Per-Fold Performance Evaluation**
   - Each fold is evaluated on its validation set (unique experiments not seen during training)
   - F1 score computed as: F1 = 2 × (Precision × Recall) / (Precision + Recall)
   - F1 score represents the fold's ability to correctly identify H. pylori patches on unseen experiments
   - Higher F1 → better fold performance → higher weight in final ensemble

2. **Weight Calculation**
   - Fold weights normalized so they sum to 1.0:
     - `weight_fold_i = F1_score_fold_i / sum(F1_scores_all_folds)`
   - Example: If fold F1 scores are [0.25, 0.22, 0.28, 0.24, 0.26], weights become [0.25, 0.22, 0.28, 0.24, 0.26]
   - Folds with superior validation performance receive proportionally higher influence in the averaged backbone

3. **Weighted Backbone Averaging**
   - For each model parameter θ (weights and biases), compute weighted average:
     - `θ_ensemble = Σ(weight_i × θ_i)` for each fold i
   - Example: ConvNeXt-Tiny backbone has ~28M parameters
   - Each parameter averaged across 5 folds, weighted by respective F1 scores
   - Result: Unified backbone that captures the best-performing feature extractors from all 5 folds

4. **Advantages Over Simple Equal-Weight Averaging**
   - ✅ **Better generalization**: Folds with poor validation performance have reduced influence
   - ✅ **Data-driven optimization**: Weights automatically adjust based on actual fold performance
   - ✅ **Noise reduction**: High-variance folds (unstable performance) have lower weight
   - ✅ **Experiment-robust features**: Folds that generalize better to unseen experiments dominate
   - ✅ **Prevents bad fold contamination**: A poorly-trained fold won't degrade final backbone

5. **Output**
   - `deephp_backbone_final_{run_id}_convnext_tiny_{iter}.pth`: F1-weighted averaged backbone ready for transfer learning
   - `{run_id}_{iter}_ensemble_weights_f1.json`: Per-fold weights used in averaging (for reproducibility)
   - Can be loaded directly into fine-tuning pipeline with `--pretrained_backbone_path` argument
   - Generates ensemble predictions on validation sets for further analysis

**Rationale**: Simple equal-weight averaging assumes all folds contribute equally, but in practice, folds with different experiment distributions may learn at different rates. F1-weighted averaging lets the data speak: folds that better generalize to unseen experiments receive proportionally more weight in the final backbone, resulting in a transfer-learning-ready feature extractor optimized for H. pylori morphology recognition.

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
python3 ensemble_voting_deepHP.py --run 32 --strategy f1
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

To Do


## 🛠️ Key Pipeline Features
- **CONFIG 87771 Stratification (DeepHP Pre-training)**: Hardcoded experiment-level 5-fold cross-validation optimized from 500,000+ greedy configuration searches. Each of 33 experiments assigned to exactly ONE fold, preventing data leakage and fold-specific artifact learning. Folds validate on different experiments (87.5K / 89.5K / 20.3K / 99.1K / 98.4K patches) with balanced ratios (1:2.06 to 1:2.81, target 1:2.28, total distance 0.6441). All folds train on ~307K patches from all other experiments.
- **Domain Adversarial Neural Networks (DANN - Optional)**: Advanced technique for DeepHP pre-training that prevents models from learning experiment-specific staining artifacts. Adversary head predicts experiment ID from features while gradient reversal layer forces the backbone to ignore experiment signals. Result: experiment-invariant representations that generalize across different H&E staining protocols and scanners. Enable with `--use_dann` flag (default: disabled for faster training).
- **Deterministic Validation Sets (Reproducible Cross-Validation)**: All folds use stratified k-fold splitting with fixed random seeds (`seed = 42 + fold_index`), ensuring validation sets are identical across training runs. This enables reliable model comparison and debugging without randomness in fold assignment. For example, Fold 0 always contains the same 20% of data as validation, while Folds 1-4 use their own consistent partitions. Training and validation indices are strictly disjoint with no overlap.
- **Macenko Stain Normalization**: Applied exclusively during DeepHP H&E pre-training to normalize color variations across different staining protocols and tissue scanners, improving backbone generalization. Reference image is automatically created/verified on login node before training starts.
- **Proactive Environment Verification**: All scripts verify virtual environment dependencies before SLURM job submission, preventing runtime failures with early, actionable error messages.
- **Centralized Configuration**: All paths and environment variables in `config.py` support dynamic resolution and environment variable overrides for portability across different systems.
- **Stride-128 Rescue Pass**: Dense-window overlap to "catch" sparse bacteria that fall in gaps at default strides.
- **Top-3 Mixed MIL**: Balances sensitivity with noise resilience by averaging the top 3 most confident tissue chunks.
- **Contrast-Boosted TTA**: 16-way transforms (8 spatial + 1.25x contrast jitter) to "pop" faint IHC signals.
- **Modern PyTorch API**: Uses current `torch.amp.GradScaler` with automatic device detection instead of deprecated `torch.cuda.amp.GradScaler`, ensuring forward compatibility with future PyTorch versions.
- **Per-Fold Threshold Calibration (Post-Processing)**: After training all 5 folds, each fold's validation predictions are analyzed to compute optimal classification thresholds that maximize F1 score. These per-fold thresholds are then applied to test predictions before ensemble fusion, improving decision boundary quality and eliminating unnecessary false positives.
- **Weighted Ensemble Voting (Post-Processing)**: Combines predictions from all 5 folds using fold-performance-weighted voting. Folds with higher validation F1 scores receive greater weight, ensuring the most reliable folds contribute more to the final patient-level decision. This hierarchical confidence weighting significantly improves robustness and generalization.
- **Hybrid Ensemble Strategy**: Intelligently fuses predictions using three parallel methods:
  - **High Confidence Zone (>0.95)**: Uses weighted ensemble voting
  - **Uncertainty Zone (0.35-0.55)**: Uses meta-classifier fallback for difficult cases
  - **Medium Confidence (0.55-0.95)**: Intelligently blends both methods (60% ensemble + 40% meta)

---

## Hardware & Optimization
- **Compute**: Optimized for **NVIDIA L40S/A40/A100 (48GB/80GB)**.
- **Precision**: `torch.set_float32_matmul_precision('high')`.
- **Data Locality**: Automated node-local `/tmp` storage sync via `run_h_pylori.sh`.
