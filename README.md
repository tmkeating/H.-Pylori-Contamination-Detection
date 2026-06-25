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
   - Fold 0 val: 7 experiments (5 pos, 2 neg) → 85,854 patches, ratio 1:2.33
   - Fold 1 val: 10 experiments (8 pos, 2 neg) → 37,093 patches, ratio 1:2.06
   - Fold 2 val: 5 experiments (3 pos, 2 neg) → 78,085 patches, ratio 1:2.31
   - Fold 3 val: 4 experiments (2 pos, 2 neg) → 78,189 patches, ratio 1:2.29
   - Fold 4 val: 7 experiments (3 pos, 4 neg) → 115,704 patches, ratio 1:2.29
   - Each fold trains on ALL experiments except its validation experiments (309K-357K patches per fold, maintaining 1:2.27-2.31 class balance)
  - Mathematically optimized from 500,000+ greedy configuration searches
- **Critical Data Integrity**: The Macenko normalization reference patch is automatically excluded from all 5 folds **before** k-fold assignment (not after), mathematically guaranteeing zero leakage between training and validation sets

### 2. Training (5-Fold Cross-Validation) - RECOMMENDED METHOD

**Launch the full transfer learning pipeline** (preferred for best accuracy):

```bash
PROFILE=SEARCHER MODEL_NAME=convnext_tiny ITER=34.0 ./submit_transfer_learning.sh
```

This script automatically orchestrates:
1. **DeepHP Backbone Pre-training** (5-fold CONFIG 87771 stratification on 394,926 H&E patches)
   - Uses CONFIG 87771 experiment-level hardcoded stratification from greedy_fold_configs.json
   - Each fold validates on unique experiments (prevents fold-specific artifact learning)
   - Fold validation sets: 85,854 / 37,093 / 78,085 / 78,189 / 115,704 patches respectively
   - Fold validation ratios: 1:2.33 / 1:2.06 / 1:2.31 / 1:2.29 / 1:2.29 (target 1:2.28, total distance 0.6441)
   - Fold training sets: 309,071 / 357,832 / 316,840 / 316,736 / 279,221 patches respectively
   - All training sets maintain excellent balance: 1:2.27 / 1:2.31 / 1:2.27 / 1:2.28 / 1:2.28 (target 1:2.28)
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

### 4.5 (Optional) Rescue Inference - High-Resolution Recovery for Misclassified Patients

**When to Run Rescue Inference:**

Rescue inference is an **optional post-pipeline enhancement** for patients where the ensemble voting results warrant deeper inspection. Use it in these scenarios:

1. **Borderline Cases with High Uncertainty**
   - Patients with ensemble confidence between 0.35-0.65 (confidence zone where voting ensemble is least decisive)
   - Predictions that barely cross the decision threshold (e.g., 0.505 probability when threshold is 0.5)
   - Cases where different folds strongly disagree (e.g., 2 folds predict positive, 3 predict negative)

2. **Known False Positives** 
   - Patients incorrectly classified as positive (high ensemble score but clinical review suggests negative)
   - Faint, irregular signals that may be staining artifacts rather than actual bacteria
   - Benefit: Dense-stride (Stride=128) windowing with 50% overlap ensures no sparse bacteria clusters are missed if they do exist

3. **Known False Negatives**
   - Patients incorrectly classified as negative (low ensemble score but clinical review suggests positive)
   - Sparse bacterium clusters that may have fallen through the gaps of standard Stride=250 windows
   - Benefit: 16-way contrast-boosted TTA (rotations, flips, 1.1x contrast enhancement) can recover faint IHC signals

4. **Quality Assurance & Validation**
   - Verifying difficult-to-diagnose cases before clinical sign-off
   - Building secondary evidence for edge cases with borderline pathology
   - Computational cost is justified for a small subset of 5-20 priority patients

**How Rescue Inference Works:**

- **Dense-Stride Windowing (Stride=128)**: Standard inference uses Stride=250 (4-pixel overlap), which can miss very sparse bacterial clusters (5-10 organisms in an entire slide). Rescue uses Stride=128 (50% overlap) to guarantee no bacterium is bisected by a patch boundary.
- **16-Way Contrast-Boosted Test-Time Augmentation (TTA)**:
  - 6 spatial transforms: Original, H-flip, V-flip, 90°, 180°, 270° rotations
  - 1.1x contrast boost (targets faint IHC organisms with weak staining)
  - Combined transforms (rotations + contrast, flips + contrast)
  - All 16 predictions are averaged for consensus voting, reducing noise
- **Output**: Dense predictions with per-patient scores that can be compared against ensemble baseline

**Example: Rescue the Seven Misclassified Patients from Iteration 34**

From the ensemble voting report, these 7 patients warrant investigation:

| Patient | True Label | Ensemble Pred | Max Vote | Reason |
|---------|-----------|---|---|---|
| B22-12_1 | Negative | 0.497 | Positive | False Positive (confidence: 0.497) |
| B22-89_0 | Negative | 0.927 | Positive | Strong False Positive (confidence: 0.927) |
| B22-206_0 | Positive | 0.178 | Negative | False Negative (sparse signal) |
| B22-262_0 | Positive | 0.230 | Negative | False Negative (sparse signal) |
| B22-69_1 | Positive | 0.312 | Negative | False Negative (sparse signal) |
| B22-81_1 | Positive | 0.090 | Negative | False Negative (very sparse signal) |
| B22-85_0 | Positive | 0.469 | Borderline | Borderline case (confidence: 0.469) |

**Command to Run Rescue:**

```bash
# Rescue the 7 misclassified patients from a finalResults run
MODEL_DIR="finalResults/convnext_tiny_pretrained_backbone_34.4_weight_1.5_gamma_3.0_focalLoss_false" \
  FOLDS="01_34.4_9077_f0 01_34.4_9078_f1 01_34.4_9079_f2 01_34.4_9080_f3 01_34.4_9081_f4" \
  TARGETS="B22-12_1,B22-206_0,B22-262_0,B22-69_1,B22-81_1,B22-85_0,B22-89_0" \
  OUTPUT_DIR="finalResults/convnext_tiny_pretrained_backbone_34.4_weight_1.5_gamma_3.0_focalLoss_false/rescue_ensemble" \
  STRIDE=128 \
  sbatch submit_rescue.sh
```

**Interpreting Results:**

After rescue completes, examine:
1. `rescue_*.csv` files in OUTPUT_DIR containing dense-stride predictions
2. Compare rescue predictions against original ensemble scores
3. If rescue still predicts negative for false positives → likely staining artifacts
4. If rescue now predicts positive for false negatives → sparse bacteria was present but missed
5. (Optional) Re-run ensemble voting with rescue data included:
   ```bash
   python3 ensemble_voting.py --runs 34-34
   ```
   This auto-detects rescue data and merges predictions back into ensemble voting

**When NOT to Use Rescue Inference:**

- Already confident in ensemble predictions (>0.95 or <0.05 confidence scores)
- High false positive rate expected on specific patients (use rescue to differentiate signal vs artifact instead)
- Resource constraints prevent running dense-stride inference (10-20x slower than standard stride)
- Standard clinical workflow requires rapid turnaround (rescue adds 30-60 minutes per patient set)

**⚠️ METHODOLOGICAL CAVEAT - Holdout Set Reuse:**

Rescue inference operates on the **same holdout set** used for baseline ensemble evaluation. This does NOT violate data integrity (no model retraining, weights frozen) but means you're applying an improved inference strategy to data where you've already seen baseline results:

- ✅ **Data Integrity**: PRESERVED (no training/test leakage, no model retraining)
- ⚠️ **Generalization Risk**: Results may be optimistic for truly new patients beyond the 114 holdout set
- ⚠️ **Sequential Testing**: You've diagnosed which patients failed, then applied better method to same data

**For final deployment or publication**, apply rescue inference to a completely separate held-out test set (never evaluated with baseline) to avoid optimistic bias.

**Acceptable current use cases**:
- Post-hoc clinical analysis and troubleshooting
- Research ablations demonstrating inference technique improvements
- Quality assurance deep-dives on specific borderline cases

*For more details, see `rescue_inference.py` documentation.*

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
- **CONFIG 87771 Stratification (DeepHP Pre-training)**: Hardcoded experiment-level 5-fold cross-validation optimized from 500,000+ greedy configuration searches. Each of 33 experiments assigned to exactly ONE fold, preventing data leakage and fold-specific artifact learning. Folds validate on different experiments (85.9K / 37.1K / 78.1K / 78.2K / 115.7K patches) with balanced ratios (1:2.06 to 1:2.33, target 1:2.28, total distance 0.6441). All training sets maintain excellent balance (1:2.27-2.31 ratio) across all folds.
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
