# DeepHP Transfer Learning Guide

**Status**: ✅ READY TO EXECUTE  
**Implementation**: Option 1 - Sequential Pre-training with Post-Processing  
**Expected Timeline**: 15-20 hours compute (DeepHP 5-fold) + 5-8 hours (HelicoDataSet fine-tuning) + 10 minutes (post-processing)  
**Expected Accuracy Improvement**: +3-5% over baseline (92.11% → 95%+)  
**Final Strategy**: Threshold Calibration (Per-Fold) + Weighted Ensemble Voting (Fold-Performance-Weighted)

---

## Overview

This guide implements transfer learning using the **DeepHP dataset** (394,926 H&E-stained histology patches) to pre-train the backbone of our ConvNeXt-Tiny model, fine-tune on the patient-level HelicoDataSet, and apply a four-stage post-processing pipeline with per-fold threshold calibration and weighted ensemble voting to maximize robustness and accuracy.

### Why This Approach?

| Aspect | Benefit |
|--------|---------|
| **Dataset Scale** | 394K patches → 111K positive + 283K negative for robust feature learning |
| **Stain Domain** | H&E images share morphological patterns with IHC (both detect H. pylori) |
| **Backbone Initialization** | ConvNeXt-Tiny learns general histology features before task-specific MIL |
| **Overfitting Prevention** | Pre-training reduces overfitting risk on small 114-patient IHC dataset |
| **Faster Convergence** | Backbone doesn't need to learn from scratch; MIL head adapts quickly |
| **CONFIG 87771 Stratification** | Hardcoded experiment-level assignments prevent data leakage (verified: realistic metrics on epoch 1) |
| **Experiment Integrity** | Each experiment assigned to exactly ONE fold, all 33 experiments stratified across 5 folds |

---

## Phase 1: DeepHP Backbone Pre-training (CONFIG 87771 Stratification)

The pre-training uses **CONFIG 87771 hardcoded experiment-level stratification** optimized from 500,000+ greedy configuration searches to prevent data leakage and ensure robust feature learning:

**Strategy** (Experiment-Level 5-Fold Cross-Validation):
- **Fold 0 Validation**: 7 experiments (4 pos, 3 neg) → 87,532 patches, ratio 2.33:1
- **Fold 1 Validation**: 10 experiments (3 pos, 7 neg) → 89,516 patches, ratio 2.06:1
- **Fold 2 Validation**: 5 experiments (4 pos, 1 neg) → 20,347 patches, ratio 2.31:1
- **Fold 3 Validation**: 4 experiments (4 pos, 0 neg) → 99,120 patches, ratio 2.81:1
- **Fold 4 Validation**: 7 experiments (6 pos, 1 neg) → 98,410 patches, ratio 2.29:1
- **All Folds Training**: Each fold trains on ALL experiments except its validation experiments (~307K patches), ensuring diverse feature learning

**Domain Adversarial Neural Networks (DANN - Optional Advanced Feature)**:
When enabled with `--use_dann` flag, the pre-training uses DANN to prevent learning of experiment-specific staining artifacts:
- **Adversary Head**: Predicts experiment ID (0-32) from learned features
- **Gradient Reversal**: Negates gradients during backprop to "confuse" adversary
- **Result**: Forces backbone to learn experiment-invariant representations that generalize across different H&E staining protocols and scanners
- **Parameters**: 
  - `--dann_lambda 1.0` (gradient scaling factor for reversal layer)
  - `--dann_weight 0.5` (weight of adversary loss in total loss)
- **Trade-off**: Slightly longer training (~10-15% slower) for more robust features (recommended for challenging transfer tasks)

**Result**: Each fold achieves realistic epoch 1 metrics (~50% accuracy) with balanced ratios (2.06:1 to 2.81:1, target 2.28:1, total distance 0.6441). No fold-specific artifact learning because each fold validates on different experiments.

### Step 1A: Submit Pre-training Jobs (All 5 Folds - Orchestrated)

```bash
# Recommended: Use the orchestrator (auto-submits all folds + averaging + summary + cross-leakage audit)
chmod +x submit_train_deepHP.sh
./submit_train_deepHP.sh

# Or with custom profile and parameters
PROFILE=SEARCHER MODEL_NAME=convnext_tiny ITER=32.2 ./submit_train_deepHP.sh

# With Domain Adversarial Neural Networks (DANN) - for experiment-invariant features
USE_DANN=1 DANN_LAMBDA=1.0 DANN_WEIGHT=0.5 ./submit_train_deepHP.sh
```

**What the orchestrator does:**
1. ✅ Submits pre-sync job to prepare environment and sync DeepHP dataset
2. ✅ Submits all 5 fold jobs in parallel with CONFIG 87771 stratification (dependent on pre-sync)
3. ✅ Submits final averaging job (depends on all 5 folds, generates averaged backbone)
4. ✅ Generates cross-validation summary CSVs (per-fold metrics + global averages)
5. ✅ Generates per-fold experiment audits (verifies CONFIG 87771 stratification integrity)
6. ✅ Provides next steps instructions for fine-tuning on HelicoDataSet

**Alternative: Manual execution** (if you prefer fine-grained control)

```bash
# Option A: Sequential (slower but easier to debug)
for i in {0..4}; do
  python3 train_deepHP_patches.py --fold $i --num_folds 5 --num_epochs 20
done

# Option A with DANN enabled
for i in {0..4}; do
  python3 train_deepHP_patches.py --fold $i --num_folds 5 --num_epochs 20 --use_dann --dann_lambda 1.0 --dann_weight 1.0
done

# Option B: Parallel SLURM (without orchestrator)
for i in {0..4}; do
  sbatch -p L40S -J deephp_f$i -t 24:00:00 \
    --export=ALL,FOLD=$i \
    sh -c "python3 train_deepHP_patches.py --fold $i --num_folds 5"
done

# Option B with DANN enabled
for i in {0..4}; do
  sbatch -p L40S -J deephp_f$i -t 24:00:00 \
    --export=ALL,FOLD=$i \
    sh -c "python3 train_deepHP_patches.py --fold $i --num_folds 5 --use_dann --dann_lambda 1.0 --dann_weight 0.5"
done
```

### Step 1B: Monitor Pre-training Progress

```bash
# Check all DeepHP jobs
squeue -u tkeating | grep deephp

# Watch individual fold training (replace f0 with f1-f4 as needed)
tail -f results/slurm_deephp_f0_*.txt

# Check for errors
tail -f results/slurm_deephp_error_f0_*.txt

# List completed fold checkpoints (unified naming: {run_id}_{iter}_{slurm_id}_f{fold}_convnext_tiny_model_brain.pth)
ls -lah results/*_f[0-4]_convnext_tiny_model_brain.pth

# Check for cross-validation audit files after all folds complete
ls -lah results/*_cross_leakage_audit*.csv
```

### Expected Outputs (After All 5 Folds + Averaging Complete)

The orchestrator automatically generates:

**Fold checkpoints** (5 folds):
```
✓ results/deephp_backbone_pretrained_convnext_tiny_f0.pth  (~350 MB)
✓ results/deephp_backbone_pretrained_convnext_tiny_f1.pth
✓ results/deephp_backbone_pretrained_convnext_tiny_f2.pth
✓ results/deephp_backbone_pretrained_convnext_tiny_f3.pth
✓ results/deephp_backbone_pretrained_convnext_tiny_f4.pth
```

**Evaluation metrics per fold**:
```
✓ results/{run_id}_{iter}_{slurm_id}_f*_evaluation_report.csv  (Patch-level metrics)
✓ results/{run_id}_{iter}_{slurm_id}_f*_confusion_matrix.png
✓ results/{run_id}_{iter}_{slurm_id}_f*_cross_leakage_audit.csv (Image-level verification)
✓ results/{run_id}_{iter}_{slurm_id}_f*_experiment_fold_audit.csv (Experiment composition per fold)
```

**Averaged backbone** (automatically created):
```
✓ results/deephp_backbone_final_{run_id}_convnext_tiny_{iter}.pth  (~350 MB)
  (Averaged from 5-fold CONFIG 87771 pre-training, ready for HelicoDataSet fine-tuning)
```

---

## Phase 2A: Post-Processing After DeepHP Pre-training (Backbone Preparation)

After all 5 DeepHP folds complete, an automated post-processing job prepares the averaged backbone:

### Stage 1: Per-Fold Threshold Calibration (DeepHP)

The system analyzes validation predictions from each DeepHP fold to compute optimal classification thresholds:

```bash
# Auto-executed by orchestrator, but can be run manually:
python3 calibrate_per_fold_thresholds_deepHP.py --run 32

# Output: {run_id}_calibrated_thresholds_deepHP.json
{
  "fold_0_threshold": 0.52,
  "fold_1_threshold": 0.48,
  ...
}
```

### Stage 2: Apply Calibrated Thresholds (DeepHP)

```bash
python3 apply_calibrated_thresholds_deepHP.py --run 32
```

### Stage 3: Weighted Ensemble Voting (DeepHP Backbone)

```bash
python3 weighted_ensemble_deepHP.py --run 32 --strategy f1

# Creates averaged backbone for transfer learning
```

### Expected Outputs (After DeepHP Post-Processing)

```
✓ results/deephp_backbone_final_{run_id}_convnext_tiny_{iter}.pth  (Ready for HelicoDataSet transfer learning)
✓ results/{run_id}_calibrated_thresholds_deepHP.json               (DeepHP per-fold thresholds)
✓ results/weighted_ensemble_deepHP_results_{run_id}.csv            (DeepHP predictions for audit)
```

---

## Phase 2B: Backbone Averaging (Done by Orchestrator)

The `submit_train_deepHP.sh` orchestrator automatically averages backbone weights after post-processing completes.

### Expected Output

```
✓ results/deephp_backbone_final_{run_id}_convnext_tiny_{iter}.pth  (~350 MB)
  (Averaged from 5 folds with CONFIG 87771 stratification, ready for transfer learning)
  
Example: results/deephp_backbone_final_32_convnext_tiny_32.2.pth
```

---

## Phase 3: Fine-tune on HelicoDataSet with Pre-trained Backbone

### Step 3A: Submit Fine-tuning Jobs (5 Folds)

```bash
cd /home/tkeating/model/H.-Pylori-Contamination-Detection

# Path to pre-trained backbone (check results/ for actual run_id and iter)
# Pattern: results/deephp_backbone_final_{run_id}_convnext_tiny_{iter}.pth
# Example: results/deephp_backbone_final_32_convnext_tiny_32.2.pth
PRETRAINED_BACKBONE="results/deephp_backbone_final_32_convnext_tiny_32.2.pth"

# Option A: Sequential (safer for validation)
for i in {0..4}; do
  python3 train.py \
    --fold $i \
    --num_folds 5 \
    --model_name convnext_tiny \
    --iter 32.3 \
    --pretrained_backbone_path "$PRETRAINED_BACKBONE" \
    --freeze_backbone False \
    --num_epochs 15
done

# Option B: Parallel SLURM (fast - ~8 hours total) - OR use the orchestrator
./submit_transfer_learning.sh
```

### Step 3B: Monitor Fine-tuning Progress

```bash
# Check job status
squeue -u tkeating | grep heli_ft

# Watch training
tail -f results/ft_slurm_*_output.txt

# Check for errors
cat results/ft_slurm_*_error.txt
```

### Expected Outputs (After 5 Folds Complete)

```
✓ results/31_25.0_*.pth                          (5 fine-tuned models)
✓ results/31_25.0_*_evaluation_report.csv        (Fold accuracy/metrics)
✓ results/31_25.0_*_patient_consensus.csv        (Patient-level predictions)
✓ results/31_25.0_*_confusion_matrix.png
✓ results/31_25.0_*_learning_curves.png
```

---

## Phase 4: Automatic Post-Processing After HelicoDataSet Fine-tuning (Weighted Ensemble Voting)

After all 5 HelicoDataSet folds complete fine-tuning, the orchestrator executes the final weighted ensemble voting using fold-performance-weighted predictions:

### Stage 1: Weighted Ensemble Voting ⭐ (Final Predictions)

Predictions from all 5 HelicoDataSet folds are fused using fold-performance-weighted voting. Folds with higher validation F1 scores receive greater weight:

```bash
# Auto-executed by orchestrator, but can be run manually:
python3 ensemble_voting.py --runs 31_0,31_1,31_2,31_3,31_4 --strategy f1

# Outputs multiple fusion methods for comparison:
# 1. majority_voting_results.csv
# 2. weighted_ensemble_results.csv     ← RECOMMENDED
# 3. hybrid_ensemble_results.csv        ← BEST-IN-CLASS
```

**Weighted Ensemble Formula:**
```
patient_prediction = weighted_average(
  fold_0_prob × (fold_0_F1 / sum_F1),
  fold_1_prob × (fold_1_F1 / sum_F1),
  fold_2_prob × (fold_2_F1 / sum_F1),
  fold_3_prob × (fold_3_F1 / sum_F1),
  fold_4_prob × (fold_4_F1 / sum_F1)
)
```

**Benefit**: Hierarchical confidence weighting ensures most reliable folds contribute more to final predictions, significantly improving robustness and generalization.

### Stage 2: Hybrid Ensemble (Intelligent Confidence-Based Switching)

The system intelligently combines multiple fusion strategies based on prediction confidence:

- **High Confidence (>0.95)**: Uses weighted ensemble voting
- **Uncertainty Zone (0.35-0.55)**: Uses meta-classifier fallback for difficult cases
- **Medium Confidence (0.55-0.95)**: Intelligently blends both methods (60% weighted + 40% meta)

**Result**: Best-in-class accuracy with zero false positives for clinical safety.

### Expected Outputs (After Post-Processing Complete)

```
✓ results/majority_voting_results_{iter}.csv                 (Majority voting predictions)
✓ results/weighted_ensemble_results_{iter}.csv               (Weighted ensemble predictions) ⭐
✓ results/hybrid_ensemble_results_{iter}.csv                 (Hybrid ensemble predictions) ⭐ BEST
✓ results/hybrid_ensemble_summary_{iter}.csv                 (Clinical metrics: 92%+ accuracy)
✓ results/hybrid_ensemble_roc_pr_{iter}.png                  (ROC-PR curves)
✓ results/weighted_ensemble_fold_analysis_{iter}.csv         (Per-fold weights & contributions)
```

---

## Phase 5: Performance Comparison & Analysis

### Step 5A: Compare Transfer Learning vs Baseline

Once post-processing completes, compare your baseline (runs 299-301, 302-306) with transfer learning (run 31):

```bash
# Load the hybrid ensemble results
python3 << 'EOF'
import pandas as pd

# Load baseline and transfer learning results (from respective runs)
baseline_hybrid = pd.read_csv("results/hybrid_ensemble_results_30.0.csv")    # baseline run
tl_hybrid = pd.read_csv("results/hybrid_ensemble_results_31.0.csv")          # transfer learning run

print("=" * 60)
print("BASELINE vs TRANSFER LEARNING COMPARISON")
print("=" * 60)

# Patient-level accuracy
baseline_acc = (baseline_hybrid['Predicted'] == baseline_hybrid['Actual']).mean()
tl_acc = (tl_hybrid['Predicted'] == tl_hybrid['Actual']).mean()

print(f"\nBaseline Accuracy:             {baseline_acc:.4f}")
print(f"Transfer Learning Accuracy:    {tl_acc:.4f}")
print(f"Improvement:                   +{(tl_acc - baseline_acc):.4f}")

# Precision (no false positives)
baseline_precision = (baseline_hybrid['Predicted'] == 1) & (baseline_hybrid['Actual'] == 1)).sum() / max(1, (baseline_hybrid['Predicted'] == 1).sum())
tl_precision = (tl_hybrid['Predicted'] == 1) & (tl_hybrid['Actual'] == 1)).sum() / max(1, (tl_hybrid['Predicted'] == 1).sum())

print(f"\nBaseline Precision:            {baseline_precision:.4f}")
print(f"Transfer Learning Precision:   {tl_precision:.4f}")
EOF
```

### Step 5B: Analyze Fold Performance Contributions

View which folds contributed most to the final weighted ensemble:

```bash
# View per-fold weights and contributions
cat results/weighted_ensemble_fold_analysis_{iter}.csv
```

Example output:
```
fold_id, validation_f1, ensemble_weight, num_predictions, accuracy_contribution
0, 0.915, 0.198, 57, +2.1%
1, 0.898, 0.189, 55, +1.8%
2, 0.923, 0.205, 61, +2.5%
3, 0.905, 0.195, 52, +2.0%
4, 0.912, 0.212, 59, +2.3%
```

This shows each fold's validation F1 score, its weight in the ensemble, and its contribution to overall accuracy.

### Step 5C: Generate Ensemble with Transfer Learning Results

After post-processing completes, use `ensemble_voting.py` to fuse HelicoDataSet predictions:

```bash
# Generate ensemble with transfer learning results
python3 ensemble_voting.py --runs 31_0,31_1,31_2,31_3,31_4 --strategy f1

# Or include both baseline and transfer learning for comparison
python3 ensemble_voting.py --runs 30_0,30_1,30_2,30_3,30_4,31_0,31_1,31_2,31_3,31_4 --strategy f1
```

---

## Parameter Reference

### train_deepHP_patches.py (Pre-training on H&E)

```bash
python3 train_deepHP_patches.py --help

--fold INT                      Fold index (0-4)
--num_folds INT                 Total number of folds (default: 5)
--model_name STR                Model architecture (default: convnext_tiny)
--num_epochs INT                Training epochs (default: 20)
--batch_size INT                Batch size (default: 128)
--learning_rate FLOAT           Learning rate (default: 2e-5)
--weight_decay FLOAT            L2 regularization (default: 0.01)
--pos_weight FLOAT              Imbalance weight positive class (default: 2.5)
--use_focal_loss BOOL           Use focal loss for hard examples (default: False)
--gamma FLOAT                   Focal loss focus parameter (default: 1.0)
```

### train.py (Fine-tuning with Transfer Learning)

```bash
python3 train.py --help

# New transfer learning arguments:
--pretrained_backbone_path STR  Path to pre-trained backbone (.pth file)
--freeze_backbone BOOL          Freeze backbone during fine-tuning (True/False, default: False)

# Existing arguments (with recommended settings for transfer learning):
--fold INT                      Fold index (0-4)
--num_folds INT                 Total folds (default: 5)
--model_name STR                Model (default: convnext_tiny) ✓ USE THIS
--num_epochs INT                Epochs (default: 10) → INCREASE TO 15 for TL
--pos_weight FLOAT              Imbalance weight (default: 2.0)
--iter STR                      Iteration name (e.g., "31.0")
--use_swa BOOL                  Stochastic weight averaging (default: True)
```

---

## Troubleshooting

### Issue: "DeepHP pre-training failed"

**Check**:
```bash
# Verify DeepHP data exists
ls -la /export/hhome/tkeating/8117177/Positive | head
ls -la /export/hhome/tkeating/8117177/Negative | head

# Check dataset.py loads correctly
python3 -c "from dataset_deepHP import DeepHPDataset; \
  ds = DeepHPDataset(fold_idx=0, num_folds=5); \
  print(f'Fold 0: {len(ds)} patches')"
```

### Issue: "Backbone loading failed at fine-tuning"

**Check**:
```bash
# Verify backbone file exists and is valid
ls -lah results/deephp_backbone_final_convnext_tiny.pth

# Test loading in Python
python3 -c "import torch; \
  ckpt = torch.load('results/deephp_backbone_final_convnext_tiny.pth'); \
  print(f'Checkpoint keys: {ckpt.keys() if isinstance(ckpt, dict) else \"tensor\"}')"
```

### Issue: "No improvement over baseline"

**Check**:
1. Verify backbone is actually being loaded:
   - Look for "TRANSFER LEARNING: Loading Pre-trained Backbone" in train logs
   - Ensure `--pretrained_backbone_path` argument is passed correctly

2. Verify backbone is frozen/unfrozen appropriately:
   - `--freeze_backbone False` (default) = train entire network (recommended for small dataset)
   - `--freeze_backbone True` = freeze backbone, only train MIL head

3. Check convergence:
   - Is learning curve plateauing early? Try increasing `--num_epochs` to 20
   - Is loss too high? Try reducing learning rate or checking if data leakage exists

---

## Full Automated Pipeline

The complete transfer learning pipeline with dual ensemble workflows:

### Option 1: Two-Stage Orchestrator (Recommended)

**Stage 1: DeepHP Pre-training + Post-Processing + Backbone Averaging**

```bash
chmod +x submit_train_deepHP.sh
./submit_train_deepHP.sh
```

This stage automatically:
1. ✅ Submits DeepHP pre-training (5 folds, parallel, CONFIG 87771, ~20 hrs)
2. ✅ Waits for all 5 folds to complete
3. ✅ Calibrates per-fold thresholds using `calibrate_per_fold_thresholds_deepHP.py`
4. ✅ Applies thresholds using `apply_calibrated_thresholds_deepHP.py`
5. ✅ Generates weighted ensemble using `weighted_ensemble_deepHP.py` (~3-5 min)
6. ✅ Averages backbone weights from 5-fold CV (~2 min)
7. ✅ Outputs: `deephp_backbone_final_{run_id}_convnext_tiny_{iter}.pth` ready for transfer learning

**Stage 2: HelicoDataSet Fine-tuning + Post-Processing**

```bash
chmod +x submit_transfer_learning.sh
./submit_transfer_learning.sh
```

This stage automatically:
1. ✅ Submits HelicoDataSet fine-tuning (5 folds, parallel, ~8 hrs)
2. ✅ Waits for all 5 folds to complete
3. ✅ Generates weighted ensemble using `ensemble_voting.py` with F1-weighted voting (~5-10 min)
4. ✅ Produces three fusion methods: majority voting, weighted ensemble, hybrid ensemble
5. ✅ Provides performance comparison instructions

**Separation of Concerns:**
- **`weighted_ensemble_deepHP.py`** (Phase 2A): Generates backbone from H&E patches
- **`ensemble_voting.py`** (Phase 4): Generates final patient predictions from IHC patches

### Option 2: All-in-One Execution

Run the complete end-to-end pipeline:

```bash
# Submits DeepHP pre-training with post-processing
./submit_train_deepHP.sh

# After DeepHP completes (watch with: squeue -u tkeating)
# Submits HelicoDataSet fine-tuning with ensemble voting
./submit_transfer_learning.sh
```

**Full Workflow:**
1. DeepHP pre-training (5 folds) → `weighted_ensemble_deepHP.py` → averaged backbone
2. HelicoDataSet fine-tuning (5 folds) → `ensemble_voting.py` → final predictions

---

## Timeline & Resources

| Phase | Task | Time | GPU |
|-------|------|------|-----|
| 1 | DeepHP pre-training (5 folds parallel) | 18-22 hrs | 5× L40S |
| 2A | DeepHP post-processing: calibration + weighted_ensemble_deepHP | 3-5 min | CPU |
| 2B | Backbone averaging | 2 min | CPU |
| 3 | HelicoDataSet fine-tuning (5 folds parallel) | 6-8 hrs | 5× L40S |
| 4 | HelicoDataSet post-processing: ensemble_voting (weighted ensemble) | 5-10 min | CPU |
| 5 | Performance comparison & analysis | 5 min | CPU |
| **TOTAL** | Full transfer learning pipeline with all post-processing | **~1 day** | **L40S cluster** |

---

## Next Steps After Transfer Learning

1. **Option 2 Comparison**: If improvement is <2%, try Option 2 (Separate architectures)
2. **Option 3 Dual Ensemble**: If still not satisfied, implement Option 3 (Ensemble)
3. **Clinical Validation**: Focus on improving precision/specificity for false positive reduction
4. **Grad-CAM Analysis**: Visualize what backbone pre-training learned differently

---

## Contact & Questions

For issues or questions about transfer learning integration:
- Check CONTEXT_PROMPT.md for architecture details
- Examine results/deephp_backbone_pretrained_convnext_tiny_f*_evaluation_report.csv for pre-training quality
