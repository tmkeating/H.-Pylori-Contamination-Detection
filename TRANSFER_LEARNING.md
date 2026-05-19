# DeepHP Transfer Learning Guide

**Status**: ✅ READY TO EXECUTE  
**Implementation**: Option 1 - Sequential Pre-training  
**Expected Timeline**: 15-20 hours compute (DeepHP 5-fold) + 5-8 hours (HelicoDataSet fine-tuning)  
**Expected Accuracy Improvement**: +3-5% over baseline (92.11% → 95%+)

---

## Overview

This guide implements transfer learning using the **DeepHP dataset** (394,926 H&E-stained histology patches) to pre-train the backbone of our ConvNeXt-Tiny model before fine-tuning on the patient-level HelicoDataSet.

### Why This Approach?

| Aspect | Benefit |
|--------|---------|
| **Dataset Scale** | 394K patches → 111K positive + 283K negative for robust feature learning |
| **Stain Domain** | H&E images share morphological patterns with IHC (both detect H. pylori) |
| **Backbone Initialization** | ConvNeXt-Tiny learns general histology features before task-specific MIL |
| **Overfitting Prevention** | Pre-training reduces overfitting risk on small 114-patient IHC dataset |
| **Faster Convergence** | Backbone doesn't need to learn from scratch; MIL head adapts quickly |

---

## Phase 1: DeepHP Backbone Pre-training

### Step 1A: Submit Pre-training Jobs (All 5 Folds - Orchestrated)

```bash
# Recommended: Use the orchestrator (auto-submits all folds + averaging + summary)
chmod +x train_deepHP.sh
./train_deepHP.sh

# Or with custom profile and parameters
PROFILE=SEARCHER MODEL_NAME=convnext_tiny ITER=31.0 ./train_deepHP.sh
```

**What the orchestrator does:**
1. ✅ Submits pre-sync job to verify environment
2. ✅ Submits all 5 fold jobs in parallel (dependent on pre-sync)
3. ✅ Submits final averaging job (depends on all 5 folds)
4. ✅ Provides next steps instructions

**Alternative: Manual execution** (if you prefer fine-grained control)

```bash
# Option A: Sequential (slower but easier to debug)
for i in {0..4}; do
  python3 train_deepHP_patches.py --fold $i --num_folds 5 --num_epochs 20
done

# Option B: Parallel SLURM (without orchestrator)
for i in {0..4}; do
  sbatch -p a40 -J deephp_f$i -t 24:00:00 \
    --export=ALL,FOLD=$i \
    sh -c "python3 train_deepHP_patches.py --fold $i --num_folds 5"
done
```

### Step 1B: Monitor Pre-training Progress

```bash
# Check all DeepHP jobs
squeue -u ricse03 | grep deephp

# Watch individual fold training (replace f0 with f1-f4 as needed)
tail -f results/slurm_deephp_f0_*.txt

# Check for errors
tail -f results/slurm_deephp_error_f0_*.txt

# List completed fold checkpoints
ls -lah results/deephp_backbone_pretrained_convnext_tiny_f*.pth
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
✓ results/deephp_backbone_pretrained_convnext_tiny_f*_evaluation.csv  (Patch-level metrics)
✓ results/deephp_backbone_pretrained_convnext_tiny_f*_confusion_matrix.png
✓ results/deephp_backbone_pretrained_convnext_tiny_f*_roc_curve.png
✓ results/deephp_backbone_pretrained_convnext_tiny_f*_pr_curve.png
✓ results/deephp_backbone_pretrained_convnext_tiny_f*_learning_curves.png
```

**Averaged backbone** (automatically created):
```
✓ results/deephp_backbone_final_convnext_tiny.pth  (~350 MB)
  (Ready for HelicoDataSet fine-tuning)
```

---

## Phase 2: Automatic Backbone Averaging (Done by Orchestrator)

The `train_deepHP.sh` orchestrator automatically averages backbone weights after all 5 folds complete. This creates a unified pre-trained backbone from all folds.

### Expected Output

```
✓ results/deephp_backbone_final_convnext_tiny.pth  (~350 MB)
  (Averaged backbone from 5 folds, ready for transfer learning)
```

---

## Phase 3: Fine-tune on HelicoDataSet with Pre-trained Backbone

### Step 3A: Submit Fine-tuning Jobs (5 Folds)

```bash
cd /hhome/ricse03/modelTwyla/H.-Pylori-Contamination-Detection

# Path to pre-trained backbone
PRETRAINED_BACKBONE="results/deephp_backbone_final_convnext_tiny.pth"

# Option A: Sequential (safer for validation)
for i in {0..4}; do
  python3 train.py \
    --fold $i \
    --num_folds 5 \
    --model_name convnext_tiny \
    --iter 31.0 \
    --pretrained_backbone_path "$PRETRAINED_BACKBONE" \
    --freeze_backbone False \
    --num_epochs 15
done

# Option B: Parallel SLURM (fast - ~8 hours total)
for i in {0..4}; do
  sbatch -J heli_ft_f$i \
    -e results/ft_slurm_%j_error.txt \
    -o results/ft_slurm_%j_output.txt \
    sh -c "python3 train.py \
      --fold $i \
      --num_folds 5 \
      --model_name convnext_tiny \
      --iter 31.0 \
      --pretrained_backbone_path '$PRETRAINED_BACKBONE' \
      --freeze_backbone False \
      --num_epochs 15" &
done
wait
```

### Step 3B: Monitor Fine-tuning Progress

```bash
# Check job status
squeue -u ricse03 | grep heli_ft

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

## Phase 4: Performance Comparison & Analysis

### Step 4A: Compare Transfer Learning vs Baseline

Once fine-tuning completes, compare your baseline (runs 299-301, 302-306) with transfer learning (run 31):

```bash
# Run evaluation comparison script (create if needed)
python3 << 'EOF'
import pandas as pd
import glob

# Load baseline and transfer learning evaluation reports
baseline_reports = glob.glob("results/30_25.0_*_evaluation_report.csv")  # or your baseline run
tl_reports = glob.glob("results/31_25.0_*_evaluation_report.csv")

baseline_df = pd.concat([pd.read_csv(f) for f in baseline_reports], ignore_index=True)
tl_df = pd.concat([pd.read_csv(f) for f in tl_reports], ignore_index=True)

print("Baseline Accuracy:      ", baseline_df['accuracy'].mean().round(4))
print("Transfer Learning Accuracy:", tl_df['accuracy'].mean().round(4))
print("Improvement: +", (tl_df['accuracy'].mean() - baseline_df['accuracy'].mean()).round(4))

print("\nBaseline Precision:     ", baseline_df['precision'].mean().round(4))
print("Transfer Learning Precision:", tl_df['precision'].mean().round(4))
EOF
```

### Step 4B: Generate Hybrid Ensemble with Transfer Learning Results

Once fine-tuning is complete, include run 31 in your ensemble:

```bash
# Generate ensemble with transfer learning results
python3 ensemble_voting.py --runs 31_0,31_1,31_2,31_3,31_4

# Or include both baseline and transfer learning for comparison
python3 ensemble_voting.py --runs 299,300,301,31_0,31_1,31_2,31_3,31_4
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
ls -la /export/hhome/ricse03/8117177/Positive | head
ls -la /export/hhome/ricse03/8117177/Negative | head

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

The transfer learning pipeline can be executed in two ways:

### Option 1: Two-Stage Orchestrator (Recommended)

**Stage 1: DeepHP Pre-training + Averaging** (automatic)
```bash
chmod +x train_deepHP.sh
./train_deepHP.sh

# Or with custom profile:
PROFILE=SEARCHER ./train_deepHP.sh
```

**Stage 2: HelicoDataSet Fine-tuning** (after pre-training completes)
```bash
chmod +x submit_transfer_learning.sh
./submit_transfer_learning.sh
```

This two-stage approach provides:
1. ✅ Clear separation of concerns (pre-training vs fine-tuning)
2. ✅ Ability to monitor/debug each phase independently
3. ✅ Profile support for both H&E and IHC training
4. ✅ Automatic orchestration of all 5 folds + averaging

### Option 2: All-in-One Orchestrator

Run the complete end-to-end pipeline in one command:
```bash
chmod +x submit_transfer_learning.sh
./submit_transfer_learning.sh
```

This automatically:
1. ✅ Submits DeepHP pre-training (5 folds, parallel, ~20 hrs)
2. ✅ Waits for completion
3. ✅ Averages backbone weights (~2 min)
4. ✅ Submits HelicoDataSet fine-tuning (5 folds, parallel, ~8 hrs)
5. ✅ Provides performance comparison instructions

---

## Timeline & Resources

| Phase | Task | Time | GPU |
|-------|------|------|-----|
| 1A | DeepHP pre-training (5 folds parallel) | 18-22 hrs | 5× A40 |
| 1B | Backbone averaging | 2 min | CPU |
| 3 | HelicoDataSet fine-tuning (5 folds parallel) | 6-8 hrs | 5× A40 |
| 4 | Ensemble & analysis | 10 min | CPU |
| **TOTAL** | Full transfer learning pipeline | **~1 day** | **A40 cluster** |

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
- Review RESEARCH_NOTES.md for experimental context
- Examine results/deephp_backbone_pretrained_convnext_tiny_f*_evaluation_report.csv for pre-training quality
