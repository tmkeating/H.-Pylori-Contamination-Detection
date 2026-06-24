#!/bin/bash
# submit_transfer_learning.sh - Complete Transfer Learning Pipeline
#
# Purpose: Orchestrate full transfer learning pipeline end-to-end:
#   Phase 1: Pre-train backbone on 394,926 H&E patches (5 folds) via submit_train_deepHP.sh
#            Uses CONFIG 87771 hardcoded experiment-level stratification
#            Post-processing: calibrate_per_fold_thresholds_deepHP.py → 
#                            apply_calibrated_thresholds_deepHP.py → 
#                            weighted_ensemble_deepHP.py (generates averaged backbone)
#   Phase 2: Sync HelicoDataSet to local scratch with blacklist exclusions
#            Cleans 2793 blacklisted items before fine-tuning
#   Phase 3: Fine-tune on HelicoDataSet using pre-trained backbone (5 folds in parallel)
#            Generates cross-leakage audits, Grad-CAM visualizations, and metrics per fold
#   Phase 4: Ensemble voting analysis for HelicoDataSet predictions
#            Runs ensemble_voting.py to combine 5-fold predictions for clinical validation
#
# Key Features:
#   - CONFIG 87771 experiment-level stratification (DeepHP pre-training)
#   - Image-level cross-leakage audits verify no patch in both train/val
#   - Experiment-level audits verify no experiment split across train/val
#   - Grad-CAM visualizations with guaranteed TP/FP/FN/TN coverage
#   - HelicoDataSet 5-fold stratification prevents patient/sample leakage
#   - Fold-level consensus files generated automatically during training
#   - Holdout consensus for proper ensemble voting on independent test set
#   - Bootstrap confidence intervals (1000 resamples) for all metrics
#   - SLURM job dependency chains ensure proper sequencing
#
# Usage:
#   PROFILE=SEARCHER MODEL_NAME=convnext_tiny ITER=30.0 ./submit_transfer_learning.sh
#   PROFILE_DEEPHP=SEARCHERDEEPHP PROFILE=SEARCHER ./submit_transfer_learning.sh
#   GPU_TYPE=a40 PROFILE=SEARCHER MODEL_NAME=convnext_tiny ./submit_transfer_learning.sh  # Run on A40 partition
#
# Environment Variables:
#   PROFILE:              Model profile for transfer learning (default: SEARCHER)
#   PROFILE_DEEPHP:       Model profile for pre-training (default: same as PROFILE)
#   MODEL_NAME:           Backbone architecture (default: convnext_tiny)
#   ITER:                 Iteration number for tracking (default: 31.0)
#   SKIP_PRETRAINING:     Skip Phase 1 if backbone already trained (default: False)
#   SKIP_PRETRAINED_BACKBONE: Skip using pretrained backbone, use base model weights (default: False)
#   SKIP_TRANSFER_LEARNING: Skip Phase 2 fine-tuning (default: False)
#   DEEPHP_SUMMARY_JOB_ID: Force specific pre-training job dependency (optional)
#   FREEZE_BACKBONE:      Keep pre-trained weights frozen (default: False)
#   GRADCAM_ONLY:         Generate only Grad-CAM visualizations (default: False)
#   USE_DANN:             Enable Domain Adversarial training (default: False)
#   DANN_LAMBDA:          Gradient reversal scaling factor (default: 1.0)
#   DANN_WEIGHT:          Weight for adversary loss (default: 0.5)
#   GPU_TYPE:             GPU type for SLURM partition (default: l40s, options: l40s, a40)
#
# Outputs:
#   DeepHP Pre-training (Phase 1) via submit_train_deepHP.sh:
#     Per-fold: model_brain.pth, cross_leakage_audit.csv, gradcam.png, metrics_summary.csv
#     Post-processing (weighted_ensemble_deepHP.py):
#       - {run_id}_calibrated_thresholds_deepHP.json (per-fold optimal thresholds)
#       - weighted_ensemble_deepHP_results_{run_id}.csv (backbone ensemble predictions)
#     Final: deephp_backbone_final_{run_id}_{model}_{iter}.pth (averaged across folds, ready for transfer)
#   HelicoDataSet Fine-tuning (Phase 3):
#     Per-fold: model_brain.pth, cross_leakage_audit.csv, gradcam.png, probabilities.json
#     Cross-val: grand_cv_summary, grand_cv_averages CSVs
#   HelicoDataSet Post-Processing (Phase 4) via ensemble_voting.py:
#     - hybrid_ensemble_results_*.csv (final patient predictions, BEST-IN-CLASS) ⭐
#     - weighted_ensemble_results_*.csv (fold-performance-weighted predictions)
#     - majority_voting_results_*.csv (simple majority voting)
#     - ensemble_voting_summary_*.csv (per-fold ensemble metrics)
#     - weighted_ensemble_fold_analysis_*.csv (per-fold weights & contributions)
#     - Calibration curves, performance dashboards, learning curve visualizations
#
# Timeline:
#   ~20-22 hours: DeepHP pre-training (5 folds parallel) [Phase 1]
#                 - Each fold: backbone training + cross-leakage audits + Grad-CAM
#                 - Stratification: CONFIG 87771 hardcoded experiment assignments (5 folds from 33 exps)
#                 - Output: averaged backbone + 5-fold cross-validation summary
#   ~3-5 min:     DeepHP post-processing (Phase 1) [weighted_ensemble_deepHP.py]
#                 - Threshold calibration, threshold application, weighted ensemble
#                 - Output: averaged backbone ready for transfer learning
#   ~2-3 hours:   Data sync to scratch                  [Phase 2]
#                 - Syncs HelicoDataSet to /tmp with rsync
#                 - Removes 2793 blacklisted items (5 bags + 2788 images)
#   ~6-8 hours:   HelicoDataSet fine-tuning (5 folds)   [Phase 3]
#                 - Each fold: fine-tune backbone + cross-leakage audits + Grad-CAM
#                 - Output: 5-fold models, metrics, and probabilities for ensemble
#   ~5-10 min:    HelicoDataSet post-processing (Phase 4) [ensemble_voting.py]
#                 - Weighted ensemble, meta-classifier, and hybrid fusion
#                 - Output: patient-level predictions with calibration
#   ~10 minutes:  Visualization generation
#                 - Calibration curves, performance dashboards, learning curves
#   Total: ~28-34 hours (depending on SKIP_PRETRAINING)
#
# Cross-Leakage Audit Strategy:
#   DeepHP Pre-training (experiment-level):
#     - CONFIG 87771: Each of 33 experiments assigned to exactly ONE fold (zero leakage)
#     - Fold 0 val: 7 experiments (4 pos, 3 neg) → 87,532 patches (2.33:1 ratio)
#     - Fold 1 val: 10 experiments (3 pos, 7 neg) → 89,516 patches (2.06:1 ratio)
#     - Fold 2 val: 5 experiments (4 pos, 1 neg) → 20,347 patches (2.31:1 ratio)
#     - Fold 3 val: 4 experiments (4 pos, 0 neg) → 99,120 patches (2.81:1 ratio)
#     - Fold 4 val: 7 experiments (6 pos, 1 neg) → 98,410 patches (2.29:1 ratio)
#   HelicoDataSet Fine-tuning (bag-level):
#     - Maintains separation of CrossValidation/Annotated, CrossValidation/Cropped, HoldOut
#     - Tracks bag membership to prevent patient/sample overlap
#     - Image-level blacklist removes conflict bags and duplicate/artifact images
#   Benefits:
#     - Prevents artifact overfitting (different staining/imaging patterns per experiment)
#     - Eliminates data leakage between train/val/holdout
#     - Enables reliable clinical validation on true holdout test set
#
# Recent Improvements:
#   - Rsync filter handles 2793 exclusion patterns (5 conflict bags + 2788 images)
#   - Consensus files auto-generated from validation set (Step 7.7 in train.py)
#   - Holdout consensus used for ensemble voting (fixes NaN/mismatch errors)
#   - Explicit SLURM dependency validation prevents silent job skipping
#   - Two-pass Grad-CAM collection guarantees all 4 prediction categories visualized
#
# Dependencies:
#   - submit_train_deepHP.sh (Phase 1): Handles DeepHP pre-training + weighted_ensemble_deepHP.py post-processing
#   - train.py (Phase 3): Fine-tuning script
#   - ensemble_voting.py (Phase 4): HelicoDataSet post-processing with weighted ensemble voting
#   - config.py (paths configuration)

set -e  # Exit on error

# Create results folder if it doesn't already exist
mkdir -p results

# Verify virtual environment before proceeding
if [ -f "./verify_venv.sh" ]; then
    source ./verify_venv.sh
else
    echo "ERROR: verify_venv.sh not found in current directory"
    exit 1
fi

MODEL_NAME=${MODEL_NAME:-"convnext_tiny"}
PROFILE=${PROFILE:-"SEARCHER"}
PROFILE_DEEPHP=${PROFILE_DEEPHP:-"$PROFILE"}  # Default to PROFILE if not specified
ITER=${ITER:-"31.0"}
FOLD_BATCH_SIZE=${FOLD_BATCH_SIZE:-"0"}  # 0=all parallel (default), N=batch in groups of N (e.g., 3 = 3+2)

# ===================================================================
# OUTPUT DIRECTORY SETUP
# ===================================================================
# Naming convention: transfer_{MODEL}_{ITER}_{PROFILE}
OUTPUT_DIR="results/transfer_${MODEL_NAME}_${ITER}_${PROFILE}"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Backbone Loading with Smart Fallback
# Priority 1: Check SKIP_PRETRAINED_BACKBONE flag - if True, use base model weights only
# Priority 2: --backbone_location flag if provided
# Priority 3: Search in results/ directory for deephp_backbone_final_*_{MODEL}_{ITER}.pth
# Priority 4: Use pattern (will fail later if doesn't exist)
SKIP_PRETRAINED_BACKBONE=${SKIP_PRETRAINED_BACKBONE:-"False"}
BACKBONE_LOCATION=${BACKBONE_LOCATION:-""}

if [ "$SKIP_PRETRAINED_BACKBONE" = "True" ] || [ "$SKIP_PRETRAINED_BACKBONE" = "true" ]; then
    PRETRAINED_BACKBONE=""
    echo "Skipping pretrained backbone - using base model weights only (ImageNet pre-trained)"
elif [ -n "$BACKBONE_LOCATION" ] && [ -f "$BACKBONE_LOCATION" ]; then
    PRETRAINED_BACKBONE="$BACKBONE_LOCATION"
    echo "Using specified backbone: $PRETRAINED_BACKBONE"
else
    # Search in results/ for first matching backbone
    PRETRAINED_BACKBONE=$(ls -t results/deephp_backbone_final_*_${MODEL_NAME}_${ITER}.pth 2>/dev/null | head -1)
    if [ -z "$PRETRAINED_BACKBONE" ]; then
        # Fallback pattern
        PRETRAINED_BACKBONE="results/deephp_backbone_final_${MODEL_NAME}_${ITER}.pth"
        echo "Note: Using backbone pattern (will verify exists when needed): $PRETRAINED_BACKBONE"
    else
        echo "Located backbone: $PRETRAINED_BACKBONE"
    fi
fi

FREEZE_BACKBONE=${FREEZE_BACKBONE:-"False"}
SKIP_PRETRAINING=${SKIP_PRETRAINING:-"False"}
SKIP_TRANSFER_LEARNING=${SKIP_TRANSFER_LEARNING:-"False"}
GRADCAM_ONLY=${GRADCAM_ONLY:-"False"}  # True to generate only Grad-CAM visualizations (skip other plots)
DEEPHP_SUMMARY_JOB_ID=${DEEPHP_SUMMARY_JOB_ID:-""}
USE_DANN=${USE_DANN:-"False"}
DANN_LAMBDA=${DANN_LAMBDA:-1.0}
DANN_WEIGHT=${DANN_WEIGHT:-0.5}
GPU_TYPE=${GPU_TYPE:-"l40s"}  # GPU type for SLURM partition (l40s or a40)

# Map GPU type to partition
# NOTE: A40 partition (pg3tfg12) requires q_pg3tfg12 QOS authorization from cluster admin
#       L40S partition (pg1tfg12) uses q_pg1tfg12 QOS (default)
if [ "$GPU_TYPE" = "a40" ]; then
    PARTITION="pg3tfg12"
    GPU_GRES="gpu:nvidia_a40:1"
    SHARD_GRES="shard:nvidia_a40:12000"
else
    PARTITION="pg1tfg12"
    GPU_GRES="gpu:l40s:1"
    SHARD_GRES="shard:l40s:12000"
fi

echo "=========================================================================="
echo "TRANSFER LEARNING: Complete End-to-End Pipeline"
echo "=========================================================================="
echo ""
echo "STRATIFICATION APPROACH:"
echo "  - Phase 1 (DeepHP Pre-training):"
echo "    CONFIG 87771 experiment-level (hardcoded 5-fold assignments)"
echo "    Each experiment assigned to exactly ONE fold (prevents leakage)"
echo "    Output: Pre-trained backbone averaged across 5 folds"
echo ""
echo "  - Phase 3 (HelicoDataSet Fine-tuning):"
echo "    5-fold bag-level stratification prevents patient/sample leakage"
echo "    Maintains separation: CrossValidation/Annotated, CrossValidation/Cropped, HoldOut"
echo "    Output: 5-fold fine-tuned models ready for ensemble voting"
echo ""
echo "Profiles:"
echo "  Pre-training (DeepHP): $PROFILE_DEEPHP"
echo "  Transfer Learning (HelicoDataSet): $PROFILE"
echo ""

# ===========================================================================
# PHASE 1: PRE-TRAINING ON DEEPHP
# ===========================================================================
if [ "$SKIP_PRETRAINING" = "True" ] || [ "$SKIP_PRETRAINING" = "true" ]; then
    echo "=========================================================================="
    echo "PHASE 1: SKIPPED (Pre-training already completed)"
    echo "=========================================================================="
    echo ""
    
    if [ -z "$DEEPHP_SUMMARY_JOB_ID" ]; then
        # Try to read from file (now uses pattern: {run_id}_{ITER}_summary_job_id.txt)
        SUMMARY_JOB_FILE=$(ls -t results/*_${ITER}_summary_job_id.txt 2>/dev/null | head -1)
        if [ -f "$SUMMARY_JOB_FILE" ]; then
            DEEPHP_SUMMARY_JOB_ID=$(cat "$SUMMARY_JOB_FILE")
            echo "Found summary job file: $SUMMARY_JOB_FILE"
        else
            echo "WARNING: No DEEPHP_SUMMARY_JOB_ID provided and file not found"
            echo "Using immediate scheduling (no dependency)"
            DEEPHP_SUMMARY_JOB_ID="0"  # Will be treated as no dependency
        fi
    fi
    
    echo "✓ Skipping pre-training"
    echo "  Summary Job ID: $DEEPHP_SUMMARY_JOB_ID"
    echo ""
else
    echo "=========================================================================="
    echo "PHASE 1: Initiating DeepHP Pre-training Pipeline"
    echo "=========================================================================="
    echo ""

    # Call submit_train_deepHP.sh to start pre-training orchestration
    if [ -f "submit_train_deepHP.sh" ]; then
        chmod +x submit_train_deepHP.sh
        # Extract RUN_ID from existing results if not specified (for parallel safety)
        if [ -z "$RUN_ID" ]; then
            # Generate next available RUN_ID
            RUN_ID=$(python3 << 'RUN_ID_GEN_EOF'
import os
import re

results_dir = "results"
if not os.path.exists(results_dir):
    print("01")
else:
    files = os.listdir(results_dir)
    max_run = 0
    
    # First priority: check for summary_job_id.txt files (generated immediately after job submission)
    for f in files:
        match = re.match(r"^(\d+)_[\d.]+_summary_job_id\.txt$", f)
        if match:
            try:
                run_id = int(match.group(1))
                max_run = max(max_run, run_id)
            except:
                pass
    
    # Fallback: if no summary files found, check other output files
    if max_run == 0:
        for f in files:
            match = re.match(r"^(\d+)_[\d.]+_(\d+)_", f)
            if match:
                try:
                    run_id = int(match.group(1))
                    max_run = max(max_run, run_id)
                except:
                    pass
    
    print(f"{max_run + 1:02d}")
RUN_ID_GEN_EOF
)
        fi
        
        echo "Phase 1 Run ID: $RUN_ID"
        echo ""
        
        # Export ITER and use PROFILE_DEEPHP for pre-training, PROFILE for transfer learning
        export MODEL_NAME PROFILE=$PROFILE_DEEPHP ITER RUN_ID USE_DANN DANN_LAMBDA DANN_WEIGHT
        PRETRAINING_OUTPUT=$(./submit_train_deepHP.sh 2>&1)
        echo "$PRETRAINING_OUTPUT"
        # Restore PROFILE for transfer learning phase
        export PROFILE
        
        # Extract summary job ID from file (written immediately by submit_train_deepHP.sh)
        # Pattern: {run_id}_{ITER}_summary_job_id.txt
        SUMMARY_JOB_FILE="results/${RUN_ID}_${ITER}_summary_job_id.txt"
        if [ -f "$SUMMARY_JOB_FILE" ]; then
            DEEPHP_SUMMARY_JOB_ID=$(cat "$SUMMARY_JOB_FILE")
            echo ""
            echo "✓ Pre-training orchestrator started"
            echo "  Summary Job ID: $DEEPHP_SUMMARY_JOB_ID"
            echo "  Summary Job File: $SUMMARY_JOB_FILE"
            echo ""
        else
            echo "ERROR: Summary job ID file not found"
            echo "       Expected: $SUMMARY_JOB_FILE"
            exit 1
        fi
    else
        echo "ERROR: submit_train_deepHP.sh not found"
        exit 1
    fi

    echo "Waiting for DeepHP pre-training to complete (this will take ~20-22 hours)..."
    echo "You can monitor progress in another terminal:"
    echo "  squeue -u \$USER | grep deephp"
    echo ""
fi

# ===========================================================================
# PHASE 2: FINE-TUNING ON HELICODATASET (depends on Phase 1 completion)
# ===========================================================================
if [ "$SKIP_TRANSFER_LEARNING" = "True" ] || [ "$SKIP_TRANSFER_LEARNING" = "true" ]; then
    echo "=========================================================================="
    echo "PHASE 2: SKIPPED (Transfer learning fine-tuning disabled)"
    echo "=========================================================================="
    echo ""
    echo "✓ Skipping HelicoDataSet fine-tuning"
    echo "  Pre-trained backbone available at: $PRETRAINED_BACKBONE"
    echo ""
    echo "To re-enable fine-tuning, run:"
    echo "  SKIP_TRANSFER_LEARNING=False ./submit_transfer_learning.sh"
    echo ""
else
    echo "=========================================================================="
    echo "PHASE 2: HelicoDataSet Transfer Learning Fine-tuning"
    echo "=========================================================================="
    echo "(This phase will start automatically after pre-training completes)"
    echo ""

# Source the Model Profiles (for consistency with HelicoDataSet training)
if [ -f "profiles.sh" ]; then
    source profiles.sh
    echo "✓ Loaded profiles from profiles.sh"
    # Call profile function if available
    if declare -f "set_profile_$PROFILE" > /dev/null; then
        "set_profile_$PROFILE"
        echo "✓ Using $PROFILE profile"
    fi
else
    echo "⚠ profiles.sh not found, using defaults"
fi

# Get virtual environment path from config
VENV_ROOT=$(python3 -c "from config import VENV_ROOT; print(VENV_ROOT)")

# Pre-training parameters (from profiles.sh, for reference and documentation)
DEEPHP_EPOCHS=${DEEPHP_EPOCHS:-20}

# HelicoDataSet fine-tuning parameters (can be overridden by profiles.sh)
NUM_EPOCHS=${NUM_EPOCHS:-15}
NEG_WEIGHT=${NEG_WEIGHT:-1.0}
POS_WEIGHT=${POS_WEIGHT:-1.0}
GAMMA=${GAMMA:-1.0}
USE_FOCAL_LOSS=${USE_FOCAL_LOSS:-"False"}
SAVER_METRIC=${SAVER_METRIC:-"loss"}
FREEZE_BN=${FREEZE_BN:-"False"}
CLIP_GRAD=${CLIP_GRAD:-1.0}
PCT_START=${PCT_START:-0.1}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
USE_SWA=${USE_SWA:-"True"}
SWA_START=${SWA_START:-10}
JITTER=${JITTER:-0.15}
POOL_TYPE=${POOL_TYPE:-"attention"}

# Export all configuration variables for presync job access
export NUM_EPOCHS NEG_WEIGHT POS_WEIGHT GAMMA USE_FOCAL_LOSS SAVER_METRIC
export FREEZE_BN FREEZE_BACKBONE CLIP_GRAD PCT_START WEIGHT_DECAY
export USE_SWA SWA_START JITTER POOL_TYPE DEEPHP_EPOCHS
export VENV_ROOT PROFILE MODEL_NAME ITER PRETRAINED_BACKBONE
export PARTITION GPU_GRES SHARD_GRES GPU_TYPE

# 1. Pre-sync handling: only submit if SKIP_PRETRAINING=true
#    If pre-training is enabled, DeepHP presync runs first, then transfer folds depend on DeepHP summary
echo "Pre-training Job handling (using PROFILE_DEEPHP=$PROFILE_DEEPHP)..."
echo ""


# Determine sbatch flags based on whether pre-training is enabled
if [ "$SKIP_PRETRAINING" = "True" ] || [ "$SKIP_PRETRAINING" = "true" ]; then
    echo "Pre-training skipped: submitting transfer learning presync with PROFILE=$PROFILE..."
    PRESYNC_SBATCH_FLAGS=""
else
    echo "Pre-training enabled: submitting transfer learning presync after pre-training..."
    echo "  (Pre-training used PROFILE_DEEPHP=$PROFILE_DEEPHP, transfer learning will use PROFILE=$PROFILE)"
    PRESYNC_SBATCH_FLAGS="--dependency=afterok:$DEEPHP_SUMMARY_JOB_ID --nodelist=dcc-gr1"
fi

PRE_SYNC_JOB=$(sbatch $PRESYNC_SBATCH_FLAGS -p $PARTITION --job-name=transfer_presync --output=$OUTPUT_DIR/slurm_transfer_presync_%j.txt <<PRESYNC_EOF
#!/bin/bash
#SBATCH -t 0-01:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:1
#SBATCH -J transfer_presync

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
export HOME=/home/tkeating
source $VENV_ROOT/bin/activate

LOCAL_SCRATCH=$(python3 -c "from config import SCRATCH_ROOT; print(SCRATCH_ROOT)" 2>/dev/null || echo "/home/tkeating/.scratch/h_pylori_data")
REMOTE_DATA=$(python3 -c "from config import DATASET_ROOT; print(DATASET_ROOT)" 2>/dev/null || echo "/home/tkeating/datasets/HelicoDataSet")

echo "=========================================================================="
echo "Configuration Summary (Transfer Learning Pipeline)"
echo "=========================================================================="
echo "Experiment Tracking:"
echo "  Profile (DeepHP): $PROFILE_DEEPHP"
echo "  Profile (HelicoDataSet): $PROFILE"
echo "  Model: $MODEL_NAME"
echo "  Iteration: $ITER"
echo ""
echo "Phase 1: DeepHP Pre-training"
echo "  Epochs: $DEEPHP_EPOCHS"
echo "  Stratification: CONFIG 87771 (experiment-level hardcoded assignments)"
echo "    - Each experiment assigned to exactly ONE fold"
echo "    - All folds train on ~307K patches from all other experiments"
echo "    - Fold ratios range 2.06:1 to 2.81:1 (target 2.28:1, distance 0.6441)"
echo "  Cross-leakage audits: Image-level AND experiment-level"
echo "  Grad-CAM: Two-pass collection (guarantees TP/FP/FN/TN coverage)"
echo ""
echo "Phase 3: HelicoDataSet Fine-tuning"
echo "  Epochs: $NUM_EPOCHS"
echo "  Stratification: 5-fold bag-level (prevents patient/sample leakage)"
echo "  Pre-trained Backbone: $PRETRAINED_BACKBONE"
echo ""
echo "Loss & Regularization:"
echo "  Neg Weight: $NEG_WEIGHT"
echo "  Pos Weight: $POS_WEIGHT"
echo "  Gamma: $GAMMA (focal loss)"
echo "  Use Focal Loss: $USE_FOCAL_LOSS"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Clip Grad: $CLIP_GRAD"
echo ""
echo "Optimization:"
echo "  Freeze BN: $FREEZE_BN"
echo "  Freeze Backbone: $FREEZE_BACKBONE"
echo "  Pct Start (LR Warmup): $PCT_START"
echo "  Use SWA: $USE_SWA"
echo "  SWA Start Epoch: $SWA_START"
echo ""
echo "Augmentation & Architecture:"
echo "  Jitter Intensity: $JITTER"
echo "  Pool Type: $POOL_TYPE"
echo "  Saver Metric: $SAVER_METRIC"
echo "=========================================================================="
echo ""

echo "=========================================================================="
echo "Transfer Learning Pre-Sync: Syncing HelicoDataSet to Local Scratch"
echo "=========================================================================="

mkdir -p "$LOCAL_SCRATCH"
mkdir -p "$LOCAL_SCRATCH/CrossValidation" "$LOCAL_SCRATCH/HoldOut" "$LOCAL_SCRATCH/HoldOut"

# Copy metadata
cp "$REMOTE_DATA"/*.xlsx "$LOCAL_SCRATCH/" 2>/dev/null || true
cp "$REMOTE_DATA"/*.csv "$LOCAL_SCRATCH/" 2>/dev/null || true

echo "[PRESYNC] Cleaning blacklisted items from scratch..."
python3 << CLEANUP_EOF
import json
import shutil
from pathlib import Path

blacklist_path = Path("./blacklist.json")
scratch_path = Path("$LOCAL_SCRATCH")

print(f"[CLEANUP] Checking for blacklisted items to remove...")
print(f"[CLEANUP] Blacklist file: {blacklist_path}")
print(f"[CLEANUP] Scratch path: {scratch_path}")

# First, remove all Thumbs.db files since rsync --exclude won't delete them
if scratch_path.exists():
    thumbs_count = 0
    for thumbs_file in scratch_path.rglob('Thumbs.db'):
        try:
            thumbs_file.unlink()
            thumbs_count += 1
        except:
            pass
    if thumbs_count > 0:
        print(f"[CLEANUP] Removed {thumbs_count} Thumbs.db files from scratch")

if blacklist_path.exists() and scratch_path.exists():
    with open(blacklist_path, 'r') as f:
        data = json.load(f)
        conflict_bags = list(data.get('conflict_blacklist', {}).keys())
        image_blacklist = data.get('image_blacklist', [])
        
        bag_removed = 0
        image_removed = 0
        
        if conflict_bags:
            print(f"[CLEANUP] Found {len(conflict_bags)} blacklisted bags to remove")
            for bag_id in conflict_bags:
                for dir_name in ['CrossValidation/Annotated', 'CrossValidation/Cropped', 'HoldOut']:
                    bag_path = scratch_path / dir_name / bag_id
                    if bag_path.exists():
                        shutil.rmtree(bag_path)
                        bag_removed += 1
            print(f"[CLEANUP] Removed {bag_removed} blacklisted bags")
        
        if image_blacklist:
            print(f"[CLEANUP] Found {len(image_blacklist)} image-level blacklist items to clean")
            for item in image_blacklist:
                if isinstance(item, dict):
                    folder = item.get('folder')
                    filename = item.get('filename')
                    for dir_name in ['CrossValidation/Annotated', 'CrossValidation/Cropped', 'HoldOut']:
                        bag_path = scratch_path / dir_name / folder
                        if bag_path.exists():
                            file_path = bag_path / filename
                            if file_path.exists():
                                file_path.unlink()
                                image_removed += 1
            print(f"[CLEANUP] Removed {image_removed} image-level files")
else:
    print(f"[CLEANUP] Scratch doesn't exist yet - no cleanup needed")
CLEANUP_EOF

echo "[PRESYNC] Generating exclude filters from blacklist..."
export EXCLUDE_FILE="/tmp/transfer_presync_exclude_filters_$$.txt"
python3 << FILTER_EOF
import json
import os

exclude_file = os.environ['EXCLUDE_FILE']
excludes = []

with open('./blacklist.json') as f:
    data = json.load(f)

conflict_bags = data.get('conflict_blacklist', {})
image_blacklist = data.get('image_blacklist', [])

for bag in conflict_bags.keys():
    excludes.append(f"{bag}/")

for item in image_blacklist:
    if isinstance(item, dict):
        folder = item.get('folder', '')
        filename = item.get('filename', '')
        if folder and filename:
            excludes.append(f"{folder}/{filename}")



# Also exclude Thumbs.db
excludes.append("*/Thumbs.db")

with open(exclude_file, 'w') as out:
    for exclude in excludes:
        out.write(exclude + "\n")

print(f"[DEBUG] Blacklist: {len(conflict_bags)} bags + {len(image_blacklist)} images")
# Show sample patterns for debugging
# Also exclude Thumbs.db
excludes.append("*/Thumbs.db")

with open(exclude_file, "r") as f:
    lines = f.readlines()
    bags = [l.strip() for l in lines if l.endswith("/\n")]
    images = [l.strip() for l in lines if "/" in l.strip() and not l.endswith("/\n")]
    other = [l.strip() for l in lines if "/" not in l.strip() and not l.endswith("/\n")]
print(f"[DEBUG] Patterns breakdown: {len(bags)} bag patterns, {len(images)} image patterns, {len(other)} other")
print(f"[DEBUG] Wrote {len(excludes)} total exclusion rules (with filter syntax)")
FILTER_EOF
echo ""
echo "[DEBUG] Verifying exclude file creation..."
if [ -f "$EXCLUDE_FILE" ]; then
    echo "[DEBUG] ✓ File exists: $EXCLUDE_FILE"
    LINES=$(wc -l < "$EXCLUDE_FILE")
    SIZE=$(wc -c < "$EXCLUDE_FILE")
    echo "[DEBUG] Size: $SIZE bytes, Lines: $LINES"
    echo "[DEBUG] First 3 patterns:"
    head -3 "$EXCLUDE_FILE" | sed 's/^/    /'
else
    echo "[DEBUG] ✗ CRITICAL: File NOT found!"
    echo "[DEBUG] Expected: $EXCLUDE_FILE"
    ls -la /tmp/transfer_presync_exclude_filters_* 2>/dev/null || echo "[DEBUG] No matching files in /tmp"
fi
echo ""

echo "[PRESYNC] Syncing HelicoDataSet to local scratch..."
mkdir -p "$LOCAL_SCRATCH/CrossValidation" "$LOCAL_SCRATCH/HoldOut"
echo "[RSYNC] Syncing with exclusion filters..."
rsync -a --delete --exclude='*/Thumbs.db' --exclude='B22-124_0' --exclude='B22-68_0' --exclude='B22-141_1' --exclude='B22-03_1' --exclude='B22-01_1' --exclude-from="$EXCLUDE_FILE" "$REMOTE_DATA/CrossValidation/Annotated/" "$LOCAL_SCRATCH/CrossValidation/Annotated/" 2>&1 | tail -5 || true
rsync -a --delete --exclude='*/Thumbs.db' --exclude='B22-124_0' --exclude='B22-68_0' --exclude='B22-141_1' --exclude='B22-03_1' --exclude='B22-01_1' --exclude-from="$EXCLUDE_FILE" "$REMOTE_DATA/CrossValidation/Cropped/" "$LOCAL_SCRATCH/CrossValidation/Cropped/" 2>&1 | tail -5 || true
rsync -a --delete --exclude='*/Thumbs.db' --exclude='B22-124_0' --exclude='B22-68_0' --exclude='B22-141_1' --exclude='B22-03_1' --exclude='B22-01_1' --exclude-from="$EXCLUDE_FILE" "$REMOTE_DATA/HoldOut/" "$LOCAL_SCRATCH/HoldOut/" 2>&1 | tail -5 || true
rm -f "$EXCLUDE_FILE"

echo "[PRESYNC] Sync complete - calculating statistics..."
echo ""
echo "=========================================================================="
echo "Pre-Sync Summary: Data Integrity & Blacklist Verification"
echo "=========================================================================="
echo ""
echo "Scratch Directory: $LOCAL_SCRATCH"
echo "Total size:"
du -sh "$LOCAL_SCRATCH" 2>/dev/null || echo "  (calculating...)"
echo ""
echo "Directory breakdown (Data Organization):"
for dir in "CrossValidation/Annotated" "CrossValidation/Cropped" "HoldOut"; do
    path="$LOCAL_SCRATCH/$dir"
    if [ -d "$path" ]; then
        size=$(du -sh "$path" 2>/dev/null | cut -f1)
        echo "  $dir:"
        echo "    $size	$path"
    fi
done
echo ""
echo "File counts (Total Images):"
annotated_count=$(find "$LOCAL_SCRATCH/CrossValidation/Annotated" -type f 2>/dev/null | wc -l)
cropped_count=$(find "$LOCAL_SCRATCH/CrossValidation/Cropped" -type f 2>/dev/null | wc -l)
holdout_count=$(find "$LOCAL_SCRATCH/HoldOut" -type f 2>/dev/null | wc -l)
echo "  Annotated: $annotated_count files (original H&E images)"
echo "  Cropped: $cropped_count files (region-of-interest crops)"
echo "  HoldOut: $holdout_count files (independent test set)"
total_count=$((annotated_count + cropped_count + holdout_count))
echo "  Total: $total_count files"
echo ""
echo "Blacklist & Data Integrity Checks:"
echo "  Conflict bags excluded: 5"
echo "    - B22-124_0, B22-68_0, B22-141_1, B22-03_1, B22-01_1"
echo "    - Reason: Conflicting annotations, quality issues, or duplicates"
echo "  Image-level exclusions: 2788"
echo "    - Duplicate/artifact images within remaining bags"
echo "  Total exclusion rules: 2793"
echo ""
echo "Stratification Verification:"
echo "  - No overlap between Annotated and Cropped (separate processing pipelines)"
echo "  - HoldOut completely independent (no patient/sample in train/val)"
echo "  - All blacklisted items removed before 5-fold fine-tuning"
echo ""
echo "✅ Pre-sync complete. Data integrity verified."
echo "   Ready to proceed with transfer learning fine-tuning on clean data."
PRESYNC_EOF
)
    
    PRE_SYNC_ID=$(echo $PRE_SYNC_JOB | awk '{print $4}')
    PRE_SYNC_DEPENDENCY="afterok:$PRE_SYNC_ID"
    
    echo "Pre-sync job ID: $PRE_SYNC_ID"
    if [ -z "$PRE_SYNC_ID" ] || [ "$PRE_SYNC_ID" = "" ]; then
        echo "ERROR: Failed to extract pre-sync job ID!"
        exit 1
fi


echo "✓ Pre-sync dependency set: $PRE_SYNC_DEPENDENCY"
echo ""

# 2. Submit 5 fine-tuning jobs (parallel or batched based on FOLD_BATCH_SIZE)
echo "Submitting transfer learning fine-tuning jobs for all 5 folds..."
if [ "$FOLD_BATCH_SIZE" != "0" ]; then
    echo "Mode: BATCH PROCESSING (groups of $FOLD_BATCH_SIZE folds)"
else
    echo "Mode: PARALLEL (all folds run simultaneously)"
fi
echo "=========================================================================="
echo ""

DEPENDENCIES=""
declare -a FOLD_IDS  # Array to track job IDs for batch dependencies

for FOLD in {0..4}
do
    # Determine this fold's dependency based on batch size
    if [ "$FOLD_BATCH_SIZE" != "0" ] && [ $FOLD -ge $FOLD_BATCH_SIZE ]; then
        # Not in first batch; depends on last fold of previous batch
        BATCH_LAST_FOLD=$(((FOLD / FOLD_BATCH_SIZE) * FOLD_BATCH_SIZE - 1))
        FOLD_DEPENDENCY="afterok:${FOLD_IDS[$BATCH_LAST_FOLD]}"
    else
        # First batch or all-parallel mode: depend on pre-sync
        FOLD_DEPENDENCY="$PRE_SYNC_DEPENDENCY"
    fi
    
    # FIND BACKBONE BEFORE SUBMITTING JOB (so it runs on login node with access to results/)
    BACKBONE_FILE=""
    
    # Try exact iteration match first
    for f in results/deephp_backbone_final_*_${MODEL_NAME}_${ITER}.pth results/*/deephp_backbone_final_*_${MODEL_NAME}_${ITER}.pth; do
        if [ -f "$f" ]; then
            BACKBONE_FILE="$f"
            break
        fi
    done
    
    # If no exact match, try any iteration of this model
    if [ -z "$BACKBONE_FILE" ]; then
        for f in results/deephp_backbone_final_*_${MODEL_NAME}_*.pth results/*/deephp_backbone_final_*_${MODEL_NAME}_*.pth; do
            if [ -f "$f" ]; then
                BACKBONE_FILE="$f"
            fi
        done
    fi
    
    # Show what we found
    if [ -n "$BACKBONE_FILE" ]; then
        echo "  ✓ Found backbone: $BACKBONE_FILE"
    else
        echo "  ⚠️  No backbone found - will use ImageNet pre-trained"
    fi
    
    echo "Submitting fold $FOLD..."
    
    JOB_OUT=$(sbatch -p $PARTITION \
        --dependency=$FOLD_DEPENDENCY \
        --job-name=transfer_f${FOLD} \
        --output=$OUTPUT_DIR/slurm_transfer_f${FOLD}_%j.txt \
        --error=$OUTPUT_DIR/slurm_transfer_error_f${FOLD}_%j.txt \
        --ntasks=1 \
        --cpus-per-task=4 \
        --gres=$GPU_GRES --gres=$SHARD_GRES \
        --mem=30G \
        --time=48:00:00 \
        <<TRAIN_EOF
#!/bin/bash
# Setup environment explicitly
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
export HOME=/home/tkeating
export FOLD=$FOLD
export MODEL_NAME=$MODEL_NAME
export NEG_WEIGHT=$NEG_WEIGHT
export POS_WEIGHT=$POS_WEIGHT
export GAMMA=$GAMMA
export NUM_EPOCHS=$NUM_EPOCHS
export SAVER_METRIC=$SAVER_METRIC
export FREEZE_BN=$FREEZE_BN
export CLIP_GRAD=$CLIP_GRAD
export PCT_START=$PCT_START
export WEIGHT_DECAY=$WEIGHT_DECAY
export USE_SWA=$USE_SWA
export SWA_START=$SWA_START
export JITTER=$JITTER
export POOL_TYPE=$POOL_TYPE
export ITER=$ITER
export SKIP_PRETRAINING=$SKIP_PRETRAINING
export SKIP_PRETRAINED_BACKBONE=$SKIP_PRETRAINED_BACKBONE
export FREEZE_BACKBONE=$FREEZE_BACKBONE
export BACKBONE_PATH=$BACKBONE_PATH
export OUTPUT_DIR=$OUTPUT_DIR

# Activate virtual environment with dependencies
source $VENV_ROOT/bin/activate

# Dynamically resolve project directory
PROJECT_DIR=\$(python3 -c "import os; print(os.path.dirname(os.path.abspath('$PWD/train.py')))" 2>/dev/null || echo "/home/tkeating/model/H.-Pylori-Contamination-Detection")
cd "\$PROJECT_DIR"

# Force all folds to use GPU 0 for memory consolidation
export CUDA_VISIBLE_DEVICES=0

# Build train.py command with optional backbone path
TRAIN_CMD="python3 -u train.py \
    --fold \$FOLD \
    --num_folds 5 \
    --model_name \$MODEL_NAME \
    --neg_weight \$NEG_WEIGHT \
    --pos_weight \$POS_WEIGHT \
    --gamma \$GAMMA \
    --num_epochs \$NUM_EPOCHS \
    --saver_metric \$SAVER_METRIC \
    --freeze_bn \$FREEZE_BN \
    --clip_grad \$CLIP_GRAD \
    --pct_start \$PCT_START \
    --weight_decay \$WEIGHT_DECAY \
    --use_swa \$USE_SWA \
    --swa_start \$SWA_START \
    --jitter \$JITTER \
    --pool_type \$POOL_TYPE \
    --iter \$ITER \
    --output_dir \$OUTPUT_DIR"

# Add backbone path if found (passed from login node)
if [ -n "\$BACKBONE_PATH" ]; then
    TRAIN_CMD="\$TRAIN_CMD --backbone_path \$BACKBONE_PATH"
fi

TRAIN_CMD="\$TRAIN_CMD --freeze_backbone \$FREEZE_BACKBONE"

TRAIN_CMD="\$TRAIN_CMD --use_focal_loss \$USE_FOCAL_LOSS"

# Execute the constructed command
eval \$TRAIN_CMD

echo ""
echo "✓ Fold \$FOLD fine-tuning complete"
TRAIN_EOF
)
    
    JOB_ID=$(echo $JOB_OUT | awk '{print $4}')
    FOLD_IDS[$FOLD]="$JOB_ID"  # Store for batch dependency lookup
    
    if [ -z "$JOB_ID" ] || [ "$JOB_ID" = "" ]; then
        echo "  ✗ ERROR: Failed to submit fold $FOLD!"
        echo "  Submission output: $JOB_OUT"
        exit 1
    fi
    
    echo "  ✓ Job ID: $JOB_ID"
    
    if [ "$FOLD_BATCH_SIZE" != "0" ]; then
        # Batching enabled: show which batch this fold belongs to
        BATCH_NUM=$((FOLD / FOLD_BATCH_SIZE))
        BATCH_POS=$((FOLD % FOLD_BATCH_SIZE))
        echo "    (Batch $((BATCH_NUM + 1)), Position $((BATCH_POS + 1)))"
    else
        # All parallel: accumulate dependencies for final summary
        if [ -z "$DEPENDENCIES" ]; then
            DEPENDENCIES="$JOB_ID"
        else
            DEPENDENCIES="$DEPENDENCIES:$JOB_ID"
        fi
    fi
done

# Set final dependency string for summary job
if [ "$FOLD_BATCH_SIZE" != "0" ]; then
    # Batching: summary depends on ALL folds (prevent race conditions if later folds finish first)
    DEPENDENCY_STRING=""
    for i in {0..4}; do
        if [ -z "$DEPENDENCY_STRING" ]; then
            DEPENDENCY_STRING="afterok:${FOLD_IDS[$i]}"
        else
            DEPENDENCY_STRING="$DEPENDENCY_STRING,afterok:${FOLD_IDS[$i]}"
        fi
    done
else
    # Parallel: convert colon-separated job IDs to SLURM dependency format
    DEPENDENCY_STRING=$(echo "$DEPENDENCIES" | sed 's/:/ /g' | awk '{for(i=1;i<=NF;i++) printf "%safterok:%s", (i>1?",":""), $i}')
fi

echo ""
echo "=========================================================================="
echo "All 5 fine-tuning jobs submitted. Scheduling final summary + ensemble job..."
echo "=========================================================================="
echo "JOB DEPENDENCY CHAIN & OUTPUTS"
echo "=========================================================================="
echo ""
echo "Pre-sync Job ID: $PRE_SYNC_ID"
echo "  - Syncs HelicoDataSet to scratch"
echo "  - Removes 2793 blacklisted items"
echo "  - Generates metadata for fine-tuning"
echo ""
if [ "$BATCHED" = "1" ]; then
    echo "Execution Mode: SEQUENTIAL BATCHING"
    echo "  1. Pre-sync ($PRE_SYNC_ID) - Syncs and validates data"
    echo "  2. Fine-tuning Fold 0 → 1 → 2 → 3 → 4 (sequential)"
    echo "  3. Ensemble/Meta-classifier (waits for fold 4)"
    echo "  4. Visualization generation (waits for step 3)"
else
    echo "Fine-tuning Job IDs: $DEPENDENCIES"
    echo "  (All depend on pre-sync: $PRE_SYNC_ID)"
    echo ""
    echo "Execution Mode: PARALLEL FOLDS"
    echo "  1. Pre-sync ($PRE_SYNC_ID) - Syncs and validates data"
    echo "  2. Fine-tuning folds 0-4 (run in parallel, wait for pre-sync)"
    echo "  3. Ensemble/Meta-classifier (waits for all 5 folds)"
    echo "  4. Visualization generation (waits for step 3)"
fi
echo ""
echo "FINE-TUNING OUTPUTS (per fold):"
echo "  - {prefix}_model_brain.pth: Fine-tuned backbone weights"
echo "  - {prefix}_cross_leakage_audit.csv: Image-level stratification verification"
echo "  - {prefix}_cross_leakage_audit_experiments.csv: Bag-level assignments (prevents patient leakage)"
echo "  - {prefix}_gradcam.png: Grad-CAM visualization (TP/FP/FN/TN guaranteed coverage)"
echo "  - {prefix}_metrics_summary.csv: Bootstrap CI metrics (Accuracy, Precision, Recall, F1, AUC)"
echo "  - {prefix}_probabilities.json: Per-sample predictions (for ensemble voting)"
echo ""
echo "CROSS-VALIDATION SUMMARIES:"
echo "  - grand_cv_summary_*.csv: Long-format per-fold metrics"
echo "  - grand_cv_averages_*.csv: Averages ± standard deviation across 5 folds"
echo "  - grand_cv_bootstrap_ci_*.csv: Bootstrap confidence intervals (1000 resamples)"
echo "  - confusion_matrices_combined.png: 5-fold confusion matrix dashboard"
echo "  - pr_roc_curves_combined.png: PR and ROC curves overlaid for all 5 folds"
echo ""
echo "ENSEMBLE ANALYSIS OUTPUTS:"
echo "  - hybrid_ensemble_*.csv: Voting results (soft/hard/meta/fusion)"
echo "  - ensemble_voting_summary_*.csv: Per-fold ensemble metrics"
echo "  - meta_classifier_results_*.csv: Meta-classifier predictions"
echo "  - calibration_curve.png: Calibration analysis for reliability"
echo "  - performance_dashboard.png: Comprehensive metrics visualization"
echo ""
echo "=========================================================================="
echo ""

# 3 & 4. Submit summary + visualization jobs (with dependency chain)
#    Summary job: runs summarize_results.py + ensemble_voting.py (HelicoDataSet Phase 4 post-processing)
#    Visualization job: runs generate_visuals.py to create calibration curves and dashboards

# Final validation of dependency string (prevent invalid sbatch syntax)
if [ -z "$DEPENDENCY_STRING" ]; then
    echo "ERROR: Failed to generate valid dependency string from fold jobs!"
    echo "Fold job IDs: $DEPENDENCIES"
    exit 1
fi

SUMMARY_JOB_ID=$(sbatch --dependency=$DEPENDENCY_STRING \
    -p $PARTITION \
    --time=0-02:00 \
    --mem=8G \
    --cpus-per-task=1 \
    --gres=$GPU_GRES --gres=$SHARD_GRES \
    --job-name=transfer_summary \
    --output=$OUTPUT_DIR/slurm_transfer_summary_%j.txt \
    --error=$OUTPUT_DIR/slurm_transfer_summary_error_%j.txt \
    <<SUMMARY_EOF
#!/bin/bash
cd /home/tkeating/model/H.-Pylori-Contamination-Detection
# Activate virtual environment for Python dependencies
VENV_ROOT=$(python3 -c "from config import VENV_ROOT; print(VENV_ROOT)")
source $VENV_ROOT/bin/activate
# Get job ID for output filename
JOB_ID=$SLURM_JOB_ID

echo "=========================================================================="
echo "All fine-tuning folds complete. Generating comprehensive ensemble analysis..."
echo "=========================================================================="
echo ""

# Extract iteration from latest checkpoint files in OUTPUT_DIR
ITER=$(python3 -c "
import glob
from pathlib import Path
# Search for model files in OUTPUT_DIR
model_pattern = '$OUTPUT_DIR/*_${MODEL_NAME}_model_brain.pth'
files = sorted(glob.glob(model_pattern))
if files:
    # Extract iteration from filename like: 31_25.0_107840_f0_convnext_tiny_model_brain.pth
    filename = Path(files[-1]).stem
    parts = filename.split('_')
    if len(parts) >= 2:
        print(parts[1])  # This is the iteration (25.0, 31.0, etc)
    else:
        print('31.0')
else:
    print('31.0')
")

echo "Iteration: $ITER"
echo ""

# Run performance summarization
echo "Step 1: Running cross-validation performance summary..."
python3 summarize_results.py --dir $OUTPUT_DIR --last 5 2>&1

echo ""
echo "Step 2: Running HelicoDataSet ensemble voting and hybrid fusion analysis..."
python3 ensemble_voting.py --dir $OUTPUT_DIR 2>&1

echo ""
echo "=========================================================================="
echo "✅ HelicoDataSet ensemble analysis completed (Phase 4 post-processing)"
echo "✅ Primary results in: results/hybrid_ensemble_* (patient-level predictions)"
echo "=========================================================================="
echo ""

# Create properly named summary file with iteration number
SUMMARY_FILE="results/slurm_summary_${ITER}_${JOB_ID}.txt"
echo "Summary report available at: $SUMMARY_FILE"
echo ""

SUMMARY_EOF
)

SUMMARY_JOB_ID=$(echo $SUMMARY_JOB_ID | awk '{print $4}')

echo "=========================================================================="
echo "✓ HelicoDataSet ensemble voting job submitted (Phase 4)!"
echo "  Job ID: $SUMMARY_JOB_ID"
echo "  Script: ensemble_voting.py"
echo "  Outputs:"
echo "    - hybrid_ensemble_*.csv (final patient predictions with confidence)"
echo "    - weighted_ensemble_*.csv (fold-performance-weighted predictions)"
echo "    - ensemble_voting_summary_*.csv (per-fold ensemble metrics)"
echo "    - weighted_ensemble_fold_analysis_*.csv (per-fold weights & contributions)"
echo "=========================================================================="
echo ""

# 4. Submit visualization generation job (depends on ensemble job)
#    Generates calibration curves, performance dashboards, and optional TL comparison
echo "Submitting automatic visualization generation job..."
echo "  Generates: Calibration curves, performance dashboards, learning curves"

VISUAL_JOB_ID=$(sbatch --dependency=afterok:$SUMMARY_JOB_ID \
    -p $PARTITION \
    --time=0-02:00 \
    --mem=16G \
    --cpus-per-task=1 \
    --gres=$GPU_GRES --gres=$SHARD_GRES \
    --job-name=transfer_visuals \
    --output=$OUTPUT_DIR/slurm_transfer_visuals_%j.txt \
    --error=$OUTPUT_DIR/slurm_transfer_visuals_error_%j.txt \
    <<VISUAL_EOF
#!/bin/bash
cd /home/tkeating/model/H.-Pylori-Contamination-Detection
# Activate virtual environment for Python dependencies
VENV_ROOT=$(python3 -c "from config import VENV_ROOT; print(VENV_ROOT)")
source $VENV_ROOT/bin/activate
JOB_ID=$SLURM_JOB_ID

echo "=========================================================================="
echo "Generating comprehensive visual reports for model analysis..."
echo "=========================================================================="
echo ""

# Extract run ID (iteration) from latest checkpoint in OUTPUT_DIR
RUN_ID=$(python3 -c "
import glob
from pathlib import Path
# Search for model files in OUTPUT_DIR
model_pattern = '$OUTPUT_DIR/*_${MODEL_NAME}_model_brain.pth'
files = sorted(glob.glob(model_pattern))
if files:
    # Extract run_id from filename like: 31_25.0_107840_f0_convnext_tiny_model_brain.pth
    filename = Path(files[-1]).stem
    parts = filename.split('_')
    if len(parts) >= 2:
        print(parts[0])  # This is the run_id (31, 30, etc)
    else:
        print('31')
else:
    print('31')
")

echo "Generating visuals for run: $RUN_ID"
echo ""

# Generate calibration curve + performance dashboard automatically (skip redundant visualizations)
echo "Step 1: Generating novel visualizations (calibration curve + performance dashboard)..."
echo "  (Skipping ROC/PR/confusion matrix - already generated during training)"

# Build visualization command with conditional Grad-CAM flag
VISUAL_CMD="python3 generate_visuals.py --run_id $RUN_ID --model_name $MODEL_NAME --dataset helicodataset --output_dir $OUTPUT_DIR"
if [ "$GRADCAM_ONLY" = "True" ] || [ "$GRADCAM_ONLY" = "true" ]; then
    echo "  [Grad-CAM Only Mode: Skipping other visualizations]"
    VISUAL_CMD="$VISUAL_CMD --gradcam_only"
else
    VISUAL_CMD="$VISUAL_CMD --pipeline_mode"
fi

eval "$VISUAL_CMD" 2>&1

echo ""
echo "=========================================================================="
echo "✅ Visual reports completed!"
echo "=========================================================================="
echo ""
echo "Generated NEW visualizations (for presentation & clinical validation):"
echo "  - Calibration Curve: results/${RUN_ID}_*_calibration_curve.png"
echo "  - Performance Dashboard: results/${RUN_ID}_*_performance_dashboard.png"
echo ""
echo "Already generated during training (available in results/):"
echo "  - Confusion Matrix"
echo "  - ROC Curve"
echo "  - PR Curve"
echo "  - Threshold Analysis"
echo "  - Probability Histogram"
echo "  - Learning Curves"
echo "  - Per-fold metrics CSV"
echo ""
echo "To generate full visualizations including Grad-CAM (larger file set):"
echo "  python3 generate_visuals.py --run_id $RUN_ID  # without --pipeline_mode"
echo ""

VISUAL_EOF
)

VISUAL_JOB_ID=$(echo $VISUAL_JOB_ID | awk '{print $4}')

echo "=========================================================================="
echo "✓ All jobs submitted successfully!"
echo "=========================================================================="
echo ""
echo "OUTPUTS GENERATED:"
echo "  Per-Fold (5 total):"
echo "    - Fine-tuned backbone: {prefix}_model_brain.pth"
echo "    - Image-level audit: {prefix}_cross_leakage_audit.csv"
echo "    - Bag-level audit: {prefix}_cross_leakage_audit_experiments.csv"
echo "    - Grad-CAM: {prefix}_gradcam.png (all 4 categories guaranteed)"
echo "    - Metrics: {prefix}_metrics_summary.csv (bootstrap CIs)"
echo "    - Predictions: {prefix}_probabilities.json (for ensemble voting)"
echo ""
echo "  Cross-Validation Summaries:"
echo "    - grand_cv_summary_*.csv (long-format metrics)"
echo "    - grand_cv_averages_*.csv (means ± std across folds)"
echo "    - grand_cv_bootstrap_ci_*.csv (1000 resamples)"
echo ""
echo "  HelicoDataSet Ensemble & Fusion (Phase 4 via ensemble_voting.py):"
echo "    - hybrid_ensemble_*.csv (BEST-IN-CLASS patient predictions)"
echo "    - weighted_ensemble_*.csv (fold-performance-weighted predictions)"
echo "    - majority_voting_*.csv (simple voting for comparison)"
echo "    - ensemble_voting_summary_*.csv (per-fold metrics)"
echo "    - weighted_ensemble_fold_analysis_*.csv (per-fold weights & contributions)"
echo ""
echo "  Visualizations:"
echo "    - Confusion matrices (5-fold dashboard)"
echo "    - PR and ROC curves (overlaid for comparison)"
echo "    - Calibration curve (reliability assessment)"
echo "    - Performance dashboard (comprehensive metrics)"
echo ""
echo "Job dependency chain:"
echo "  1. Summary/Ensemble ($SUMMARY_JOB_ID)"
echo "  2. Visualizations ($VISUAL_JOB_ID) ← depends on step 1"
echo ""
echo "Monitor progress with:"
echo "  squeue -u $USER | grep transfer"
echo ""
echo "View logs with:"
echo "  tail -f results/slurm_transfer_summary_*.txt"
echo "  tail -f results/slurm_transfer_visuals_*.txt"
echo ""
echo "STRATIFICATION VERIFICATION (after completion):"
echo "  1. Check image-level audits:"
echo "     grep LEAKAGE_DETECTED results/*_cross_leakage_audit.csv"
echo "  2. Check bag-level audits:"
echo "     cat results/*_cross_leakage_audit_experiments.csv"
echo "  3. View fold-level confusion matrices and metrics:"
echo "     ls -lh results/*_confusion_matrices_combined.png"
echo "     cat results/grand_cv_averages_*.csv"
echo ""
echo "Expected timeline:"
echo "  - Pre-sync: ~2-3 minutes"
echo "  - Fine-tuning (5 folds parallel): ~6-8 hours"
echo "  - Ensemble/Summary: ~10 minutes"
echo "  - Visualization generation: ~10 minutes"
echo "  Total: ~6-8 hours"
echo ""
fi
