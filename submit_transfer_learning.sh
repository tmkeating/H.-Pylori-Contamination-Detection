#!/bin/bash
# submit_transfer_learning.sh - Complete Transfer Learning Pipeline
#
# Purpose: Orchestrate full transfer learning pipeline end-to-end:
#   Phase 1: Pre-train backbone on 394,926 H&E patches (5 folds) via submit_train_deepHP.sh
#   Phase 2: Sync data to local scratch and clean blacklisted items
#   Phase 3: Fine-tune on HelicoDataSet using pre-trained backbone (5 folds in parallel)
#   Phase 4: Generate ensemble voting, meta-classifier, and hybrid fusion results
#
# Key Features:
#   - Automatic data syncing to /tmp with rsync + exclusion filters
#   - Fold-level consensus files generated automatically during training
#   - Holdout consensus for proper ensemble voting on independent test set
#   - Bootstrap confidence intervals (1000 resamples) for all metrics
#   - SLURM job dependency chains ensure proper sequencing
#
# Usage:
#   PROFILE=SEARCHER MODEL_NAME=convnext_tiny ITER=30.0 ./submit_transfer_learning.sh
#
# Environment Variables:
#   PROFILE:              Model profile from profiles.sh (default: SEARCHER)
#   MODEL_NAME:           Backbone architecture (default: convnext_tiny)
#   ITER:                 Iteration number for tracking (default: 31.0)
#   SKIP_PRETRAINING:     Skip Phase 1 if backbone already trained (default: False)
#   DEEPHP_SUMMARY_JOB_ID: Force specific pre-training job dependency (optional)
#   FREEZE_BACKBONE:      Keep pre-trained weights frozen (default: False)
#
# Timeline:
#   ~20-22 hours: DeepHP pre-training (5 folds parallel) [Phase 1]
#   ~2-3 hours:  Data sync to scratch                  [Phase 2]
#   ~6-8 hours:  HelicoDataSet fine-tuning (5 folds)   [Phase 3]
#   ~30 minutes: Ensemble voting + meta-classifier     [Phase 4]
#   Total: ~28-34 hours (depending on SKIP_PRETRAINING)
#
# Recent Fixes:
#   - Rsync filter now includes '+ **' rule to ensure all files are copied to scratch
#   - Consensus files auto-generated from validation set (Step 7.7 in train.py)
#   - Holdout consensus used for ensemble voting (fixes NaN/mismatch errors)
#   - Explicit SLURM dependency validation prevents silent job skipping
#
# Dependencies:
#   - submit_train_deepHP.sh (Phase 1)
#   - train.py (Phase 3)
#   - ensemble_voting.py (Phase 4)
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
ITER=${ITER:-"31.0"}
FOLD_BATCH_SIZE=${FOLD_BATCH_SIZE:-"0"}  # 0=all parallel (default), N=batch in groups of N (e.g., 3 = 3+2)
PRETRAINED_BACKBONE="results/deephp_backbone_final_${MODEL_NAME}.pth"
FREEZE_BACKBONE=${FREEZE_BACKBONE:-"False"}
SKIP_PRETRAINING=${SKIP_PRETRAINING:-"False"}
GRADCAM_ONLY=${GRADCAM_ONLY:-"False"}  # True to generate only Grad-CAM visualizations (skip other plots)
DEEPHP_SUMMARY_JOB_ID=${DEEPHP_SUMMARY_JOB_ID:-""}

echo "=========================================================================="
echo "TRANSFER LEARNING: Complete End-to-End Pipeline (Option B)"
echo "=========================================================================="
echo ""
echo "Phase 1: DeepHP H&E Pre-training"
echo "Phase 2: HelicoDataSet Transfer Learning Fine-tuning"
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
        # Try to read from file
        if [ -f "results/deephp_summary_job_id.txt" ]; then
            DEEPHP_SUMMARY_JOB_ID=$(cat results/deephp_summary_job_id.txt)
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
        # Export ITER so submit_train_deepHP.sh picks it up
        export MODEL_NAME PROFILE ITER
        PRETRAINING_OUTPUT=$(./submit_train_deepHP.sh 2>&1)
        echo "$PRETRAINING_OUTPUT"
        
        # Extract summary job ID from the output or file
        if [ -f "results/deephp_summary_job_id.txt" ]; then
            DEEPHP_SUMMARY_JOB_ID=$(cat results/deephp_summary_job_id.txt)
            echo ""
            echo "✓ Pre-training orchestrator started"
            echo "  Summary Job ID: $DEEPHP_SUMMARY_JOB_ID"
            echo ""
        else
            echo "ERROR: submit_train_deepHP.sh did not produce summary job ID file"
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

echo "=========================================================================="
echo "Configuration Summary"
echo "=========================================================================="
echo "Configuration:"
echo "  Profile: $PROFILE"
echo "  Model: $MODEL_NAME"
echo "  Iteration: $ITER"
echo ""
echo "Pre-training (DeepHP):"
echo "  Pre-training Epochs: $DEEPHP_EPOCHS"
echo "  Pre-trained Backbone: $PRETRAINED_BACKBONE"
echo ""
echo "Fine-tuning (HelicoDataSet):"
echo "  Epochs: $NUM_EPOCHS"
echo "  Neg Weight: $NEG_WEIGHT"
echo "  Pos Weight: $POS_WEIGHT"
echo "  Gamma: $GAMMA"
echo "  Use Focal Loss: $USE_FOCAL_LOSS"
echo "  Saver Metric: $SAVER_METRIC"
echo "  Freeze BN: $FREEZE_BN"
echo "  Freeze Backbone: $FREEZE_BACKBONE"
echo "  Clip Grad: $CLIP_GRAD"
echo "  Pct Start: $PCT_START"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Use SWA: $USE_SWA"
echo "  SWA Start: $SWA_START"
echo "  Jitter: $JITTER"
echo "  Pool Type: $POOL_TYPE"
echo "=========================================================================="
echo ""


# 1. Pre-sync handling: only submit if SKIP_PRETRAINING=true
#    If pre-training is enabled, DeepHP presync runs first, then transfer folds depend on DeepHP summary
echo "Pre-sync job handling..."
echo ""


# Determine sbatch flags based on whether pre-training is enabled
if [ "$SKIP_PRETRAINING" = "True" ] || [ "$SKIP_PRETRAINING" = "true" ]; then
    echo "Pre-training skipped: submitting transfer learning presync..."
    PRESYNC_SBATCH_FLAGS=""
else
    echo "Pre-training enabled: submitting transfer learning presync after pre-training..."
    PRESYNC_SBATCH_FLAGS="--dependency=afterok:$DEEPHP_SUMMARY_JOB_ID --nodelist=dcc-gr1"
fi

PRE_SYNC_JOB=$(sbatch $PRESYNC_SBATCH_FLAGS -p pg1tfg12 --job-name=transfer_presync --output=results/slurm_transfer_presync_%j.txt <<'PRESYNC_EOF'
#!/bin/bash
#SBATCH -p pg1tfg12
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
echo "Configuration Summary"
echo "=========================================================================="
echo "Configuration:"
echo "  Profile: $PROFILE"
echo "  Model: $MODEL_NAME"
echo "  Iteration: $ITER"
echo ""
echo "Pre-training (DeepHP):"
echo "  Pre-training Epochs: $DEEPHP_EPOCHS"
echo "  Pre-trained Backbone: $PRETRAINED_BACKBONE"
echo ""
echo "Fine-tuning (HelicoDataSet):"
echo "  Epochs: $NUM_EPOCHS"
echo "  Neg Weight: $NEG_WEIGHT"
echo "  Pos Weight: $POS_WEIGHT"
echo "  Gamma: $GAMMA"
echo "  Use Focal Loss: $USE_FOCAL_LOSS"
echo "  Saver Metric: $SAVER_METRIC"
echo "  Freeze BN: $FREEZE_BN"
echo "  Freeze Backbone: $FREEZE_BACKBONE"
echo "  Clip Grad: $CLIP_GRAD"
echo "  Pct Start: $PCT_START"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Use SWA: $USE_SWA"
echo "  SWA Start: $SWA_START"
echo "  Jitter: $JITTER"
echo "  Pool Type: $POOL_TYPE"
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
echo "Pre-Sync Statistics"
echo "=========================================================================="
echo ""
echo "Scratch Directory: $LOCAL_SCRATCH"
echo "Total size:"
du -sh "$LOCAL_SCRATCH" 2>/dev/null || echo "  (calculating...)"
echo ""
echo "Directory breakdown:"
for dir in "CrossValidation/Annotated" "CrossValidation/Cropped" "HoldOut"; do
    path="$LOCAL_SCRATCH/$dir"
    if [ -d "$path" ]; then
        size=$(du -sh "$path" 2>/dev/null | cut -f1)
        echo "  $dir:"
        echo "    $size	$path"
    fi
done
echo ""
echo "File counts:"
annotated_count=$(find "$LOCAL_SCRATCH/CrossValidation/Annotated" -type f 2>/dev/null | wc -l)
cropped_count=$(find "$LOCAL_SCRATCH/CrossValidation/Cropped" -type f 2>/dev/null | wc -l)
holdout_count=$(find "$LOCAL_SCRATCH/HoldOut" -type f 2>/dev/null | wc -l)
echo "  Annotated: $annotated_count files"
echo "  Cropped: $cropped_count files"
echo "  HoldOut: $holdout_count files"
total_count=$((annotated_count + cropped_count + holdout_count))
echo "  Total: $total_count files"
echo ""
echo "Blacklist Summary:"
echo "  Conflict bags excluded: 5"
echo "  Image-level exclusions: 2788"
echo "  Total exclusions: 2793"
echo ""
echo "✓ Pre-sync complete. Ready for transfer learning fine-tuning."
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
    echo "Submitting fold $FOLD..."
    
    JOB_OUT=$(sbatch -p pg1tfg12 \
        --dependency=$FOLD_DEPENDENCY \
        --job-name=transfer_f${FOLD} \
        --output=results/slurm_transfer_f${FOLD}_%j.txt \
        --error=results/slurm_transfer_error_f${FOLD}_%j.txt \
        --ntasks=1 \
        --cpus-per-task=4 \
        --gres=gpu:l40s:1 --gres=shard:l40s:12000 \
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
export FREEZE_BACKBONE=$FREEZE_BACKBONE

# Activate virtual environment with dependencies
source $VENV_ROOT/bin/activate

# Dynamically resolve project directory
PROJECT_DIR=$(python3 -c "import os; print(os.path.dirname(os.path.abspath('${PWD}/train.py')))" 2>/dev/null || echo "/home/tkeating/model/H.-Pylori-Contamination-Detection")
cd "$PROJECT_DIR"

# Force all folds to use GPU 0 for memory consolidation
export CUDA_VISIBLE_DEVICES=0

# Build train.py command with conditional backbone path
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
    --iter \$ITER"

# Only include backbone path if not skipping pre-training
if [ "\$SKIP_PRETRAINING" != "True" ] && [ "\$SKIP_PRETRAINING" != "true" ]; then
    TRAIN_CMD="\$TRAIN_CMD --pretrained_backbone_path results/deephp_backbone_final_${MODEL_NAME}.pth"
fi

TRAIN_CMD="\$TRAIN_CMD --freeze_backbone \$FREEZE_BACKBONE"

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
echo "DEPENDENCY CHAIN SUMMARY"
echo "=========================================================================="
echo ""
echo "Pre-sync Job:     $PRE_SYNC_ID"
if [ "$BATCHED" = "1" ]; then
    echo "Execution Order (SEQUENTIAL BATCHING):"
    echo "  1. Pre-sync ($PRE_SYNC_ID) - Syncs data to scratch"
    echo "  2. Fold 0 → Fold 1 → Fold 2 → Fold 3 → Fold 4 (sequential)"
    echo "  3. Summary & ensemble (waits for last fold)"
    echo "  4. Visualization generation (waits for summary)"
else
    echo "Fine-tuning Jobs: $DEPENDENCIES"
    echo "  (All depend on pre-sync: $PRE_SYNC_ID)"
    echo ""
    echo "Execution Order (PARALLEL):"
    echo "  1. Pre-sync ($PRE_SYNC_ID) - Syncs data to scratch"
    echo "  2. Fine-tuning folds (parallel, wait for step 1)"
    echo "  3. Summary & ensemble (waits for step 2)"
    echo "  4. Visualization generation (waits for step 3)"
fi
echo ""
echo "=========================================================================="
echo ""

# 3 & 4. Submit summary + visualization jobs (with dependency chain)
#    Summary job: runs summarize_results.py, ensemble_voting.py
#    Visualization job: runs generate_visuals.py to create calibration curves and dashboards

# Final validation of dependency string (prevent invalid sbatch syntax)
if [ -z "$DEPENDENCY_STRING" ]; then
    echo "ERROR: Failed to generate valid dependency string from fold jobs!"
    echo "Fold job IDs: $DEPENDENCIES"
    exit 1
fi

SUMMARY_JOB_ID=$(sbatch --dependency=$DEPENDENCY_STRING \
    -p pg1tfg12 \
    --time=0-02:00 \
    --mem=8G \
    --cpus-per-task=1 \
    --gres=gpu:l40s:1 --gres=shard:l40s:12000 \
    --job-name=transfer_summary \
    --output=results/slurm_transfer_summary_%j.txt \
    --error=results/slurm_transfer_summary_error_%j.txt \
    <<'SUMMARY_EOF'
#!/bin/bash
#SBATCH -p pg1tfg12
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

# Extract iteration from latest checkpoint files
ITER=$(python3 -c "
import glob
from pathlib import Path
# Search for model files matching the current model architecture
model_pattern = 'results/*_${MODEL_NAME}_model_brain.pth'
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
python3 summarize_results.py --dir results --last 5 2>&1

echo ""
echo "Step 2: Generating ensemble voting, meta-classifier, and hybrid fusion results..."
python3 ensemble_voting.py 2>&1

echo ""
echo "=========================================================================="
echo "Clinical analysis and hybrid ensemble fusion completed."
echo "✅ Primary results in: results/hybrid_ensemble_*"
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
echo "✓ Ensemble voting job submitted!"
echo "  Job ID: $SUMMARY_JOB_ID"
echo "=========================================================================="
echo ""

# 4. Submit visualization generation job (depends on ensemble job)
#    Generates calibration curves, performance dashboards, and optional TL comparison
echo "Submitting automatic visualization generation job..."
echo ""

VISUAL_JOB_ID=$(sbatch --dependency=afterok:$SUMMARY_JOB_ID \
    -p pg1tfg12 \
    --time=0-02:00 \
    --mem=16G \
    --cpus-per-task=1 \
    --gres=gpu:l40s:1 --gres=shard:l40s:12000 \
    --job-name=transfer_visuals \
    --output=results/slurm_transfer_visuals_%j.txt \
    --error=results/slurm_transfer_visuals_error_%j.txt \
    <<'VISUAL_EOF'
#!/bin/bash
#SBATCH -p pg1tfg12
cd /home/tkeating/model/H.-Pylori-Contamination-Detection
# Activate virtual environment for Python dependencies
VENV_ROOT=$(python3 -c "from config import VENV_ROOT; print(VENV_ROOT)")
source $VENV_ROOT/bin/activate
JOB_ID=$SLURM_JOB_ID

echo "=========================================================================="
echo "Generating comprehensive visual reports for model analysis..."
echo "=========================================================================="
echo ""

# Extract run ID (iteration) from latest checkpoint
RUN_ID=$(python3 -c "
import glob
from pathlib import Path
# Search for model files matching the current model architecture
model_pattern = 'results/*_${MODEL_NAME}_model_brain.pth'
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
VISUAL_CMD="python3 generate_visuals.py --run_id $RUN_ID --model_name $MODEL_NAME --dataset helicodataset"
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
echo "Expected timeline:"
echo "  - Pre-sync: ~2 minutes"
echo "  - Fine-tuning (5 folds parallel): ~6-8 hours"
echo "  - Ensemble/Summary: ~10 minutes"
echo "  - Visualization generation: ~10 minutes"
echo "  Total: ~6-8 hours"
echo ""
