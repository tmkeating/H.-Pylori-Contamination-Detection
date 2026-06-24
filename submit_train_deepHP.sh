#!/bin/bash
# submit_train_deepHP.sh - DeepHP 5-Fold SLURM Orchestrator
#
# Purpose: Orchestrate full transfer learning pipeline with CONFIG 87771 experiment-level stratification:
#   1. Pre-sync DeepHP dataset to scratch with blacklist exclusions
#   2. Pre-train backbone on 394,926 H&E patches with 5 folds using CONFIG 87771
#   3. Automatically average backbone weights across folds
#   4. Generate cross-validation summaries and visualizations
#
# STRATIFICATION: CONFIG 87771 (Experiment-Level 5-Fold Cross-Validation)
#   PROBLEM SOLVED: Naive fold assignment caused data leakage (0%-99% recall variance on epoch 1)
#   Solution: Hardcoded experiment-to-fold assignment optimized from 500,000+ greedy searches
#   
#   CONFIG 87771 HARDCODED EXPERIMENT ASSIGNMENTS:
#   Each of the 33 experiments assigned to exactly ONE fold (zero data leakage):
#   
#   - Fold 0 val: 7 experiments (4 pos, 3 neg) → 87,532 patches, ratio 1:2.33
#   - Fold 1 val: 10 experiments (3 pos, 7 neg) → 89,516 patches, ratio 1:2.06
#   - Fold 2 val: 5 experiments (4 pos, 1 neg) → 20,347 patches, ratio 1:2.31
#   - Fold 3 val: 4 experiments (4 pos, 0 neg) → 99,120 patches, ratio 1:2.81
#   - Fold 4 val: 7 experiments (6 pos, 1 neg) → 98,410 patches, ratio 1:2.29
#   
#   All 33 experiments assigned to exactly ONE fold (total: 394,925 patches)
#   Training data for each fold: All experiments NOT assigned to this fold (~307K patches)
#   
#   TOTAL DISTANCE: 0.6441 (sum of distances from target ratio 1:2.28)
#
# RESULT:
#   - Each fold validates on UNIQUE experiments (prevents fold-specific artifact learning)
#   - Each fold trains on diverse experiments (same ~307K patches across all folds)
#   - Experiment integrity: No experiment split between train and val (prevents leakage)
#   - Balanced ratios: All folds 1:2.06 to 1:2.81 (target 1:2.28)
#   - Realistic metrics: ~50% epoch 1 accuracy across all folds (no 0%-99% variance)
#   - Cross-leakage audits: image-level (VERIFIED_UNIQUE) + experiment distribution verification
#   - Grad-CAM visualizations: guaranteed TP/FP/FN/TN coverage
#   - Mathematically optimized: Selected from 500,000+ configurations
#
# Usage:
#   PROFILE=SEARCHER MODEL_NAME=convnext_tiny ITER=31.0 ./submit_train_deepHP.sh
#
# Environment Variables:
#   PROFILE:    Model profile from profiles.sh (default: SEARCHER)
#   MODEL_NAME: Backbone architecture (default: convnext_tiny)
#   ITER:       Iteration number for tracking (default: 31.0)
#   RUN_ID:     Run ID for parallel job safety (auto-generated if not provided)

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
FOLD_BATCH_SIZE=${FOLD_BATCH_SIZE:-"0"}  # 0=all parallel (default), N=batch in groups of N

# ===================================================================
# OUTPUT DIRECTORY SETUP
# ===================================================================
# Naming convention: deephp_{MODEL}_{ITER}_{PROFILE_DEEPHP}
OUTPUT_DIR="results/deephp_${MODEL_NAME}_${ITER}_${PROFILE_DEEPHP}"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# RUN_ID for parallel job safety (optional; auto-generated if not provided)
RUN_ID=${RUN_ID:-""}

# Source the Model Profiles (for consistency with HelicoDataSet training)
if [ -f "profiles.sh" ]; then
    source profiles.sh
    echo "✓ Loaded profiles from profiles.sh"
    # For DeepHP, we mainly care about the base learning parameters
    # If a profile function exists, call it to load parameters
    if declare -f "set_profile_$PROFILE" > /dev/null; then
        "set_profile_$PROFILE"
        echo "✓ Using $PROFILE profile"
    fi
else
    echo "⚠ profiles.sh not found, using defaults"
fi

# Get virtual environment path from config
VENV_ROOT=$(python3 -c "from config import VENV_ROOT; print(VENV_ROOT)")

# DeepHP-specific parameters (can be overridden by profiles.sh)
DEEPHP_EPOCHS=${DEEPHP_EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-32}  # Training mini-batch size (reduced to fit in 11.5GB GPU memory limit)
LEARNING_RATE=${LEARNING_RATE:-2e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
NEG_WEIGHT=${NEG_WEIGHT:-1.0}
POS_WEIGHT=${POS_WEIGHT:-1.5}
USE_FOCAL_LOSS=${USE_FOCAL_LOSS:-"False"}
GAMMA=${GAMMA:-3.0}
USE_SWA=${USE_SWA:-"True"}
SWA_START=${SWA_START:-12}
JITTER=${JITTER:-0.15}
PCT_START=${PCT_START:-0.1}
CLIP_GRAD=${CLIP_GRAD:-0.0}
SAVER_METRIC=${SAVER_METRIC:-"f1"}
USE_DANN=${USE_DANN:-"False"}
DANN_LAMBDA=${DANN_LAMBDA:-1.0}
DANN_WEIGHT=${DANN_WEIGHT:-0.5}

echo "=========================================================================="
echo "DeepHP H&E Pre-training Pipeline (5-Fold Cross-Validation)"
echo "=========================================================================="
echo "Configuration:"
echo "  Profile: $PROFILE"
echo "  Model: $MODEL_NAME"
echo "  Iteration: $ITER"
echo "  Pre-training Epochs: $DEEPHP_EPOCHS"
echo "  Training Batch Size: $BATCH_SIZE"
echo "  Fold Batching Mode: $FOLD_BATCH_SIZE (0=all parallel)"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Pos Weight: $POS_WEIGHT"
echo "  Focal Loss: $USE_FOCAL_LOSS (gamma=$GAMMA)"
echo "  Gamma: $GAMMA"
echo "  Use SWA: $USE_SWA (SWA start epoch: $SWA_START)"
echo "  Jitter Intensity: $JITTER"
echo "  Dropout: $DROPOUT"
echo "  PCT Start (LR Warmup): $PCT_START"
echo "  Clip Grad: $CLIP_GRAD"
echo "  Saver Metric: $SAVER_METRIC"
echo "  Freeze BN: $FREEZE_BN"
echo "  Freeze Backbone: $FREEZE_BACKBONE"
echo "  DANN: $USE_DANN (lambda=$DANN_LAMBDA, weight=$DANN_WEIGHT)"
echo "=========================================================================="
echo ""

# Check for Macenko reference image (used for stain normalization)
echo "Checking for Macenko reference image..."
if [ ! -f "macenko_reference.png" ]; then
    echo "⚠️  macenko_reference.png not found. Creating it now..."
    source $VENV_ROOT/bin/activate
    python3 create_macenko_reference.py
    if [ $? -eq 0 ]; then
        echo "✓ Macenko reference created successfully"
    else
        echo "ERROR: Failed to create Macenko reference image"
        exit 1
    fi
else
    echo "✓ Macenko reference found"
fi
echo ""

# Generate or validate RUN_ID for parallel job safety
if [ -z "$RUN_ID" ]; then
    # Auto-generate next available RUN_ID based on existing results
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

echo "Using RUN_ID: $RUN_ID"
echo ""

# Export all configuration variables for presync job access
export NUM_EPOCHS NEG_WEIGHT POS_WEIGHT GAMMA USE_FOCAL_LOSS SAVER_METRIC
export FREEZE_BN FREEZE_BACKBONE CLIP_GRAD PCT_START WEIGHT_DECAY
export USE_SWA SWA_START JITTER DROPOUT POOL_TYPE DEEPHP_EPOCHS BATCH_SIZE USE_COMPILE LEARNING_RATE
export VENV_ROOT PROFILE MODEL_NAME ITER PRETRAINED_BACKBONE RUN_ID

# 1. Submit pre-sync job (prepares scratch directory for data)
echo "Submitting pre-sync job to prepare environment..."
PRE_SYNC_JOB=$(sbatch -p pg1tfg12 --job-name=deephp_presync --output=$OUTPUT_DIR/slurm_deephp_presync_%j.txt <<'PRESYNC_EOF'
#!/bin/bash
#SBATCH -p pg1tfg12
#SBATCH -t 0-02:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -J deephp_presync

# Setup environment explicitly
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
export HOME=/home/tkeating

# Activate virtual environment with dependencies
source $VENV_ROOT/bin/activate

echo "=========================================================================="
echo "Configuration Summary (DeepHP Pre-training)"
echo "=========================================================================="
echo "Configuration:"
echo "  Profile: $PROFILE"
echo "  Model: $MODEL_NAME"
echo "  Iteration: $ITER"
echo ""
echo "Training Parameters:"
echo "  Epochs: $DEEPHP_EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Pct Start (LR Warmup): $PCT_START"
echo ""
echo "Loss & Regularization:"
echo "  Neg Weight: $NEG_WEIGHT"
echo "  Pos Weight: $POS_WEIGHT"
echo "  Use Focal Loss: $USE_FOCAL_LOSS"
echo "  Gamma: $GAMMA"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Clip Grad: $CLIP_GRAD"
echo "  Dropout: $DROPOUT"
echo ""
echo "Augmentation & Normalization:"
echo "  Jitter Intensity: $JITTER"
echo ""
echo "Model Selection & SWA:"
echo "  Saver Metric: $SAVER_METRIC"
echo "  Use SWA: $USE_SWA"
echo "  SWA Start Epoch: $SWA_START"
echo ""
echo "Cross-Validation:"
echo "  Fold Batching Mode: $FOLD_BATCH_SIZE (0=all parallel)"
echo "  5 Folds with Class-First Weighted Round-Robin Stratification"
echo "=========================================================================="
echo ""

echo "Pre-sync job: Setting up DeepHP dataset for training..."
echo "This pre-sync job performs:"
echo "  1. Syncs DeepHP dataset (394,926 patches) to scratch directory"
echo "  2. Applies blacklist exclusions from blacklistDeepHP.json"
echo "  3. Verifies no blacklisted items were synced"
echo "  4. Exports DEEPHP_DATASET_ROOT for all training jobs"
echo ""
echo "After sync, training folds will use CONFIG 87771 stratification:"
echo "  - Each fold validates on unique experiments (prevents fold-specific artifacts)"
echo "  - Each fold trains on all other experiments (~307K patches, diverse)"
echo "  - Fold 0 val: 7 exps (4 pos, 3 neg) → 87,532 patches (1:2.33 ratio)"
echo "  - Fold 1 val: 10 exps (3 pos, 7 neg) → 89,516 patches (1:2.06 ratio)"
echo "  - Fold 2 val: 5 exps (4 pos, 1 neg) → 20,347 patches (1:2.31 ratio)"
echo "  - Fold 3 val: 4 exps (4 pos, 0 neg) → 99,120 patches (1:2.81 ratio)"
echo "  - Fold 4 val: 7 exps (6 pos, 1 neg) → 98,410 patches (1:2.29 ratio)"
echo ""
DEEPHP_ROOT=$(python3 -c "from config import DEEPHP_DATASET_ROOT; print(DEEPHP_DATASET_ROOT)")
echo "✓ Source dataset root: $DEEPHP_ROOT"

# Setup scratch directory from config (use home dir since /tmp is too small)
DEEPHP_SCRATCH=$(python3 -c "from config import DEEPHP_SCRATCH_ROOT; print(DEEPHP_SCRATCH_ROOT)")
mkdir -p "$DEEPHP_SCRATCH"
echo "✓ Created scratch directory: $DEEPHP_SCRATCH"

# Check source exists
if [ ! -d "$DEEPHP_ROOT/Positive" ] || [ ! -d "$DEEPHP_ROOT/Negative" ]; then
    echo "ERROR: DeepHP dataset not found at $DEEPHP_ROOT"
    exit 1
fi

# Clean up any previously synced blacklisted items from scratch
echo "Cleaning blacklisted items from scratch..."
python3 << CLEANUP_EOF
import json
import os
from pathlib import Path

blacklist_path = Path("./blacklistDeepHP.json")
scratch_path = Path("$DEEPHP_SCRATCH")

if blacklist_path.exists() and scratch_path.exists():
    try:
        with open(blacklist_path) as f:
            data = json.load(f)
        
        if 'macenko_reference_patch' in data:
            ref = data['macenko_reference_patch']
            folder = ref.get('folder')
            filename = ref.get('filename')
            
            if folder and filename:
                file_path = scratch_path / folder / filename
                if file_path.exists():
                    file_path.unlink()
                    print(f"[CLEANUP] Removed existing blacklisted file: {folder}/{filename}")
    except Exception as e:
        print(f"[CLEANUP] Warning: Could not clean blacklisted items: {e}")
CLEANUP_EOF

# Check cleanup exit status
if [ $? -ne 0 ]; then
    echo "[PRESYNC] FATAL ERROR: Blacklist cleanup failed!"
    exit 1
fi

# Generate rsync exclude filters from blacklistDeepHP.json
EXCLUDE_FILTER_FILE="/tmp/deephp_exclude_filters_$$.txt"
export EXCLUDE_FILTER_FILE
echo "Generating exclude filters from blacklistDeepHP.json..."
python3 << FILTER_EOF
import json
import os

exclude_filter_file = os.environ.get('EXCLUDE_FILTER_FILE', '/tmp/deephp_exclude_filters.txt')
excludes = []

try:
    with open('./blacklistDeepHP.json') as f:
        data = json.load(f)
    
    if 'macenko_reference_patch' in data:
        ref = data['macenko_reference_patch']
        folder = ref.get('folder')
        filename = ref.get('filename')
        if folder and filename:
            # For rsync filters: when syncing Positive/, use just the filename (no folder prefix)
            # since paths are relative to the source directory being synced
            if folder == 'Positive':
                excludes.append(f"- {filename}")
                print(f"[FILTER] Excluding from Positive: {filename}")
except FileNotFoundError:
    print(f"[FILTER] blacklistDeepHP.json not found - no exclusions needed")
except Exception as e:
    print(f"[FILTER] Error reading blacklist: {e}")

# Write filter file with proper rsync syntax
with open(exclude_filter_file, 'w') as f:
    for exclude in excludes:
        f.write(exclude + '\n')
    # Critical: include directories and all other files
    f.write('+ */\n')
    f.write('+ **\n')
    f.write('- *\n')

print(f"[FILTER] Wrote exclude filter file: {exclude_filter_file}")
FILTER_EOF

# Sync dataset to scratch with exclusion filters
echo "Syncing DeepHP dataset to scratch (with blacklist exclusions)..."
mkdir -p "$DEEPHP_SCRATCH/Positive" "$DEEPHP_SCRATCH/Negative"

if [ -f "$EXCLUDE_FILTER_FILE" ]; then
    echo "[RSYNC] Syncing Positive patches with exclusion filters..."
    rsync -aq --filter="merge $EXCLUDE_FILTER_FILE" "$DEEPHP_ROOT/Positive/" "$DEEPHP_SCRATCH/Positive/" || { echo "ERROR: Sync failed for Positive"; exit 1; }
    echo "[RSYNC] Syncing Negative patches..."
    rsync -aq "$DEEPHP_ROOT/Negative/" "$DEEPHP_SCRATCH/Negative/" || { echo "ERROR: Sync failed for Negative"; exit 1; }
    rm -f "$EXCLUDE_FILTER_FILE"
else
    echo "ERROR: Exclude filter file not generated"
    exit 1
fi

# Verify sync
echo "Verifying sync..."
SCRATCH_POS_COUNT=$(find "$DEEPHP_SCRATCH/Positive" -type f | wc -l)
SCRATCH_NEG_COUNT=$(find "$DEEPHP_SCRATCH/Negative" -type f | wc -l)
TOTAL_PATCHES=$((SCRATCH_POS_COUNT + SCRATCH_NEG_COUNT))

echo "✓ Positive patches synced: $SCRATCH_POS_COUNT"
echo "✓ Negative patches synced: $SCRATCH_NEG_COUNT"
echo "✓ Total patches: $TOTAL_PATCHES"

# Count original dataset
ORIG_POS_COUNT=$(find "$DEEPHP_ROOT/Positive" -type f | wc -l)
ORIG_NEG_COUNT=$(find "$DEEPHP_ROOT/Negative" -type f | wc -l)
ORIG_TOTAL=$((ORIG_POS_COUNT + ORIG_NEG_COUNT))
EXCLUDED=$((ORIG_TOTAL - TOTAL_PATCHES))

echo "✓ Original dataset total: $ORIG_TOTAL"
echo "✓ Excluded by blacklist: $EXCLUDED"

# Export for training jobs to use
export DEEPHP_DATASET_ROOT="$DEEPHP_SCRATCH"
echo "✓ DeepHP dataset ready at: $DEEPHP_SCRATCH"

# Print blacklist information
echo ""
echo "Blacklist Status:"
python3 << 'BLACKLIST_CHECK'
import json
import os
from pathlib import Path

try:
    with open('./blacklistDeepHP.json') as f:
        blacklist_data = json.load(f)
    
    if 'macenko_reference_patch' in blacklist_data:
        ref = blacklist_data['macenko_reference_patch']
        filename = ref.get('filename', 'unknown')
        folder = ref.get('folder', 'unknown')
        reason = ref.get('reason', 'unknown')
        score = ref.get('score', 'unknown')
        
        print(f"  ✓ Macenko Reference Excluded from Sync:")
        print(f"    File: {folder}/{filename}")
        print(f"    Quality Score: {score}")
        print(f"    Reason: {reason}")
        
        # Verification: Search for the blacklisted file in scratch to prove it was excluded
        print(f"\n  Verification - Searching for blacklisted file in scratch:")
        scratch_root = os.path.expandvars("$DEEPHP_SCRATCH")
        search_path = Path(scratch_root) / folder / filename
        found = search_path.exists()
        
        if found:
            print(f"    ❌ ERROR: Blacklisted file FOUND in scratch: {search_path}")
            print(f"       This indicates the exclusion filter did not work!")
        else:
            print(f"    ✓ Blacklisted file NOT found in scratch (as expected)")
            print(f"    ✓ Exclusion filter working correctly")
    else:
        print(f"  (No Macenko reference in blacklist)")
except FileNotFoundError:
    print(f"  ⚠ blacklistDeepHP.json not found (Macenko reference will be created during training)")
except Exception as e:
    print(f"  ERROR: Failed to read blacklist: {e}")
BLACKLIST_CHECK

PRESYNC_EOF
)
PRE_SYNC_JOB_ID=$(echo $PRE_SYNC_JOB | awk '{print $4}')
echo "Pre-sync job ID: $PRE_SYNC_JOB_ID"
PRE_SYNC_DEPENDENCY="afterok:$PRE_SYNC_JOB_ID"

# 2. Submit 5 fold training jobs (parallel or batched based on FOLD_BATCH_SIZE)
# Each fold uses CONFIG 87771 experiment-level stratification to:
# - Train on ~307K patches from all experiments NOT assigned to this fold
# - Validate on unique experiments assigned to this fold (~87K patches)
# - See diverse experiments across all folds (prevents fold-specific artifacts)
# - Maintain balanced class ratio (~1:2.3) across folds
# - Achieve realistic epoch 1 metrics (~50% accuracy, no 0%-99% variance)
#
# Each fold generates:
# - {prefix}_model_brain.pth: Trained backbone weights
# - {prefix}_cross_leakage_audit.csv: Image-level audit (verifies VERIFIED_UNIQUE across train/val)
# - {prefix}_gradcam.png: Grad-CAM visualization (guaranteed TP/FP/FN/TN coverage)
# - {prefix}_metrics_summary.csv: Bootstrap CI metrics per fold
#
echo ""
echo "Submitting DeepHP pre-training jobs for all 5 folds..."
if [ "$FOLD_BATCH_SIZE" != "0" ]; then
    echo "Mode: BATCH PROCESSING (groups of $FOLD_BATCH_SIZE folds)"
else
    echo "Mode: PARALLEL (all folds run simultaneously)"
fi
echo "=========================================================================="

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
        --job-name=deephp_f${FOLD} \
        --output=$OUTPUT_DIR/slurm_deephp_f${FOLD}_%j.txt \
        --error=$OUTPUT_DIR/slurm_deephp_error_f${FOLD}_%j.txt \
        --ntasks=1 \
        --cpus-per-task=4 \
        --gres=shard:l40s:12000 \
        --mem=20G \
        --time=36:00:00 \
        <<TRAIN_EOF
#!/bin/bash
# Setup environment explicitly
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
export HOME=/home/tkeating

# Activate virtual environment with dependencies
source $VENV_ROOT/bin/activate

FOLD=$FOLD
MODEL_NAME=$MODEL_NAME
DEEPHP_EPOCHS=$DEEPHP_EPOCHS
BATCH_SIZE=$BATCH_SIZE
USE_COMPILE=$USE_COMPILE
LEARNING_RATE=$LEARNING_RATE
WEIGHT_DECAY=$WEIGHT_DECAY
POS_WEIGHT=$POS_WEIGHT
USE_FOCAL_LOSS=$USE_FOCAL_LOSS
GAMMA=$GAMMA
ITER=$ITER
RUN_ID=$RUN_ID

cd /home/tkeating/model/H.-Pylori-Contamination-Detection

# Let SLURM shard scheduler distribute jobs across available GPUs
# Do not force GPU 0 - shards will be assigned by scheduler

# Get scratch directory at runtime from config
DEEPHP_SCRATCH=\$(python3 -c "from config import DEEPHP_SCRATCH_ROOT; print(DEEPHP_SCRATCH_ROOT)")

# Use synced DeepHP dataset from scratch
export DEEPHP_DATASET_ROOT="\$DEEPHP_SCRATCH"

# Parse per-fold pos_weight if comma-separated, otherwise use single value
FOLD_POS_WEIGHT=\$POS_WEIGHT
if [[ \$POS_WEIGHT == *","* ]]; then
    # Comma-separated per-fold weights - extract for this fold
    IFS=',' read -ra WEIGHTS <<< "\$POS_WEIGHT"
    FOLD_POS_WEIGHT=\${WEIGHTS[\$FOLD]}
fi

python3 -u train_deepHP_patches.py \
    --fold \$FOLD \
    --num_folds 5 \
    --model_name \$MODEL_NAME \
    --num_epochs \$DEEPHP_EPOCHS \
    --batch_size \$BATCH_SIZE \
    --learning_rate \$LEARNING_RATE \
    --weight_decay \$WEIGHT_DECAY \
    --neg_weight \$NEG_WEIGHT \
    --pos_weight \$FOLD_POS_WEIGHT \
    --use_focal_loss \$USE_FOCAL_LOSS \
    --gamma \$GAMMA \
    --use_swa \$USE_SWA \
    --swa_start \$SWA_START \
    --jitter \$JITTER \
    --dropout \$DROPOUT \
    --use_compile \$USE_COMPILE \
    --pct_start \$PCT_START \
    --clip_grad \$CLIP_GRAD \
    --saver_metric \$SAVER_METRIC \
    --iter \$ITER \
    --output_dir $OUTPUT_DIR \
    --run_id \$RUN_ID \
    --use_dann \$USE_DANN \
    --dann_lambda \$DANN_LAMBDA \
    --dann_weight \$DANN_WEIGHT

echo ""
echo "✓ Fold \$FOLD complete: results/deephp_backbone_pretrained_\${MODEL_NAME}_f\${FOLD}.pth"
TRAIN_EOF
)
    
    JOB_ID=$(echo $JOB_OUT | awk '{print $4}')
    FOLD_IDS[$FOLD]="$JOB_ID"  # Store for batch dependency lookup
    echo "  → Job ID: $JOB_ID"
    
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

echo ""
echo "=========================================================================="
echo "All 5 fold jobs submitted. Scheduling final averaging + summary job..."
echo "=========================================================================="

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

# Validate that all folds were successfully submitted
if [ ${#FOLD_IDS[@]} -eq 0 ]; then
    echo "ERROR: No fold jobs were successfully submitted!"
    echo "Cannot proceed with summary job."
    exit 1
fi

# 3. Submit final summary job (depends on all 5 folds)
# This job performs post-training analysis:
# - Averages backbone weights across all 5 folds (ensemble preprocessing)
# - Generates cross-validation summary CSVs:
#   * grand_cv_pretraining_summary_{run_id}_{iter}.csv: Long-format fold metrics
#   * grand_cv_pretraining_averages_{run_id}_{iter}.csv: Cross-fold averages ± std
#   * grand_cv_pretraining_bootstrap_ci_{run_id}_{iter}.csv: Bootstrap confidence intervals
# - Creates combined visualization dashboards:
#   * Confusion matrices (4-panel: TP/FP/FN/TN) across all 5 folds
#   * PR and ROC curves overlaid for all 5 folds
# - Prepares for fine-tuning on HelicoDataSet using averaged backbone

# Final validation of dependency string
if [ -z "$DEPENDENCY_STRING" ]; then
    echo "ERROR: Failed to generate valid dependency string from: $DEPENDENCIES"
    exit 1
fi

SUMMARY_JOB_OUT=$(sbatch --dependency=$DEPENDENCY_STRING \
    --export=ALL \
    -p pg1tfg12 \
    --time=0-00:30 \
    --mem=16G \
    --cpus-per-task=1 \
    --gres=shard:l40s:4000 \
    --job-name=deephp_summary \
    --output=$OUTPUT_DIR/slurm_deephp_summary_%j.txt \
    --error=$OUTPUT_DIR/slurm_deephp_summary_error_%j.txt \
    <<'SUMMARY_EOF'
#!/bin/bash
#SBATCH -p pg1tfg12
cd /home/tkeating/model/H.-Pylori-Contamination-Detection

# Get virtual environment path from config
VENV_ROOT=$(python3 -c "from config import VENV_ROOT; print(VENV_ROOT)")

# Activate virtual environment for Python dependencies
source $VENV_ROOT/bin/activate

echo "=========================================================================="
echo "All DeepHP pre-training folds complete. Averaging backbone weights..."
echo "=========================================================================="
echo ""

python3 << 'PYTHON_EOF'
from load_pretrained_backbone import average_backbone_weights
import os
import glob
import re

# Find all fold checkpoints with the new naming pattern
# Pattern: {run_id}_{iter}_{slurm_id}_f{fold}_{model}_model_brain.pth
checkpoint_files = sorted(glob.glob("results/*_model_brain.pth"))

# Filter to only DeepHP backbone files (contain _f0_, _f1_, etc. and convnext_tiny)
fold_checkpoints = {}
for f in checkpoint_files:
    # Match pattern like: 302_31.0_113456_f0_convnext_tiny_model_brain.pth
    match = re.search(r'_f(\d+)_convnext_tiny_model_brain\.pth$', f)
    if match:
        fold_idx = int(match.group(1))
        if fold_idx not in fold_checkpoints or fold_idx < 5:  # We want folds 0-4
            fold_checkpoints[fold_idx] = f

# Sort by fold index and extract paths
if len(fold_checkpoints) < 5:
    print(f"ERROR: Expected 5 folds, found {len(fold_checkpoints)}")
    print(f"Found folds: {sorted(fold_checkpoints.keys())}")
    exit(1)

fold_paths = [fold_checkpoints[i] for i in range(5)]

# Extract run_id from first fold checkpoint filename
# Pattern: {run_id}_{iter_name}_{slurm_id}_f{fold}_{model_name}_model_brain.pth
# E.g., "302_31.0_113456_f0_convnext_tiny_model_brain.pth"

# Use RUN_ID and ITER from environment variables (determined at script start)
run_id = os.environ.get('RUN_ID', 'unknown')
model_name = os.environ.get('MODEL_NAME', 'convnext_tiny')
iter_name = os.environ.get('ITER', '31.0')
output_path = f"results/deephp_backbone_final_{run_id}_{model_name}_{iter_name}.pth"

print(f"Found backbone checkpoints:")
for i, path in enumerate(fold_paths):
    print(f"  Fold {i}: {path}")
print("")

# Verify all fold checkpoints exist
missing = [p for p in fold_paths if not os.path.exists(p)]
if missing:
    print(f"ERROR: Missing checkpoints: {missing}")
    exit(1)

print("Averaging 5-fold backbone weights...")
average_backbone_weights(fold_paths, output_path)

print(f"\n{'='*80}")
print(f"✓ Backbone averaging complete!")
print(f"{'='*80}")
print(f"\nNext steps:")
print(f"1. Fine-tune on HelicoDataSet using pre-trained backbone:")
print(f"   PRETRAINED_BACKBONE={output_path}")
print(f"   for i in {{0..4}}; do")
print(f"     sbatch -J heli_ft_f\\$i run_h_pylori.sh \\$i")
print(f"   done")
print(f"\n2. Or use automated fine-tuning:")
print(f"   ./submit_transfer_learning.sh")
print(f"\n3. Generate ensemble and analysis:")
print(f"   python3 ensemble_voting.py --runs 31_0,31_1,31_2,31_3,31_4")
print(f"{'='*80}\n")

# Output run_id for reference
print(f"__RUN_ID_OUTPUT__:{run_id}")

PYTHON_EOF

# Generate cross-validation summary CSVs for DeepHP pre-training
echo "Generating pre-training cross-validation summary reports..."
python3 << 'CSV_GEN_EOF'
import pandas as pd
import numpy as np
import glob
import os
import re
from pathlib import Path

# Extract iteration number from environment variable (for parallel-safe CSV naming)
iter_num = os.environ.get('ITER', '31.0')
print(f"[CSV] Processing iteration: {iter_num}")

# Find all DeepHP metrics_summary reports for THIS specific iteration (one per fold)
# Pattern: {run_id}_{iter}_{slurm_id}_f{fold}_{model_name}_metrics_summary.csv
all_reports = sorted(glob.glob("results/*_f[0-4]_convnext_tiny_metrics_summary.csv"))

# Filter to only reports matching this iteration
eval_reports = []
for report_path in all_reports:
    # Extract iteration from filename (pattern: ..._31.0_{slurm_id}_f{fold}_...)
    match = re.search(r'_(\d+\.\d+)_\d+_f\d+_', report_path)
    if match:
        report_iter = match.group(1)
        if report_iter == iter_num:
            eval_reports.append(report_path)

print(f"[CSV] Found {len(eval_reports)} evaluation reports for iteration {iter_num}")

if not eval_reports:
    print(f"WARNING: No evaluation reports found for DeepHP iteration {iter_num}")
else:
    # Load all fold metrics_summary reports
    fold_data = []
    for report_path in eval_reports:
        try:
            df = pd.read_csv(report_path)
            # metrics_summary.csv has: Metric, Point_Estimate, Bootstrap_Mean, Bootstrap_Std, CI_Lower_95%, CI_Upper_95%, CI_Margin
            # Extract fold number from filename (pattern: ..._f{N}_...)
            match = re.search(r'_f(\d+)_', report_path)
            if match:
                fold_idx = int(match.group(1))
                # Rename columns to include fold
                df['Fold'] = fold_idx
                # Only keep Metric and Point_Estimate columns for aggregation (ignore per-fold CI)
                if 'Metric' in df.columns and 'Point_Estimate' in df.columns:
                    fold_data.append(df[['Fold', 'Metric', 'Point_Estimate']])
                    print(f"  ✓ Loaded fold {fold_idx}: {Path(report_path).name}")
                else:
                    print(f"  WARNING: Missing Metric or Point_Estimate column in {report_path}")
        except Exception as e:
            print(f"  WARNING: Could not load {report_path}: {e}")
    if fold_data:
        # Combine all fold data (now in long format: Fold, Metric, Point_Estimate)
        all_metrics = pd.concat(fold_data, ignore_index=True)
        
        # Rename Point_Estimate to Mean for consistency in downstream processing
        all_metrics = all_metrics.rename(columns={'Point_Estimate': 'Mean'})
        
        # Extract run_id from first metrics file filename
        # Pattern: {run_id}_{iter}_{slurm_id}_f{fold}_{model_name}_metrics_summary.csv
        run_id = None
        if eval_reports:
            match = re.search(r'^(\d+)_[\d.]+_\d+_f\d+_', os.path.basename(eval_reports[0]))
            if match:
                run_id = match.group(1)
        
        # Generate summary CSV with per-fold metrics in long format
        if run_id:
            summary_output = f"results/grand_cv_pretraining_summary_{run_id}_{iter_num}.csv"
        else:
            summary_output = f"results/grand_cv_pretraining_summary_{iter_num}.csv"
        all_metrics.to_csv(summary_output, index=False)
        print(f"\n✓ Pre-training summary CSV: {summary_output}")
        print(f"  ({len(fold_data)} folds, {len(all_metrics)} total metric rows)")
        
        # Generate averages CSV from long-format data
        # Group by Metric and calculate mean/std across folds
        averages_data = []
        for metric in all_metrics['Metric'].unique():
            metric_values = all_metrics[all_metrics['Metric'] == metric]['Mean'].values
            
            averages_data.append({
                'Metric': metric,
                'Mean': metric_values.mean(),
                'Std': metric_values.std()
            })
        
        avg_df = pd.DataFrame(averages_data)
        
        # Add formatted column
        avg_df['Formatted'] = avg_df.apply(
            lambda row: f"{row['Mean']:.4f} ± {row['Std']:.4f}", axis=1
        )
        avg_df['Run_Range'] = iter_num
        
        if run_id:
            avg_output = f"results/grand_cv_pretraining_averages_{run_id}_{iter_num}.csv"
        else:
            avg_output = f"results/grand_cv_pretraining_averages_{iter_num}.csv"
        avg_df.to_csv(avg_output, index=False)
        print(f"✓ Pre-training averages CSV: {avg_output}\n")
        
        # Print formatted averages table
        print(f"{'='*80}")
        print(f"CROSS-VALIDATION AVERAGES (Iteration {iter_num})")
        print(f"{'='*80}")
        print(f"{'Metric':<40} {'Mean':>15} {'Std Dev':>15}")
        print(f"{'-'*80}")
        for _, row in avg_df.iterrows():
            metric_name = row['Metric']
            mean_val = row['Mean']
            std_val = row['Std']
            print(f"{metric_name:<40} {mean_val:>15.4f} {std_val:>15.4f}")
        print(f"{'='*80}\n")
        
        # Generate bootstrap CI CSV from long-format data
        from scipy import stats
        
        ci_data = []
        for metric in all_metrics['Metric'].unique():
            try:
                values = all_metrics[all_metrics['Metric'] == metric]['Mean'].values
                point_est = values.mean()
                fold_std = values.std()
                
                # Bootstrap CI (resample from fold-level means)
                bootstrap_samples = np.random.choice(values, size=(1000, len(values)), replace=True)
                bootstrap_means = bootstrap_samples.mean(axis=1)
                bootstrap_mean = bootstrap_means.mean()
                bootstrap_std = bootstrap_means.std()
                
                ci_lower = np.percentile(bootstrap_means, 2.5)
                ci_upper = np.percentile(bootstrap_means, 97.5)
                ci_margin = (ci_upper - ci_lower) / 2
                
                ci_data.append({
                    'Metric': metric,
                    'Point_Estimate': point_est,
                    'Fold_Std': fold_std,
                    'Bootstrap_Mean': bootstrap_mean,
                    'Bootstrap_Std': bootstrap_std,
                    'CI_Lower_95%': ci_lower,
                    'CI_Upper_95%': ci_upper,
                    'CI_Margin': ci_margin
                })
            except:
                pass
        
        ci_df = pd.DataFrame(ci_data)
        if run_id:
            ci_output = f"results/grand_cv_pretraining_bootstrap_ci_{run_id}_{iter_num}.csv"
        else:
            ci_output = f"results/grand_cv_pretraining_bootstrap_ci_{iter_num}.csv"
        ci_df.to_csv(ci_output, index=False)
        print(f"✓ Pre-training bootstrap CI CSV: {ci_output}")
        
        print(f"\n{'='*80}")
        print(f"Pre-training cross-validation summary complete!")
        print(f"Generated {len(fold_data)} fold metrics across all evaluation measures")
        print(f"Iteration: {iter_num}")
        print(f"{'='*80}\n")
    else:
        print(f"ERROR: No fold evaluation reports could be loaded for iteration {iter_num}")

CSV_GEN_EOF
echo ""
python3 << 'CM_DASH_EOF'
import pandas as pd
import numpy as np
import json
import glob
import os
import re
from pathlib import Path
from sklearn.metrics import confusion_matrix
from visualization_utils import plot_combined_confusion_matrices

# Extract iteration number
iter_num = os.environ.get('ITER', '31.0')
print(f"[CM] Processing iteration: {iter_num}")

# Find all DeepHP evaluation reports for THIS specific iteration
all_reports = sorted(glob.glob("results/*_f[0-4]_convnext_tiny_evaluation_report.csv"))

# Filter to only reports matching this iteration
eval_reports = []
for report_path in all_reports:
    match = re.search(r'_(\d+\.\d+)_\d+_f\d+_', report_path)
    if match:
        report_iter = match.group(1)
        if report_iter == iter_num:
            eval_reports.append(report_path)

print(f"[CM] Found {len(eval_reports)} evaluation reports for iteration {iter_num}")

if len(eval_reports) == 5:
    # Load probabilities JSON files to get labels and predictions for confusion matrices
    fold_cms = []
    prob_files = glob.glob(f"results/*{iter_num}*_probabilities.json")
    
    for prob_file in sorted(prob_files):
        try:
            with open(prob_file, 'r') as f:
                data = json.load(f)
            
            # Extract fold number
            match = re.search(r'_f(\d+)_', prob_file)
            fold_idx = int(match.group(1)) if match else -1
            
            # Get labels and predictions from probabilities file
            labels = np.array(data['labels'])
            predictions = np.array(data['predictions_at_0_5'])
            cm = confusion_matrix(labels, predictions, labels=[0, 1])
            fold_cms.append(cm)
            print(f"  ✓ Fold {fold_idx}: {cm.shape} confusion matrix")
        except Exception as e:
            print(f"  ✗ Error loading {prob_file}: {e}")
    
    # Generate combined dashboard if all 5 folds loaded successfully
    if len(fold_cms) == 5:
        model_name = os.environ.get('MODEL_NAME', 'convnext_tiny')
        run_id = os.environ.get('RUN_ID', '')
        
        # Build filename with run_id if available
        if run_id:
            dashboard_output = f"results/{run_id}_{iter_num}_confusion_matrices_combined_{model_name}.png"
        else:
            dashboard_output = f"results/{iter_num}_confusion_matrices_combined_{model_name}.png"
        
        plot_combined_confusion_matrices(fold_cms, dashboard_output, figsize=(16, 10))
        print(f"\n✓ Combined confusion matrices dashboard saved:")
        print(f"  {dashboard_output}")
    else:
        print(f"WARNING: Only {len(fold_cms)}/5 folds loaded. Skipping dashboard.")

CM_DASH_EOF
echo ""
echo "=========================================================================="
echo "Generating combined PR and ROC curves dashboard..."
python3 << 'PR_ROC_DASH_EOF'
import json
import numpy as np
import glob
import os
import re
from pathlib import Path
from visualization_utils import plot_combined_pr_roc_curves

# Extract iteration number
iter_num = os.environ.get('ITER', '31.0')
print(f"[PRROC] Processing iteration: {iter_num}")

# Find all probabilities.json files for THIS specific iteration
all_probs = sorted(glob.glob("results/*_f[0-4]_convnext_tiny_probabilities.json"))

# Filter to only reports matching this iteration
prob_files = []
for probs_path in all_probs:
    match = re.search(r'_(\d+\.\d+)_\d+_f\d+_', probs_path)
    if match:
        report_iter = match.group(1)
        if report_iter == iter_num:
            prob_files.append(probs_path)

print(f"[PRROC] Found {len(prob_files)} probability files for iteration {iter_num}")

if len(prob_files) == 5:
    # Load probability files for each fold
    fold_data_list = []
    for probs_path in sorted(prob_files):
        try:
            with open(probs_path, 'r') as f:
                prob_data = json.load(f)
            
            labels = np.array(prob_data['labels'])
            probs = np.array(prob_data['probabilities'])
            
            # Extract fold index
            match = re.search(r'_f(\d+)_', probs_path)
            fold_idx = int(match.group(1)) if match else -1
            
            fold_data_list.append({
                'fold_idx': fold_idx,
                'labels': labels,
                'probs': probs
            })
            print(f"  ✓ Fold {fold_idx}: {len(labels)} samples loaded")
        except Exception as e:
            print(f"  ✗ Error loading {probs_path}: {e}")
    
    # Generate combined dashboard if all 5 folds loaded successfully
    if len(fold_data_list) == 5:
        # Sort by fold index to ensure correct order
        fold_data_list = sorted(fold_data_list, key=lambda x: x['fold_idx'])
        
        model_name = os.environ.get('MODEL_NAME', 'convnext_tiny')
        run_id = os.environ.get('RUN_ID', '')
        
        # Build filename with run_id if available
        if run_id:
            dashboard_output = f"results/{run_id}_{iter_num}_pr_roc_curves_combined_{model_name}.png"
        else:
            dashboard_output = f"results/{iter_num}_pr_roc_curves_combined_{model_name}.png"
        
        plot_combined_pr_roc_curves(fold_data_list, dashboard_output, figsize=(16, 14))
        print(f"\n✓ Combined PR and ROC curves dashboard saved:")
        print(f"  {dashboard_output}")
    else:
        print(f"WARNING: Only {len(fold_data_list)}/5 folds loaded. Skipping dashboard.")
else:
    print(f"WARNING: Expected 5 probability files for iteration {iter_num}, found {len(prob_files)}")

PR_ROC_DASH_EOF
echo ""
echo "=========================================================================="
echo "DeepHP pre-training pipeline finished!"
echo "=========================================================================="
echo ""
echo "Backbone checkpoints:"
ls -lah results/deephp_backbone_pretrained_convnext_tiny_f*.pth 2>/dev/null | tail -5
echo ""
echo "Averaged backbone:"
model_name=$(python3 -c "import os; print(os.environ.get('MODEL_NAME', 'convnext_tiny'))")
iter_name=$(python3 -c "import os; print(os.environ.get('ITER', '31.0'))")
# Note: Run ID is extracted in the summary script and embedded in the filename
ls -lah results/deephp_backbone_final_*_${model_name}_${iter_name}.pth 2>/dev/null
echo ""

SUMMARY_EOF
)

SUMMARY_JOB_ID=$(echo $SUMMARY_JOB_OUT | awk '{print $4}')

# Write summary job ID file immediately (for parallel job safety)
# This allows transfer_learning.sh to proceed without waiting
SUMMARY_JOB_ID_FILE="results/${RUN_ID}_${ITER}_summary_job_id.txt"
echo "$SUMMARY_JOB_ID" > "$SUMMARY_JOB_ID_FILE"

echo "=========================================================================="
echo "✓ All jobs submitted successfully!"
echo "=========================================================================="
echo ""
echo "OUTPUTS GENERATED PER FOLD:"
echo "  - {prefix}_model_brain.pth: Trained backbone weights"
echo "  - {prefix}_cross_leakage_audit.csv: Image-level stratification verification"
echo "  - {prefix}_experiment_fold_audit.csv: Per-fold experiment composition"
echo "  - {prefix}_gradcam.png: Grad-CAM (TP/FP/FN/TN guaranteed coverage)"
echo "  - {prefix}_metrics_summary.csv: Bootstrap CI metrics"
echo "  - {prefix}_probabilities.json: Per-sample predictions for post-hoc analysis"
echo ""
echo "CROSS-VALIDATION SUMMARIES (Generated by Summary Job):"
echo "  - grand_cv_pretraining_summary_{run_id}_{iter}.csv: Long-format fold metrics"
echo "  - grand_cv_pretraining_averages_{run_id}_{iter}.csv: Averages ± std"
echo "  - grand_cv_pretraining_bootstrap_ci_{run_id}_{iter}.csv: Bootstrap confidence intervals"
echo "  - {run_id}_{iter}_confusion_matrices_combined_{model}.png: 5-fold confusion matrix dashboard"
echo "  - {run_id}_{iter}_pr_roc_curves_combined_{model}.png: 5-fold PR and ROC dashboard"
echo ""
echo "FINAL OUTPUT:"
echo "  - deephp_backbone_final_{run_id}_{model}_{iter}.pth: Averaged backbone (all 5 folds)"
echo ""
echo "Pre-sync Job ID: $PRE_SYNC_JOB_ID"
echo "Fold Jobs: $DEPENDENCIES"
echo "Summary Job ID: $SUMMARY_JOB_ID"
echo ""
echo "Run ID: $RUN_ID"
echo "Iteration: $ITER"
echo ""
echo "Summary job ID file written immediately:"
echo "  $SUMMARY_JOB_ID_FILE"
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER | grep deephp"
echo ""
echo "View logs with:"
echo "  tail -f results/slurm_deephp_f0_*.txt"
echo ""

# Submit post-processing job after summary completes
POST_PROCESS_JOB=$(sbatch --dependency=afterok:$SUMMARY_JOB_ID \
    --export=ALL \
    -p pg1tfg12 --time=0-00:30 --mem=16G --cpus-per-task=2 \
    --job-name=deephp_postprocess --output=$OUTPUT_DIR/slurm_postprocess_%j.txt \
    <<'POSTPROCESS_EOF'
#!/bin/bash
# Get virtual environment path from config
VENV_ROOT=$(python3 -c "from config import VENV_ROOT; print(VENV_ROOT)")
source $VENV_ROOT/bin/activate
cd /home/tkeating/model/H.-Pylori-Contamination-Detection

echo "Post-Processing Pipeline"
echo "======================="

# Step 1: Calibrate thresholds
echo "Step 1: Calibrating per-fold thresholds..."
python3 calibrate_per_fold_thresholds_deepHP.py --run ${RUN_ID}_${ITER} || exit 1

# Step 2: Apply thresholds
echo "Step 2: Applying calibrated thresholds..."
python3 apply_calibrated_thresholds_deepHP.py --run ${RUN_ID}_${ITER} --model convnext_tiny || exit 1

# Step 3: Load backbone with thresholds
echo "Step 3: Loading backbone models with thresholds..."
python3 deephp_backbone_with_threshold.py --run ${RUN_ID}_${ITER} --model convnext_tiny

# Step 4: Weighted ensemble voting
echo "Step 4: Computing weighted ensemble..."
python3 ensemble_voting_deepHP.py --run ${RUN_ID}_${ITER} --strategy f1 || exit 1

# Step 5: Regenerate backbone using F1-weighted ensemble weights (improved over equal-weight average)
echo "Step 5: Regenerating backbone with F1-weighted ensemble averaging..."
python3 << 'WEIGHTED_BACKBONE_EOF'
import json
import glob
from pathlib import Path
from load_pretrained_backbone import weighted_average_backbone_weights

# Use variables from environment
run_id = "${RUN_ID}"
iter_name = "${ITER}"
model_name = "convnext_tiny"
run_iter_combined = f"{run_id}_{iter_name}"

weights_file = Path("results") / f"{run_iter_combined}_ensemble_weights_f1.json"

if not weights_file.exists():
    print(f"ERROR: Ensemble weights file not found: {weights_file}")
    exit(1)

with open(weights_file) as f:
    ensemble_data = json.load(f)
    fold_weights = ensemble_data.get("fold_weights", {})

print(f"Loaded F1-based ensemble weights:")
for fold_idx, weight in sorted(fold_weights.items()):
    print(f"  Fold {fold_idx}: {float(weight):.4f}")

# Find all fold checkpoints
# Pattern: {run_id}_{iter}_{slurm_id}_f{fold}_{model}_model_brain.pth
results_dir = Path("results")
fold_paths = []
for fold_idx in range(5):
    fold_files = list(results_dir.glob(f"{run_iter_combined}_*_f{fold_idx}_{model_name}_model_brain.pth"))
    if fold_files:
        # Get the most recent one (in case multiple exist)
        fold_path = sorted(fold_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        fold_paths.append(str(fold_path))
    else:
        print(f"WARNING: Fold {fold_idx} checkpoint not found")

if len(fold_paths) < 5:
    print(f"ERROR: Expected 5 folds, found {len(fold_paths)}")
    exit(1)

# Regenerate backbone with weighted averaging using correct naming convention
output_path = f"results/deephp_backbone_final_{run_id}_{model_name}_{iter_name}.pth"
weighted_average_backbone_weights(fold_paths, fold_weights, output_path)
print(f"✓ Backbone regenerated with F1-weighted ensemble averaging!")
WEIGHTED_BACKBONE_EOF

echo "Post-processing complete!"
POSTPROCESS_EOF
)

POST_JOB_ID=$(echo $POST_PROCESS_JOB | awk '{print $4}')
echo "Post-processing job ID: $POST_JOB_ID"
echo ""

echo "STRATIFICATION VERIFICATION (CONFIG 87771):"
echo "  1. Per-fold image-level audits: Verify no patch appears in both train and val for that fold"
echo "  2. Per-fold experiment audits: Show which experiments assigned to each fold"
echo "     - Each experiment assigned to exactly ONE fold (experiment-level integrity)"
echo "     - Each fold validates on different experiments (prevents fold-specific artifacts)"
echo "     - All folds train on all experiments except their validation experiments"
echo ""