#!/bin/bash
# train_deepHP.sh - DeepHP 5-Fold SLURM Orchestrator
#
# Purpose: Orchestrate full transfer learning pipeline:
#   1. Pre-train backbone on 394,926 H&E patches (5 folds)
#   2. Average backbone weights across folds
#   3. Schedule final summary job
#
# Usage:
#   PROFILE=SEARCHER MODEL_NAME=convnext_tiny ITER=31.0 ./train_deepHP.sh
#
# Environment Variables:
#   PROFILE:    Model profile from profiles.sh (default: SEARCHER)
#   MODEL_NAME: Backbone architecture (default: convnext_tiny)
#   ITER:       Iteration number for tracking (default: 31.0)

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
FOLD_BATCH_SIZE=${FOLD_BATCH_SIZE:-"0"}  # 0=all parallel (default), N=batch in groups of N

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
POS_WEIGHT=${POS_WEIGHT:-2.5}
USE_FOCAL_LOSS=${USE_FOCAL_LOSS:-"False"}
GAMMA=${GAMMA:-1.0}

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

# 1. Submit pre-sync job (prepares scratch directory for data)
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

echo "Submitting pre-sync job to prepare environment..."
PRE_SYNC_JOB=$(sbatch -p pg1tfg12 --job-name=deephp_presync --output=results/slurm_deephp_presync_%j.txt <<'PRESYNC_EOF'
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

echo "Pre-sync job: Setting up DeepHP dataset for training..."

# Get dataset root from config
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
        --output=results/slurm_deephp_f${FOLD}_%j.txt \
        --error=results/slurm_deephp_error_f${FOLD}_%j.txt \
        --ntasks=1 \
        --cpus-per-task=4 \
        --gres=gpu:l40s:1 --gres=shard:l40s:12000 \
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
LEARNING_RATE=$LEARNING_RATE
WEIGHT_DECAY=$WEIGHT_DECAY
POS_WEIGHT=$POS_WEIGHT
USE_FOCAL_LOSS=$USE_FOCAL_LOSS
GAMMA=$GAMMA
ITER=$ITER

cd /home/tkeating/model/H.-Pylori-Contamination-Detection

# Force all folds to use GPU 0 for memory consolidation
export CUDA_VISIBLE_DEVICES=0

# Get scratch directory at runtime from config
DEEPHP_SCRATCH=\$(python3 -c "from config import DEEPHP_SCRATCH_ROOT; print(DEEPHP_SCRATCH_ROOT)")

# Use synced DeepHP dataset from scratch
export DEEPHP_DATASET_ROOT="\$DEEPHP_SCRATCH"

python3 -u train_deepHP_patches.py \
    --fold \$FOLD \
    --num_folds 5 \
    --model_name \$MODEL_NAME \
    --num_epochs \$DEEPHP_EPOCHS \
    --batch_size \$BATCH_SIZE \
    --learning_rate \$LEARNING_RATE \
    --weight_decay \$WEIGHT_DECAY \
    --pos_weight \$POS_WEIGHT \
    --use_focal_loss \$USE_FOCAL_LOSS \
    --gamma \$GAMMA \
    --iter \$ITER

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
#    This job averages the backbone and prepares for fine-tuning

# Final validation of dependency string
if [ -z "$DEPENDENCY_STRING" ]; then
    echo "ERROR: Failed to generate valid dependency string from: $DEPENDENCIES"
    exit 1
fi

SUMMARY_JOB_OUT=$(sbatch --dependency=$DEPENDENCY_STRING \
    -p pg1tfg12 \
    --time=0-00:30 \
    --mem=16G \
    --cpus-per-task=1 \
    --gres=gpu:l40s:1 --gres=shard:l40s:4000 \
    --job-name=deephp_summary \
    --output=results/slurm_deephp_summary_%j.txt \
    --error=results/slurm_deephp_summary_error_%j.txt \
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
output_path = "results/deephp_backbone_final_convnext_tiny.pth"

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

PYTHON_EOF

echo "✓ Summary job complete"
echo ""
echo "=========================================================================="
echo "DeepHP pre-training pipeline finished!"
echo "=========================================================================="
echo ""
echo "Backbone checkpoints:"
ls -lah results/deephp_backbone_pretrained_convnext_tiny_f*.pth 2>/dev/null | tail -5
echo ""
echo "Averaged backbone:"
ls -lah results/deephp_backbone_final_convnext_tiny.pth 2>/dev/null
echo ""

SUMMARY_EOF
)

SUMMARY_JOB_ID=$(echo $SUMMARY_JOB_OUT | awk '{print $4}')

echo "=========================================================================="
echo "✓ All jobs submitted successfully!"
echo "=========================================================================="
echo ""
echo "Pre-sync Job ID: $PRE_SYNC_JOB_ID"
echo "Fold Jobs: $DEPENDENCIES"
echo "Summary Job ID: $SUMMARY_JOB_ID"
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER | grep deephp"
echo ""
echo "View logs with:"
echo "  tail -f results/slurm_deephp_f0_*.txt"
echo ""

# Output summary job ID to file for orchestrator scripts to read
echo "$SUMMARY_JOB_ID" > results/deephp_summary_job_id.txt
echo "Summary job ID written to: results/deephp_summary_job_id.txt"
echo ""