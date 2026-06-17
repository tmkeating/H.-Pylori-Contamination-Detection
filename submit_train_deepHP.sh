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

# Export all configuration variables for presync job access
export NUM_EPOCHS NEG_WEIGHT POS_WEIGHT GAMMA USE_FOCAL_LOSS SAVER_METRIC
export FREEZE_BN FREEZE_BACKBONE CLIP_GRAD PCT_START WEIGHT_DECAY
export USE_SWA SWA_START JITTER POOL_TYPE DEEPHP_EPOCHS BATCH_SIZE LEARNING_RATE
export VENV_ROOT PROFILE MODEL_NAME ITER PRETRAINED_BACKBONE

# 1. Submit pre-sync job (prepares scratch directory for data)
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
echo "  5 Folds with Experiment-Level Stratification"
echo "=========================================================================="
echo ""

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
    --neg_weight \$NEG_WEIGHT \
    --pos_weight \$POS_WEIGHT \
    --use_focal_loss \$USE_FOCAL_LOSS \
    --gamma \$GAMMA \
    --use_swa \$USE_SWA \
    --swa_start \$SWA_START \
    --jitter \$JITTER \
    --pct_start \$PCT_START \
    --clip_grad \$CLIP_GRAD \
    --saver_metric \$SAVER_METRIC \
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

# Build output path using MODEL_NAME and ITER from environment
model_name = os.environ.get('MODEL_NAME', 'convnext_tiny')
iter_name = os.environ.get('ITER', '31.0')
output_path = f"results/deephp_backbone_final_{model_name}_{iter_name}.pth"

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

# Find all DeepHP evaluation reports for THIS specific iteration (one per fold)
# Pattern: {run_id}_{iter}_{slurm_id}_f{fold}_{model_name}_evaluation_report.csv
all_reports = sorted(glob.glob("results/*_f[0-4]_convnext_tiny_evaluation_report.csv"))

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
    # Load all fold evaluation reports
    fold_data = []
    for report_path in eval_reports:
        try:
            df = pd.read_csv(report_path)
            # Extract fold number from filename (pattern: ..._f{N}_...)
            match = re.search(r'_f(\d+)_', report_path)
            if match:
                fold_idx = int(match.group(1))
                # Add fold column
                df['Fold'] = fold_idx
                fold_data.append(df)
                print(f"  ✓ Loaded fold {fold_idx}: {Path(report_path).name}")
        except Exception as e:
            print(f"  WARNING: Could not load {report_path}: {e}")
    
    if fold_data:
        # Combine all fold data
        all_metrics = pd.concat(fold_data, ignore_index=True)
        
        # Generate summary CSV with per-fold metrics (include iteration in filename)
        summary_output = f"results/grand_cv_pretraining_summary_{iter_num}.csv"
        all_metrics.to_csv(summary_output, index=False)
        print(f"\n✓ Pre-training summary CSV: {summary_output}")
        print(f"  ({len(fold_data)} folds, {len(all_metrics)} total rows)")
        
        # Generate averages CSV
        metric_cols = [col for col in all_metrics.columns if col not in ['Fold', 'RunID', 'FoldIdx']]
        
        averages = {}
        stds = {}
        
        for col in metric_cols:
            if all_metrics[col].dtype in [np.float64, np.int64]:
                try:
                    averages[col] = all_metrics[col].mean()
                    stds[col] = all_metrics[col].std()
                except:
                    pass
        
        avg_df = pd.DataFrame({
            'Metric': list(averages.keys()),
            'Mean': list(averages.values()),
            'Std': list(stds.values())
        })
        
        # Add formatted column
        avg_df['Formatted'] = avg_df.apply(
            lambda row: f"{row['Mean']:.4f} ± {row['Std']:.4f}", axis=1
        )
        avg_df['Run_Range'] = iter_num
        
        avg_output = f"results/grand_cv_pretraining_averages_{iter_num}.csv"
        avg_df.to_csv(avg_output, index=False)
        print(f"✓ Pre-training averages CSV: {avg_output}")
        
        # Generate bootstrap CI CSV
        from scipy import stats
        
        ci_data = []
        for col in metric_cols:
            if all_metrics[col].dtype in [np.float64, np.int64]:
                try:
                    values = all_metrics[col].dropna().values
                    point_est = values.mean()
                    fold_std = values.std()
                    
                    # Bootstrap CI
                    bootstrap_samples = np.random.choice(values, size=(1000, len(values)), replace=True)
                    bootstrap_means = bootstrap_samples.mean(axis=1)
                    bootstrap_mean = bootstrap_means.mean()
                    bootstrap_std = bootstrap_means.std()
                    
                    ci_lower = np.percentile(bootstrap_means, 2.5)
                    ci_upper = np.percentile(bootstrap_means, 97.5)
                    ci_margin = (ci_upper - ci_lower) / 2
                    
                    ci_data.append({
                        'Metric': col,
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
echo "=========================================================================="
echo "Generating combined confusion matrices dashboard..."
python3 << 'CM_DASH_EOF'
import pandas as pd
import numpy as np
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
    # Load evaluation reports and extract confusion matrices
    fold_cms = []
    for report_path in sorted(eval_reports):
        try:
            df = pd.read_csv(report_path)
            # Extract fold number
            match = re.search(r'_f(\d+)_', report_path)
            fold_idx = int(match.group(1)) if match else -1
            
            # Get labels and predictions from evaluation report
            if 'Label' in df.columns and 'Prediction' in df.columns:
                labels = df['Label'].values
                preds = df['Prediction'].values
                cm = confusion_matrix(labels, preds, labels=[0, 1])
                fold_cms.append(cm)
                print(f"  ✓ Fold {fold_idx}: {cm.shape} confusion matrix")
            else:
                print(f"  ✗ Fold {fold_idx}: Missing Label or Prediction columns")
        except Exception as e:
            print(f"  ✗ Error loading {report_path}: {e}")
    
    # Generate combined dashboard if all 5 folds loaded successfully
    if len(fold_cms) == 5:
        model_name = os.environ.get('MODEL_NAME', 'convnext_tiny')
        dashboard_output = f"results/{iter_num}_confusion_matrices_combined_{model_name}.png"
        
        plot_combined_confusion_matrices(fold_cms, dashboard_output, figsize=(16, 10))
        print(f"\n✓ Combined confusion matrices dashboard saved:")
        print(f"  {dashboard_output}")
    else:
        print(f"WARNING: Only {len(fold_cms)}/5 folds loaded. Skipping dashboard.")
else:
    print(f"WARNING: Expected 5 reports for iteration {iter_num}, found {len(eval_reports)}")

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
ls -lah "results/deephp_backbone_final_${model_name}_${iter_name}.pth" 2>/dev/null
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