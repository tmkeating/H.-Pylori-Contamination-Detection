#!/bin/bash
# H. Pylori Core Training & Inference Worker
# -----------------------------------------
# This script handles the heavy lifting for a single fold: data synchronization, 
# environment setup, and execution of the train.py script. It is designed to run 
# as a SLURM batch job but supports manual execution via a 'manual' flag.
#
# What it does:
#   1. Synchronizes the dataset from network storage to local NVMe SSD (/tmp) 
#      using rsync for high-speed I/O.
#   2. Activates the Python virtual environment.
#   3. Executes train.py with specific Profile (SEARCHER/AUDITOR) hyperparameters.
#   4. Manages specialized logging for SLURM vs. manual runs.
#
# Usage:
#   sbatch run_h_pylori.sh
#   (Usually invoked by submit_all_folds.sh)
#
# Environment Variables (Inherited from submit_all_folds.sh):
#   FOLD, MODEL_NAME, NEG_WEIGHT, POS_WEIGHT, GAMMA, NUM_EPOCHS, FREEZE_BN, 
#   CLIP_GRAD, PCT_START, WD, USE_SWA, JITTER, POOL_TYPE, ITER.
# -----------------------------------------
#SBATCH --job-name=h_pylori_fast
#SBATCH -D .
#SBATCH -n 1                           # One task
#SBATCH -c 8                          # Request 8 CPU cores per task for fast data loading
#SBATCH -N 1
#SBATCH -t 0-06:00                     # It will likely finish in < 6 hours now
#SBATCH -p dcca40
#SBATCH --mem=48G                      # Increased to 48GB as requested
#SBATCH --gres=gpu:1
#SBATCH -o results/output_%j.txt
#SBATCH -e results/error_%j.txt

# Create results directory if it doesn't exist
mkdir -p results

# --- LOCAL SCRATCH SETUP ---
# We use /tmp as it's on a local NVMe SSD (faster than network storage)
LOCAL_SCRATCH="/tmp/ricse03_h_pylori_data"
REMOTE_DATA="/import/fhome/vlia/HelicoDataSet"

echo "Synchronizing dataset to local scratch: $LOCAL_SCRATCH"
mkdir -p "$LOCAL_SCRATCH"

# Skip sync if this is a fold job (pre-sync job already synced)
if [ "$SKIP_SYNC" = "1" ]; then
    echo "[SKIP_SYNC] Skipping data sync - fold job will use pre-synced data"
    echo "Data synchronization complete (blacklisted items and orphaned patches excluded)."
else

# Copy metadata (fast)
cp "$REMOTE_DATA"/*.xlsx "$LOCAL_SCRATCH/"
cp "$REMOTE_DATA"/*.csv "$LOCAL_SCRATCH/"

# Clean up any previously synced blacklisted items from scratch before re-syncing
# This ensures we don't accumulate old blacklisted files across runs
BLACKLIST_FILE="./blacklist.json"

# Pre-rsync cleanup: Remove any existing blacklisted bags
python3 << CLEANUP_EOF
import json
import shutil
from pathlib import Path
import sys

blacklist_path = Path("$BLACKLIST_FILE")
scratch_path = Path("$LOCAL_SCRATCH")

print(f"[CLEANUP] Checking for blacklisted items to remove...")
print(f"[CLEANUP] Blacklist file: {blacklist_path}")
print(f"[CLEANUP] Scratch path: {scratch_path}")

if not blacklist_path.exists():
    print(f"[CLEANUP] WARNING: Blacklist file not found at {blacklist_path}")
    # Not fatal - just continue without cleaning up
else:
    if scratch_path.exists():
        with open(blacklist_path, 'r') as f:
            data = json.load(f)
            conflict_bags = list(data.get('conflict_blacklist', {}).keys())
            
            if conflict_bags:
                print(f"[CLEANUP] Found {len(conflict_bags)} blacklisted bags to remove")
                removed_count = 0
                
                # Remove entire bag folders if they exist in scratch
                for bag_id in conflict_bags:
                    # Blacklisted bags could be in any of these directories
                    for dir_name in ['CrossValidation/Annotated', 'CrossValidation/Cropped', 'HoldOut']:
                        bag_path = scratch_path / dir_name / bag_id
                        if bag_path.exists():
                            print(f"[CLEANUP] Removing: {bag_path}")
                            shutil.rmtree(bag_path)
                            removed_count += 1
                
                print(f"[CLEANUP] Removed {removed_count} blacklisted bags from scratch")
            else:
                print(f"[CLEANUP] No blacklisted bags found in blacklist.json")
    else:
        print(f"[CLEANUP] Scratch path doesn't exist yet - no cleanup needed")
CLEANUP_EOF

# Generate rsync exclude filters from blacklist.json AND identify orphaned patches
EXCLUDE_FILTER_FILE="/tmp/h_pylori_exclude_filters_$$.txt"
export EXCLUDE_FILTER_FILE

echo "[DEBUG] Filter file will be at: $EXCLUDE_FILTER_FILE" >&2

# Create a basic rsync filter file using jq (if available) or Python
if command -v jq &> /dev/null; then
    echo "[DEBUG] Using jq to parse blacklist.json" >&2
    {
        jq -r '.conflict_blacklist | keys[]' blacklist.json | sed 's/^/- /'
        jq -r '.image_blacklist[]? | "- \(.folder)/\(.filename)"' blacklist.json
        echo "+ */"
        echo "- *"
        echo "+ *.png"
        echo "- *"
    } > "$EXCLUDE_FILTER_FILE"
    echo "[DEBUG] Created filter file with jq" >&2
else
    echo "[DEBUG] jq not available, using Python" >&2
    python3 << 'PYTHON_EOF'
import json
import os
from pathlib import Path

exclude_filter_file = os.environ['EXCLUDE_FILTER_FILE']
remote_data = os.environ.get('REMOTE_DATA', '/import/fhome/vlia/HelicoDataSet')

excludes = []

# --- PART 1: Blacklist exclusions ---
with open('./blacklist.json') as f:
    data = json.load(f)

for bag in data.get('conflict_blacklist', {}).keys():
    excludes.append(f"- {bag}/")
for item in data.get('image_blacklist', []):
    if isinstance(item, dict):
        excludes.append(f"- {item.get('folder')}/{item.get('filename')}")

print(f"[DEBUG] Blacklist: {len(data.get('conflict_blacklist', {}))} bags + {len(data.get('image_blacklist', []))} images")

# --- PART 2: Detect orphaned patches ---
try:
    import pandas as pd
    p_csv = os.path.join(remote_data, "PatientDiagnosis.csv")
    patch_xlsx = os.path.join(remote_data, "HP_WSI-CoordAnnotatedAllPatches.xlsx")
    
    if os.path.exists(p_csv) and os.path.exists(patch_xlsx):
        print(f"[DEBUG] Loading clinical metadata...")
        patient_df = pd.read_csv(p_csv)
        # Column name is 'CODI', not 'PatientID'
        clinical_patients = set(patient_df['CODI'].unique()) if 'CODI' in patient_df.columns else set()
        
        patch_df = pd.read_excel(patch_xlsx)
        annotated_patients = set()
        # Column name is 'Pat_ID', not 'Patient_ID'
        if 'Pat_ID' in patch_df.columns:
            annotated_patients = set(patch_df['Pat_ID'].dropna().unique())
        
        valid_patients = clinical_patients.union(annotated_patients)
        print(f"[DEBUG] Found {len(valid_patients)} valid patients in clinical metadata")
        
        # Scan for orphaned patches
        orphaned_count = 0
        remote_path = Path(remote_data)
        for dir_name in ['CrossValidation/Annotated', 'CrossValidation/Cropped', 'HoldOut']:
            dir_path = remote_path / dir_name
            if dir_path.exists():
                for bag_dir in dir_path.iterdir():
                    if bag_dir.is_dir():
                        bag_name = bag_dir.name
                        patient_id = '_'.join(bag_name.split('_')[:-1]) if '_' in bag_name else bag_name
                        
                        if patient_id not in valid_patients:
                            excludes.append(f"- {bag_name}/")
                            orphaned_count += 1
        
        print(f"[DEBUG] Detected {orphaned_count} orphaned bags to exclude")
    else:
        print(f"[DEBUG] Clinical metadata not found - skipping orphan detection")
except ImportError:
    print(f"[DEBUG] pandas not available - skipping orphan detection")
except Exception as e:
    print(f"[DEBUG] Warning: Could not detect orphaned patches: {e}")

# Write filter file
with open(exclude_filter_file, 'w') as out:
    for exclude in excludes:
        out.write(exclude + "\n")
    out.write("+ */\n")
    out.write("- *\n")
    out.write("+ *.png\n")
    out.write("- *\n")

print(f"[DEBUG] Wrote {len(excludes)} total exclusion rules to filter file")
PYTHON_EOF
fi

echo "[DEBUG] Filter file created: $(ls -lh $EXCLUDE_FILTER_FILE 2>/dev/null || echo 'NOT FOUND')" >&2

# Sync folders with filter file
# Skip blacklisted bags AND orphaned patches based on clinical metadata
# Use an exclusive lock to serialize rsync operations across concurrent folds
# This prevents race conditions when multiple folds sync to the same scratch directory
SYNC_LOCK_FILE="/tmp/h_pylori_sync.lock"
{
    echo "[RSYNC] Waiting for exclusive lock on scratch sync..."
    flock -x 200
    echo "[RSYNC] Acquired lock - proceeding with sync"
    
    mkdir -p "$LOCAL_SCRATCH/CrossValidation"
    if [ -f "$EXCLUDE_FILTER_FILE" ]; then
        echo "[RSYNC] Syncing with exclude filter file ($EXCLUDE_FILTER_FILE)"
        rsync -aq --filter="merge $EXCLUDE_FILTER_FILE" "$REMOTE_DATA/CrossValidation/Annotated" "$LOCAL_SCRATCH/CrossValidation/"
        rsync -aq --filter="merge $EXCLUDE_FILTER_FILE" "$REMOTE_DATA/CrossValidation/Cropped" "$LOCAL_SCRATCH/CrossValidation/"
        rsync -aq --filter="merge $EXCLUDE_FILTER_FILE" "$REMOTE_DATA/HoldOut" "$LOCAL_SCRATCH/"
        rm -f "$EXCLUDE_FILTER_FILE"
    else
        echo "[RSYNC] Filter file not found - syncing all files"
        rsync -aq "$REMOTE_DATA/CrossValidation/Annotated" "$LOCAL_SCRATCH/CrossValidation/"
    rsync -aq "$REMOTE_DATA/CrossValidation/Cropped" "$LOCAL_SCRATCH/CrossValidation/"
    rsync -aq "$REMOTE_DATA/HoldOut" "$LOCAL_SCRATCH/"
fi
    echo "[RSYNC] Sync complete, releasing lock"
} 200>"$SYNC_LOCK_FILE"

echo "Data synchronization complete (blacklisted items and orphaned patches excluded)."
fi

# Exit early if this is a pre-sync-only job (all folds will depend on this)
if [ "$PRE_SYNC_ONLY" = "1" ]; then
    echo "[PRE_SYNC] Pre-sync job complete. Exiting to allow dependent fold jobs to start training."
    exit 0
fi

# 1. Load necessary modules (Common on clusters)
# module load python/3.8
# module load cuda/11.8

# 2. Activate your virtual environment
# The venv is located in the parent directory
source ../venv/bin/activate

# 3. Ensure you have the GPU version of PyTorch
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Run the training script
# Use environment variables if set, otherwise default values
FOLD=${FOLD:-0}
NUM_FOLDS=${NUM_FOLDS:-5}
MODEL_NAME=${MODEL_NAME:-"convnext_tiny"}
NEG_WEIGHT=${NEG_WEIGHT:-1.0}
POS_WEIGHT=${POS_WEIGHT:-7.5}
GAMMA=${GAMMA:-1.0}
NUM_EPOCHS=${NUM_EPOCHS:-15}
SAVER_METRIC=${SAVER_METRIC:-"recall"}
FREEZE_BN=${FREEZE_BN:-"False"}
CLIP_GRAD=${CLIP_GRAD:-0.0}
PCT_START=${PCT_START:-0.1}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
USE_SWA=${USE_SWA:-"False"}
SWA_START=${SWA_START:-15}
JITTER=${JITTER:-0.15}
POOL_TYPE=${POOL_TYPE:-"attention"}
ITER=${ITER:-"25.0"}
PRE_SYNC_ONLY=${PRE_SYNC_ONLY:-0}
SKIP_SYNC=${SKIP_SYNC:-0}

# Capture standard output and error to the results directory manually if not on SLURM
SLURM_JOB_ID=${SLURM_JOB_ID:-"manual"}

# If running on SLURM, SBATCH already handles -o and -e logic.
# Only use manual redirection if SLURM_JOB_ID is "manual".
if [ "$SLURM_JOB_ID" == "manual" ]; then
    OUTPUT_LOG="results/output_${FOLD}_manual.txt"
    ERROR_LOG="results/error_${FOLD}_manual.txt"
    echo "Starting Training for Fold: $FOLD of $NUM_FOLDS using $MODEL_NAME (NegWeight=$NEG_WEIGHT, PosWeight=$POS_WEIGHT, Gamma=$GAMMA, Epochs=$NUM_EPOCHS, Saver=$SAVER_METRIC, FreezeBN=$FREEZE_BN, ClipGrad=$CLIP_GRAD, PctStart=$PCT_START, WD=$WEIGHT_DECAY, SWA=$USE_SWA, SWAStart=$SWA_START, Jitter=$JITTER, Pool=$POOL_TYPE)" | tee -a "$OUTPUT_LOG"
    python train.py --fold $FOLD --num_folds $NUM_FOLDS --model_name "$MODEL_NAME" \
                    --neg_weight "$NEG_WEIGHT" --pos_weight "$POS_WEIGHT" --gamma "$GAMMA" \
                    --num_epochs "$NUM_EPOCHS" --saver_metric "$SAVER_METRIC" \
                    --freeze_bn "$FREEZE_BN" --clip_grad "$CLIP_GRAD" --pct_start "$PCT_START" \
                    --weight_decay "$WEIGHT_DECAY" --use_swa "$USE_SWA" --swa_start "$SWA_START" \
                    --jitter "$JITTER" --pool_type "$POOL_TYPE" --iter "$ITER" > >(tee -a "$OUTPUT_LOG") 2> >(tee -a "$ERROR_LOG" >&2)
else
    echo "Starting Training for Fold: $FOLD of $NUM_FOLDS using $MODEL_NAME (NegWeight=$NEG_WEIGHT, PosWeight=$POS_WEIGHT, Gamma=$GAMMA, Epochs=$NUM_EPOCHS, Saver=$SAVER_METRIC, FreezeBN=$FREEZE_BN, ClipGrad=$CLIP_GRAD, PctStart=$PCT_START, WD=$WEIGHT_DECAY, SWA=$USE_SWA, SWAStart=$SWA_START, Jitter=$JITTER, Pool=$POOL_TYPE, Iter=$ITER)"
    python train.py --fold $FOLD --num_folds $NUM_FOLDS --model_name "$MODEL_NAME" \
                    --neg_weight "$NEG_WEIGHT" --pos_weight "$POS_WEIGHT" --gamma "$GAMMA" \
                    --num_epochs "$NUM_EPOCHS" --saver_metric "$SAVER_METRIC" \
                    --freeze_bn "$FREEZE_BN" --clip_grad "$CLIP_GRAD" --pct_start "$PCT_START" \
                    --weight_decay "$WEIGHT_DECAY" --use_swa "$USE_SWA" --swa_start "$SWA_START" \
                    --jitter "$JITTER" --pool_type "$POOL_TYPE" --iter "$ITER"
fi
