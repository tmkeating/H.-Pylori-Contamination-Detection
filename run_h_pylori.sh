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

# Generate rsync exclude filters from blacklist.json to skip all blacklisted items
EXCLUDE_FILTERS=$(python3 << PYTHON_EOF
import json
from pathlib import Path

blacklist_path = Path("./blacklist.json")

if blacklist_path.exists():
    with open(blacklist_path, 'r') as f:
        data = json.load(f)
        conflict_bags = list(data.get('conflict_blacklist', {}).keys())
        
        excludes = []
        # Exclude entire blacklisted bags (folders)
        for bag_id in conflict_bags:
            excludes.append(f"--exclude={bag_id}")
        
        if excludes:
            print(' '.join(excludes))
PYTHON_EOF
)

# Sync folders (rsync is efficient - only copies missing/changed files)
# Skip all blacklisted bags listed in blacklist.json
mkdir -p "$LOCAL_SCRATCH/CrossValidation"
if [ -n "$EXCLUDE_FILTERS" ]; then
    echo "[RSYNC] Syncing with ${EXCLUDE_FILTERS}"
    eval "rsync -aq $EXCLUDE_FILTERS '$REMOTE_DATA/CrossValidation/Annotated' '$LOCAL_SCRATCH/CrossValidation/'"
    eval "rsync -aq $EXCLUDE_FILTERS '$REMOTE_DATA/CrossValidation/Cropped' '$LOCAL_SCRATCH/CrossValidation/'"
    eval "rsync -aq $EXCLUDE_FILTERS '$REMOTE_DATA/HoldOut' '$LOCAL_SCRATCH/'"
else
    echo "[RSYNC] No exclude filters found - syncing all files"
    rsync -aq "$REMOTE_DATA/CrossValidation/Annotated" "$LOCAL_SCRATCH/CrossValidation/"
    rsync -aq "$REMOTE_DATA/CrossValidation/Cropped" "$LOCAL_SCRATCH/CrossValidation/"
    rsync -aq "$REMOTE_DATA/HoldOut" "$LOCAL_SCRATCH/"
fi

echo "Data synchronization complete (blacklisted items removed before sync, rsync excludes prevent re-sync)."

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
