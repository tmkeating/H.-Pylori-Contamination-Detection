#!/bin/bash
# High-Resolution SLURM Rescue Batch Script
# ------------------------------------------
# Submits a GPU job to perform dense-stride inference (Rescue Pass) on specifically 
# identified difficult patients (e.g., the 'Unreachable Six').
#
# What it does:
#   1. Allocates a GPU and 32GB RAM on the cluster.
#   2. Iterates through all 5 Searcher model folds.
#   3. Runs rescue_inference.py with a dense stride (128px) for each fold.
#   4. Extends 16-way contrast-boosted TTA for maximum signal recovery.
#
# Usage:
#   sbatch submit_rescue.sh
#
# Configurable Variables:
#   STRIDE:  The dense window overlap (Default: 128).
#   TARGETS: Comma-separated list of PatientIDs to recover.
# ------------------------------------------
#SBATCH --job-name=hpy_rescue
#SBATCH -D .
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -t 0-02:00
#SBATCH -p dcca40
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o results/rescue_ensemble/slurm_rescue_%j.txt
#SBATCH -e results/rescue_ensemble/error_rescue_%j.txt

# Create output directory
mkdir -p results/rescue_ensemble

# --- LOCAL SCRATCH SETUP (Skip image sync if already present or just use remote) ---
# We'll rely on the existing dataset paths in rescue_inference.py which already 
# check for /tmp/ricse03_h_pylori_data first.

# Activate virtual environment
source ../venv/bin/activate

# Parameters
STRIDE=${STRIDE:-128}
TARGETS=${TARGETS:-"B22-206,B22-262,B22-69,B22-81,B22-85,B22-01"}
MODEL_DIR="results/"
OUTPUT_DIR="results/rescue_ensemble"

# Explicit fold list from current Searcher Iteration (25.1)
# High-resolution pass (Stride 128) for the identified 'Unreachable Six' patients.
# Fold 0: 302_25.1_106069_f0
# Fold 1: 303_25.1_106070_f1
# Fold 2: 304_25.1_106071_f2
# Fold 3: 305_25.1_106072_f3
# Fold 4: 306_25.1_106073_f4

FOLDS=("302_25.1_106069_f0" "303_25.1_106070_f1" "304_25.1_106071_f2" "305_25.1_106072_f3" "306_25.1_106073_f4")

echo "--- 🏥 Starting Full-Ensemble Rescue SLURM (Stride: $STRIDE) ---"
echo "Targets: $TARGETS"

for FOLD_BASE in "${FOLDS[@]}"; do
    MODEL_PATH="${MODEL_DIR}/${FOLD_BASE}_convnext_tiny_model_brain.pth"
    OUTPUT_CSV="${OUTPUT_DIR}/rescue_${FOLD_BASE}.csv"
    
    echo "-------------------------------------------"
    echo "Processing $FOLD_BASE..."
    python3 rescue_inference.py \
        --model "$MODEL_PATH" \
        --output "$OUTPUT_CSV" \
        --stride $STRIDE \
        --targets "$TARGETS"
done

echo "--- 🏁 Rescue Ensemble SLURM Job Completed ---"
