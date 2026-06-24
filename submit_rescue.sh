#!/bin/bash
# High-Resolution SLURM Rescue Batch Script
# ------------------------------------------
# Submits a GPU job to perform dense-stride inference (Rescue Pass) on specifically 
# identified difficult patients (e.g., the 'Unreachable Six').
#
# What it does:
#   1. Allocates a GPU and 30GB RAM on the cluster.
#   2. Iterates through all 5 model folds.
#   3. Runs rescue_inference.py with a dense stride (128px) for each fold.
#   4. Extends 16-way contrast-boosted TTA for maximum signal recovery.
#
# Usage:
#   sbatch submit_rescue.sh
#
# Example Command (Rescue misclassified patients from finalResults run):
#   MODEL_DIR="finalResults/convnext_tiny_pretrained_backbone_34.4_weight_1.5_gamma_3.0_focalLoss_false" \
#     FOLDS="01_34.4_9077_f0 01_34.4_9078_f1 01_34.4_9079_f2 01_34.4_9080_f3 01_34.4_9081_f4" \
#     TARGETS="B22-12_1,B22-206_0,B22-262_0,B22-69_1,B22-81_1,B22-85_0,B22-89_0" \
#     OUTPUT_DIR="finalResults/convnext_tiny_pretrained_backbone_34.4_weight_1.5_gamma_3.0_focalLoss_false/rescue_ensemble" \
#     STRIDE=128 \
#     sbatch submit_rescue.sh
#
# Configurable Variables:
#   MODEL_DIR:  Directory containing trained model folds (Default: results/).
#               Use finalResults/DIR_NAME for archived results.
#   FOLDS:      Space-separated list of fold model base names (optional).
#               If not specified, auto-discovers from MODEL_DIR.
#   STRIDE:     The dense window overlap (Default: 128).
#   TARGETS:    Comma-separated list of PatientIDs to recover.
# ------------------------------------------
#SBATCH --job-name=hpy_rescue
#SBATCH -D .
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -t 0-02:00
#SBATCH -p pg1tfg12
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:l40s:12000

set -e  # Exit on error

# Determine output directory for SLURM logs
OUTPUT_DIR=${OUTPUT_DIR:-"results/rescue_ensemble"}
SLURM_LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$SLURM_LOG_DIR"

# Redirect SLURM output to the OUTPUT_DIR
exec 1>"${SLURM_LOG_DIR}/slurm_rescue_$$.txt"
exec 2>"${SLURM_LOG_DIR}/error_rescue_$$.txt"

# Verify virtual environment before proceeding
if [ -f "./verify_venv.sh" ]; then
    source ./verify_venv.sh
else
    echo "ERROR: verify_venv.sh not found in current directory"
    exit 1
fi

# Activate virtual environment
source $VENV_ROOT/bin/activate

# Parameters
STRIDE=${STRIDE:-128}
TARGETS=${TARGETS:-"B22-206,B22-262,B22-69,B22-81,B22-85,B22-01"}
MODEL_DIR=${MODEL_DIR:-"results/"}
# OUTPUT_DIR already defined above, use default if not set

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# If FOLDS not specified, use default Searcher Iteration (25.1)
# Otherwise, use provided FOLDS (space-separated list of model base names)
if [ -z "$FOLDS" ]; then
    # Default fold list from current Searcher Iteration (25.1)
    # High-resolution pass (Stride 128) for the identified 'Unreachable Six' patients.
    # Fold 0: 302_25.1_106069_f0
    # Fold 1: 303_25.1_106070_f1
    # Fold 2: 304_25.1_106071_f2
    # Fold 3: 305_25.1_106072_f3
    # Fold 4: 306_25.1_106073_f4
    FOLDS=("302_25.1_106069_f0" "303_25.1_106070_f1" "304_25.1_106071_f2" "305_25.1_106072_f3" "306_25.1_106073_f4")
else
    # Convert space-separated FOLDS string to array
    read -ra FOLDS <<< "$FOLDS"
fi

echo "--- 🏥 Starting Full-Ensemble Rescue SLURM (Stride: $STRIDE) ---"
echo "Model Directory: $MODEL_DIR"
echo "Targets: $TARGETS"
echo "Folds: ${FOLDS[@]}"

for FOLD_BASE in "${FOLDS[@]}"; do
    MODEL_PATH="${MODEL_DIR}/${FOLD_BASE}_convnext_tiny_model_brain.pth"
    
    # Verify model exists before processing
    if [ ! -f "$MODEL_PATH" ]; then
        echo "⚠️  WARNING: Model not found: $MODEL_PATH"
        echo "   Skipping fold $FOLD_BASE"
        continue
    fi
    
    echo "-------------------------------------------"
    echo "Processing $FOLD_BASE..."
    python3 rescue_inference.py \
        --model "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --stride $STRIDE \
        --targets "$TARGETS"
done

echo "--- 🏁 Rescue Ensemble SLURM Job Completed ---"
