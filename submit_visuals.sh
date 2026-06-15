#!/bin/bash
# H. Pylori Visual Report Generation & Grad-CAM Utility
# =====================================================
# Generates comprehensive visual reports and spatial heatmaps (Grad-CAM) to verify 
# that the model is detecting actual bacterial morphology and not tissue artifacts.
#
# What it does:
#   1. Loads trained model checkpoint and validation/test data
#   2. Generates patient-level performance metrics with bootstrap confidence intervals
#   3. Creates ROC, PR, confusion matrix, probability histogram, and threshold analysis plots
#   4. Generates Grad-CAM heatmaps showing high-attention image regions
#   5. Produces CSV reports with detailed predictions and metrics
#
# Supports:
#   - HelicoDataSet (IHC H. Pylori stain) fine-tuning visualizations
#   - DeepHP (H&E histology) pre-training visualizations
#   - Batch processing both datasets
#   - Custom fold selection
#   - Different backbone architectures (ConvNeXt-Tiny, ResNet50)
#
# Usage:
#   sbatch run_visuals.sh [OPTIONS]
#   
#   Or submit with custom parameters (command-line style):
#   sbatch run_visuals.sh --RUN_ID=62 --FOLD=0 --PIPELINE_MODE
#   
#   Or via environment variables (sbatch --export):
#   sbatch --export=RUN_ID=62,FOLD=0,PIPELINE_MODE=true run_visuals.sh
#   
#   Or set environment variables then submit (model auto-detected from training):
#   export RUN_ID=62 PIPELINE_MODE=true
#   sbatch run_visuals.sh
#
# Environment Variables:
#   RUN_ID (optional)      - Experiment run ID (e.g., "62_102498"). Defaults to latest run.
#   FOLD (optional)        - Which fold to visualize (0-4). Default: 0
#   NUM_FOLDS (optional)   - Total folds in CV (usually 5). Default: 5
#   DATASET (optional)     - Dataset type: "helicodataset", "deephp", or "both". Default: helicodataset
#   MODEL (optional)       - Backbone model: "convnext_tiny" or "convnext_small". 
#                           Auto-detected from training checkpoints if not provided.
#   PIPELINE_MODE (optional) - Generate calibration + dashboard visualizations. Default: true
#   GRADCAM_ONLY (optional) - Generate only Grad-CAM (skip other plots). Default: false
# =====================================================
#SBATCH --job-name=h_pylori_visuals
#SBATCH -D .
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -t 0-02:00
#SBATCH -p pg1tfg12
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH -o results/visuals_output_%j.txt
#SBATCH -e results/visuals_error_%j.txt

set -e  # Exit on error
set +x  # Disable debug output (comment this line to enable)

# Enable unbuffered Python output
export PYTHONUNBUFFERED=1

# ========================================================================
# AUTO-REDIRECT OUTPUT IF RUNNING WITH BASH (NOT SBATCH)
# ========================================================================
if [ -z "$SLURM_JOB_ID" ]; then
    # Running with bash, not sbatch - redirect output to results folder
    mkdir -p results
    TIMESTAMP=$(date +%s)
    OUTPUT_LOG="results/visuals_output_manual_${TIMESTAMP}.txt"
    ERROR_LOG="results/visuals_error_manual_${TIMESTAMP}.txt"
    
    # Initialize log files with header
    {
        echo "========================================================================="
        echo "H. Pylori Visual Report Generation - Execution Log"
        echo "========================================================================="
        echo "Start Time: $(date)"
        echo "Log File: $OUTPUT_LOG"
        echo "========================================================================="
        echo ""
    } | tee "$OUTPUT_LOG" "$ERROR_LOG" > /dev/null
    
    echo "Running with bash. Logs will be saved to:"
    echo "   Output: $OUTPUT_LOG"
    echo "   Errors: $ERROR_LOG"
    
    # Redirect stdout to output log, stderr to both error log and console
    exec 1> >(tee -a "$OUTPUT_LOG")
    exec 2> >(tee -a "$ERROR_LOG" >&2)
else
    # Running with SLURM - SLURM directives handle output redirection
    echo "✓ Running under SLURM job ID: $SLURM_JOB_ID"
    echo "  Output: results/visuals_output_${SLURM_JOB_ID}.txt"
    echo "  Errors: results/visuals_error_${SLURM_JOB_ID}.txt"
fi

# Verify virtual environment before proceeding
if [ -f "./verify_venv.sh" ]; then
    source ./verify_venv.sh
else
    echo "ERROR: verify_venv.sh not found in current directory"
    exit 1
fi

# Set defaults for optional parameters (from environment variables)
RUN_ID=${RUN_ID:-}           # Empty = use latest run
FOLD=${FOLD:-0}
NUM_FOLDS=${NUM_FOLDS:-5}
DATASET=${DATASET:-helicodataset}
# Model will be auto-detected below if not provided
MODEL=${MODEL_NAME:-${MODEL:-}}
PIPELINE_MODE=${PIPELINE_MODE:-true}  # Default to true (full visualizations including calibration + dashboard)
GRADCAM_ONLY=${GRADCAM_ONLY:-false}    # true = Grad-CAM visualizations only

# Function to detect model name from checkpoint files
detect_model_from_checkpoints() {
    local run_id="$1"
    local fold_idx="${2:-0}"
    
    # If run_id is empty, find the latest run
    if [ -z "$run_id" ]; then
        # Get latest model checkpoint across all folds
        local latest_ckpt=$(ls -t results/*_f*_*_model_brain.pth 2>/dev/null | head -1)
        if [ -z "$latest_ckpt" ]; then
            echo "convnext_tiny"  # Fallback default
            return
        fi
        run_id=$(basename "$latest_ckpt" | sed 's/_.*$//')
    fi
    
    # Find checkpoint for this run and fold
    local ckpt_pattern="results/${run_id}_*_*_f${fold_idx}_*_model_brain.pth"
    local ckpt=$(ls $ckpt_pattern 2>/dev/null | head -1)
    
    if [ -z "$ckpt" ]; then
        # Fallback: try any fold for this run
        ckpt_pattern="results/${run_id}_*_*_f*_*_model_brain.pth"
        ckpt=$(ls $ckpt_pattern 2>/dev/null | head -1)
    fi
    
    if [ -n "$ckpt" ]; then
        # Extract model name from filename
        # Format: RUN_ID_xxx_xxx_fX_MODEL_NAME_model_brain.pth
        local basename=$(basename "$ckpt")
        local model_part=$(echo "$basename" | sed 's/_model_brain\.pth$//' | awk -F'_f[0-9]+_' '{print $2}')
        echo "$model_part"
    else
        echo "convnext_tiny"  # Fallback default
    fi
}

# Setup environment explicitly for SLURM jobs
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
export HOME=/home/tkeating

# Activate virtual environment with dependencies
source $VENV_ROOT/bin/activate

# Parse command-line arguments (override environment variables)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --RUN_ID=*|--run_id=*)
            RUN_ID="${1#*=}"
            shift
            ;;
        --FOLD=*|--fold=*)
            FOLD="${1#*=}"
            shift
            ;;
        --NUM_FOLDS=*|--num_folds=*)
            NUM_FOLDS="${1#*=}"
            shift
            ;;
        --DATASET=*|--dataset=*)
            DATASET="${1#*=}"
            shift
            ;;
        --MODEL=*|--model=*|--MODEL_NAME=*|--model_name=*)
            MODEL="${1#*=}"
            shift
            ;;
        --PIPELINE_MODE|--pipeline_mode)
            PIPELINE_MODE="true"
            shift
            ;;
        --GRADCAM_ONLY|--gradcam_only)
            GRADCAM_ONLY="true"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            shift
            ;;
    esac
done

# Create results directory
mkdir -p results

# Auto-detect model name if not explicitly provided
if [ -z "$MODEL" ]; then
    MODEL=$(detect_model_from_checkpoints "$RUN_ID" "$FOLD")
    echo "✓ Auto-detected model: $MODEL"
fi

echo "========================================================================="
echo "H. Pylori Visual Report Generation"
echo "========================================================================="
echo "RUN_ID: ${RUN_ID:-latest}"
echo "FOLD: $FOLD"
echo "NUM_FOLDS: $NUM_FOLDS"
echo "DATASET: $DATASET"
echo "MODEL: $MODEL"
echo "PIPELINE_MODE: $PIPELINE_MODE"
echo "GRADCAM_ONLY: $GRADCAM_ONLY"
echo "========================================================================="
echo ""

# Build python command with unbuffered output
PYTHON_CMD="python3 -u generate_visuals.py"
PYTHON_CMD="$PYTHON_CMD --fold $FOLD"
PYTHON_CMD="$PYTHON_CMD --num_folds $NUM_FOLDS"
PYTHON_CMD="$PYTHON_CMD --dataset $DATASET"
PYTHON_CMD="$PYTHON_CMD --model_name $MODEL"

if [ -n "$RUN_ID" ]; then
    PYTHON_CMD="$PYTHON_CMD --run_id $RUN_ID"
fi

if [ "$GRADCAM_ONLY" = "true" ] || [ "$GRADCAM_ONLY" = "True" ]; then
    PYTHON_CMD="$PYTHON_CMD --gradcam_only"
elif [ "$PIPELINE_MODE" = "true" ] || [ "$PIPELINE_MODE" = "True" ]; then
    PYTHON_CMD="$PYTHON_CMD --pipeline_mode"
fi

echo "Running: $PYTHON_CMD"
echo ""

# Execute visual generation with error handling
if $PYTHON_CMD; then
    EXIT_CODE=$?
    echo ""
    echo "========================================================================="
    echo "✓ Visual report generation complete!"
    echo "  Outputs saved to: results/*_gradcam_samples/"
    echo "  Exit Code: $EXIT_CODE"
    echo "========================================================================="
    echo "End Time: $(date)" >> "${ERROR_LOG:-/dev/null}" 2>&1
else
    EXIT_CODE=$?
    echo ""
    echo "========================================================================="
    echo "✗ Visual report generation failed!"
    echo "  Exit Code: $EXIT_CODE"
    echo "========================================================================="
    echo "" >&2
    echo "=========================================================================" >&2
    echo "ERROR: Visual report generation failed with exit code $EXIT_CODE" >&2
    echo "=========================================================================" >&2
    echo "Check output log for details: ${OUTPUT_LOG:-results/visuals_output_*.txt}" >&2
    echo "End Time: $(date)" >&2
    exit $EXIT_CODE
fi
