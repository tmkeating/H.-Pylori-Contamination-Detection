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
#   FOLD (optional)        - Which fold to visualize (0-4). Default: 0. Ignored if USE_ENSEMBLE_BACKBONE=true
#   NUM_FOLDS (optional)   - Total folds in CV (usually 5). Default: 5
#   DATASET (optional)     - Dataset type: "helicodataset", "deephp", or "both". Default: helicodataset
#   MODEL (optional)       - Backbone model: "convnext_tiny" or "convnext_small". 
#                           Auto-detected from training checkpoints if not provided.
#   USE_ENSEMBLE_BACKBONE (optional) - Use ensemble weighted backbone instead of fold-specific. Default: false
#   BACKBONE_PATH (optional)   - Full path to ensemble backbone file (e.g., results/deephp_backbone_final_01_convnext_tiny_34.4.pth)
#                           If provided, overrides fold-specific model loading. Must exist or job will fail.
#   ITERATION (optional)   - DEPRECATED: Use --backbone_path instead. Legacy iteration parameter.
#   PIPELINE_MODE (optional) - Generate calibration + dashboard visualizations. Default: true
#   GRADCAM_ONLY (optional) - Generate only Grad-CAM (skip other plots). Default: false
# =====================================================
#SBATCH --job-name=h_pylori_visuals
#SBATCH -D .
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -t 0-06:00
#SBATCH -p pg1tfg12
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:l40s:12000
#SBATCH -o results/visuals_output_%j.txt
#SBATCH -e results/visuals_error_%j.txt
# Note: Output directory can be overridden via --output_dir parameter

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
OUTPUT_DIR=${OUTPUT_DIR:-results}       # Output directory for all visualizations
RUN_ID=${RUN_ID:-}           # Empty = use latest run
FOLD=${FOLD:-0}
NUM_FOLDS=${NUM_FOLDS:-5}
DATASET=${DATASET:-helicodataset}
# Model will be auto-detected below if not provided
MODEL=${MODEL_NAME:-${MODEL:-}}
USE_ENSEMBLE_BACKBONE=${USE_ENSEMBLE_BACKBONE:-false}
ITERATION=${ITERATION:-}     # Required if USE_ENSEMBLE_BACKBONE=true
BACKBONE_PATH=${BACKBONE_PATH:-}  # Direct path to ensemble backbone (bypasses parsing)
PIPELINE_MODE=${PIPELINE_MODE:-true}  # Default to true (full visualizations including calibration + dashboard)
GRADCAM_ONLY=${GRADCAM_ONLY:-false}    # true = Grad-CAM visualizations only
GENERATE_ENSEMBLE_GRADCAM=${GENERATE_ENSEMBLE_GRADCAM:-false}  # Generate ensemble Grad-CAM after fold completes
DRY_RUN=${DRY_RUN:-false}    # If true, just print the command without executing

# Function to detect model name from checkpoint files
detect_model_from_checkpoints() {
    local run_id="$1"
    local fold_idx="${2:-0}"
    local output_dir="${3:-results}"
    
    # If run_id is empty, find the latest run
    if [ -z "$run_id" ]; then
        # Get latest model checkpoint across all folds
        local latest_ckpt=$(ls -t $output_dir/*_f*_*_model_brain.pth 2>/dev/null | head -1)
        if [ -z "$latest_ckpt" ]; then
            echo "convnext_tiny"  # Fallback default
            return
        fi
        run_id=$(basename "$latest_ckpt" | sed 's/_.*$//')
    fi
    
    # Find checkpoint for this run and fold
    local ckpt_pattern="$output_dir/${run_id}_*_*_f${fold_idx}_*_model_brain.pth"
    local ckpt=$(ls $ckpt_pattern 2>/dev/null | head -1)
    
    if [ -z "$ckpt" ]; then
        # Fallback: try any fold for this run
        ckpt_pattern="$output_dir/${run_id}_*_*_f*_*_model_brain.pth"
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
        --OUTPUT_DIR=*|--output_dir=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --OUTPUT_DIR|--output_dir)
            shift
            OUTPUT_DIR="$1"
            shift
            ;;
        --RUN_ID=*|--run_id=*)
            RUN_ID="${1#*=}"
            shift
            ;;
        --RUN_ID|--run_id)
            shift
            RUN_ID="$1"
            shift
            ;;
        --FOLD=*|--fold=*)
            FOLD="${1#*=}"
            shift
            ;;
        --FOLD|--fold)
            shift
            FOLD="$1"
            shift
            ;;
        --NUM_FOLDS=*|--num_folds=*)
            NUM_FOLDS="${1#*=}"
            shift
            ;;
        --NUM_FOLDS|--num_folds)
            shift
            NUM_FOLDS="$1"
            shift
            ;;
        --DATASET=*|--dataset=*)
            DATASET="${1#*=}"
            shift
            ;;
        --DATASET|--dataset)
            shift
            DATASET="$1"
            shift
            ;;
        --MODEL=*|--model=*|--MODEL_NAME=*|--model_name=*)
            MODEL="${1#*=}"
            shift
            ;;
        --MODEL|--model|--MODEL_NAME|--model_name)
            shift
            MODEL="$1"
            shift
            ;;
        --USE_ENSEMBLE_BACKBONE|--use_ensemble_backbone)
            USE_ENSEMBLE_BACKBONE="true"
            shift
            ;;
        --ITERATION=*|--iteration=*)
            ITERATION="${1#*=}"
            shift
            ;;
        --ITERATION|--iteration)
            shift
            ITERATION="$1"
            shift
            ;;
        --BACKBONE_PATH=*|--backbone_path=*)
            BACKBONE_PATH="${1#*=}"
            shift
            ;;
        --BACKBONE_PATH|--backbone_path)
            shift
            BACKBONE_PATH="$1"
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
        --ENSEMBLE|--ensemble)
            GENERATE_ENSEMBLE_GRADCAM="true"
            shift
            ;;
        --DRY_RUN|--dry_run|--dry-run)
            DRY_RUN="true"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            shift
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Auto-detect dataset from backbone file if provided
if [ -n "$BACKBONE_PATH" ] && grep -q "deephp" <<< "$BACKBONE_PATH"; then
    if [ "$DATASET" = "helicodataset" ]; then
        DATASET="deephp"
        echo "✓ Auto-detected dataset from backbone file: $DATASET"
    fi
fi

# Auto-detect model name if not explicitly provided
if [ -z "$MODEL" ]; then
    MODEL=$(detect_model_from_checkpoints "$RUN_ID" "$FOLD" "$OUTPUT_DIR")
    echo "✓ Auto-detected model: $MODEL"
fi

echo "========================================================================="
echo "H. Pylori Visual Report Generation"
echo "========================================================================="echo "OUTPUT_DIR: $OUTPUT_DIR"echo "RUN_ID: ${RUN_ID:-latest}"
if [ "$USE_ENSEMBLE_BACKBONE" = "true" ]; then
    echo "BACKBONE: Ensemble Weighted (Iteration: $ITERATION)"
else
    echo "FOLD: $FOLD"
fi
echo "NUM_FOLDS: $NUM_FOLDS"
echo "DATASET: $DATASET"
echo "MODEL: $MODEL"
echo "PIPELINE_MODE: $PIPELINE_MODE"
echo "GRADCAM_ONLY: $GRADCAM_ONLY"
echo "GENERATE_ENSEMBLE_GRADCAM: $GENERATE_ENSEMBLE_GRADCAM"
echo "========================================================================="
echo ""

# Build python command with unbuffered output
PYTHON_CMD="python3 -u generate_visuals.py"

if [ "$USE_ENSEMBLE_BACKBONE" = "true" ] || [ -n "$BACKBONE_PATH" ]; then
    # Ensemble weighted backbone mode
    if [ -z "$BACKBONE_PATH" ]; then
        echo "ERROR: BACKBONE_PATH required when USE_ENSEMBLE_BACKBONE=true"
        echo "Example: sbatch submit_visuals.sh --backbone_path results/deephp_backbone_final_01_convnext_tiny_34.4.pth"
        exit 1
    fi
    if [ ! -f "$BACKBONE_PATH" ]; then
        echo "ERROR: Ensemble backbone not found: $BACKBONE_PATH"
        exit 1
    fi
    PYTHON_CMD="$PYTHON_CMD --backbone_path $BACKBONE_PATH"
    PYTHON_CMD="$PYTHON_CMD --fold 0"  # Use fold 0 for validation data
else
    # Standard fold-specific backbone mode
    PYTHON_CMD="$PYTHON_CMD --fold $FOLD"
fi

PYTHON_CMD="$PYTHON_CMD --num_folds $NUM_FOLDS"
PYTHON_CMD="$PYTHON_CMD --dataset $DATASET"
PYTHON_CMD="$PYTHON_CMD --model_name $MODEL"
PYTHON_CMD="$PYTHON_CMD --output_dir $OUTPUT_DIR"

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

# Execute visual generation with error handling (or dry run)
if [ "$DRY_RUN" = "true" ] || [ "$DRY_RUN" = "True" ]; then
    echo "[DRY_RUN] Command would execute as:"
    echo "  $PYTHON_CMD"
    echo ""
    echo "To execute this command, run without --dry_run flag or set DRY_RUN=false"
    echo ""
    exit 0
fi

if $PYTHON_CMD; then
    EXIT_CODE=$?
    echo ""
    echo "========================================================================="
    echo "✓ Visual report generation complete!"
    echo "  Outputs saved to: $OUTPUT_DIR/*_gradcam_samples/"
    echo "  Exit Code: $EXIT_CODE"
    echo "========================================================================="
    
    # Submit ensemble Grad-CAM job if requested
    if [ "$GENERATE_ENSEMBLE_GRADCAM" = "true" ]; then
        echo ""
        echo "Submitting ensemble Grad-CAM generation job..."
        
        ENSEMBLE_CMD="sbatch -p pg1tfg12"
        ENSEMBLE_CMD="$ENSEMBLE_CMD --job-name=h_pylori_ensemble_gradcam"
        ENSEMBLE_CMD="$ENSEMBLE_CMD -n 1 -N 1 -t 0-04:00 --mem=32G --gres=gpu:1"
        ENSEMBLE_CMD="$ENSEMBLE_CMD -o $OUTPUT_DIR/ensemble_gradcam_%j.out"
        ENSEMBLE_CMD="$ENSEMBLE_CMD -e $OUTPUT_DIR/ensemble_gradcam_%j.err"
        
        # Build Grad-CAM command with output directory
        if [ -n "$BACKBONE_PATH" ]; then
            ENSEMBLE_CMD="$ENSEMBLE_CMD --wrap='python3 generate_deephp_gradcam.py --output_dir $OUTPUT_DIR --backbone_path $BACKBONE_PATH --fold 0-4 --model $MODEL'"
        else
            ENSEMBLE_CMD="$ENSEMBLE_CMD --wrap='python3 generate_deephp_gradcam.py --output_dir $OUTPUT_DIR --run ${RUN_ID:-latest} --fold 0-4 --model $MODEL'"
        fi
        
        # If running under SLURM, add dependency on current job
        if [ -n "$SLURM_JOB_ID" ]; then
            ENSEMBLE_CMD="$ENSEMBLE_CMD --dependency=afterok:$SLURM_JOB_ID"
        fi
        
        ENSEMBLE_OUTPUT=$(eval $ENSEMBLE_CMD 2>&1)
        ENSEMBLE_EXIT=$?
        
        if [ $ENSEMBLE_EXIT -eq 0 ]; then
            echo "$ENSEMBLE_OUTPUT"
            echo "✓ Ensemble Grad-CAM job submitted successfully"
        else
            echo "✗ Failed to submit ensemble Grad-CAM job"
            echo "Error output: $ENSEMBLE_OUTPUT"
        fi
    fi
    
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
    echo "Check output log for details: ${OUTPUT_LOG:-$OUTPUT_DIR/visuals_output_*.txt}" >&2
    echo "End Time: $(date)" >&2
    exit $EXIT_CODE
fi
