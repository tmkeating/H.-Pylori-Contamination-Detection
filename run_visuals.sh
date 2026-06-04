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
#   sbatch run_visuals.sh --RUN_ID=62_102498 --FOLD=0 --PIPELINE_MODE
#   
#   Or via environment variables (sbatch --export):
#   sbatch --export=RUN_ID=62_102498,FOLD=0,PIPELINE_MODE=true run_visuals.sh
#   
#   Or set environment variables then submit:
#   export RUN_ID=62_102498 PIPELINE_MODE=true
#   sbatch run_visuals.sh
#
# Environment Variables:
#   RUN_ID (optional)      - Experiment run ID (e.g., "62_102498"). Defaults to latest run.
#   FOLD (optional)        - Which fold to visualize (0-4). Default: 0
#   NUM_FOLDS (optional)   - Total folds in CV (usually 5). Default: 5
#   DATASET (optional)     - Dataset type: "helicodataset", "deephp", or "both". Default: helicodataset
#   MODEL (optional)       - Backbone model: "convnext_tiny" or "resnet50". Default: convnext_tiny
# =====================================================
#SBATCH --job-name=h_pylori_visuals
#SBATCH -D .
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -t 0-02:00
#SBATCH -p pg1tfg12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o results/visuals_output_%j.txt
#SBATCH -e results/visuals_error_%j.txt

# Set defaults for optional parameters (from environment variables)
RUN_ID=${RUN_ID:-}           # Empty = use latest run
FOLD=${FOLD:-0}
NUM_FOLDS=${NUM_FOLDS:-5}
DATASET=${DATASET:-helicodataset}
MODEL=${MODEL:-convnext_tiny}
PIPELINE_MODE=${PIPELINE_MODE:-false}  # true = calibration curve + dashboard only

# Setup environment explicitly for SLURM jobs
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
export HOME=/home/tkeating

# Activate virtual environment with dependencies
source /home/tkeating/venv/bin/activate

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
        --MODEL=*|--model=*)
            MODEL="${1#*=}"
            shift
            ;;
        --PIPELINE_MODE|--pipeline_mode)
            PIPELINE_MODE="true"
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

echo "========================================================================="
echo "H. Pylori Visual Report Generation"
echo "========================================================================="
echo "RUN_ID: ${RUN_ID:-latest}"
echo "FOLD: $FOLD"
echo "NUM_FOLDS: $NUM_FOLDS"
echo "DATASET: $DATASET"
echo "MODEL: $MODEL"
echo "PIPELINE_MODE: $PIPELINE_MODE"
echo "========================================================================="
echo ""

# Activate environment if available
if [ -f ../venv/bin/activate ]; then
    source ../venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Build python command
PYTHON_CMD="python3 generate_visuals.py"
PYTHON_CMD="$PYTHON_CMD --fold $FOLD"
PYTHON_CMD="$PYTHON_CMD --num_folds $NUM_FOLDS"
PYTHON_CMD="$PYTHON_CMD --dataset $DATASET"
PYTHON_CMD="$PYTHON_CMD --model_name $MODEL"

if [ -n "$RUN_ID" ]; then
    PYTHON_CMD="$PYTHON_CMD --run_id $RUN_ID"
fi

if [ "$PIPELINE_MODE" = "true" ]; then
    PYTHON_CMD="$PYTHON_CMD --pipeline_mode"
fi

echo "Running: $PYTHON_CMD"
echo ""

# Execute visual generation
$PYTHON_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================="
    echo "✓ Visual report generation complete!"
    echo "  Outputs saved to: results/*_gradcam_samples/"
    echo "========================================================================="
else
    echo ""
    echo "========================================================================="
    echo "✗ Visual report generation failed!"
    echo "========================================================================="
    exit 1
fi 
