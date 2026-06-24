#!/bin/bash
# H. Pylori Visual Report Generation - All Folds Sequential
# ===========================================================
# Submits visualization job 5 times (one per fold) with proper SLURM dependency chaining.
# Each fold job waits for the previous fold to complete successfully before starting.
#
# Usage:
#   sbatch submit_visuals_all_folds.sh [RUN_ID] [MODEL] [DATASET] [--ensemble] [--after-job JOB_ID]
#   
#   Examples:
#   sbatch submit_visuals_all_folds.sh                    # Uses latest run, defaults to helicodataset
#   sbatch submit_visuals_all_folds.sh 62_102498          # Specify run ID
#   sbatch submit_visuals_all_folds.sh 62_102498 convnext_small
#   sbatch submit_visuals_all_folds.sh 62_102498 convnext_tiny helicodataset
#   sbatch submit_visuals_all_folds.sh 62_102498 convnext_tiny deephp    # DeepHP visualizations
#   sbatch submit_visuals_all_folds.sh 62_102498 convnext_tiny both      # Both datasets
#   sbatch submit_visuals_all_folds.sh 62_102498 convnext_tiny both --ensemble  # + ensemble Grad-CAM
#   sbatch submit_visuals_all_folds.sh 62_102498 convnext_tiny both --backbone_path results/deephp_backbone_final_01_convnext_tiny_34.4.pth
#   sbatch submit_visuals_all_folds.sh 62_102498 convnext_tiny both --backbone_path results/deephp_backbone_final_01_convnext_tiny_34.4.pth --ensemble
#   sbatch submit_visuals_all_folds.sh 62_102498 convnext_tiny both --after-job 9000
#   sbatch submit_visuals_all_folds.sh 62_102498 convnext_tiny both --job-id-range 9110-9114 --ensemble
#
# Environment Variables (override defaults):
#   RUN_ID       - Experiment run ID (e.g., "62_102498")
#   MODEL        - Backbone model (convnext_tiny, convnext_small). Default: convnext_tiny
#   DATASET      - Dataset type (helicodataset, deephp, both). Default: helicodataset
#   AFTER_JOB_ID - Job ID to depend on before starting fold 0. Default: none (folds start immediately)
#   BACKBONE_PATH - Full path to ensemble backbone checkpoint. If provided, only fold 0 is submitted
#   JOB_ID_RANGE - SLURM job ID range for ensemble Grad-CAM (e.g., "9110-9114"). Overrides standard dependency chain.
#   GENERATE_ENSEMBLE_GRADCAM - If "true", generate Grad-CAMs for ensemble weighted backbone. Default: false
#   DRY_RUN      - If set to "true", prints commands without submitting. Default: false

# SBATCH Directives - Minimal resources for job submission wrapper
#SBATCH --job-name=h_pylori_visuals_orchestrator
#SBATCH -D .
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-00:10
#SBATCH -p pg1tfg12
#SBATCH --mem=1G
#SBATCH -o results/visuals_orchestrator_output_%j.txt
#SBATCH -e results/visuals_orchestrator_error_%j.txt

set -e

# =====================================================
# Configuration
# =====================================================
RUN_ID="${1:-${RUN_ID}}"              # First positional arg or env var
MODEL="${2:-${MODEL:-convnext_tiny}}"  # Second positional arg or env var, default: convnext_tiny
DATASET="${3:-${DATASET:-helicodataset}}"  # Third positional arg or env var, default: helicodataset
DRY_RUN="${DRY_RUN:-false}"            # If true, just print commands
AFTER_JOB_ID="${AFTER_JOB_ID:-}"       # Job ID to depend on before starting fold 0
BACKBONE_PATH="${BACKBONE_PATH:-}"     # Full path to ensemble backbone file
JOB_ID_RANGE="${JOB_ID_RANGE:-}"       # SLURM job ID range (e.g., "9110-9114") for ensemble Grad-CAM dependency
DEPEND_STR=""                          # Will be set if JOB_ID_RANGE is provided

# Check for --ensemble, --after-job, --backbone_path, and --job-id-range flags in remaining arguments
GENERATE_ENSEMBLE_GRADCAM="${GENERATE_ENSEMBLE_GRADCAM:-false}"
for arg in "$@"; do
    if [ "$arg" = "--ensemble" ]; then
        GENERATE_ENSEMBLE_GRADCAM="true"
    elif [ "$arg" = "--after-job" ]; then
        NEXT_IS_JOB_ID=true
    elif [ "$NEXT_IS_JOB_ID" = "true" ]; then
        AFTER_JOB_ID="$arg"
        NEXT_IS_JOB_ID=false
    elif [ "$arg" = "--backbone_path" ] || [ "$arg" = "--backbone-path" ]; then
        NEXT_IS_BACKBONE=true
    elif [ "$NEXT_IS_BACKBONE" = "true" ]; then
        BACKBONE_PATH="$arg"
        NEXT_IS_BACKBONE=false
    elif [ "$arg" = "--job-id-range" ]; then
        NEXT_IS_RANGE=true
    elif [ "$NEXT_IS_RANGE" = "true" ]; then
        JOB_ID_RANGE="$arg"
        NEXT_IS_RANGE=false
    fi
done

NUM_FOLDS=5

# =====================================================
# Helper Functions
# =====================================================

print_header() {
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
}

log_info() {
    echo "[INFO] $1" >&2
}

log_error() {
    echo "[ERROR] $1" >&2
}

log_success() {
    echo "[✓] $1" >&2
}

submit_fold_job() {
    local fold_idx=$1
    local prev_job_id=$2
    
    # Build sbatch command
    local cmd="sbatch -p pg1tfg12"
    
    # Add dependency
    if [ $fold_idx -eq 0 ] && [ -n "$AFTER_JOB_ID" ]; then
        # First fold depends on AFTER_JOB_ID if specified
        cmd="$cmd --dependency=afterok:$AFTER_JOB_ID"
    elif [ -n "$prev_job_id" ] && [ "$prev_job_id" != "" ]; then
        # Subsequent folds depend on previous fold
        cmd="$cmd --dependency=afterok:$prev_job_id"
    fi
    
    # Add job name with fold info
    cmd="$cmd --job-name=h_pylori_visuals_fold${fold_idx}"
    
    # Add script and all arguments
    cmd="$cmd submit_visuals.sh"
    cmd="$cmd --run_id $RUN_ID"
    cmd="$cmd --fold $fold_idx"
    cmd="$cmd --model_name $MODEL"
    cmd="$cmd --dataset $DATASET"
    
    # Add backbone path if provided
    if [ -n "$BACKBONE_PATH" ]; then
        cmd="$cmd --backbone_path $BACKBONE_PATH"
    fi
    
    log_info "Submitting fold $fold_idx..."
    if [ $fold_idx -eq 0 ] && [ -n "$AFTER_JOB_ID" ]; then
        log_info "  (depends on job: $AFTER_JOB_ID)"
    elif [ "$fold_idx" != "0" ]; then
        log_info "  (depends on job: $prev_job_id)"
    fi
    log_info "  Command: $cmd"
    
    if [ "$DRY_RUN" = "true" ]; then
        echo "DRY_RUN_FOLD_${fold_idx}"
        return 0
    fi
    
    # Execute and capture job ID
    local output
    output=$(eval "$cmd" 2>&1)
    
    if [ $? -eq 0 ]; then
        # Extract job ID from sbatch output
        local job_id=$(echo "$output" | grep -oP 'Submitted batch job \K[0-9]+' || true)
        
        if [ -z "$job_id" ]; then
            log_error "Could not extract job ID from output: $output"
            return 1
        fi
        
        log_success "Fold $fold_idx submitted successfully (Job ID: $job_id)"
        echo "$job_id"
    else
        log_error "Failed to submit fold $fold_idx"
        log_error "Output: $output"
        return 1
    fi
}

submit_ensemble_gradcam_job() {
    local prev_job_id=$1
    local depend_str=$2  # Optional: custom dependency string (e.g., "afterok:9110:9111:9112:9113:9114")
    
    # Extract RUN_ID components for iteration name
    local run_base=$(echo "$RUN_ID" | cut -d'_' -f1,2)
    local iter_name=$(echo "$RUN_ID" | cut -d'_' -f3-)
    
    log_info "Submitting ensemble Grad-CAM generation..."
    if [ -n "$depend_str" ]; then
        log_info "  (depends on job range: $depend_str)"
    else
        log_info "  (depends on job: $prev_job_id)"
    fi
    
    # Build sbatch command
    local cmd="sbatch -p pg1tfg12"
    if [ -n "$depend_str" ]; then
        cmd="$cmd --dependency=$depend_str"
    else
        cmd="$cmd --dependency=afterok:$prev_job_id"
    fi
    cmd="$cmd --job-name=h_pylori_ensemble_gradcam"
    cmd="$cmd -n 1 -N 1 -t 0-02:00 --mem=16G"
    cmd="$cmd --wrap='python3 generate_deephp_gradcam.py --run $RUN_ID --fold 0-4 --model $MODEL'"
    
    log_info "  Command: $cmd"
    
    if [ "$DRY_RUN" = "true" ]; then
        echo "DRY_RUN_ENSEMBLE"
        return 0
    fi
    
    # Execute and capture job ID
    local output
    output=$(eval "$cmd" 2>&1)
    
    if [ $? -eq 0 ]; then
        # Extract job ID from sbatch output
        local job_id=$(echo "$output" | grep -oP 'Submitted batch job \K[0-9]+' || true)
        
        if [ -z "$job_id" ]; then
            log_error "Could not extract job ID from output: $output"
            return 1
        fi
        
        log_success "Ensemble Grad-CAM job submitted successfully (Job ID: $job_id)"
        echo "$job_id"
    else
        log_error "Failed to submit ensemble Grad-CAM job"
        log_error "Output: $output"
        return 1
    fi
}

# =====================================================
# Main Script
# =====================================================

print_header "H. Pylori Visualization Generation - All Folds"

# Print configuration
echo "Configuration:"
echo "  RUN_ID:                     $RUN_ID"
echo "  MODEL:                      $MODEL"
echo "  DATASET:                    $DATASET"
if [ -n "$BACKBONE_PATH" ]; then
    echo "  BACKBONE_PATH:              $BACKBONE_PATH"
    echo "  NUM_FOLDS:                  1 (fold 0 only, using ensemble backbone)"
else
    echo "  NUM_FOLDS:                  $NUM_FOLDS"
fi
if [ -n "$AFTER_JOB_ID" ]; then
    echo "  AFTER_JOB_ID:               $AFTER_JOB_ID"
fi
if [ -n "$JOB_ID_RANGE" ]; then
    echo "  JOB_ID_RANGE:               $JOB_ID_RANGE"
fi
echo "  GENERATE_ENSEMBLE_GRADCAM:  $GENERATE_ENSEMBLE_GRADCAM"
echo "  DRY_RUN:                    $DRY_RUN"
echo ""

# Validate script exists
if [ ! -f "submit_visuals.sh" ]; then
    log_error "submit_visuals.sh not found in current directory"
    exit 1
fi

# Submit all folds with dependencies (or just fold 0 if backbone_path provided)
prev_job_id=""
declare -a submitted_jobs

# Determine which folds to process
if [ -n "$BACKBONE_PATH" ]; then
    # Ensemble backbone mode: only fold 0
    FOLDS_TO_PROCESS="0"
else
    # Fold-specific mode: all folds
    FOLDS_TO_PROCESS=$(seq 0 $((NUM_FOLDS - 1)))
fi

for fold in $FOLDS_TO_PROCESS; do
    job_id=$(submit_fold_job $fold "$prev_job_id")
    
    if [ $? -ne 0 ]; then
        log_error "Failed to submit fold $fold. Aborting."
        exit 1
    fi
    
    submitted_jobs+=("$job_id")
    prev_job_id="$job_id"
    
    # Small delay between submissions to avoid race conditions
    if [ "$fold" != "$(echo $FOLDS_TO_PROCESS | awk '{print $NF}')" ]; then
        sleep 1
    fi
done

# Handle job ID range for ensemble Grad-CAM (if provided)
if [ -n "$JOB_ID_RANGE" ]; then
    # Convert range to dependency format: "9110-9114" -> "afterok:9110:9111:9112:9113:9114"
    START=$(echo "$JOB_ID_RANGE" | cut -d'-' -f1)
    END=$(echo "$JOB_ID_RANGE" | cut -d'-' -f2)
    
    # Build dependency list
    DEPEND_STR="afterok"
    for job in $(seq $START $END); do
        DEPEND_STR="${DEPEND_STR}:${job}"
    done
fi

# Submit ensemble Grad-CAM job if requested
ensemble_job_id=""
if [ "$GENERATE_ENSEMBLE_GRADCAM" = "true" ]; then
    # Pass DEPEND_STR if using job ID range, otherwise use prev_job_id
    if [ -n "$DEPEND_STR" ]; then
        ensemble_job_id=$(submit_ensemble_gradcam_job "" "$DEPEND_STR")
    else
        ensemble_job_id=$(submit_ensemble_gradcam_job "$prev_job_id" "")
    fi
    
    if [ $? -ne 0 ]; then
        log_error "Failed to submit ensemble Grad-CAM job"
        exit 1
    fi
fi

# =====================================================
# Summary
# =====================================================

print_header "Submission Summary"

if [ -n "$BACKBONE_PATH" ]; then
    echo "Submitted fold 0 with ensemble backbone:"
    echo "  Backbone: $(basename $BACKBONE_PATH)"
else
    echo "Successfully submitted all folds with dependency chain:"
fi
echo ""

for i in "${!submitted_jobs[@]}"; do
    job_id=${submitted_jobs[$i]}
    
    if [ $i -eq 0 ]; then
        if [ -n "$AFTER_JOB_ID" ]; then
            echo "  Fold $i: Job $job_id (depends on $AFTER_JOB_ID)"
        else
            echo "  Fold $i: Job $job_id (starts immediately)"
        fi
    else
        prev_id=${submitted_jobs[$((i-1))]}
        echo "  Fold $i: Job $job_id (depends on $prev_id)"
    fi
done

if [ -n "$ensemble_job_id" ] && [ "$ensemble_job_id" != "" ]; then
    echo ""
    echo "Ensemble Grad-CAM Generation:"
    if [ -n "$DEPEND_STR" ]; then
        echo "  Job: $ensemble_job_id (depends on job range: $DEPEND_STR)"
    else
        echo "  Job: $ensemble_job_id (depends on Fold $((NUM_FOLDS-1)) Job ${submitted_jobs[-1]})"
    fi
fi

echo ""
echo "Check status with:"
echo "  squeue -j ${submitted_jobs[0]}  # Show all fold jobs"
if [ -n "$ensemble_job_id" ] && [ "$ensemble_job_id" != "" ]; then
    echo "  squeue -j $ensemble_job_id        # Show ensemble Grad-CAM job"
fi
echo ""
echo "View individual fold results in:"
echo "  results/{run}_{fold}_*_evaluation_report.csv"
echo "  results/{run}_{fold}_*_learning_curves.json"
if [ -n "$ensemble_job_id" ] && [ "$ensemble_job_id" != "" ]; then
    echo ""
    echo "View ensemble Grad-CAM results in:"
    echo "  results/{run}_f{fold}_{model}_gradcam/"
fi
echo ""

log_success "All fold jobs submitted with dependency chain!"
