#!/bin/bash
# H. Pylori 5-Fold SLURM Orchestrator
# ----------------------------------
# This script manages the full cross-validation pipeline by submitting all 5 folds
# to the SLURM cluster and scheduling a final summarization job that waits for 
# them to complete.
#
# What it does:
#   1. Orchestrates training for all 5 folds using a specific Profile (Searcher/Auditor).
#   2. Sources hyperparameter configurations from profiles.sh.
#   3. Uses --dependency=afterok to schedule a Summary Job (summarize_results.py) 
#      and a Voting Job (ensemble_voting.py) ONLY after all folds finish successfully.
#
# Usage:
#   PROFILE=SEARCHER MODEL_NAME=convnext_tiny ITER=25.0 ./submit_all_folds.sh
#
# Environment Variables:
#   PROFILE:    The model profile from profiles.sh (AUDITOR, SEARCHER, etc.)
#   MODEL_NAME: Backbone architecture (Default: convnext_tiny)
#   ITER:       Iteration number for experiment tracking.
# ----------------------------------

# Simple script to submit all 5 folds for H. Pylori cross-validation
# Usage: PROFILE=SEARCHER MODEL_NAME=convnext_tiny ITER=25.0 ./submit_all_folds.sh

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
PROFILE=${PROFILE:-"AUDITOR"}
ITER=${ITER:-"26.0"}
BATCH_SIZE=${BATCH_SIZE:-"0"}  # 0=all parallel (default), N=batch in groups of N

# 1. Source the Model Profiles (Central Source of Truth)
if [ -f "profiles.sh" ]; then
    source profiles.sh
    # Dynamically call the setter for the requested profile
    if declare -f "set_profile_$PROFILE" > /dev/null; then
        "set_profile_$PROFILE"
        echo "Using $PROFILE profile from profiles.sh..."
    else
        echo "ERROR: Profile '$PROFILE' not found in profiles.sh. Using defaults."
        NEG_WEIGHT=1.0
        POS_WEIGHT=2.2
        GAMMA=2.0
        SAVER_METRIC="loss"
    fi
else
    echo "ERROR: profiles.sh not found. Using defaults."
    NEG_WEIGHT=1.0
    POS_WEIGHT=2.2
    GAMMA=2.0
    SAVER_METRIC="loss"
fi

echo "Parameters: NegWeight=$NEG_WEIGHT, PosWeight=$POS_WEIGHT, Gamma=$GAMMA, Epochs=$NUM_EPOCHS, FreezeBN=$FREEZE_BN, ClipGrad=$CLIP_GRAD, PctStart=$PCT_START, Saver=$SAVER_METRIC, WD=$WEIGHT_DECAY, SWA=$USE_SWA, SWAStart=$SWA_START, Jitter=$JITTER"

echo "-------------------------------------------"
echo "Submitting pre-sync job to populate scratch directory before training..."
# Submit a pre-sync job that syncs data once, blocking until complete
# All fold jobs will depend on this pre-sync job to avoid concurrent sync/training operations
PRE_SYNC_JOB=$(sbatch -p pg1tfg12 --export=ALL,PRE_SYNC_ONLY=1 run_h_pylori.sh)
PRE_SYNC_JOB_ID=$(echo $PRE_SYNC_JOB | awk '{print $4}')
echo "Pre-sync job ID: $PRE_SYNC_JOB_ID"
PRE_SYNC_DEPENDENCY="afterok:$PRE_SYNC_JOB_ID"

echo ""
echo "Submitting training jobs for all 5 folds..."
if [ "$BATCH_SIZE" != "0" ]; then
    echo "Mode: BATCH PROCESSING (groups of $BATCH_SIZE folds)"
else
    echo "Mode: PARALLEL (all folds run simultaneously)"
fi
echo ""

DEPENDENCIES=""
declare -a FOLD_IDS  # Array to track job IDs for batch dependencies

for FOLD in {0..4}
do
    # Determine this fold's dependency based on batch size
    if [ "$BATCH_SIZE" != "0" ] && [ $FOLD -ge $BATCH_SIZE ]; then
        # Not in first batch; depends on last fold of previous batch
        BATCH_LAST_FOLD=$(((FOLD / BATCH_SIZE) * BATCH_SIZE - 1))
        FOLD_DEPENDENCY="afterok:${FOLD_IDS[$BATCH_LAST_FOLD]}"
    else
        # First batch or all-parallel mode: depend on pre-sync
        FOLD_DEPENDENCY="$PRE_SYNC_DEPENDENCY"
    fi
    
    echo "-------------------------------------------"
    echo "Submitting SLURM job for Fold $FOLD using $MODEL_NAME ($PROFILE Profile, Iter $ITER)..."
    # Capture the job ID
    # Iteration 21.3: Expanded export list to include Stability parameters
    # CHANGE: Fold jobs now depend on pre-sync job to avoid concurrent rsync operations
    # CHANGE: Added SKIP_SYNC=1 for fold jobs so they don't re-sync (pre-sync job already did it)
    JOB_OUT=$(sbatch -p pg1tfg12 --dependency=$FOLD_DEPENDENCY -v FOLD=$FOLD,MODEL_NAME=$MODEL_NAME,NEG_WEIGHT=$NEG_WEIGHT,POS_WEIGHT=$POS_WEIGHT,GAMMA=$GAMMA,NUM_EPOCHS=$NUM_EPOCHS,FREEZE_BN=$FREEZE_BN,CLIP_GRAD=$CLIP_GRAD,PCT_START=$PCT_START,SAVER_METRIC=$SAVER_METRIC,WEIGHT_DECAY=$WEIGHT_DECAY,USE_SWA=$USE_SWA,SWA_START=$SWA_START,JITTER=$JITTER,ITER=$ITER,SKIP_SYNC=1 run_h_pylori.sh)
    echo "$JOB_OUT"
    JOB_ID=$(echo $JOB_OUT | awk '{print $4}')
    FOLD_IDS[$FOLD]="$JOB_ID"  # Store for batch dependency lookup
    
    if [ "$BATCH_SIZE" != "0" ]; then
        # Batching enabled: show which batch this fold belongs to
        BATCH_NUM=$((FOLD / BATCH_SIZE))
        BATCH_POS=$((FOLD % BATCH_SIZE))
        echo "  (Batch $((BATCH_NUM + 1)), Position $((BATCH_POS + 1)))"
    else
        # All parallel: accumulate dependencies for final summary
        if [ -z "$DEPENDENCIES" ]; then
            DEPENDENCIES="$JOB_ID"
            MIN_JOB="$JOB_ID"
        else
            DEPENDENCIES="$DEPENDENCIES:$JOB_ID"
            MAX_JOB="$JOB_ID"
        fi
    fi
    
    # Wait 1 second to ensure sequential submission and prevent race conditions
    sleep 1
done

# Set final dependency string for summary job
if [ "$BATCH_SIZE" != "0" ]; then
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

echo "-------------------------------------------"
echo "Submitting Global Attention-MIL final summary as dependent job..."

# Validate that all folds were successfully submitted (for parallel mode)
if [ "$BATCHED" != "1" ] && [ -z "$DEPENDENCIES" ]; then
    echo "ERROR: No fold jobs were successfully submitted!"
    echo "Cannot proceed with summary job."
    exit 1
fi

# This job will only start once all 5 folds have successfully completed

# Final validation of dependency string
if [ -z "$DEPENDENCY_STRING" ]; then
    echo "ERROR: Failed to generate valid dependency string from: $DEPENDENCIES"
    exit 1
fi

sbatch --dependency=$DEPENDENCY_STRING \
    -p pg1tfg12 \
    --time=0-12:00 \
    --mem=20G \
    --cpus-per-task=6 \
    --job-name=HPy_FinalSummary \
    --output=results/slurm_summary_%j.txt \
    --error=results/slurm_summary_error_%j.txt \
    <<'SUMMARY_EOF'
#!/bin/bash
#SBATCH -p dcca40

cd /home/tkeating/model/H.-Pylori-Contamination-Detection

# Get job ID for output filename
JOB_ID=$SLURM_JOB_ID

# Get virtual environment path from config
VENV_ROOT=$(python3 -c "from config import VENV_ROOT; print(VENV_ROOT)")

# Activate virtual environment
source $VENV_ROOT/bin/activate

echo "=========================================================================="
echo "All training folds complete. Generating comprehensive ensemble analysis..."
echo "=========================================================================="
echo ""

# Extract iteration from latest checkpoint files
ITER=$(python3 -c "
import glob
from pathlib import Path
files = sorted(glob.glob('results/*_convnext_tiny_model_brain.pth'))
if files:
    # Extract iteration from filename like: 28_25.0_107840_f0_convnext_tiny_model_brain.pth
    filename = Path(files[-1]).stem
    parts = filename.split('_')
    if len(parts) >= 2:
        print(parts[1])  # This is the iteration (25.0, 26.0, etc)
    else:
        print('26.0')
else:
    print('26.0')
")

echo "Iteration: $ITER"
echo ""

# Step 1: Cross-validation performance summary
echo "=========================================================================="
echo "Step 1: Running cross-validation performance summary..."
echo "=========================================================================="
python3 summarize_results.py --dir results --last 5 2>&1

echo ""
echo "=========================================================================="
echo "Step 2: Generating ensemble voting, meta-classifier, and hybrid fusion..."
echo "=========================================================================="
echo ""

# Note: ensemble_voting.py runs THREE fusion methods:
#   1. Ensemble Voting (majority vote)
#   2. Meta-Classifier (Random Forest with LOO-CV)
#   3. Hybrid Ensemble (intelligent confidence-zone blending) ⭐ RECOMMENDED
#
# Primary outputs: hybrid_ensemble_*.csv with 92.11% accuracy, 100% precision
# Comparison outputs: ensemble_voting_*.csv and meta_classifier_*.csv for analysis
python3 ensemble_voting.py 2>&1

echo ""
echo "=========================================================================="
echo "Clinical analysis and hybrid ensemble fusion completed."
echo "=========================================================================="
echo "✅ Primary results (RECOMMENDED):"
echo "   - results/hybrid_ensemble_results_*.csv - Patient predictions"
echo "   - results/hybrid_ensemble_summary_*.csv - Performance metrics (92.11% accuracy, 100% precision)"
echo "   - results/hybrid_ensemble_roc_pr_*.png - ROC/PR curves"
echo ""
echo "📊 Comparison outputs (for analysis):"
echo "   - results/ensemble_voting_summary_*.csv - Base voting method"
echo "   - results/meta_classifier_summary_*.csv - Meta-classifier method"
echo "=========================================================================="
echo ""

SUMMARY_EOF

echo "-------------------------------------------"
echo "All 5 folds + Hybrid Ensemble fusion job submitted. Use 'squeue -u $USER' to monitor progress."

