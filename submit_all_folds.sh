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

MODEL_NAME=${MODEL_NAME:-"convnext_tiny"}
PROFILE=${PROFILE:-"AUDITOR"}
ITER=${ITER:-"26.0"}

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

for FOLD in {0..4}
do
    echo "-------------------------------------------"
    echo "Submitting SLURM job for Fold $FOLD using $MODEL_NAME ($PROFILE Profile, Iter $ITER)..."
    # Capture the job ID
    # Iteration 21.3: Expanded export list to include Stability parameters
    JOB_OUT=$(sbatch -p dcca40 --export=ALL,FOLD=$FOLD,MODEL_NAME=$MODEL_NAME,NEG_WEIGHT=$NEG_WEIGHT,POS_WEIGHT=$POS_WEIGHT,GAMMA=$GAMMA,NUM_EPOCHS=$NUM_EPOCHS,FREEZE_BN=$FREEZE_BN,CLIP_GRAD=$CLIP_GRAD,PCT_START=$PCT_START,SAVER_METRIC=$SAVER_METRIC,WEIGHT_DECAY=$WEIGHT_DECAY,USE_SWA=$USE_SWA,SWA_START=$SWA_START,JITTER=$JITTER,ITER=$ITER run_h_pylori.sh)
    echo "$JOB_OUT"
    JOB_ID=$(echo $JOB_OUT | awk '{print $4}')
    
    # Add to dependency list
    if [ -z "$DEPENDENCIES" ]; then
        DEPENDENCIES="$JOB_ID"
        MIN_JOB="$JOB_ID"
    else
        DEPENDENCIES="$DEPENDENCIES:$JOB_ID"
        MAX_JOB="$JOB_ID"
    fi
    
    # Wait 1 second to ensure sequential submission and prevent race conditions
    sleep 1
done

echo "-------------------------------------------"
echo "Submitting Global Attention-MIL final summary as dependent job..."
# This job will only start once all 5 folds have successfully completed
sbatch --dependency=afterok:$DEPENDENCIES <<EOF
#!/bin/bash
#SBATCH -p dcca40                    # Submit to the dcca40 partition
#SBATCH -t 0-00:30                   # Set a 30-minute time limit for summarization
#SBATCH --mem=16G                    # Allocate 16GB of RAM for data aggregation
#SBATCH -c 4                         # Request 4 CPU cores for parallel processing
#SBATCH -J HPy_FinalSummary          # Set the SLURM job name for monitoring
#SBATCH -o results/slurm_summary_%j.txt # Direct output to results folder with Job ID suffix

# Activate virtual environment
source ../venv/bin/activate

echo "All folds finished. Iteration 24.9: Robust Generalization Summary..."
# Fix: Summary script expects results dir and --last 5 for the latest fold set
python summarize_results.py --dir results --last 5

echo \"Generating Ensemble Voting Summary (OR-Logic) for runs ${MIN_JOB}-${MAX_JOB}...\"
python ensemble_voting.py --runs ${MIN_JOB}-${MAX_JOB}

echo \"Clinical analysis and visualization generated.\"
EOF

echo "-------------------------------------------"
echo "All 5 folds + Summary job submitted. Use 'squeue -u $USER' to monitor progress."
