#!/bin/bash

# Simple script to submit all 5 folds for H. Pylori cross-validation
# Usage: MODEL_NAME=convnext_tiny ./submit_all_folds.sh

MODEL_NAME=${MODEL_NAME:-"convnext_tiny"}

for FOLD in {0..4}
do
    echo "-------------------------------------------"
    echo "Submitting SLURM job for Fold $FOLD using $MODEL_NAME..."
    # Capture the job ID
    JOB_OUT=$(sbatch --export=ALL,FOLD=$FOLD,MODEL_NAME=$MODEL_NAME run_h_pylori.sh)
    echo "$JOB_OUT"
    JOB_ID=$(echo $JOB_OUT | awk '{print $4}')
    
    # Add to dependency list
    if [ -z "$DEPENDENCIES" ]; then
        DEPENDENCIES="$JOB_ID"
    else
        DEPENDENCIES="$DEPENDENCIES:$JOB_ID"
    fi
    
    # Wait 1 second to ensure sequential submission and prevent race conditions
    sleep 1
done

echo "-------------------------------------------"
echo "Submitting Global Meta-Classifier training as dependent job..."
# This job will only start once all 5 folds have successfully completed
sbatch --dependency=afterok:$DEPENDENCIES <<EOF
#!/bin/bash
#SBATCH -p dcca40
#SBATCH -t 0-00:30
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -J HPy_FinalSummary
#SBATCH -o slurm-%j.out

# Activate virtual environment
source ../venv/bin/activate

echo "All folds finished. Executing Global Meta-Classifier training..."
python meta_classifier.py
echo "Clinical analysis complete."
EOF

echo "-------------------------------------------"
echo "All 5 folds + Summary job submitted. Use 'squeue -u $USER' to monitor progress."
