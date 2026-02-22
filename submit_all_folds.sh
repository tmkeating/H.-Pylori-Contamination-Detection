#!/bin/bash

# Simple script to submit all 5 folds for H. Pylori cross-validation
# Usage: MODEL_NAME=convnext_tiny ./submit_all_folds.sh

MODEL_NAME=${MODEL_NAME:-"convnext_tiny"}

for FOLD in {0..4}
do
    echo "-------------------------------------------"
    echo "Submitting SLURM job for Fold $FOLD using $MODEL_NAME..."
    sbatch --export=ALL,FOLD=$FOLD,MODEL_NAME=$MODEL_NAME run_h_pylori.sh
    # Wait 1 second to ensure sequential submission and prevent race conditions
    sleep 1
done

echo "-------------------------------------------"
echo "All 5 folds submitted. Use 'squeue -u $USER' to monitor progress."
