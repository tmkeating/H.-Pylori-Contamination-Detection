#!/bin/bash

# Simple script to submit all 5 folds for H. Pylori cross-validation
# Usage: ./submit_all_folds.sh

for FOLD in {0..4}
do
    echo "-------------------------------------------"
    echo "Submitting SLURM job for Fold $FOLD..."
    sbatch --export=ALL,FOLD=$FOLD run_h_pylori.sh
done

echo "-------------------------------------------"
echo "All 5 folds submitted. Use 'squeue -u $USER' to monitor progress."
