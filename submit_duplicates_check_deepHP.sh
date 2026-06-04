#!/bin/bash
#SBATCH --job-name=deephp_duplicates_check
#SBATCH -p pg1tfg12
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-02:00
#SBATCH --mem=32G
#SBATCH -o results/slurm_deephp_duplicates_%j.txt
#SBATCH -e results/slurm_deephp_duplicates_error_%j.txt

# ⚠️  SUBMIT SCRIPT FOR DEEPHP DATASET AUDIT ONLY - NOT FOR HELICODATASET
#
# This script submits the DeepHP duplicate detection audit to SLURM
# Generates:
#   - deephp_image_inventory.csv (all ~394,926 patches)
#   - deephp_image_duplicates.csv (duplicate patches if found)
#   - deephp_class_distribution.csv (Positive vs Negative counts)
#   - deephp_patch_duplicate_audit.csv (duplicate summary by class)
#   - suggested_deephp_blacklist.json (patches to exclude)
#
# Usage:
#   sbatch submit_deephp_duplicates_check.sh
#
# Output files: results/*.csv and results/*.json

# Create results folder if it doesn't already exist
mkdir -p results

# Activate environment
source ../venv/bin/activate

echo "Starting DeepHP Global Duplicate Detection Audit..."
echo "Dataset: /home/tkeating/datasets/8117177 (Positive + Negative folders)"
echo "Expected patches: ~394,926 total (111,005 Positive + 283,921 Negative)"
echo ""

# Using absolute path for accuracy
python3 /home/tkeating/model/H.-Pylori-Contamination-Detection/check_global_duplicates_deepHP.py

echo ""
echo "✓ DeepHP duplicate detection audit complete!"
echo "Output files:"
echo "  - deephp_image_inventory.csv"
echo "  - deephp_image_duplicates.csv (if duplicates found)"
echo "  - deephp_class_distribution.csv"
echo "  - deephp_patch_duplicate_audit.csv"
echo "  - suggested_deephp_blacklist.json"
