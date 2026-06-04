#!/bin/bash
#SBATCH --job-name=global_duplicates_check
#SBATCH -p pg1tfg12
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-02:00
#SBATCH --mem=32G
#SBATCH -o results/slurm_duplicates_%j.txt
#SBATCH -e results/slurm_duplicates_error_%j.txt

# ⚠️  SUBMIT SCRIPT FOR HELICODATASET AUDIT - CROSS-DATASET DUPLICATE DETECTION
#
# This script submits the global duplicate detection audit to SLURM
# Scans ALL images across ALL folders in the entire workspace
# Generates:
#   - image_inventory.csv (all scanned images with file paths)
#   - image_duplicates.csv (duplicate images if found)
#   - class_distribution.csv (Contaminated vs Clean counts if labeled)
#   - patch_duplicate_audit.csv (duplicate summary by class/folder)
#   - suggested_blacklist.json (images to exclude for data integrity)
#
# Usage:
#   sbatch submit_duplicates_check.sh
#
# Output files: results/*.csv and results/*.json

# Create results folder if it doesn't already exist
mkdir -p results

set -e  # Exit on error

# Verify virtual environment before proceeding
if [ -f "./verify_venv.sh" ]; then
    source ./verify_venv.sh
else
    echo "ERROR: verify_venv.sh not found in current directory"
    exit 1
fi

# Activate environment
source $VENV_ROOT/bin/activate

echo "Starting Global Deduplication Audit (All images, All folders)..."
# Using absolute path for accuracy
python3 /home/tkeating/model/H.-Pylori-Contamination-Detection/check_global_duplicates.py

echo ""
echo "✓ Global duplicate detection audit complete!"
echo "Output files:"
echo "  - image_inventory.csv"
echo "  - image_duplicates.csv (if duplicates found)"
echo "  - class_distribution.csv (if labeled)"
echo "  - patch_duplicate_audit.csv"
echo "  - suggested_blacklist.json"
