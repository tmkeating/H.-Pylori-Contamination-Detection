#!/bin/bash
#SBATCH --job-name=global_dedup
#SBATCH -p dcca40
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-02:00
#SBATCH --mem=32G
#SBATCH -o results/slurm_dedupe_%j.txt
#SBATCH -e results/slurm_dedupe_error_%j.txt

# Create results folder if it doesn't already exist
mkdir -p results

# Activate environment
source ../venv/bin/activate

echo "Starting Global Deduplication Audit (All images, All folders)..."
# Using absolute path for accuracy
python3 /hhome/ricse03/modelTwyla/H.-Pylori-Contamination-Detection/global_duplicates_check.py

echo "Audit Complete. Reports generated: global_image_inventory.csv, global_image_duplicates.csv, dataset_presence_matrix.csv"
