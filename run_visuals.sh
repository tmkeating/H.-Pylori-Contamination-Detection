#!/bin/bash
#SBATCH --job-name=h_pylori_visuals     # Name of the job
#SBATCH -D .                           # Use CURRENT directory as working directory
#SBATCH -n 4                           # Number of cores
#SBATCH -N 1                           # Ensure that all cores are on one machine
#SBATCH -t 0-01:00                     # Runtime (1 hour is plenty)
#SBATCH -p dcca40                     # Submit to the dcca40 partition
#SBATCH --mem=16G                     # Total memory
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH -o results/visuals_output_%j.txt # Standard output log
#SBATCH -e results/visuals_error_%j.txt  # Error log

# Create results directory if it doesn't exist
mkdir -p results

# Run the visual generation script
echo "Generating Visual Report for Run 10..."
/bin/python3 generate_visuals.py
