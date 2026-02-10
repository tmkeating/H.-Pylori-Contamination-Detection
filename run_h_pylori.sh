#!/bin/bash
#SBATCH --job-name=h_pylori_train       # Name of the job
#SBATCH -D .                           # Use CURRENT directory as working directory
#SBATCH -n 4                           # Number of cores
#SBATCH -N 1                           # Ensure that all cores are on one machine
#SBATCH -t 0-12:00                     # Runtime in D-HH:MM (12 hours)
#SBATCH -p dcca40                      # Submit to the A40 node partition
#SBATCH --mem=32G                      # Total memory
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH -o results/output_%j.txt       # Standard output log (JobID %j)
#SBATCH -e results/error_%j.txt        # Error log

# Create results directory if it doesn't exist
mkdir -p results

# 1. Load necessary modules (Common on clusters)
# module load python/3.8
# module load cuda/11.8

# 2. Activate your virtual environment
# Assumes 'venv' is in the same directory as this script
source venv/bin/activate

# 3. Ensure you have the GPU version of PyTorch
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Run the training script
python train.py
