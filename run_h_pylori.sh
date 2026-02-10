#!/bin/bash
#SBATCH --job-name=h_pylori_train       # Name of the job
#SBATCH --output=results/output_%j.txt # Standard output and error log (%j = JobID)
#SBATCH --error=results/error_%j.txt
#SBATCH -n 4                           # Number of cores
#SBATCH -N 1                           # Ensure that all cores are on one machine
#SBATCH -t 0-12:00                     # Runtime in D-HH:MM (12 hours)
#SBATCH -p dcca40                 # Partition to submit to (Master High/Low)
#SBATCH -q masterlow                   # Quality of Service (QOS)
#SBATCH --mem=32G                      # Total memory
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH -o results/%x_%j.out           # Standard output log
#SBATCH -e results/%x_%j.err           # Standard error log

# 1. Load necessary modules (Uncomment and adjust based on your cluster)
# module load python/3.8
# module load cuda/11.8

# 2. Activate your virtual environment
source venv/bin/activate

# 3. Ensure you have the GPU version of PyTorch (not the +cpu version)
# This only needs to run once, but doesn't hurt to keep
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Run the training script
python train.py
