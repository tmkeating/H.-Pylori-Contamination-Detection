#!/bin/bash
#SBATCH --job-name=h_pylori_train       # Name of the job
#SBATCH --output=results/output_%j.txt # Standard output and error log (%j = JobID)
#SBATCH --error=results/error_%j.txt
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --mem=32G                      # Total memory
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --time=12:00:00                # Time limit hrs:min:sec
#SBATCH --partition=gpu                # Partition name (change if your cluster uses a different name)

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
