#!/bin/bash
#SBATCH --job-name=h_pylori_fast
#SBATCH -D .
#SBATCH -n 1                           # One task
#SBATCH -c 8                          # Request 8 CPU cores per task for fast data loading
#SBATCH -N 1
#SBATCH -t 0-06:00                     # It will likely finish in < 6 hours now
#SBATCH -p dcca40
#SBATCH --mem=48G                      # Increased to 48GB as requested
#SBATCH --gres=gpu:1
#SBATCH -o results/output_%j.txt
#SBATCH -e results/error_%j.txt

# Create results directory if it doesn't exist
mkdir -p results

# --- LOCAL SCRATCH SETUP ---
# We use /tmp as it's on a local NVMe SSD (faster than network storage)
LOCAL_SCRATCH="/tmp/ricse03_h_pylori_data"
REMOTE_DATA="/import/fhome/vlia/HelicoDataSet"

echo "Synchronizing dataset to local scratch: $LOCAL_SCRATCH"
mkdir -p "$LOCAL_SCRATCH"

# Copy metadata (fast)
cp "$REMOTE_DATA"/*.xlsx "$LOCAL_SCRATCH/"
cp "$REMOTE_DATA"/*.csv "$LOCAL_SCRATCH/"

# Sync folders (rsync is efficient - only copies missing/changed files)
mkdir -p "$LOCAL_SCRATCH/CrossValidation"
rsync -aq "$REMOTE_DATA/CrossValidation/Annotated" "$LOCAL_SCRATCH/CrossValidation/"
rsync -aq "$REMOTE_DATA/CrossValidation/Cropped" "$LOCAL_SCRATCH/CrossValidation/"
rsync -aq "$REMOTE_DATA/HoldOut" "$LOCAL_SCRATCH/"

echo "Data synchronization complete."

# 1. Load necessary modules (Common on clusters)
# module load python/3.8
# module load cuda/11.8

# 2. Activate your virtual environment
# The venv is located in the parent directory
source ../venv/bin/activate

# 3. Ensure you have the GPU version of PyTorch
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Run the training script
# Use environment variables if set, otherwise default to Fold 0
FOLD=${FOLD:-0}
NUM_FOLDS=${NUM_FOLDS:-5}

echo "Starting Training for Fold: $FOLD of $NUM_FOLDS"
python train.py --fold $FOLD --num_folds $NUM_FOLDS

# 5. Integrate Meta-Classifier: Rebuild the diagnostic layer with the new data
echo "Updating Meta-Classifier with latest results..."
python meta_classifier.py
