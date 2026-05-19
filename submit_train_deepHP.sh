#!/bin/bash
# train_deepHP.sh - DeepHP 5-Fold SLURM Orchestrator
#
# Purpose: Orchestrate full transfer learning pipeline:
#   1. Pre-train backbone on 394,926 H&E patches (5 folds)
#   2. Average backbone weights across folds
#   3. Schedule final summary job
#
# Usage:
#   PROFILE=SEARCHER MODEL_NAME=convnext_tiny ITER=31.0 ./train_deepHP.sh
#
# Environment Variables:
#   PROFILE:    Model profile from profiles.sh (default: SEARCHER)
#   MODEL_NAME: Backbone architecture (default: convnext_tiny)
#   ITER:       Iteration number for tracking (default: 31.0)

set -e  # Exit on error

MODEL_NAME=${MODEL_NAME:-"convnext_tiny"}
PROFILE=${PROFILE:-"SEARCHER"}
ITER=${ITER:-"31.0"}

# Source the Model Profiles (for consistency with HelicoDataSet training)
if [ -f "profiles.sh" ]; then
    source profiles.sh
    echo "✓ Loaded profiles from profiles.sh"
    # For DeepHP, we mainly care about the base learning parameters
    # If a profile function exists, call it to load parameters
    if declare -f "set_profile_$PROFILE" > /dev/null; then
        "set_profile_$PROFILE"
        echo "✓ Using $PROFILE profile"
    fi
else
    echo "⚠ profiles.sh not found, using defaults"
fi

# DeepHP-specific parameters (can be overridden by profiles.sh)
NUM_EPOCHS=${NUM_EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-128}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
POS_WEIGHT=${POS_WEIGHT:-2.5}
USE_FOCAL_LOSS=${USE_FOCAL_LOSS:-"False"}
GAMMA=${GAMMA:-1.0}

echo "=========================================================================="
echo "DeepHP H&E Pre-training Pipeline (5-Fold Cross-Validation)"
echo "=========================================================================="
echo "Configuration:"
echo "  Profile: $PROFILE"
echo "  Model: $MODEL_NAME"
echo "  Iteration: $ITER"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Pos Weight: $POS_WEIGHT"
echo "  Focal Loss: $USE_FOCAL_LOSS (gamma=$GAMMA)"
echo "=========================================================================="
echo ""

# 1. Submit pre-sync job (prepares scratch directory for data)
echo "Submitting pre-sync job to prepare environment..."
PRE_SYNC_JOB=$(sbatch -p dcca40 --job-name=deephp_presync --output=results/slurm_deephp_presync_%j.txt <<'PRESYNC_EOF'
#!/bin/bash
#SBATCH -p dcca40
#SBATCH -t 0-01:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -J deephp_presync

echo "Pre-sync job: Verifying DeepHP dataset accessibility..."

DEEPHP_ROOT=$(python3 -c "from config import DEEPHP_DATASET_ROOT; print(DEEPHP_DATASET_ROOT)")
echo "✓ DeepHP dataset root: $DEEPHP_ROOT"

if [ ! -d "$DEEPHP_ROOT/Positive" ] || [ ! -d "$DEEPHP_ROOT/Negative" ]; then
    echo "ERROR: DeepHP dataset not found at $DEEPHP_ROOT"
    exit 1
fi

echo "✓ DeepHP dataset verified: ready for training"

PRESYNC_EOF
)
PRE_SYNC_JOB_ID=$(echo $PRE_SYNC_JOB | awk '{print $4}')
echo "Pre-sync job ID: $PRE_SYNC_JOB_ID"
PRE_SYNC_DEPENDENCY="afterok:$PRE_SYNC_JOB_ID"

# 2. Submit 5 fold training jobs (dependent on pre-sync)
echo ""
echo "Submitting DeepHP pre-training jobs for all 5 folds..."
echo "=========================================================================="

DEPENDENCIES=""
for FOLD in {0..4}
do
    echo "Submitting fold $FOLD..."
    
    JOB_OUT=$(sbatch -p dcca40 \
        --dependency=$PRE_SYNC_DEPENDENCY \
        --job-name=deephp_f${FOLD} \
        --output=results/slurm_deephp_f${FOLD}_%j.txt \
        --error=results/slurm_deephp_error_f${FOLD}_%j.txt \
        --ntasks=1 \
        --cpus-per-task=8 \
        --gres=gpu:1 \
        --mem=48G \
        --time=24:00:00 \
        --export=ALL,FOLD=$FOLD,MODEL_NAME=$MODEL_NAME,NUM_EPOCHS=$NUM_EPOCHS,BATCH_SIZE=$BATCH_SIZE,LEARNING_RATE=$LEARNING_RATE,WEIGHT_DECAY=$WEIGHT_DECAY,POS_WEIGHT=$POS_WEIGHT,USE_FOCAL_LOSS=$USE_FOCAL_LOSS,GAMMA=$GAMMA \
        <<TRAIN_EOF
#!/bin/bash
cd /hhome/ricse03/modelTwyla/H.-Pylori-Contamination-Detection

python3 train_deepHP_patches.py \
    --fold \$FOLD \
    --num_folds 5 \
    --model_name \$MODEL_NAME \
    --num_epochs \$NUM_EPOCHS \
    --batch_size \$BATCH_SIZE \
    --learning_rate \$LEARNING_RATE \
    --weight_decay \$WEIGHT_DECAY \
    --pos_weight \$POS_WEIGHT \
    --use_focal_loss \$USE_FOCAL_LOSS \
    --gamma \$GAMMA

echo ""
echo "✓ Fold \$FOLD complete: results/deephp_backbone_pretrained_\${MODEL_NAME}_f\${FOLD}.pth"
TRAIN_EOF
)
    
    JOB_ID=$(echo $JOB_OUT | awk '{print $4}')
    echo "  → Job ID: $JOB_ID"
    
    # Add to dependency list
    if [ -z "$DEPENDENCIES" ]; then
        DEPENDENCIES="$JOB_ID"
    else
        DEPENDENCIES="$DEPENDENCIES:$JOB_ID"
    fi
    
    sleep 1  # Prevent race conditions
done

echo ""
echo "=========================================================================="
echo "All 5 fold jobs submitted. Scheduling final averaging + summary job..."
echo "=========================================================================="
echo ""

# 3. Submit final summary job (depends on all 5 folds)
#    This job averages the backbone and prepares for fine-tuning
sbatch --dependency=afterok:$DEPENDENCIES \
    -p dcca40 \
    --time=0-00:30 \
    --mem=16G \
    --cpus-per-task=4 \
    --job-name=deephp_summary \
    --output=results/slurm_deephp_summary_%j.txt \
    --error=results/slurm_deephp_summary_error_%j.txt \
    <<'SUMMARY_EOF'
#!/bin/bash
#SBATCH -p dcca40
cd /hhome/ricse03/modelTwyla/H.-Pylori-Contamination-Detection

echo "=========================================================================="
echo "All DeepHP pre-training folds complete. Averaging backbone weights..."
echo "=========================================================================="
echo ""

python3 << 'PYTHON_EOF'
from load_pretrained_backbone import average_backbone_weights
import os

fold_paths = [f"results/deephp_backbone_pretrained_convnext_tiny_f{i}.pth" for i in range(5)]
output_path = "results/deephp_backbone_final_convnext_tiny.pth"

# Verify all fold checkpoints exist
missing = [p for p in fold_paths if not os.path.exists(p)]
if missing:
    print(f"ERROR: Missing checkpoints: {missing}")
    exit(1)

print("Averaging 5-fold backbone weights...")
average_backbone_weights(fold_paths, output_path)

print(f"\n{'='*80}")
print(f"✓ Backbone averaging complete!")
print(f"{'='*80}")
print(f"\nNext steps:")
print(f"1. Fine-tune on HelicoDataSet using pre-trained backbone:")
print(f"   PRETRAINED_BACKBONE={output_path}")
print(f"   for i in {{0..4}}; do")
print(f"     sbatch -J heli_ft_f\$i run_h_pylori.sh \$i")
print(f"   done")
print(f"\n2. Or use automated fine-tuning:")
print(f"   ./submit_transfer_learning.sh")
print(f"\n3. Generate ensemble and analysis:")
print(f"   python3 ensemble_voting.py --runs 31_0,31_1,31_2,31_3,31_4")
print(f"{'='*80}\n")

PYTHON_EOF

echo "✓ Summary job complete"
echo ""
echo "=========================================================================="
echo "DeepHP pre-training pipeline finished!"
echo "=========================================================================="
echo ""
echo "Backbone checkpoints:"
ls -lah results/deephp_backbone_pretrained_convnext_tiny_f*.pth 2>/dev/null | tail -5
echo ""
echo "Averaged backbone:"
ls -lah results/deephp_backbone_final_convnext_tiny.pth 2>/dev/null
echo ""

SUMMARY_EOF

echo "=========================================================================="
echo "✓ All jobs submitted successfully!"
echo "=========================================================================="
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER | grep deephp"
echo ""
echo "View logs with:"
echo "  tail -f results/slurm_deephp_f0_*.txt"
echo ""

