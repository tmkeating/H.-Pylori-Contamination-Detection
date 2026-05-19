#!/bin/bash
# submit_transfer_learning.sh - Complete transfer learning pipeline
#
# Orchestrates the full Option 1 transfer learning workflow:
# 1. Pre-train backbone on DeepHP (5-fold)
# 2. Average across folds
# 3. Fine-tune on HelicoDataSet (5-fold)
#
# Usage:
#   ./submit_transfer_learning.sh

set -e  # Exit on error

echo "========================================================================"
echo "H. Pylori Transfer Learning Pipeline (DeepHP → HelicoDataSet)"
echo "========================================================================"
echo ""

WORK_DIR="/hhome/ricse03/modelTwyla/H.-Pylori-Contamination-Detection"
cd $WORK_DIR

# ========================================================================
# PHASE 1: Pre-train on DeepHP (5-fold)
# ========================================================================
echo "PHASE 1: Submitting DeepHP backbone pre-training (5 folds in parallel)"
echo "========================================================================="
echo ""

JOB_IDS=()
for i in {0..4}; do
    JOB_ID=$(sbatch -J deephp_f$i train_deepHP.sh $i | awk '{print $NF}')
    JOB_IDS+=($JOB_ID)
    echo "Submitted fold $i: Job ID $JOB_ID"
done

echo ""
echo "Pre-training jobs submitted: ${JOB_IDS[@]}"
echo "Waiting for all folds to complete..."
echo ""

# Wait for all pre-training jobs to complete
for JOB_ID in "${JOB_IDS[@]}"; do
    while squeue -j $JOB_ID &>/dev/null; do
        sleep 60
    done
    echo "✓ Job $JOB_ID completed"
done

echo ""
echo "✓ All DeepHP pre-training folds complete!"
echo ""

# ========================================================================
# PHASE 2: Average backbone weights across folds
# ========================================================================
echo "PHASE 2: Averaging backbone weights across 5 folds"
echo "========================================================================="
echo ""

python3 << 'EOF'
from load_pretrained_backbone import average_backbone_weights

fold_paths = [f"results/deephp_backbone_pretrained_convnext_tiny_f{i}.pth" for i in range(5)]
output_path = "results/deephp_backbone_final_convnext_tiny.pth"

average_backbone_weights(fold_paths, output_path)
print(f"✓ Averaged backbone saved to: {output_path}")
EOF

echo ""
echo "✓ Backbone averaging complete!"
echo ""

# ========================================================================
# PHASE 3: Fine-tune on HelicoDataSet (5-fold) using pre-trained backbone
# ========================================================================
echo "PHASE 3: Fine-tuning on HelicoDataSet with pre-trained backbone (5 folds)"
echo "========================================================================="
echo ""

PRETRAINED_BACKBONE="results/deephp_backbone_final_convnext_tiny.pth"

if [ ! -f "$PRETRAINED_BACKBONE" ]; then
    echo "ERROR: Pre-trained backbone not found at $PRETRAINED_BACKBONE"
    exit 1
fi

echo "Using pre-trained backbone: $PRETRAINED_BACKBONE"
echo ""

FT_JOB_IDS=()
for i in {0..4}; do
    # Modify run_h_pylori.sh to accept pretrained backbone path, or use train.py directly
    # For now, we'll submit with updated parameters
    JOB_ID=$(sbatch -J heli_ft_f$i -e results/ft_slurm_%j_error.txt -o results/ft_slurm_%j_output.txt \
        sh -c "cd $WORK_DIR && python3 train.py \
            --fold $i \
            --num_folds 5 \
            --model_name convnext_tiny \
            --iter 31.0 \
            --pretrained_backbone_path $PRETRAINED_BACKBONE \
            --freeze_backbone False" | awk '{print $NF}')
    
    FT_JOB_IDS+=($JOB_ID)
    echo "Submitted fine-tuning fold $i: Job ID $JOB_ID"
done

echo ""
echo "Fine-tuning jobs submitted: ${FT_JOB_IDS[@]}"
echo "Waiting for all folds to complete..."
echo ""

# Wait for all fine-tuning jobs to complete
for JOB_ID in "${FT_JOB_IDS[@]}"; do
    while squeue -j $JOB_ID &>/dev/null; do
        sleep 60
    done
    echo "✓ Job $JOB_ID completed"
done

echo ""
echo "========================================================================"
echo "✓✓✓ TRANSFER LEARNING PIPELINE COMPLETE! ✓✓✓"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "1. Run final ensemble fusion:"
echo "   python3 ensemble_voting.py --runs 31_0,31_1,31_2,31_3,31_4"
echo "2. Compare transfer learning vs baseline results in:"
echo "   - results/31*.pth (fine-tuned models)"
echo "   - results/31*_evaluation_report.csv"
echo ""
