#!/bin/bash
# run_rescue_ensemble.sh
# ----------------------
# Runs the Rescue Inference across all 5 folds of the Searcher ensemble.

STRIDE=128
TARGETS="B22-206,B22-262,B22-69,B22-81,B22-85,B22-01"
MODEL_DIR="finalResults/searcher"
OUTPUT_DIR="finalResults/rescue_ensemble"

mkdir -p $OUTPUT_DIR

# Loop through all 5 fold models (f0 to f4)
# Note: Using the Searcher 'model_brain' (non-SWA) weights for consistency
FOLDS=("297_25.0_105773_f0" "298_25.0_105774_f1" "299_25.0_105775_f2" "300_25.0_105776_f3" "301_25.0_105777_f4")

echo "--- 🏥 Starting Full-Ensemble Rescue (Stride: $STRIDE) ---"

for FOLD_BASE in "${FOLDS[@]}"; do
    MODEL_PATH="${MODEL_DIR}/${FOLD_BASE}_convnext_tiny_model_brain.pth"
    OUTPUT_CSV="${OUTPUT_DIR}/rescue_${FOLD_BASE}.csv"
    
    echo "Processing $FOLD_BASE..."
    python3 rescue_inference.py \
        --model "$MODEL_PATH" \
        --output "$OUTPUT_CSV" \
        --stride $STRIDE \
        --targets "$TARGETS"
done

echo "--- 🏁 Rescue Ensemble Completed. Results in $OUTPUT_DIR ---"
