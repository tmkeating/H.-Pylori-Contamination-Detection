# H. Pylori Experiment Hyperparameter Profiles
# ---------------------------------------------
# This is the CENTRAL SOURCE of TRUTH for all experiment configurations.
# Definitions here are sourced by submit_all_folds.sh and run_h_pylori.sh 
# to ensure consistency across the 5-fold cross-validation pipeline.
#
# Profiles:
#   AUDITOR:  Precision focus (High PosWeight, Low Gamma, High WD).
#   SEARCHER: Recall focus (Target 100% Recall, Balanced PosWeight, High Gamma).
#   EXTREME:  Diagnostic safety mode (Max-MIL pooling, massive PosWeight).
# ---------------------------------------------

# Profile: AUDITOR (Clinical Grade - 100% Precision)
function set_profile_AUDITOR() {
    export NEG_WEIGHT=1.0
    export POS_WEIGHT=7.5
    export GAMMA=1.0
    export NUM_EPOCHS=20
    export SAVER_METRIC="f1"
    export FREEZE_BN="False"
    export CLIP_GRAD=0.0
    export PCT_START=0.1
    export WEIGHT_DECAY=0.1
    export USE_SWA="True"
    export SWA_START=15
    export JITTER=0.15
    export POOL_TYPE="attention"
}

# Profile: SEARCHER (Iteration 24.9: Robust Generalization - Target 100% Recall)
# Using Max-MIL + Stabilized Weighted Training + ReduceLROnPlateau
function set_profile_SEARCHER() {
    export NEG_WEIGHT=1.0
    # Iteration 24.9: Balanced the PosWeight to 5.0 to prevent gradient saturation
    export POS_WEIGHT=5.0 
    # Maintained 3.0 to focus gradients on 'Hard' sparse bacterial targets
    export GAMMA=3.0
    # Higher WD to prevent 100% Training Accuracy (Overfitting)
    export WEIGHT_DECAY=0.05
    # Reduced epochs with early stopping potential
    export NUM_EPOCHS=20
    export SAVER_METRIC="f1"
    export FREEZE_BN="True"
    export CLIP_GRAD=1.0
    export PCT_START=0.1
    export WEIGHT_DECAY=0.05
    export USE_SWA="True"
    export SWA_START=12
    export JITTER=0.25
    export POOL_TYPE="attention"
}

# Profile: EXTREME (Diagnostic Safety Mode)
function set_profile_EXTREME() {
    export POS_WEIGHT=25.0
    export GAMMA=5.0
    export SAVER_METRIC="recall"
    export POOL_TYPE="max"
}
