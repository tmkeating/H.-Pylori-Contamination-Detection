# Model Profiles for H. Pylori Detection
# This is the CENTRAL SOURCE of TRUTH for all experiment hyperparameters.

# Profile: AUDITOR (Clinical Grade - 100% Precision)
# Restored to the "Golden" Iteration 17 configuration
function set_profile_AUDITOR() {
    export NEG_WEIGHT=1.0
    export POS_WEIGHT=7.5
    export GAMMA=1.0
    export NUM_EPOCHS=20
    export SAVER_METRIC="recall"
    export FREEZE_BN="False"
    export CLIP_GRAD=0.0
    export PCT_START=0.1
    export WEIGHT_DECAY=0.1
    export USE_SWA="True"
    export SWA_START=15
    export JITTER=0.15
}

# Profile: SEARCHER (Iteration 21 Calibration - Rebalanced for ResNet Stability)
# Fixed collapse by reducing POS_WEIGHT and using POS_PRECISION focus
function set_profile_SEARCHER() {
    export NEG_WEIGHT=1.0
    export POS_WEIGHT=0.25
    export GAMMA=1.5
    export NUM_EPOCHS=25
    export SAVER_METRIC="f1"
    # --- Stability Patch (Iteration 21.2) ---
    export FREEZE_BN="True"
    export CLIP_GRAD=0.3
    export PCT_START=0.4
    export WEIGHT_DECAY=0.1
    export USE_SWA="True"
    export SWA_START=18
    export JITTER=0.45
}

# Profile: EXTREME (Legacy Iteration 17)
function set_profile_EXTREME() {
    export POS_WEIGHT=7.5
    export GAMMA=1.0
    export SAVER_METRIC="recall"
}

