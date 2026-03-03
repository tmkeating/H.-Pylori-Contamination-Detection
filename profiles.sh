# Model Profiles for H. Pylori Detection
# This is the CENTRAL SOURCE of TRUTH for all experiment hyperparameters.

# Profile: AUDITOR (Clinical Grade - 100% Precision)
# Restored to the "Golden" Iteration 17 configuration
function set_profile_AUDITOR() {
    export NEG_WEIGHT=1.0
    export POS_WEIGHT=7.5
    export GAMMA=2.0
    export NUM_EPOCHS=20
    export SAVER_METRIC="recall"
}

# Profile: SEARCHER (Iteration 21 Calibration - Rebalanced for ResNet Stability)
# Fixed collapse by reducing POS_WEIGHT and using POS_PRECISION focus
function set_profile_SEARCHER() {
    export NEG_WEIGHT=1.0
    export POS_WEIGHT=2.2
    export GAMMA=2.0
    export NUM_EPOCHS=15
    export SAVER_METRIC="f1"
}

# Profile: EXTREME (Legacy Iteration 17)
function set_profile_EXTREME() {
    export POS_WEIGHT=7.5
    export GAMMA=1.0
    export SAVER_METRIC="recall"
}

