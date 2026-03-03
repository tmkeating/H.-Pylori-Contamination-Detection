# Model Profiles for H. Pylori Detection
# This is the CENTRAL SOURCE of TRUTH for all experiment hyperparameters.

# Profile: AUDITOR (Clinical Grade - 100% Precision)
# Restored to the "Golden" Iteration 17 configuration
function set_profile_AUDITOR() {
    export NEG_WEIGHT=1.0
    export POS_WEIGHT=7.5
    export GAMMA=2.0
    export SAVER_METRIC="recall"
}

# Profile: SEARCHER (Iteration 20 Calibration - Balanced Recall/Precision)
function set_profile_SEARCHER() {
    export NEG_WEIGHT=1.0
    export POS_WEIGHT=4.0
    export GAMMA=2.0
    export SAVER_METRIC="f1"
}

# Profile: EXTREME (Legacy Iteration 17)
function set_profile_EXTREME() {
    export POS_WEIGHT=7.5
    export GAMMA=1.0
    export SAVER_METRIC="recall"
}

