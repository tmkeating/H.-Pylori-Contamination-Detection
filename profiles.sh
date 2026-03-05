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
    export POOL_TYPE="attention"
}

# Profile: SEARCHER (Iteration 22: Moderate Precision Searcher - Max-MIL)
# Re-calibrated for stability: Balanced Jitter and 25 Epochs.
function set_profile_SEARCHER() {
    export NEG_WEIGHT=1.0
    # Moderate PosWeight to anchor the background while Max-Pooling isolates signal
    export POS_WEIGHT=0.5
    export GAMMA=2.0
    export NUM_EPOCHS=25
    export SAVER_METRIC="recall"
    export FREEZE_BN="True"
    export CLIP_GRAD=0.5
    export PCT_START=0.3
    export WEIGHT_DECAY=0.1
    export USE_SWA="True"
    export SWA_START=18
    export JITTER=0.25
    export POOL_TYPE="max"
}

# Profile: EXTREME (Maximum Awareness)
function set_profile_EXTREME() {
    export POS_WEIGHT=5.0
    export GAMMA=1.0
    export SAVER_METRIC="recall"
    export POOL_TYPE="max"
}
