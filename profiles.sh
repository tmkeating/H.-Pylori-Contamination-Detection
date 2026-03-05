# Model Profiles for H. Pylori Detection
# This is the CENTRAL SOURCE of TRUTH for all experiment hyperparameters.

# Profile: AUDITOR (Clinical Grade - 100% Precision)
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

# Profile: SEARCHER (Iteration 24: Sensitivity Squeeze - Target 100% Recall)
# Using Max-MIL + High PosWeight to anchor sparse signals
function set_profile_SEARCHER() {
    export NEG_WEIGHT=1.0
    # Increased to 10.0 to force the optimizer to ignore no "Ghost" patients
    export POS_WEIGHT=10.0 
    # Increased to 3.0 to focus gradients on 'Hard' sparse bacterial targets
    export GAMMA=3.0
    # Maintained 30 epochs and 0.4 warmup for Max-MIL stability
    export NUM_EPOCHS=30 
    export SAVER_METRIC="recall"
    export FREEZE_BN="True"
    export CLIP_GRAD=0.5
    export PCT_START=0.4
    export WEIGHT_DECAY=0.05
    export USE_SWA="True"
    export SWA_START=22
    export JITTER=0.25
    export POOL_TYPE="max"
}

# Profile: EXTREME (Diagnostic Safety Mode)
function set_profile_EXTREME() {
    export POS_WEIGHT=25.0
    export GAMMA=5.0
    export SAVER_METRIC="recall"
    export POOL_TYPE="max"
}
