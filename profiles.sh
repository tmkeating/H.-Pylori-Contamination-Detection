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

# Profile: SEARCHER (Iteration 23: Stability Searcher - Max-MIL)
# Focus: Stabilizing Fold 2/3 convergence while maintaining 100% Precision (+) in Fold 4.
function set_profile_SEARCHER() {
    export NEG_WEIGHT=1.0
    # Moderate PosWeight to anchor the background while Max-Pooling isolates signal
    export POS_WEIGHT=0.75 # Increased from 0.5 to anchor gradient in sparse folds
    export GAMMA=2.0
    export NUM_EPOCHS=30 # Extended from 25 to allow slow Max-MIL feature mining
    export SAVER_METRIC="recall"
    export FREEZE_BN="True"
    export CLIP_GRAD=0.5
    export PCT_START=0.4 # Extended warmup (from 0.3) for gradient stability
    export WEIGHT_DECAY=0.05 # Slightly reduced WD to allow easier gradient flow
    export USE_SWA="True"
    export SWA_START=22 # Offset to account for 30-epoch duration
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
