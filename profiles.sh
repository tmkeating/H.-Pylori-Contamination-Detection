# H. Pylori Experiment Hyperparameter Profiles
# ---------------------------------------------
# This is the CENTRAL SOURCE of TRUTH for all experiment configurations.
# Definitions here are sourced by submit_all_folds.sh and run_h_pylori.sh 
# to ensure consistency across the 5-fold cross-validation pipeline.
#
# Default Hyperparameters (can be overridden by profiles):
#   DROPOUT=0.4 (Default dropout rate for regularization)
#
# Profiles:
#   SEARCHER: Recall focus (Target 100% Recall, Balanced PosWeight, High Gamma). FREEZE_BN=True, FREEZE_BACKBONE=False, USE_DANN=False
#   SEARCHERDEEPHP:  The profile for DeepHP integration (Target 100% Recall, Higher PosWeight, High Gamma). FREEZE_BN=False, FREEZE_BACKBONE=False, USE_DANN=True
#   TEST: Minimal test profile (1 epoch, no DANN)
#   TESTDEEPHP: Minimal test profile for DeepHP integration (1 epoch, USE_DANN=True)
# ---------------------------------------------

# Set default values (can be overridden by individual profiles)
export DROPOUT=${DROPOUT:-0.25}
export LEARNING_RATE=${LEARNING_RATE:-2e-5}
# Only for DeepHP training
export BATCH_SIZE=${BATCH_SIZE:-32} # Only for DeepHP training
export USE_COMPILE=${USE_COMPILE:-False}

# Profile: SEARCHER (Iteration 24.9: Robust Generalization - Target 100% Recall)
# Using Max-MIL + Stabilized Weighted Training + ReduceLROnPlateau
function set_profile_SEARCHER() {
    export NEG_WEIGHT=1.0
    export POS_WEIGHT=1.5 
    # Maintained 3.0 to focus gradients on 'Hard' sparse bacterial targets
    export GAMMA=3.0
    export USE_FOCAL_LOSS="False"
    # Higher WD to prevent 100% Training Accuracy (Overfitting)
    export WEIGHT_DECAY=0.05
    export NUM_EPOCHS=20
    export DEEPHP_EPOCHS=20
    export LEARNING_RATE=2e-5
    export SAVER_METRIC="f1"
    export FREEZE_BN="True"
    export FREEZE_BACKBONE="False"
    export CLIP_GRAD=1.0
    export PCT_START=0.1
    export USE_SWA="True"
    export SWA_START=12
    export JITTER=0.25
    export DROPOUT=0.25
    export POOL_TYPE="attention"
    export USE_DANN="False"
    export DANN_LAMBDA=1.0
    export DANN_WEIGHT=1.0
}

function set_profile_SEARCHER1() {
    export NEG_WEIGHT=1.0
    export POS_WEIGHT=2.0 
    # Maintained 3.0 to focus gradients on 'Hard' sparse bacterial targets
    export GAMMA=3.0
    export USE_FOCAL_LOSS="False"
    # Higher WD to prevent 100% Training Accuracy (Overfitting)
    export WEIGHT_DECAY=0.05
    export NUM_EPOCHS=20
    export DEEPHP_EPOCHS=20
    export LEARNING_RATE=2e-5
    export SAVER_METRIC="f1"
    export FREEZE_BN="True"
    export FREEZE_BACKBONE="False"
    export CLIP_GRAD=1.0
    export PCT_START=0.1
    export USE_SWA="True"
    export SWA_START=12
    export JITTER=0.25
    export DROPOUT=0.25
    export POOL_TYPE="attention"
    export USE_DANN="False"
    export DANN_LAMBDA=1.0
    export DANN_WEIGHT=1.0
}

function set_profile_SEARCHER2() {
    export NEG_WEIGHT=1.0
    export POS_WEIGHT=2.5
    export GAMMA=3.0
    export USE_FOCAL_LOSS="False"
    # Higher WD to prevent 100% Training Accuracy (Overfitting)
    export WEIGHT_DECAY=0.05
    export NUM_EPOCHS=20
    export DEEPHP_EPOCHS=20
    export LEARNING_RATE=2e-5
    export SAVER_METRIC="f1"
    export FREEZE_BN="True"
    export FREEZE_BACKBONE="False"
    export CLIP_GRAD=1.0
    export PCT_START=0.1
    export USE_SWA="True"
    export SWA_START=12
    export JITTER=0.25
    export DROPOUT=0.25
    export POOL_TYPE="attention"
    export USE_DANN="False"
    export DANN_LAMBDA=1.0
    export DANN_WEIGHT=1.0
}

function set_profile_SEARCHER3() {
    export NEG_WEIGHT=1.0
    export POS_WEIGHT=1.5 
    export GAMMA=3.5
    export USE_FOCAL_LOSS="False"
    # Higher WD to prevent 100% Training Accuracy (Overfitting)
    export WEIGHT_DECAY=0.05
    export NUM_EPOCHS=20
    export DEEPHP_EPOCHS=20
    export LEARNING_RATE=2e-5
    export SAVER_METRIC="f1"
    export FREEZE_BN="True"
    export FREEZE_BACKBONE="False"
    export CLIP_GRAD=1.0
    export PCT_START=0.1
    export USE_SWA="True"
    export SWA_START=12
    export JITTER=0.25
    export DROPOUT=0.25
    export POOL_TYPE="attention"
    export USE_DANN="False"
    export DANN_LAMBDA=1.0
    export DANN_WEIGHT=1.0
}

function set_profile_SEARCHER4() {
    export NEG_WEIGHT=1.0
    export POS_WEIGHT=2.0
    export GAMMA=3.5
    export USE_FOCAL_LOSS="False"
    # Higher WD to prevent 100% Training Accuracy (Overfitting)
    export WEIGHT_DECAY=0.05
    export NUM_EPOCHS=20
    export DEEPHP_EPOCHS=20
    export LEARNING_RATE=2e-5
    export SAVER_METRIC="f1"
    export FREEZE_BN="True"
    export FREEZE_BACKBONE="False"
    export CLIP_GRAD=1.0
    export PCT_START=0.1
    export USE_SWA="True"
    export SWA_START=12
    export JITTER=0.25
    export DROPOUT=0.25
    export POOL_TYPE="attention"
    export USE_DANN="False"
    export DANN_LAMBDA=1.0
    export DANN_WEIGHT=1.0
}

function set_profile_SEARCHER5() {
    export NEG_WEIGHT=1.0
    export POS_WEIGHT=2.5 
    export GAMMA=3.5
    export USE_FOCAL_LOSS="False"
    # Higher WD to prevent 100% Training Accuracy (Overfitting)
    export WEIGHT_DECAY=0.05
    export NUM_EPOCHS=20
    export DEEPHP_EPOCHS=20
    export LEARNING_RATE=2e-5
    export SAVER_METRIC="f1"
    export FREEZE_BN="True"
    export FREEZE_BACKBONE="False"
    export CLIP_GRAD=1.0
    export PCT_START=0.1
    export USE_SWA="True"
    export SWA_START=12
    export JITTER=0.25
    export DROPOUT=0.25
    export POOL_TYPE="attention"
    export USE_DANN="False"
    export DANN_LAMBDA=1.0
    export DANN_WEIGHT=1.0
}

function set_profile_SEARCHER6() {
    export NEG_WEIGHT=1.0
    export POS_WEIGHT=1.5
    export GAMMA=4.0
    export USE_FOCAL_LOSS="False"
    # Higher WD to prevent 100% Training Accuracy (Overfitting)
    export WEIGHT_DECAY=0.05
    export NUM_EPOCHS=20
    export DEEPHP_EPOCHS=20
    export LEARNING_RATE=2e-5
    export SAVER_METRIC="f1"
    export FREEZE_BN="True"
    export FREEZE_BACKBONE="False"
    export CLIP_GRAD=1.0
    export PCT_START=0.1
    export USE_SWA="True"
    export SWA_START=12
    export JITTER=0.25
    export DROPOUT=0.25
    export POOL_TYPE="attention"
    export USE_DANN="False"
    export DANN_LAMBDA=1.0
    export DANN_WEIGHT=1.0
}

function set_profile_SEARCHER7() {
    export NEG_WEIGHT=1.0
    export POS_WEIGHT=2.0
    export GAMMA=4.0
    export USE_FOCAL_LOSS="False"
    # Higher WD to prevent 100% Training Accuracy (Overfitting)
    export WEIGHT_DECAY=0.05
    export NUM_EPOCHS=20
    export DEEPHP_EPOCHS=20
    export LEARNING_RATE=2e-5
    export SAVER_METRIC="f1"
    export FREEZE_BN="True"
    export FREEZE_BACKBONE="False"
    export CLIP_GRAD=1.0
    export PCT_START=0.1
    export USE_SWA="True"
    export SWA_START=12
    export JITTER=0.25
    export DROPOUT=0.25
    export POOL_TYPE="attention"
    export USE_DANN="False"
    export DANN_LAMBDA=1.0
    export DANN_WEIGHT=1.0
}

function set_profile_SEARCHER8() {
    export NEG_WEIGHT=1.0
    export POS_WEIGHT=2.5
    export GAMMA=4.0
    export USE_FOCAL_LOSS="False"
    # Higher WD to prevent 100% Training Accuracy (Overfitting)
    export WEIGHT_DECAY=0.05
    export NUM_EPOCHS=20
    export DEEPHP_EPOCHS=20
    export LEARNING_RATE=2e-5
    export SAVER_METRIC="f1"
    export FREEZE_BN="True"
    export FREEZE_BACKBONE="False"
    export CLIP_GRAD=1.0
    export PCT_START=0.1
    export USE_SWA="True"
    export SWA_START=12
    export JITTER=0.25
    export DROPOUT=0.25
    export POOL_TYPE="attention"
    export USE_DANN="False"
    export DANN_LAMBDA=1.0
    export DANN_WEIGHT=1.0
}

function set_profile_SEARCHER9() {
    export NEG_WEIGHT=1.0
    export POS_WEIGHT=1.5
    export GAMMA=4.5
    export USE_FOCAL_LOSS="False"
    # Higher WD to prevent 100% Training Accuracy (Overfitting)
    export WEIGHT_DECAY=0.05
    export NUM_EPOCHS=20
    export DEEPHP_EPOCHS=20
    export LEARNING_RATE=2e-5
    export SAVER_METRIC="f1"
    export FREEZE_BN="True"
    export FREEZE_BACKBONE="False"
    export CLIP_GRAD=1.0
    export PCT_START=0.1
    export USE_SWA="True"
    export SWA_START=12
    export JITTER=0.25
    export DROPOUT=0.25
    export POOL_TYPE="attention"
    export USE_DANN="False"
    export DANN_LAMBDA=1.0
    export DANN_WEIGHT=1.0
}

function set_profile_SEARCHER10() {
    export NEG_WEIGHT=1.0
    export POS_WEIGHT=2.0
    export GAMMA=4.5
    export USE_FOCAL_LOSS="False"
    # Higher WD to prevent 100% Training Accuracy (Overfitting)
    export WEIGHT_DECAY=0.05
    export NUM_EPOCHS=20
    export DEEPHP_EPOCHS=20
    export LEARNING_RATE=2e-5
    export SAVER_METRIC="f1"
    export FREEZE_BN="True"
    export FREEZE_BACKBONE="False"
    export CLIP_GRAD=1.0
    export PCT_START=0.1
    export USE_SWA="True"
    export SWA_START=12
    export JITTER=0.25
    export DROPOUT=0.25
    export POOL_TYPE="attention"
    export USE_DANN="False"
    export DANN_LAMBDA=1.0
    export DANN_WEIGHT=1.0
}

function set_profile_SEARCHER11() {
    export NEG_WEIGHT=1.0
    export POS_WEIGHT=3.5
    export GAMMA=4.5
    export USE_FOCAL_LOSS="False"
    # Higher WD to prevent 100% Training Accuracy (Overfitting)
    export WEIGHT_DECAY=0.05
    export NUM_EPOCHS=20
    export DEEPHP_EPOCHS=20
    export LEARNING_RATE=2e-5
    export SAVER_METRIC="f1"
    export FREEZE_BN="True"
    export FREEZE_BACKBONE="False"
    export CLIP_GRAD=1.0
    export PCT_START=0.1
    export USE_SWA="True"
    export SWA_START=12
    export JITTER=0.25
    export DROPOUT=0.25
    export POOL_TYPE="attention"
    export USE_DANN="False"
    export DANN_LAMBDA=1.0
    export DANN_WEIGHT=1.0
}

function set_profile_SEARCHERDEEPHP() {
    export NEG_WEIGHT=1.0
    export POS_WEIGHT="6.0,2.5,1.5,8.0,2.5"
    export GAMMA=4.0
    export USE_FOCAL_LOSS="False"
    # Higher WD to prevent 100% Training Accuracy (Overfitting)
    export WEIGHT_DECAY=0.05
    export NUM_EPOCHS=20
    export DEEPHP_EPOCHS=20
    export BATCH_SIZE=64
    export LEARNING_RATE=1e-5
    export SAVER_METRIC="f1"
    export FREEZE_BN="False"
    export FREEZE_BACKBONE="False"
    export CLIP_GRAD=1.0
    export PCT_START=0.1
    export USE_SWA="True"
    export SWA_START=12
    export JITTER=0.25
    export DROPOUT=0.4
    export POOL_TYPE="attention"
    export USE_DANN="True"
    export DANN_LAMBDA=1.0
    export DANN_WEIGHT=1.0
    export USE_COMPILE="True"
}

function set_profile_TEST() {
    export NEG_WEIGHT=1.0
    export POS_WEIGHT=1.0 
    export GAMMA=3.0
    export USE_FOCAL_LOSS="False"
    export NUM_EPOCHS=1
    export DEEPHP_EPOCHS=1
    export BATCH_SIZE=64
    export LEARNING_RATE=2e-5
    export SAVER_METRIC="f1"
    export FREEZE_BN="True"
    export FREEZE_BACKBONE="False"
    export POOL_TYPE="attention"
    export USE_DANN="False"
    export DANN_LAMBDA=1.0
    export DANN_WEIGHT=1.0
}

function set_profile_TESTDEEPHP() {
    export NEG_WEIGHT=1.0
    export POS_WEIGHT="6.0,2.5,1.5,8.0,2.5"
    export GAMMA=4.0
    export USE_FOCAL_LOSS="False"
    export NUM_EPOCHS=1
    export DEEPHP_EPOCHS=1
    export BATCH_SIZE=64
    export LEARNING_RATE=1e-5
    export SAVER_METRIC="f1"
    export FREEZE_BN="False"
    export FREEZE_BACKBONE="False"
    export DROPOUT=0.4
    export POOL_TYPE="attention"
    export USE_DANN="True"
    export DANN_LAMBDA=1.0
    export DANN_WEIGHT=1.0
    export USE_COMPILE="True"
}
