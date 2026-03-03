# Model Profiles for H. Pylori Detection
# Format: PROFILE_NAME=(POS_WEIGHT GAMMA SAVER_METRIC DROPOUT SWA_LR)

# 1. Auditor: Focuses on 100% Precision (Clinical Grade)
AUDITOR=(2.2 2.0 "loss" 0.5 1e-5)

# 2. Searcher: Focuses on 100% Recall (Triage Stage)
# Note: Iteration 19 calibration for ResNet stability
SEARCHER=(3.5 2.0 "f1" 0.5 1e-5)

# 3. Extreme (Original Iteration 17): High-penalty recall
EXTREME=(7.5 1.0 "recall" 0.5 1e-5)
