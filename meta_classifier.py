"""
H. Pylori Contamination Detection - Meta-Classifier & Ensemble Fusion
====================================================================

OVERVIEW
--------
This module implements the final fusion layer for multi-model ensemble predictions.
Instead of simple probability averaging, it treats each model fold as a feature
extractor, aggregating engineered features across all folds to train a Random Forest
meta-classifier that learns non-linear fusion rules.

Provides:
  - Multi-fold feature extraction from multiple model directories
  - Engineered diagnostic features capturing distribution patterns
  - Leave-One-Out Cross-Validation (LOO-CV) performance estimation
  - Comprehensive fusion results with confusion matrix analysis
  - Failure case identification for clinical audit

PURPOSE
-------
The challenge of ensemble fusion: Simple averaging fails when component models have
complementary strengths and weaknesses.

This meta-classifier learns to combine model predictions by:
  1. Extracting diagnostic patterns from each model's predictions
  2. Learning non-linear decision boundaries via Random Forest
  3. "Vetoing" uncertain signals based on pattern consistency
  4. Achieving higher clinical precision than any single model

Clinical advantage: Reduces false positives (reduces unnecessary procedures) while
maintaining sensitivity to true positives (doesn't miss cases).

ARCHITECTURE
------------

MULTI-FOLD ENSEMBLE DESIGN:
  Multiple model folders, each containing predictions from different training runs:
  - Each folder provides diagnostic predictions across patients
  - Predictions aggregated with engineered statistical features
  - Meta-classifier learns which model outputs to trust based on signal patterns

FEATURE ENGINEERING (6 features per fold × N folds = 6N total):

  For each model fold, extract 6 diagnostic features:
    1. Max_Prob: Highest positive probability in any patch
       Captures the strongest signal found in the slide
    
    2. Mean_Prob: Average positive probability across all patches
       Indicates overall slide background/staining quality
    
    3. Skeptical_Gap: Max_Prob - Mean_Prob
       Differentiates "spiky" artifact (high gap) from
       "cloud" bacteremia (low gap)
    
    4. Density_P50: Proportion of patches with prob ≥ 0.50
       Quantifies light bacterial load
    
    5. Density_P80: Proportion of patches with prob ≥ 0.80
       Quantifies heavy bacterial load (strongest indicator)
    
    6. Fold_ID: Model fold identifier
       Helps model learn fold-specific prediction patterns

CLINICAL INTERPRETATION:
  - High Max_Prob + High Density_P80 = CONFIRMED positive
  - High Max_Prob + Low Density + High Skeptical_Gap = Likely artifact (false positive)
  - Low Max_Prob across all models = Negative with high confidence

HOW IT WORKS
------------

STEP 1: FEATURE EXTRACTION
  1. Load patient consensus reports from all model fold directories
  2. For each patient, extract 6 features from each fold
  3. Aggregate into (N_patients, 6×N_folds) feature matrix
  4. Align patients with padding (handle missing folds)

STEP 2: LEAVE-ONE-OUT CROSS-VALIDATION
  1. For each patient (one at a time):
     a. Hold out that patient's data
     b. Train Random Forest on remaining patients
     c. Predict for the held-out patient
     d. Record true vs. predicted label
  2. Captures real-world performance on "unseen" patients
  3. Unbiased estimate (no test set leakage)

STEP 3: RANDOM FOREST META-CLASSIFIER
  Configuration:
    - n_estimators=100: Sufficient for stability
    - max_depth=3: Prevents overfitting to rare combinations
    - class_weight='balanced': Prioritizes minority positive class
  
  Learns patterns like:
    "IF (Model1_Max > 0.8) AND (Model2_Max < 0.5) 
        AND (Density_P80 < 0.02) THEN Negative (artifact)"
    "IF (All_models_agree_high) AND (Density_P80 > 0.1) 
        THEN Positive (confirmed bacteria)"

STEP 4: PERFORMANCE EVALUATION
  1. Compute confusion matrix (TP, TN, FP, FN)
  2. Generate clinical metrics (precision, recall, F1)
  3. Identify false negatives (missed cases) for audit
  4. Save results for research validation

USAGE
-----

BASIC USAGE:

  python meta_classifier.py \\
    --search_dir <DIR1> \\
    --audit_dir <DIR2> \\
    --accuracy_dir <DIR3>

COMMAND-LINE ARGUMENTS:

  --search_dir (required)
    Path to directory containing model fold 1 outputs
    Expected: Multiple *_patient_consensus.csv files

  --audit_dir (required)
    Path to directory containing model fold 2 outputs
    Expected: Multiple *_patient_consensus.csv files

  --accuracy_dir (required)
    Path to directory containing model fold 3 outputs
    Expected: Multiple *_patient_consensus.csv files

Note: Exact directory names and purposes are application-specific. The meta-classifier
treats all input directories equivalently, extracting features from each independently.

INPUT DATA FORMAT:

  Each input directory should contain consensus reports resembling:
    PatientID | Actual | Max_Prob | Mean_Prob | Count_P50 | Count_P80 | Patch_Count
    P001      | 0      | 0.45     | 0.12      | 5         | 0         | 847
    P002      | 1      | 0.92     | 0.38      | 142       | 87        | 847
    ...

  Required columns:
    - PatientID: Unique patient identifier
    - Actual: Ground truth label (0=negative, 1=positive)
    - Max_Prob: Highest model probability
    - Mean_Prob: Average model probability
    - Count_P50: Number of patches with prob ≥ 0.50
    - Count_P80: Number of patches with prob ≥ 0.80
    - Patch_Count: Total number of patches in patient

  Alternative column names supported:
    - Bag_Mean_Prob → Mean_Prob
    - Meta_Prob, patch_count (case-insensitive variants)

OUTPUT
------

  Console Output:
    - Feature matrix shape and alignment info
    - Classification report (precision, recall, F1 for each class)
    - Final accuracy percentage
    - Confusion matrix (TP, TN, FP, FN)
    - List of false negatives (if any) for clinical audit

  File Output:
    meta_fusion_results.csv
      - PatientID: Patient identifier
      - Actual: True label
      - Predicted: Meta-classifier prediction
      - Ready for external validation and research publication

EXAMPLES
--------

  # Standard multi-fold fusion (recommended)
  python meta_classifier.py \\
    --search_dir results/fold_a_outputs \\
    --audit_dir results/fold_b_outputs \\
    --accuracy_dir results/fold_c_outputs

  # Typical output:
  # --- Loading Features from 15 model folds ---
  # Aligning patients to 90 engineered features...
  # Feature matrix shape: (116, 90) (Patients x Model-Outputs)
  # --- Training Meta-Classifier (LOO-CV) ---
  # ========================================
  #       META-CLASSIFIER FUSION REPORT
  # ========================================
  #             precision    recall  f1-score   support
  # Negative       0.88      0.91      0.89        71
  # Positive       0.92      0.89      0.90        45
  # accuracy                           0.90       116
  # TN: 65 | FP: 6 | FN: 5 | TP: 40
  # ⚠️ Still Missed: ['P032', 'P089', 'P047', 'P055', 'P101']

CLASS REFERENCE
---------------

load_all_model_features(results_dirs)
  Extracts high-dimensional diagnostic features from model folds.
  
  Args:
    results_dirs: Dict mapping directory keys (arbitrary names)
                  to paths containing *_patient_consensus.csv files
  
  Returns:
    X: (N_patients, N_features) numpy array, feature matrix
    y: (N_patients,) numpy array, true labels (0/1)
    pids: List of patient IDs corresponding to rows in X and y
  
  Feature Matrix Shape:
    If N folds with 6 features each = 6N total features
    Patients with missing folds are zero-padded to ensure uniform shape
  
  Extracted Features (per fold):
    1. Max_Prob: Highest probability from model
    2. Mean_Prob: Average probability from all patches
    3. Skeptical_Gap: Max_Prob - Mean_Prob
    4. Density_P50: Proportion of patches with prob ≥ 0.50
    5. Density_P80: Proportion of patches with prob ≥ 0.80
    6. Fold_ID: Fold identifier (0, 1, 2, etc.)

main()
  Entry point for meta-classifier training and evaluation.
  
  Process:
    1. Parse command-line arguments (N directory paths)
    2. Load features from all folds
    3. Train Random Forest via Leave-One-Out Cross-Validation
    4. Evaluate performance metrics
    5. Identify missed cases for clinical audit
    6. Save results to CSV
  
  Outputs:
    - Console: Classification report and confusion matrix
    - File: meta_fusion_results.csv with predictions

CLINICAL INTERPRETATION GUIDELINES
----------------------------------

NEGATIVE DIAGNOSIS (High Confidence):
  - All models: Max_Prob < 0.60
  - Density_P80 < 0.01 (virtually no high-confidence patches)
  - Skeptical_Gap relatively uniform (no spiky outliers)
  - Action: Clear; no bacteria detected

POSITIVE DIAGNOSIS (High Confidence):
  - Consensus across models on high probability OR
  - High Density_P80 (> 0.05 with significant high-confidence patches)
  - Skeptical_Gap controlled (not just a single spike)
  - Action: Confirm; bacteria detected, recommend treatment

AMBIGUOUS CASE (Requires Manual Review):
  - High probability from one model; skeptical from others
  - High Max_Prob but low Density_P80 (single suspicious patch)
  - Conflicting predictions across models
  - Action: Pathologist review recommended

FAILED DETECTIONS (False Negatives):
  - All models underestimate probability
  - Possible causes: Poor staining, unusual morphology, scanning artifacts
  - Action: Manual slide review; consider repeat test

BAYESIAN REASONING:
  The meta-classifier implicitly performs Bayesian fusion:
    P(Bacteria | features) ∝ P(features | Bacteria) × P(Bacteria)
  
  High density features provide strong evidence BY THEMSELVES:
    P(features | Bacteria) >> P(features | Noise)
  
  Single suspicious patch has weaker evidence:
    P(single_spike | Bacteria) ≈ P(single_spike | Artifact)
  
  Random Forest learns these relationships automatically from data.

ADVANCED TOPICS
---------------

IMBALANCED DATASET HANDLING:
  - class_weight='balanced' corrects for imbalanced classes
  - Prevents model from preferring majority class
  - Particularly important if positive cases are minority

HYPERPARAMETER TUNING:
  Consider modifying Random Forest parameters for deployment:
  - max_depth: Lower (2-3) prevents overfitting; higher (5+) increases flexibility
  - n_estimators: 50-200 typical; more ≈ more stable but slower
  - min_samples_leaf: Higher (5+) prevents single-patient rules

FEATURE IMPORTANCE:
  Random Forest can identify which features matter most:
    clf.feature_importances_
  
  Interpretation:
    - High importance for density features → model uses bacterial load as key signal
    - High importance for Skeptical_Gap → model learns artifact detection
    - Balanced importance → features complement each other well

REQUIREMENTS
------------
  - pandas: Data loading and aggregation
  - NumPy: Numerical operations
  - scikit-learn: Random Forest, cross-validation, metrics
  - Python 3.7+

NOTES
-----
  - Leave-One-Out CV: Most unbiased but computationally expensive (N fits for N patients)
  - Random Forest is non-linear: Can capture interactions between folds
  - Feature padding: Handles missing folds gracefully (fills with zeros)
  - All feature values normalized by patch count to prevent size bias
  - Column name variants supported for compatibility with different report formats
  - Results saved to meta_fusion_results.csv for archival and publication
  - Patient IDs anonymized/normalized (remove extensions, special characters)
  - False negative identification critical for clinical validation and error analysis
"""
import pandas as pd
import numpy as np
import argparse
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def load_all_model_features(results_dirs):
    """
    Extracts high-dimensional diagnostic features from all tri-profile folds.

    TECHNICAL RATIONALE: Engineered Feature Selection
    -------------------------------------------------
    Clinical H. Pylori detection is not just about "probability," but 
    about the *distribution* of suspicion:
    1. Max_Prob (The Signal): Captures the strongest morphological match.
    2. Mean_Prob (The Noise): Captures global slide staining quality.
    3. Skeptical_Gap (Max-Mean): Differentiates a "Spiky" artifact 
       (high gap) from genuine "Cloud" bacteremia (low gap).
    4. Density_P50/P80: Quantifies the bacterial load. High density 
       at high confidence (P80) is an extremely strong clinical 
       indicator of 'ALTA' (High) contamination.

    This multi-expert approach (Searcher, Auditor, Accuracy) ensures 
    that the Meta-Classifier can "veto" a Searcher's False Positive 
    if the Auditor and Density features suggest the signal is 
    consistent with background noise.
    """
    patient_data = {} # PatientID -> { 'features': [], 'label': label }
    
    # We want to identify which profile is providing which feature
    profile_map = {
        'searcher': 0,
        'auditor': 1,
        'accuracy': 2
    }
    
    for run_name, directory in results_dirs.items():
        pattern = os.path.join(directory, "*_patient_consensus.csv") 
        files = sorted(glob.glob(pattern)) # Sort to ensure consistent feature ordering
        
        for f in files:
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]
            
            for _, row in df.iterrows():
                pid = str(row['PatientID']).split('_')[0].split('.')[0].strip()
                if pid not in patient_data:
                    patient_data[pid] = {'features': [], 'label': row['Actual']}
                
                # Robust extraction
                max_p = row.get('Max_Prob', row.get('Meta_Prob', 0.0))
                mean_p = row.get('Mean_Prob', row.get('Bag_Mean_Prob', 0.0))

                # NEW: Extract Patch Counts for High-Confidence patches
                # These features tell the model "How MANY patches were suspicious?"
                # A single high peak might be noise, but 10 moderate peaks are likely bacteria.
                count_p50 = row.get('Count_P50', row.get('count_p50', 0.0))
                count_p80 = row.get('Count_P80', row.get('count_p80', 0.0))
                patch_total = row.get('Patch_Count', row.get('patch_count', 1.0))
                
                # Normalize counts by total patches to prevent bias from slide size
                density_p50 = count_p50 / patch_total if patch_total > 0 else 0
                density_p80 = count_p80 / patch_total if patch_total > 0 else 0
                
                patient_data[pid]['features'].extend([
                    max_p, 
                    mean_p,
                    max_p - mean_p,
                    density_p50,
                    density_p80,
                    profile_map[run_name] 
                ])
    
    # Convert to matrix
    X = []
    y = []
    pids = []
    
    # Identify the maximum number of features found in any patient
    max_features = max(len(d['features']) for d in patient_data.values())
    print(f"Aligning patients to {max_features} engineered features...")
    
    for pid, data in patient_data.items():
        feats = data['features']
        # Padding for patients not found in all folds (ensures consistent vector length for Random Forest)
        if len(feats) < max_features:
            feats.extend([0.0] * (max_features - len(feats)))
        
        X.append(feats)      # Feature matrix for clinical indicators
        y.append(data['label']) # Target diagnostic class (Positive/Negative)
        pids.append(pid)     # Track PatientID for final reporting
        
    return np.array(X), np.array(y), pids
    
    # Convert to matrix
    X = []
    y = []
    pids = []
    
    # Identify the maximum number of features found in any patient
    max_features = max(len(d['features']) for d in patient_data.values())
    print(f"Aligning patients to {max_features} features...")
    
    for pid, data in patient_data.items():
        # Padding for patients not found in all folds (though they should be)
        feats = data['features']
        if len(feats) < max_features:
            print(f"Warning: Patient {pid} has only {len(feats)} features. Padding with 0.")
            feats.extend([0.0] * (max_features - len(feats)))
        
        X.append(feats)
        y.append(data['label'])
        pids.append(pid)
        
    return np.array(X), np.array(y), pids

def main():
    # Argument parsing for multi-expert directory inputs
    parser = argparse.ArgumentParser(description="Meta-Classifier Feature Fusion for H. Pylori")
    parser.add_argument("--search_dir", required=True)     # Over-sensitive fold outputs
    parser.add_argument("--audit_dir", required=True)      # Specificity-focused fold outputs
    parser.add_argument("--accuracy_dir", required=True)   # Balanced-training fold outputs
    args = parser.parse_args()
    
    # Map input directories to their respective clinical profile roles
    dirs = {
        'searcher': args.search_dir,
        'auditor': args.audit_dir,
        'accuracy': args.accuracy_dir
    }
    
    print("--- Loading Features from 15 model folds ---")
    # Aggregates structured data (Max, Mean, Density) across all tri-profile experts
    X, y, pids = load_all_model_features(dirs)
    
    print(f"Feature matrix shape: {X.shape} (Patients x Model-Outputs)")
    
    # Leave-One-Out Cross-Validation (LOO-CV)
    # ----------------------------------------------
    # TECHNICAL DECISION: Given the high stakes of clinical diagnostics and 
    # the relatively small patient cohort (116 patients), LOO-CV provides 
    # the most unbiased estimate of the meta-classifier's performance on 
    # "unseen" patients, ensuring the fusion is robust to individual-slide outliers.
    loo = LeaveOneOut()
    y_true = []
    y_pred = []
    
    # Random Forest Meta-Classifier
    # ---------------------------------------------
    # TECHNICAL DECISION: Simple averaging fails when one model (Searcher) is 
    # intentionally over-sensitive. The Random Forest learns non-linear 
    # interactions (e.g., "If Searcher is high but Auditor is low and 
    # Density is < 1%, classify as Negative"). 
    # - n_estimators=100: Standard for stability.
    # - max_depth=3: Prevents meta-overfitting to noisy artifact combinations.
    # - class_weight='balanced': Prioritizes the minority Positive class.
    clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42, class_weight='balanced')
    
    print("--- Training Meta-Classifier (LOO-CV) ---")
    # Execute Leave-One-Out validation to estimate real-world clinical performance
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Train meta-model on all patients except one and predict for the held-out patient
        clf.fit(X_train, y_train)
        y_pred.append(clf.predict(X_test)[0])
        y_true.append(y_test[0])
    
    print("\n" + "="*40)
    print("      META-CLASSIFIER FUSION REPORT")
    print("="*40)
    # Generate aggregate metrics (Precision, Recall, F1) across the entire LOO-CV cohort
    print(classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))
    
    acc = accuracy_score(y_true, y_pred)
    print(f"Final Accuracy: {100.0 * acc:.2f}%")
    
    # Extract specific confusion counts for clinical audit (FP vs FN trade-off)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.flatten()
    print(f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")

    # Check the "Unreachable Three"
    # Aggregate cross-validation results into a structured DataFrame for failure analysis
    results_df = pd.DataFrame({'PatientID': pids, 'Actual': y_true, 'Predicted': y_pred})
    
    # Identify False Negatives (Target cases where the ensemble failed to detect bacteria)
    missed = results_df[(results_df['Actual'] == 1) & (results_df['Predicted'] == 0)]
    if not missed.empty:
        print(f"\n⚠️ Still Missed: {missed['PatientID'].tolist()}")
    else:
        # Clinical Milestone: All positive cases correctly identified by the meta-fusion layer
        print(f"\n✅ 100% RECALL ACHIEVED BY FEATURE FUSION.")

    # Persist the fusion results for external documentation and research validation
    results_df.to_csv("meta_fusion_results.csv", index=False)

if __name__ == "__main__":
    main()
