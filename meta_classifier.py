"""
# H. Pylori Meta-Classifier & Feature Fusion Engine
# ------------------------------------------------
# Implements the final fusion layer for the H. Pylori diagnostic pipeline.
# Instead of simple probability averaging, this script treats each model 
# fold (Searcher, Auditor, Accuracy) as a high-dimensional feature extractor.
#
# What it does:
#   1. Aggregates 90 specialized features (15 folds x 6 features) per patient:
#      - Peak Probability: Highest signal found in any patch.
#      - Mean Probability: Background confidence.
#      - Density P50/P80: Percentage of suspicious vs. high-confidence patches.
#        (Differentiates sparse bacterial load from single-patch noise).
#      - Skeptical Gap: Peak-to-Mean difference (detects "spiky" artifacts).
#   2. Trains a Random Forest meta-classifier using Leave-One-Out (LOO) CV 
#      to find the non-linear boundary that separates bacteria from artifacts.
#   3. Identifies the "Unreachable" patients remaining after fusion.
#
# Usage:
#   python3 meta_classifier.py --search_dir finalResults/searcher \
#                               --audit_dir finalResults/auditor \
#                               --accuracy_dir finalResults/accuracy_extra
# ------------------------------------------------
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
        if len(feats) < max_features:
            feats.extend([0.0] * (max_features - len(feats)))
        
        X.append(feats)
        y.append(data['label'])
        pids.append(pid)
        
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
    parser = argparse.ArgumentParser(description="Meta-Classifier Feature Fusion for H. Pylori")
    parser.add_argument("--search_dir", required=True)
    parser.add_argument("--audit_dir", required=True)
    parser.add_argument("--accuracy_dir", required=True)
    args = parser.parse_args()
    
    dirs = {
        'searcher': args.search_dir,
        'auditor': args.audit_dir,
        'accuracy': args.accuracy_dir
    }
    
    print("--- Loading Features from 15 model folds ---")
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
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf.fit(X_train, y_train)
        y_pred.append(clf.predict(X_test)[0])
        y_true.append(y_test[0])
    
    print("\n" + "="*40)
    print("      META-CLASSIFIER FUSION REPORT")
    print("="*40)
    print(classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))
    
    acc = accuracy_score(y_true, y_pred)
    print(f"Final Accuracy: {100.0 * acc:.2f}%")
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.flatten()
    print(f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")

    # Check the "Unreachable Three"
    results_df = pd.DataFrame({'PatientID': pids, 'Actual': y_true, 'Predicted': y_pred})
    missed = results_df[(results_df['Actual'] == 1) & (results_df['Predicted'] == 0)]
    if not missed.empty:
        print(f"\n⚠️ Still Missed: {missed['PatientID'].tolist()}")
    else:
        print(f"\n✅ 100% RECALL ACHIEVED BY FEATURE FUSION.")

    results_df.to_csv("meta_fusion_results.csv", index=False)

if __name__ == "__main__":
    main()
