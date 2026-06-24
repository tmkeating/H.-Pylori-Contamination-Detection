#!/usr/bin/env python3
"""
Generate Rescue Inference Comparison Report
============================================
Compares ensemble voting results with rescue inference results to produce
a detailed impact analysis without rerunning ensemble voting.

Usage:
    python3 generate_rescue_comparison.py --ensemble_dir <dir> --run_label <label>

Example:
    python3 generate_rescue_comparison.py \
        --ensemble_dir "finalResults/convnext_tiny_pretrained_backbone_34.4_weight_1.5_gamma_3.0_focalLoss_false" \
        --run_label "01-01"
"""
import pandas as pd
import numpy as np
import os
import glob
import argparse
from pathlib import Path


def generate_rescue_comparison(ensemble_dir, run_label):
    """Generate rescue comparison report from existing ensemble voting results."""
    
    # Load ensemble voting results
    report_file = os.path.join(ensemble_dir, f"ensemble_voting_report_{run_label}.csv")
    if not os.path.exists(report_file):
        print(f"Error: Ensemble voting report not found at {report_file}")
        return
    
    ensemble_df = pd.read_csv(report_file)
    print(f"✓ Loaded ensemble voting results: {len(ensemble_df)} patients")
    
    # Load rescue data
    rescue_dir = os.path.join(ensemble_dir, "rescue_ensemble")
    if not os.path.exists(rescue_dir):
        print(f"Error: Rescue directory not found at {rescue_dir}")
        return
    
    rescue_files = sorted(glob.glob(os.path.join(rescue_dir, "rescue_*_f[0-4].csv")))
    if not rescue_files:
        print(f"Error: No rescue files found in {rescue_dir}")
        return
    
    print(f"✓ Found {len(rescue_files)} rescue files")
    
    # Build rescue map: (PatientID, fold) -> Max_Prob
    rescue_map = {}
    for rf in rescue_files:
        # Extract fold number from filename: rescue_01_34.4_9077_f0.csv -> f0 -> 0
        fold_part = rf.split('_')[-1].replace('.csv', '')  # 'f0'
        fold_idx = int(fold_part[1:])
        
        rdf = pd.read_csv(rf)
        for _, row in rdf.iterrows():
            rescue_map[(row['PatientID'], fold_idx)] = row['Max_Prob']
    
    print(f"✓ Loaded {len(rescue_map)} rescue data points across folds")
    
    # Generate comparison
    rescue_comparisons = []
    
    for _, row in ensemble_df.iterrows():
        patient_id = row['PatientID']
        ensemble_prob = row['Max_Ensemble_Prob']
        ensemble_actual = row['Actual']
        ensemble_pred = row['Ensemble_Pred']
        
        # Find rescue probability for this patient (check all folds)
        rescue_probs = []
        for fold_idx in range(5):  # 5-fold CV
            if (patient_id, fold_idx) in rescue_map:
                rescue_probs.append(rescue_map[(patient_id, fold_idx)])
        
        if rescue_probs:
            rescue_prob = np.mean(rescue_probs)
            prob_change = rescue_prob - ensemble_prob
            pct_change = (prob_change / ensemble_prob * 100) if ensemble_prob > 0 else 0
            
            # Clinical interpretation
            status = ""
            finding = ""
            
            # Determine status based on actual vs predicted and prob change
            if ensemble_actual == 1:  # True positive or false negative
                if ensemble_pred == 1:
                    status = "✓ TP Confirmed"
                    finding = "True positive confirmed by rescue"
                else:
                    if pct_change > 50:
                        status = "✓ FN RECOVERED"
                        finding = f"BACTERIA FOUND - Dense windowing recovered bacteria (Ensemble {ensemble_prob:.3f} → Rescue {rescue_prob:.3f})"
                    elif pct_change > 20:
                        status = "⚠ FN Partial"
                        finding = f"Improved but detection still weak (Ensemble {ensemble_prob:.3f} → Rescue {rescue_prob:.3f})"
                    else:
                        status = "❌ FN Confirmed"
                        finding = f"Bacteria not recovered by rescue (Ensemble {ensemble_prob:.3f} → Rescue {rescue_prob:.3f})"
            else:  # True negative or false positive
                if ensemble_pred == 0:
                    status = "✓ TN Confirmed"
                    finding = "True negative confirmed by rescue"
                else:
                    if prob_change < -0.3:  # Large drop
                        status = "✓ Strong FP"
                        finding = f"CONFIRMED ARTIFACT - Confidence collapsed (Ensemble {ensemble_prob:.3f} → Rescue {rescue_prob:.3f}), indicating staining artifact not bacteria"
                    elif prob_change < -0.1:
                        status = "⚠ Weak FP"
                        finding = f"Likely false positive - Requires clinical review (Ensemble {ensemble_prob:.3f} → Rescue {rescue_prob:.3f})"
                    else:
                        status = "⚠ Borderline"
                        finding = f"Confirms borderline status (Ensemble {ensemble_prob:.3f} → Rescue {rescue_prob:.3f})"
            
            rescue_comparisons.append({
                'PatientID': patient_id,
                'Actual': ensemble_actual,
                'Ensemble_Pred': ensemble_pred,
                'Ensemble_Prob': ensemble_prob,
                'Rescue_Prob': rescue_prob,
                'Prob_Change': prob_change,
                'Pct_Change': pct_change,
                'Status': status,
                'Finding': finding
            })
    
    if rescue_comparisons:
        rescue_comp_df = pd.DataFrame(rescue_comparisons)
        
        # Print summary
        print("\n" + "="*80)
        print("📊 RESCUE INFERENCE IMPACT ANALYSIS")
        print("="*80)
        print(f"\n📋 Summary of Rescue Impact on {len(rescue_comp_df)} Patients:\n")
        
        for _, row in rescue_comp_df.iterrows():
            print(f"{row['Status']:<20} {row['PatientID']:<12} {row['Finding']}")
        
        # Save detailed comparison to CSV
        comp_csv = os.path.join(ensemble_dir, f"ensemble_voting_rescue_comparison_{run_label}.csv")
        rescue_comp_df.to_csv(comp_csv, index=False)
        print(f"\n✓ Rescue comparison report saved: {comp_csv}")
        print(f"  - Total patients analyzed: {len(rescue_comp_df)}")
        
        # Count categories
        recovered = len(rescue_comp_df[rescue_comp_df['Status'].str.contains('FN RECOVERED')])
        artifacts = len(rescue_comp_df[rescue_comp_df['Status'].str.contains('Strong FP')])
        tp_confirmed = len(rescue_comp_df[rescue_comp_df['Status'] == '✓ TP Confirmed'])
        tn_confirmed = len(rescue_comp_df[rescue_comp_df['Status'] == '✓ TN Confirmed'])
        
        print(f"  - FN RECOVERED: {recovered}")
        print(f"  - Strong FP (Artifacts): {artifacts}")
        print(f"  - TP Confirmed: {tp_confirmed}")
        print(f"  - TN Confirmed: {tn_confirmed}")
        
    else:
        print("No rescue data found for patients in ensemble report")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate rescue inference comparison report")
    parser.add_argument("--ensemble_dir", type=str, required=True,
                       help="Directory containing ensemble voting results")
    parser.add_argument("--run_label", type=str, required=True,
                       help="Run label (e.g., '01-01')")
    args = parser.parse_args()
    
    generate_rescue_comparison(args.ensemble_dir, args.run_label)
