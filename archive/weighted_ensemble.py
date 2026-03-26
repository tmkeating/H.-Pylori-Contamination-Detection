import pandas as pd
import numpy as np
import argparse
import os
import glob
from sklearn.metrics import classification_report, confusion_matrix

def load_patient_results(results_dir):
    """
    Search for all *_patient_consensus.csv files in the directory and load them.
    Returns a combined dataframe with multi-model probabilities.
    """
    pattern = os.path.join(results_dir, "*_patient_consensus.csv")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No *_patient_consensus.csv files found in {results_dir}")
    
    # We want to aggregate the 'Max_Prob' from each fold/model
    all_dfs = []
    for f in files:
        df = pd.read_csv(f)
        # Identify the fold or model index from filename if possible
        # Typically looks like '167_104966_f0_..._patient_consensus.csv'
        model_id = os.path.basename(f).split('_')[2] if 'f' in os.path.basename(f) else os.path.basename(f)
        
        # Keep only relevant columns and rename Max_Prob to be unique per fold
        df = df[['PatientID', 'Actual', 'Max_Prob']]
        df = df.rename(columns={'Max_Prob': f'Max_Prob_{model_id}'})
        all_dfs.append(df.set_index('PatientID'))
    
    # Join all models on PatientID
    combined = pd.concat(all_dfs, axis=1)
    
    # Remove duplicate 'Actual' columns and keep one
    actual_cols = [col for col in combined.columns if 'Actual' in col]
    # Handle potentially multiple columns from concat by taking the slice carefully
    combined['GroundTruth'] = combined[actual_cols].iloc[:, 0]
    combined = combined.drop(columns=actual_cols)
    
    # Compute mean Max_Prob across all source models
    prob_cols = [col for col in combined.columns if 'Max_Prob' in col]
    combined['Mean_Prob'] = combined[prob_cols].mean(axis=1)
    combined['Any_High_Conf'] = combined[prob_cols].max(axis=1)
    
    return combined

def main():
    parser = argparse.ArgumentParser(description="Multi-Stage Searcher-Auditor-Accuracy Ensemble")
    parser.add_argument("--search_dir", required=True, help="Directory containing Searcher (High Recall) results")
    parser.add_argument("--audit_dir", required=True, help="Directory containing Auditor (High Precision) results")
    parser.add_argument("--accuracy_dir", help="Directory containing High Accuracy (ResNet50) results")
    parser.add_argument("--s_pass_th", type=float, default=0.10, help="Searcher first-pass minimum probability")
    parser.add_argument("--w_search", type=float, default=0.40, help="Weight for Searcher probability")
    parser.add_argument("--w_audit", type=float, default=0.60, help="Weight for Auditor probability")
    parser.add_argument("--w_acc", type=float, default=0.0, help="Weight for Accuracy model (if directory provided)")
    parser.add_argument("--final_th", type=float, default=0.35, help="Final weighted probability threshold")
    parser.add_argument("--audit_safety_th", type=float, default=0.70, help="Auditor individual-run safety threshold")
    parser.add_argument("--s_safety_th", type=float, default=0.50, help="Searcher high-confidence safety threshold (Overrule Auditor)")
    
    args = parser.parse_args()
    
    print(f"--- Loading Searcher Results from: {args.search_dir} ---")
    df_search = load_patient_results(args.search_dir)
    
    print(f"--- Loading Auditor Results from: {args.audit_dir} ---")
    df_audit = load_patient_results(args.audit_dir)

    df_acc = None
    if args.accuracy_dir:
        print(f"--- Loading Accuracy Results from: {args.accuracy_dir} ---")
        df_acc = load_patient_results(args.accuracy_dir)
    
    # Merge Searcher and Auditor results
    df_final = pd.DataFrame(index=df_search.index)
    df_final['Actual'] = df_search['GroundTruth']
    df_final['P_search'] = df_search['Mean_Prob']
    df_final['P_audit'] = df_audit['Mean_Prob']
    df_final['Audit_Safety'] = df_audit['Any_High_Conf']
    
    if df_acc is not None:
        df_final['P_acc'] = df_acc['Mean_Prob']
    
    # Implement Multi-Stage Logic
    
    # Stage 1: Searcher First Pass
    df_final['Stage1_Pass'] = df_final['P_search'] >= args.s_pass_th
    
    # Stage 2: Weighted Score
    if df_acc is not None:
        total_w = args.w_search + args.w_audit + args.w_acc
        df_final['Weighted_Score'] = (args.w_search * df_final['P_search'] + 
                                     args.w_audit * df_final['P_audit'] + 
                                     args.w_acc * df_final['P_acc']) / total_w
    else:
        df_final['Weighted_Score'] = (args.w_search * df_final['P_search']) + (args.w_audit * df_final['P_audit'])
    
    # Stage 3: Final Prediction
    # Positive if (Stage1_Pass AND Score > final_th) OR (Audit_Safety > safety_th) OR (P_search > s_safety_th)
    # The Safety Net handles cases where a single Auditor fold is extremely sure
    # The Searcher Safety Net (Overrule) ensures high-confidence Searcher hits aren't vetoed
    stage2_pred = (df_final['Stage1_Pass']) & (df_final['Weighted_Score'] >= args.final_th)
    a_safety_pred = (df_final['Audit_Safety'] >= args.audit_safety_th)
    s_safety_pred = (df_final['P_search'] >= args.s_safety_th)
    
    df_final['Predicted'] = (stage2_pred | a_safety_pred | s_safety_pred).astype(int)
    
    # Metrics Calculation
    y_true = df_final['Actual']
    y_pred = df_final['Predicted']
    
    print("\n" + "="*40)
    print("      WEIGHTED ENSEMBLE REPORT")
    print("="*40)
    print(classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.flatten()
    
    print(f"Confusion Matrix:\nTN: {tn} | FP: {fp}\nFN: {fn} | TP: {tp}")
    print(f"\nFinal Statistics:")
    print(f"Recall: {100.0 * tp / (tp + fn):.2f}%")
    print(f"Precision: {100.0 * tp / (tp + fp):.2f}%")
    
    # Identify False Negatives (Crucial for Recall debugging)
    fn_patients = df_final[ (df_final['Actual'] == 1) & (df_final['Predicted'] == 0) ].index.tolist()
    if fn_patients:
        print(f"\n⚠️ WARNING: False Negatives Detected: {fn_patients}")
    else:
        print(f"\n✅ 100% RECALL MAINTAINED.")

    # Save detailed report
    report_path = "weighted_ensemble_summary.csv"
    df_final.to_csv(report_path)
    print(f"\nDetailed report saved to: {report_path}")

    # NEW: Save summarized statistics
    stats_path = "weighted_ensemble_stats.csv"
    stats_data = {
        "Metric": ["Recall", "Precision", "Accuracy", "TN", "FP", "FN", "TP", "Total"],
        "Value": [
            f"{100.0 * tp / (tp + fn):.2f}%",
            f"{100.0 * tp / (tp + fp):.2f}%",
            f"{100.0 * (tp + tn) / (tp + tn + fp + fn):.2f}%",
            tn, fp, fn, tp, (tp + tn + fp + fn)
        ],
        "Searcher_Dir": [args.search_dir] * 8,
        "Auditor_Dir": [args.audit_dir] * 8
    }
    pd.DataFrame(stats_data).to_csv(stats_path, index=False)
    print(f"Summarized statistics saved to: {stats_path}")

if __name__ == "__main__":
    main()
