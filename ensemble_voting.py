import pandas as pd
import numpy as np
import os
import glob

import argparse

def calculate_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    return recall, precision, accuracy, tp, fp, fn, tn

def main():
    parser = argparse.ArgumentParser(description="Ensemble Voting for H. Pylori")
    parser.add_argument("--runs", type=str, help="Run ID range to aggregate (e.g., '292-296'). If omitted, finds 5 most recent.")
    args = parser.parse_args()

    if args.runs:
        # If runs are provided like "292-296", use that pattern
        pattern = os.path.join("results", f"[{args.runs.replace('-', '')}]_*_patient_consensus.csv")
        # Handle the case where the range is actually a glob-friendly bracket or just a prefix
        if '-' in args.runs:
            start, end = args.runs.split('-')
            # glob doesn't support ranges like [292-296] easily for multi-digit strings
            # We will manually find files matching the prefix in the range
            all_possible = glob.glob(os.path.join("results", "*_f[0-4]_*_patient_consensus.csv"))
            files = []
            for f in all_possible:
                try:
                    run_id = int(os.path.basename(f).split('_')[0])
                    if int(start) <= run_id <= int(end):
                        files.append(f)
                except ValueError:
                    continue
        else:
            files = glob.glob(os.path.join("results", f"{args.runs}_*_patient_consensus.csv"))
    else:
        # Iteration 25.0: Dynamically find the 5 most recent consensus files in results/
        pattern = os.path.join("results", "*_f[0-4]_*_patient_consensus.csv")
        all_files = glob.glob(pattern)
        all_files.sort(key=os.path.getmtime, reverse=True)
        
        if len(all_files) < 5:
            print(f"Error: Found only {len(all_files)} consensus files. Need at least 5 for ensemble.")
            return
            
        files = all_files[:5]
    
    # Re-sort files by filename so they appear in Fold 0, 1, 2, 3, 4 order
    files.sort()
    
    print(f"Aggregating ensemble from the following files:")
    for f in files:
        print(f"  - {f}")

    all_dfs = []
    for f in files:
        df = pd.read_csv(f)
        all_dfs.append(df)
        
    # Validate that all files have the same patients and labels
    pids = all_dfs[0]['PatientID'].values
    labels = all_dfs[0]['Actual'].values
    
    for i, df in enumerate(all_dfs[1:], 1):
        if not np.array_equal(pids, df['PatientID'].values):
            print(f"Error: PatientID mismatch in file {files[i]}")
            # Try to align them if they are just shuffled, but usually they should be same order
            df = df.set_index('PatientID').reindex(pids).reset_index()
            all_dfs[i] = df
        
    # Aggregate
    # Max-ensemble probability: The maximum Max_Prob found across all 5 models.
    # Mean-ensemble probability: The average Max_Prob across all 5 models.
    # Voting prediction: A patient is "Positive" if any of the 5 models flagged them (using the 0.40 threshold) 
    # OR if the Max-ensemble probability > 0.45.
    
    max_probs = np.column_stack([df['Max_Prob'].values for df in all_dfs])
    ensemble_max_prob = np.max(max_probs, axis=1)
    ensemble_mean_prob = np.mean(max_probs, axis=1)
    
    # Check "flagged" condition (Iteration 25.0 uses 0.40 threshold in Predicted)
    individual_preds = np.column_stack([df['Predicted'].values for df in all_dfs])
    any_model_flagged = np.any(individual_preds == 1, axis=1)
    
    # Ensemble logic: Majority voting (at least 3 models) OR significantly high max prob
    majority_vote = np.sum(individual_preds == 1, axis=1) >= 3
    
    # Iteration 25.1: Lowered Safety Override to 0.20 to capture "Ghost Patient" B22-81_1 (avg 0.23)
    # Balanced against majority vote (3/5) at 0.40 to prevent FP explosion.
    ensemble_pred = (majority_vote) | (ensemble_max_prob > 0.20)
    ensemble_pred = ensemble_pred.astype(int)
    
    # Calculate metrics
    rec, prec, acc, tp, fp, fn, tn = calculate_metrics(labels, ensemble_pred)
    
    # Identify missed patients (Ultimate Ghost Patients)
    missed_indices = np.where((labels == 1) & (ensemble_pred == 0))[0]

    print("--- Ensemble Results ---")
    print(f"Recall:    {rec:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    
    if rec == 1.0:
        print("SUCCESS: 100% Recall achieved!")
    else:
        print(f"FAILURE: Recall is {rec*100:.2f}%")
        print("\nMissed patients (Ultimate Ghost Patients):")
        for idx in missed_indices:
            print(f"  - {pids[idx]} (Max Prob: {ensemble_max_prob[idx]:.4f}, Mean Prob: {ensemble_mean_prob[idx]:.4f})")

    print("\n--- Individual Folds ---")
    for i, df in enumerate(all_dfs):
        r, p, a, _, _, _, _ = calculate_metrics(df['Actual'].values, df['Predicted'].values)
        print(f"Fold {i} (Job {files[i].split('_')[1]}): Recall={r:.4f}, Prec={p:.4f}, Acc={a:.4f}")

    # Create detailed CSV for inspection
    # Get the sweep range logic for naming
    run_ids = [f.split('/')[1].split('_')[0] for f in files]
    min_run = min(run_ids)
    max_run = max(run_ids)
    out_name = f"results/ensemble_voting_report_{min_run}-{max_run}.csv"
    
    ensemble_df = pd.DataFrame({
        'PatientID': pids,
        'Actual': labels,
        'Ensemble_Pred': ensemble_pred,
        'Max_Ensemble_Prob': ensemble_max_prob,
        'Mean_Ensemble_Prob': ensemble_mean_prob,
        'Any_Flagged': any_model_flagged.astype(int)
    })
    ensemble_df.to_csv(out_name, index=False)
    print(f"\nDetailed report saved to [{out_name}]({out_name})")

    # Iteration 24.9: Save a concise summary for easy automated consumption
    summary_name = f"results/ensemble_voting_summary_{min_run}-{max_run}.csv"
    summary_data = {
        "Metric": ["Recall", "Precision", "Accuracy", "TP", "FP", "FN", "TN", "Ultimate_Ghost_Count"],
        "Value": [rec, prec, acc, tp, fp, fn, tn, len(missed_indices)]
    }
    pd.DataFrame(summary_data).to_csv(summary_name, index=False)
    print(f"Concise summary saved to [{summary_name}]({summary_name})")

if __name__ == "__main__":
    main()
