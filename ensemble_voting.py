import pandas as pd
import numpy as np
import os
import glob

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
    files = glob.glob("results/27[2-6]_*_patient_consensus.csv")
    files.sort()
    
    if len(files) != 5:
        print(f"Warning: Expected 5 files, found {len(files)}")
        print(files)

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
    # Voting prediction: A patient is "Positive" if any of the 5 models flagged them (using the 0.07 threshold) 
    # OR if the Max-ensemble probability > 0.10.
    
    max_probs = np.column_stack([df['Max_Prob'].values for df in all_dfs])
    ensemble_max_prob = np.max(max_probs, axis=1)
    ensemble_mean_prob = np.mean(max_probs, axis=1)
    
    # Check "flagged" condition (usually Predicted or Searcher_Flag represents the 0.07 threshold in these reports)
    # The prompt says: "any of the 5 models flagged them (using the 0.07 threshold)"
    # Let's verify if 'Predicted' uses 0.07. 
    # Usually in these Iteration 24 series, 0.07 is the threshold for Searcher_Flag or Predicted.
    
    individual_preds = np.column_stack([df['Predicted'].values for df in all_dfs])
    any_model_flagged = np.any(individual_preds == 1, axis=1)
    
    ensemble_pred = (any_model_flagged) | (ensemble_max_prob > 0.10)
    ensemble_pred = ensemble_pred.astype(int)
    
    # Calculate metrics
    rec, prec, acc, tp, fp, fn, tn = calculate_metrics(labels, ensemble_pred)
    
    print("--- Ensemble Results ---")
    print(f"Recall:    {rec:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    
    if rec == 1.0:
        print("SUCCESS: 100% Recall achieved!")
    else:
        print(f"FAILURE: Recall is {rec*100:.2f}%")
        # Identify missed patients
        missed_indices = np.where((labels == 1) & (ensemble_pred == 0))[0]
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
