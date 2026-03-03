import pandas as pd
import glob
import os
import argparse

def merge_searcher_ensemble(results_dir="results", run_ids=None):
    """
    Combines multiple model outputs (e.g., ConvNeXt and ResNet) using Union Logic.
    If EITHER model flags a patient (P > 0.1), the Ensemble flags them.
    """
    if run_ids is None:
        print("Please provide the RunIDs to ensemble (e.g. 167 for ConvNeXt, 172 for ResNet)")
        return

    # 1. Find all consensus files for the specified runs
    all_consensus = []
    for rid in run_ids:
        pattern = os.path.join(results_dir, f"{rid}_*_patient_consensus.csv")
        files = glob.glob(pattern)
        for f in files:
            fold = os.path.basename(f).split('_')[2]
            df = pd.read_csv(f)
            df['RunID'] = rid
            df['Fold'] = fold
            all_consensus.append(df)

    if not all_consensus:
        print("No consensus files found.")
        return

    full_df = pd.concat(all_consensus)
    
    # 2. Group by Patient and Fold to perform the Union
    # We want to keep the highest probability across models (The 'Optimist' Union)
    ensemble_df = full_df.groupby(['PatientID', 'Fold', 'Actual']).agg({
        'Max_Prob': 'max',
        'Mean_Prob': 'max',
        'Searcher_Flag': 'max' # Union Logic: 1 if ANY model is 1
    }).reset_index()

    # 3. Re-calculate metrics for the Ensemble
    # Searcher recall is based on Searcher_Flag (P > 0.1)
    positives = ensemble_df[ensemble_df['Actual'] == 1]
    negatives = ensemble_df[ensemble_df['Actual'] == 0]
    
    total_pos = len(positives)
    total_neg = len(negatives)
    
    tp = positives['Searcher_Flag'].sum()
    fp = negatives['Searcher_Flag'].sum()
    
    recall = tp / total_pos if total_pos > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"\n{'='*40}")
    print(f"{'ENSEMBLE SEARCHER PERFORMANCE':^40}")
    print(f"{'='*40}")
    print(f"Models: {run_ids}")
    print(f"Total Patients: {len(ensemble_df)}")
    print(f"Searcher Recall (Union): {recall:.4f} ({tp}/{total_pos})")
    print(f"Searcher Precision (Union): {precision:.4f} ({tp}/{tp+fp})")
    print(f"{'='*40}\n")
    
    # Identify persisting Ghost Patients (The ones both models missed)
    missed = positives[positives['Searcher_Flag'] == 0]
    if not missed.empty:
        print(f"Remaining Ghost Patients (Missed by BOTH models):")
        print(missed[['PatientID', 'Max_Prob']])
    else:
        print("Success! All positive patients caught by the Ensemble.")

    # Save ensemble result
    ensemble_df.to_csv(os.path.join(results_dir, "ensemble_searcher_results.csv"), index=False)
    print(f"Results saved to {results_dir}/ensemble_searcher_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs='+', required=True, help="List of RunIDs to ensemble")
    parser.add_argument("--dir", type=str, default="results")
    args = parser.parse_args()
    merge_searcher_ensemble(args.dir, args.runs)
