"""
# H. Pylori Result Summarization Utility
# --------------------------------------
# Aggregates individual evaluation reports from 5-fold cross-validation runs
# to generate a grand summary of the model's performance.
#
# What it does:
#   1. Looks for '*_evaluation_report.csv' files in a specified directory.
#   2. Parses key metrics (Accuracy, Precision(+), Recall(+), F1-Macro).
#   3. Calculates mean and standard deviation across folds to ensure stability.
#   4. Saves the final aggregation to 'grand_cv_summary.csv'.
#
# Usage:
#   python3 summarize_results.py --dir results --last 5
#
# Arguments:
#   --dir:  Directory containing the *_evaluation_report.csv files (Default: results).
#   --last: Only summarize the last N reports found (useful for 5-fold ensembles).
# --------------------------------------
"""
import pandas as pd
import glob
import os
import argparse

def generate_grand_summary(results_dir="results", last_n=None):
    # 1. Find all evaluation report CSVs
    # Pattern: *_f[0-4]_*_evaluation_report.csv
    report_files = sorted(glob.glob(os.path.join(results_dir, "*_f[0-4]_*_evaluation_report.csv")))
    
    if not report_files:
        print(f"No evaluation reports found in {results_dir}")
        return

    # Filter to only the most recent N files if requested
    if last_n:
        print(f"Filtering to the last {last_n} reports...")
        report_files = report_files[-last_n:]

    all_metrics = []
    
    print(f"\n{'='*60}")
    print(f"{'H. Pylori Iteration Summary':^60}")
    if last_n:
        print(f"{f'(Showing Last {last_n} Runs)':^60}")
    print(f"{'='*60}\n")

    for file in report_files:
        fold_name = os.path.basename(file).split('_')[2] 
        run_id = os.path.basename(file).split('_')[0]
        
        df = pd.read_csv(file, index_col=0)
        
        # Extract key metrics
        acc = df.loc['accuracy', 'precision'] # Accuracy is stored as precision/recall/f1 in the accuracy row
        pos_recall = df.loc['Positive', 'recall']
        pos_precision = df.loc['Positive', 'precision']
        neg_recall = df.loc['Negative', 'recall']
        f1_macro = df.loc['macro avg', 'f1-score']
        
        metrics = {
            'RunID': run_id,
            'Fold': fold_name,
            'Accuracy': acc,
            'Precision(+)': pos_precision,
            'Recall(+)': pos_recall,
            'Recall(-)': neg_recall,
            'F1_Macro': f1_macro
        }
        all_metrics.append(metrics)
        
        print(f"[{run_id} {fold_name}] Acc: {acc:.4f} | Prec(+): {pos_precision:.4f} | Rec(+): {pos_recall:.4f} | Rec(-): {neg_recall:.4f}")

    # 2. Calculate Averages
    summary_df = pd.DataFrame(all_metrics)
    # Ensure only numeric columns are selected for mean/std
    numeric_cols = ['Accuracy', 'Precision(+)', 'Recall(+)', 'Recall(-)', 'F1_Macro']
    averages = summary_df[numeric_cols].mean()
    stds = summary_df[numeric_cols].std()

    print(f"\n{'-'*60}")
    print(f"{'MEAN CROSS-VALIDATION RESULTS':^60}")
    print(f"{'-'*60}")
    print(f"Accuracy:     {averages['Accuracy']:.4f} ± {stds['Accuracy']:.4f}")
    print(f"Precision(+): {averages['Precision(+)']:.4f} ± {stds['Precision(+)']:.4f}")
    print(f"Recall(+):    {averages['Recall(+)']:.4f} ± {stds['Recall(+)']:.4f}")
    print(f"Recall(-):    {averages['Recall(-)']:.4f} ± {stds['Recall(-)']:.4f}")
    print(f"F1 Macro:     {averages['F1_Macro']:.4f} ± {stds['F1_Macro']:.4f}")
    print(f"{'='*60}\n")

    # 3. Save to CSV for long-term tracking
    # Detect run range for filename
    if all_metrics:
        run_ids = sorted([int(m['RunID']) for m in all_metrics])
        min_run, max_run = run_ids[0], run_ids[-1]
        run_suffix = f"_{min_run}-{max_run}"
    else:
        run_suffix = ""

    # Save individual fold records
    summary_filename = f"grand_cv_summary{run_suffix}.csv"
    summary_df.to_csv(os.path.join(results_dir, summary_filename), index=False)
    
    # Also save a 'grand_averages.csv' with the ± scores
    avg_stds_df = pd.DataFrame({
        'Metric': numeric_cols,
        'Mean': averages.values,
        'Std': stds.values,
        'Formatted': [f"{m:.4f} \u00b1 {s:.4f}" for m, s in zip(averages, stds)]
    })
    
    # Add Run Range metadata as a column or row
    if all_metrics:
        avg_stds_df['Run_Range'] = f"{min_run}-{max_run}"

    averages_filename = f"grand_cv_averages{run_suffix}.csv"
    avg_stds_df.to_csv(os.path.join(results_dir, averages_filename), index=False)
    
    print(f"Grand summary saved to {results_dir}/{summary_filename}")
    print(f"Grand averages with \u00b1 saved to {results_dir}/{averages_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="results")
    parser.add_argument("--last", type=int, default=None, help="Only summarize the last N reports")
    args = parser.parse_args()
    generate_grand_summary(args.dir, args.last)
