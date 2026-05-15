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
#   4. Computes bootstrap confidence intervals across folds.
#   5. Saves summaries to 'grand_cv_summary.csv' and 'grand_cv_averages.csv'.
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
import numpy as np
import glob
import os
import argparse

def compute_bootstrap_ci_from_folds(summary_df, numeric_cols, n_bootstrap=500):
    """
    Compute bootstrap confidence intervals by resampling across folds.
    
    Args:
        summary_df: DataFrame with fold-level metrics
        numeric_cols: List of metric column names to compute CIs for
        n_bootstrap: Number of bootstrap resamples
    
    Returns:
        Dictionary with CI statistics for each metric
    """
    n_folds = len(summary_df)
    bootstrap_stats = {col: [] for col in numeric_cols}
    
    print(f"  Computing bootstrap CIs ({n_bootstrap} resamples)...", end='', flush=True)
    for b in range(n_bootstrap):
        # Resample with replacement across folds
        indices = np.random.choice(n_folds, size=n_folds, replace=True)
        
        for col in numeric_cols:
            # Filter out NaN values in this column
            fold_values = summary_df[col].values[indices]
            fold_values = fold_values[~np.isnan(fold_values)]
            
            if len(fold_values) > 0:
                bootstrap_stats[col].append(np.mean(fold_values))
            else:
                bootstrap_stats[col].append(np.nan)
        
        if (b + 1) % 100 == 0:
            print(f" {b+1}", end='', flush=True)
    
    print(" ✓")
    
    # Compute statistics
    ci_results = {}
    for col in numeric_cols:
        values = [v for v in bootstrap_stats[col] if not np.isnan(v)]
        if values:
            values_array = np.array(values)
            ci_results[col] = {
                'mean': np.mean(values_array),
                'std': np.std(values_array),
                'ci_lower': np.percentile(values_array, 2.5),
                'ci_upper': np.percentile(values_array, 97.5),
                'ci_margin': (np.percentile(values_array, 97.5) - np.percentile(values_array, 2.5)) / 2
            }
        else:
            ci_results[col] = {
                'mean': np.nan, 'std': np.nan,
                'ci_lower': np.nan, 'ci_upper': np.nan, 'ci_margin': np.nan
            }
    
    return ci_results

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
        
        # Extract key metrics from classification report
        try:
            acc = df.loc['accuracy', df.columns[0]]
            pos_recall = df.loc['Positive', 'recall']
            pos_precision = df.loc['Positive', 'precision']
            neg_recall = df.loc['Negative', 'recall']
            f1_macro = df.loc['macro avg', 'f1-score']
        except (KeyError, IndexError):
            print(f"Warning: Could not parse standard metrics from {file}")
            continue
        
        # Extract clinical metrics (added as additional rows)
        clinical_metrics_dict = {}
        clinical_metric_names = [
            'Sensitivity_(Recall)', 'Specificity', 'Balanced_Accuracy',
            'PPV_(Positive_Predictive_Value)', 'NPV_(Negative_Predictive_Value)',
            'FPR_(False_Positive_Rate)', 'FNR_(False_Negative_Rate)',
            'Matthews_Correlation_Coefficient', 'Cohen_Kappa'
        ]
        
        for metric_name in clinical_metric_names:
            try:
                clinical_metrics_dict[metric_name] = df.loc[metric_name, df.columns[0]]
            except KeyError:
                clinical_metrics_dict[metric_name] = None
        
        metrics = {
            'RunID': run_id,
            'Fold': fold_name,
            'Accuracy': acc,
            'Precision(+)': pos_precision,
            'Recall(+)': pos_recall,
            'Recall(-)': neg_recall,
            'F1_Macro': f1_macro
        }
        
        # Add clinical metrics to the metrics dict
        metrics.update(clinical_metrics_dict)
        all_metrics.append(metrics)
        
        print(f"[{run_id} {fold_name}] Acc: {acc:.4f} | Prec(+): {pos_precision:.4f} | Rec(+): {pos_recall:.4f} | Rec(-): {neg_recall:.4f}")

    # 2. Calculate Averages
    summary_df = pd.DataFrame(all_metrics)
    
    # Drop columns that are completely empty (no clinical metrics in old reports)
    summary_df = summary_df.dropna(axis=1, how='all')
    
    # Ensure only numeric columns are selected for mean/std
    numeric_cols = [
        'Accuracy', 'Precision(+)', 'Recall(+)', 'Recall(-)', 'F1_Macro',
        'Sensitivity_(Recall)', 'Specificity', 'Balanced_Accuracy',
        'PPV_(Positive_Predictive_Value)', 'NPV_(Negative_Predictive_Value)',
        'FPR_(False_Positive_Rate)', 'FNR_(False_Negative_Rate)',
        'Matthews_Correlation_Coefficient', 'Cohen_Kappa'
    ]
    
    # Filter to only columns that exist in the dataframe and have data
    numeric_cols = [col for col in numeric_cols if col in summary_df.columns]
    
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
    
    # Compute bootstrap confidence intervals across folds
    print(f"\n{'Computing Bootstrap Confidence Intervals':^60}")
    print(f"{'-'*60}")
    bootstrap_ci = compute_bootstrap_ci_from_folds(summary_df, numeric_cols)

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
    # Add Run Range metadata to the summary for experiment tracking
    if all_metrics:
        avg_stds_df['Run_Range'] = f"{min_run}-{max_run}"

    # Save the consolidated averages (Mean ± Std) for iteration benchmarking
    averages_filename = f"grand_cv_averages{run_suffix}.csv"
    avg_stds_df.to_csv(os.path.join(results_dir, averages_filename), index=False)
    
    # Save bootstrap confidence intervals
    bootstrap_ci_filename = f"grand_cv_bootstrap_ci{run_suffix}.csv"
    bootstrap_data = {
        "Metric": numeric_cols,
        "Point_Estimate": [averages[col] for col in numeric_cols],
        "Fold_Std": [stds[col] for col in numeric_cols],
        "Bootstrap_Mean": [bootstrap_ci[col]['mean'] for col in numeric_cols],
        "Bootstrap_Std": [bootstrap_ci[col]['std'] for col in numeric_cols],
        "CI_Lower_95%": [bootstrap_ci[col]['ci_lower'] for col in numeric_cols],
        "CI_Upper_95%": [bootstrap_ci[col]['ci_upper'] for col in numeric_cols],
        "CI_Margin": [bootstrap_ci[col]['ci_margin'] for col in numeric_cols]
    }
    
    bootstrap_ci_df = pd.DataFrame(bootstrap_data)
    bootstrap_ci_df.to_csv(os.path.join(results_dir, bootstrap_ci_filename), index=False)
    
    # Final console notification for audit trails
    print(f"Grand summary saved to {results_dir}/{summary_filename}")
    print(f"Grand averages with ± saved to {results_dir}/{averages_filename}")
    print(f"Bootstrap CIs saved to {results_dir}/{bootstrap_ci_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="results")
    parser.add_argument("--last", type=int, default=None, help="Only summarize the last N reports")
    args = parser.parse_args()
    generate_grand_summary(args.dir, args.last)
