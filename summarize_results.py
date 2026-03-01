import pandas as pd
import glob
import os
import argparse

def generate_grand_summary(results_dir="results"):
    # 1. Find all evaluation report CSVs
    # Pattern: *_f[0-4]_convnext_tiny_evaluation_report.csv
    report_files = glob.glob(os.path.join(results_dir, "*_f[0-4]_*_evaluation_report.csv"))
    
    if not report_files:
        print(f"No evaluation reports found in {results_dir}")
        return

    all_metrics = []
    
    print(f"\n{'='*60}")
    print(f"{'H. Pylori Iteration 11: Cross-Validation Grand Summary':^60}")
    print(f"{'='*60}\n")

    for file in sorted(report_files):
        fold_name = os.path.basename(file).split('_')[2] # Extracts 'f0', 'f1', etc.
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
    averages = summary_df[['Accuracy', 'Precision(+)', 'Recall(+)', 'Recall(-)', 'F1_Macro']].mean()
    stds = summary_df[['Accuracy', 'Precision(+)', 'Recall(+)', 'Recall(-)', 'F1_Macro']].std()

    print(f"\n{'-'*60}")
    print(f"{'MEAN CROSS-VALIDATION RESULTS':^60}")
    print(f"{'-'*60}")
    print(f"Accuracy:     {averages['Accuracy']:.4f} ± {stds['Accuracy']:.4f}")
    print(f"Precision(+): {averages['Precision(+)']:.4f} ± {stds['Precision(+)']:.4f}")
    print(f"Recall(+):    {averages['Recall(+)']:.4f} ± {stds['Recall(+)']:.4f}")
    print(f"Recall(-):    {averages['Recall(-']:.4f} ± {stds['Recall(-']:.4f}")
    print(f"F1 Macro:     {averages['F1_Macro']:.4f} ± {stds['F1_Macro']:.4f}")
    print(f"{'='*60}\n")

    # 3. Save to CSV for long-term tracking
    summary_df.to_csv(os.path.join(results_dir, "grand_cv_summary.csv"), index=False)
    print(f"Grand summary saved to {results_dir}/grand_cv_summary.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="results")
    args = parser.parse_args()
    generate_grand_summary(args.dir)
