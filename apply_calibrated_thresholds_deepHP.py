#!/usr/bin/env python3
"""
Post-Process DeepHP Validation Predictions with Calibrated Thresholds

This script applies per-fold thresholds to saved predictions and generates
corrected evaluation metrics.

Usage:
    python3 apply_calibrated_thresholds.py --run 01_34.0
"""

import json
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, cohen_kappa_score,
    roc_auc_score, roc_curve, auc
)


def apply_thresholds_to_fold(fold_idx, probabilities_json_path, thresholds_data, output_dir, run_id="", iteration=""):
    """
    Apply calibrated threshold to a fold's predictions and generate new metrics.
    
    Args:
        fold_idx: Fold index (0-4)
        probabilities_json_path: Path to probability JSON file
        thresholds_data: Loaded thresholds from calibration
        output_dir: Output directory for corrected predictions
        run_id: Run ID for output filename
        iteration: Iteration for output filename
    """
    # Load probabilities
    with open(probabilities_json_path, 'r') as f:
        data = json.load(f)
    
    labels = np.array(data['labels'], dtype=int)
    probabilities = np.array(data['probabilities'], dtype=float)
    
    # Get threshold for this fold
    fold_key = str(fold_idx)
    if fold_key not in thresholds_data['fold_thresholds']:
        print(f"ERROR: No threshold for fold {fold_idx}")
        return None
    
    threshold = thresholds_data['fold_thresholds'][fold_key]['threshold']
    
    # Apply threshold
    predictions = (probabilities >= threshold).astype(int)
    
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (recall + specificity) / 2
    
    mcc = matthews_corrcoef(labels, predictions)
    kappa = cohen_kappa_score(labels, predictions)
    
    # PPV/NPV
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # FPR/FNR
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    metrics = {
        'fold': fold_idx,
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'sensitivity': recall,
        'specificity': specificity,
        'balanced_accuracy': balanced_accuracy,
        'ppv': ppv,
        'npv': npv,
        'fpr': fpr,
        'fnr': fnr,
        'mcc': mcc,
        'kappa': kappa,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'total_samples': len(labels),
        'num_positive': int(np.sum(labels)),
        'num_negative': len(labels) - int(np.sum(labels))
    }
    
    # Save corrected predictions
    corrected_data = {
        **data,
        'threshold_applied': float(threshold),
        'predictions_at_threshold': predictions.tolist()
    }
    
    # Build output filename with run_id and iteration if provided
    if run_id and iteration:
        output_filename = f"{run_id}_{iteration}_f{fold_idx}_predictions_corrected.json"
    else:
        output_filename = f"fold_{fold_idx}_predictions_corrected.json"
    
    output_json = Path(output_dir) / output_filename
    with open(output_json, 'w') as f:
        json.dump(corrected_data, f, indent=2)
    
    print(f"\nFold {fold_idx} (threshold={threshold:.3f}):")
    print(f"  Accuracy:   {accuracy:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  F1 Score:   {f1:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"  MCC: {mcc:.4f}, Kappa: {kappa:.4f}")
    print(f"  Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Apply calibrated thresholds to DeepHP predictions')
    parser.add_argument('--run', default='01_34.0', help='Run ID (e.g., 01_34.0)')
    parser.add_argument('--model', default='convnext_tiny', help='Model name')
    parser.add_argument('--output_dir', default='results', help='Output directory for results (default: results)')
    args = parser.parse_args()
    
    # Parse run ID and iteration from args.run (format: "{run_id}_{iteration}" e.g., "01_34.0")
    run_parts = args.run.split('_', 1)  # Split on first underscore only
    run_id = run_parts[0] if len(run_parts) > 0 else ""
    iteration = run_parts[1] if len(run_parts) > 1 else ""
    
    # Use output_dir parameter
    output_dir = args.output_dir
    
    # Load calibrated thresholds
    threshold_file = Path(output_dir) / f'{args.run}_calibrated_thresholds.json'
    with open(threshold_file) as f:
        thresholds_data = json.load(f)
    
    print("="*80)
    print(f"Applying Calibrated Thresholds to {args.run}")
    print(f"Run ID: {run_id}, Iteration: {iteration}")
    print("="*80)
    
    # Process all folds
    all_metrics = []
    for fold_idx in range(5):
        prob_files = glob.glob(str(Path(output_dir) / f'{args.run}_*_f{fold_idx}_{args.model}_probabilities.json'))
        if prob_files:
            metrics = apply_thresholds_to_fold(fold_idx, prob_files[0], thresholds_data, output_dir, run_id=run_id, iteration=iteration)
            if metrics:
                all_metrics.append(metrics)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: Metrics After Threshold Calibration")
    print(f"{'='*80}")
    
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        print("\n" + df[['fold', 'threshold', 'accuracy', 'precision', 'recall', 'f1_score', 'balanced_accuracy']].to_string(index=False))
        
        # Save summary
        summary_csv = Path(output_dir) / f"{args.run}_calibrated_metrics.csv"
        df.to_csv(summary_csv, index=False)
        print(f"\n✓ Saved detailed metrics to {summary_csv}")
        
        # Print cross-fold statistics
        print(f"\n{'='*80}")
        print("Cross-Fold Statistics (Mean ± Std)")
        print(f"{'='*80}")
        print(f"Accuracy:          {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
        print(f"Precision:         {df['precision'].mean():.4f} ± {df['precision'].std():.4f}")
        print(f"Recall:            {df['recall'].mean():.4f} ± {df['recall'].std():.4f}")
        print(f"F1 Score:          {df['f1_score'].mean():.4f} ± {df['f1_score'].std():.4f}")
        print(f"Balanced Accuracy: {df['balanced_accuracy'].mean():.4f} ± {df['balanced_accuracy'].std():.4f}")


if __name__ == '__main__':
    main()
