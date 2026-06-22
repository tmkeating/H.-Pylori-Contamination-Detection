#!/usr/bin/env python3
"""
Per-Fold Threshold Calibration for DeepHP 5-Fold Cross-Validation

Problem: CONFIG 87771 folds have different validation set compositions,
so threshold 0.5 is suboptimal for most folds.

Solution: Search for optimal threshold per fold using F1 score on the
validation set, then save thresholds for inference.

Usage:
    python3 calibrate_per_fold_thresholds.py                # Auto-detect most recent run
    python3 calibrate_per_fold_thresholds.py --run 01_34.1  # Specific run
    python3 calibrate_per_fold_thresholds.py --run 01_34.2 --model convnext_tiny

Output: {run_id}_calibrated_thresholds.json with per-fold thresholds
"""

import json
import glob
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

def calibrate_fold_threshold(fold_idx, probabilities_json_path, threshold_range=np.arange(0.1, 1.0, 0.01)):
    """
    Find optimal threshold for a single fold.
    
    Searches threshold range and returns threshold that maximizes F1 score.
    """
    with open(probabilities_json_path, 'r') as f:
        data = json.load(f)
    
    labels = np.array(data['labels'])
    probabilities = np.array(data['probabilities'])
    
    best_threshold = 0.5
    best_f1 = 0.0
    best_metrics = {}
    
    print(f"\n{'='*80}")
    print(f"FOLD {fold_idx} Threshold Calibration")
    print(f"{'='*80}")
    print(f"Searching {len(threshold_range)} thresholds from {threshold_range[0]:.2f} to {threshold_range[-1]:.2f}")
    
    for threshold in threshold_range:
        predictions = (probabilities >= threshold).astype(int)
        
        # Compute metrics
        f1 = f1_score(labels, predictions, zero_division=0)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        
        # Track best
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'threshold': float(threshold),
                'f1_score': float(f1),
                'precision': float(precision),
                'recall': float(recall),
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn),
                'total_samples': len(labels),
                'num_positive': int(np.sum(labels)),
                'num_negative': len(labels) - int(np.sum(labels))
            }
    
    # Print results
    print(f"\nOptimal Threshold: {best_threshold:.3f}")
    print(f"  F1 Score:  {best_metrics['f1_score']:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall:    {best_metrics['recall']:.4f}")
    print(f"  TP: {best_metrics['tp']}, FP: {best_metrics['fp']}, FN: {best_metrics['fn']}, TN: {best_metrics['tn']}")
    print(f"  Positive class: {best_metrics['num_positive']} / {best_metrics['total_samples']}")
    
    return best_metrics


def main():
    """
    Search for and calibrate thresholds for all folds.
    """
    parser = argparse.ArgumentParser(description='Calibrate per-fold thresholds for DeepHP predictions')
    parser.add_argument('--run', default='', help='Run ID (e.g., 01_34.1). If not provided, auto-detects most recent run')
    parser.add_argument('--model', default='convnext_tiny', help='Model name')
    args = parser.parse_args()
    
    # Find all probabilities files
    prob_files = sorted(glob.glob('/home/tkeating/model/H.-Pylori-Contamination-Detection/results/*probabilities.json'))
    
    if not prob_files:
        print("ERROR: No probabilities JSON files found!")
        return
    
    # Get unique run IDs
    runs = {}
    for f in prob_files:
        filename = Path(f).name
        # Extract run ID (e.g., "01_34.0" from "01_34.0_8810_f0_...")
        # Pattern: {run_id}_{iter}_{slurm_id}_f{fold}_{model}_probabilities.json
        parts = filename.split('_')
        if len(parts) >= 4:
            run_id = f"{parts[0]}_{parts[1]}"  # e.g., "01_34"
            fold_idx = int(parts[3].replace('f', ''))
            
            if run_id not in runs:
                runs[run_id] = {}
            runs[run_id][fold_idx] = f
    
    # Determine which run to process
    if args.run:
        # User specified a run
        run_id = args.run
        if run_id not in runs:
            print(f"ERROR: Run ID '{run_id}' not found in probability files")
            print(f"Available run IDs: {sorted(runs.keys())}")
            return
        print(f"\nUsing specified run: {run_id}")
    else:
        # Auto-detect most recent run
        if not runs:
            print("ERROR: No runs found in probability files")
            return
        run_id = sorted(runs.keys())[-1]
        print(f"\nAuto-detected most recent run: {run_id}")
    
    print(f"Processing {len(runs[run_id])} folds from run {run_id}")
    print(f"Files: {sorted(runs[run_id].values())}")
    
    # Calibrate all folds
    thresholds_data = {
        'run_id': run_id,
        'model_name': 'convnext_tiny',
        'strategy': 'per-fold threshold optimization on validation set',
        'fold_thresholds': {}
    }
    
    for fold_idx in sorted(runs[run_id].keys()):
        prob_file = runs[run_id][fold_idx]
        metrics = calibrate_fold_threshold(fold_idx, prob_file)
        thresholds_data['fold_thresholds'][str(fold_idx)] = metrics
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: Optimal Thresholds by Fold")
    print(f"{'='*80}")
    for fold_idx in sorted(thresholds_data['fold_thresholds'].keys()):
        metrics = thresholds_data['fold_thresholds'][fold_idx]
        print(f"Fold {fold_idx}: threshold={metrics['threshold']:.3f}, F1={metrics['f1_score']:.4f}, "
              f"Recall={metrics['recall']:.4f}, Precision={metrics['precision']:.4f}")
    
    # Save thresholds
    output_path = f"/home/tkeating/model/H.-Pylori-Contamination-Detection/results/{run_id}_calibrated_thresholds.json"
    with open(output_path, 'w') as f:
        json.dump(thresholds_data, f, indent=2)
    
    print(f"\n✓ Saved calibrated thresholds to {output_path}")
    print(f"\nImplementation: These thresholds should be applied during inference:")
    print(f"  - For each fold's backbone model, use the corresponding threshold instead of 0.5")
    print(f"  - During ensemble voting, apply fold-specific thresholds before aggregation")


if __name__ == '__main__':
    main()
