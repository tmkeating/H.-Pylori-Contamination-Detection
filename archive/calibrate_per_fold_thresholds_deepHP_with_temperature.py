#!/usr/bin/env python3
"""
Per-Fold Threshold + Temperature Calibration for DeepHP 5-Fold Cross-Validation

EXPERIMENTAL: Tests temperature scaling to improve probability calibration.

Temperature scaling applies: P_scaled = softmax(logit / T)
Higher T (e.g., 2.0-5.0) softens overconfident predictions.

Usage:
    python3 calibrate_per_fold_thresholds_with_temperature.py --run 02_34.3
    python3 calibrate_per_fold_thresholds_with_temperature.py --run 02_34.3 --model convnext_tiny

Output: {run_id}_calibrated_thresholds_temp.json with per-fold thresholds AND temperatures
"""

import json
import glob
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

def apply_temperature_scaling(probabilities, temperature):
    """
    Apply temperature scaling to probabilities.
    
    For probabilities already in [0, 1], we approximate by:
    P_scaled = 1 / (1 + exp((1/P - 1) / T))
    
    More accurately, we need the logits. But given only probs, we can use:
    logit ≈ log(P / (1-P))
    
    Then: P_scaled = 1 / (1 + exp(-logit/T))
    """
    # Clip to avoid log(0)
    probs_clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    
    # Recover logits from probabilities
    logits = np.log(probs_clipped / (1 - probs_clipped))
    
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Convert back to probabilities
    scaled_probs = 1.0 / (1.0 + np.exp(-scaled_logits))
    
    return scaled_probs


def calibrate_fold_threshold_with_temperature(fold_idx, probabilities_json_path, 
                                               temperature_range=np.arange(1.0, 5.1, 0.5),
                                               threshold_range=np.arange(0.1, 1.0, 0.01)):
    """
    Find optimal temperature AND threshold for a single fold.
    
    Grid searches temperatures, then for each temperature finds best threshold.
    Returns the combination that maximizes F1 score.
    """
    with open(probabilities_json_path, 'r') as f:
        data = json.load(f)
    
    labels = np.array(data['labels'])
    probabilities = np.array(data['probabilities'])
    
    best_temperature = 1.0
    best_threshold = 0.5
    best_f1 = 0.0
    best_metrics = {}
    results_by_temp = {}
    
    print(f"\n{'='*80}")
    print(f"FOLD {fold_idx} Threshold + Temperature Calibration")
    print(f"{'='*80}")
    print(f"Searching {len(temperature_range)} temperatures and {len(threshold_range)} thresholds")
    
    for temperature in temperature_range:
        # Apply temperature scaling
        if temperature != 1.0:
            scaled_probs = apply_temperature_scaling(probabilities, temperature)
        else:
            scaled_probs = probabilities.copy()
        
        # Find best threshold for this temperature
        best_f1_for_temp = 0.0
        best_threshold_for_temp = 0.5
        best_metrics_for_temp = {}
        
        for threshold in threshold_range:
            predictions = (scaled_probs >= threshold).astype(int)
            
            # Compute metrics
            f1 = f1_score(labels, predictions, zero_division=0)
            precision = precision_score(labels, predictions, zero_division=0)
            recall = recall_score(labels, predictions, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
            
            # Track best for this temperature
            if f1 > best_f1_for_temp:
                best_f1_for_temp = f1
                best_threshold_for_temp = threshold
                best_metrics_for_temp = {
                    'threshold': float(threshold),
                    'f1_score': float(f1),
                    'precision': float(precision),
                    'recall': float(recall),
                    'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
                }
        
        results_by_temp[float(temperature)] = {
            'threshold': best_threshold_for_temp,
            'f1_score': best_f1_for_temp,
            'metrics': best_metrics_for_temp
        }
        
        # Track overall best
        if best_f1_for_temp > best_f1:
            best_f1 = best_f1_for_temp
            best_temperature = temperature
            best_threshold = best_threshold_for_temp
            best_metrics = best_metrics_for_temp
    
    # Print results
    print(f"\nTemperature sweep results:")
    print(f"{'Temp':<8} {'Threshold':<12} {'F1 Score':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 60)
    for temp in sorted(results_by_temp.keys()):
        res = results_by_temp[temp]
        print(f"{temp:<8.1f} {res['threshold']:<12.3f} {res['f1_score']:<12.4f} "
              f"{res['metrics']['precision']:<12.4f} {res['metrics']['recall']:<12.4f}")
    
    print(f"\n✓ Optimal: Temperature={best_temperature:.1f}, Threshold={best_threshold:.3f}, F1={best_f1:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}, Recall: {best_metrics['recall']:.4f}")
    
    best_metrics['temperature'] = float(best_temperature)
    best_metrics['threshold'] = float(best_threshold)
    # Don't save all_temperatures to avoid circular reference
    
    return best_metrics


def main():
    """
    Search for and calibrate thresholds + temperatures for all folds.
    """
    parser = argparse.ArgumentParser(description='Calibrate per-fold thresholds + temperatures for DeepHP predictions')
    parser.add_argument('--run', default='', help='Run ID (e.g., 02_34.3). If not provided, auto-detects most recent run')
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
        parts = filename.split('_')
        if len(parts) >= 4:
            run_id = f"{parts[0]}_{parts[1]}"
            fold_idx = int(parts[3].replace('f', ''))
            
            if run_id not in runs:
                runs[run_id] = {}
            runs[run_id][fold_idx] = f
    
    # Determine which run to process
    if args.run:
        run_id = args.run
        if run_id not in runs:
            print(f"ERROR: Run ID '{run_id}' not found in probability files")
            print(f"Available run IDs: {sorted(runs.keys())}")
            return
        print(f"\nUsing specified run: {run_id}")
    else:
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
        'model_name': args.model,
        'strategy': 'per-fold threshold + temperature optimization on validation set',
        'fold_thresholds': {}
    }
    
    for fold_idx in sorted(runs[run_id].keys()):
        prob_file = runs[run_id][fold_idx]
        metrics = calibrate_fold_threshold_with_temperature(fold_idx, prob_file)
        thresholds_data['fold_thresholds'][str(fold_idx)] = metrics
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: Optimal Temperatures and Thresholds by Fold")
    print(f"{'='*80}")
    for fold_idx in sorted(thresholds_data['fold_thresholds'].keys()):
        metrics = thresholds_data['fold_thresholds'][fold_idx]
        print(f"Fold {fold_idx}: T={metrics['temperature']:.1f}, threshold={metrics['threshold']:.3f}, F1={metrics['f1_score']:.4f}, "
              f"Recall={metrics['recall']:.4f}, Precision={metrics['precision']:.4f}")
    
    # Save thresholds + temperatures
    output_path = f"/home/tkeating/model/H.-Pylori-Contamination-Detection/results/{run_id}_calibrated_thresholds_temp.json"
    with open(output_path, 'w') as f:
        json.dump(thresholds_data, f, indent=2)
    
    print(f"\n✓ Saved calibrated thresholds + temperatures to {output_path}")
    print(f"\nImplementation: These should be applied during inference:")
    print(f"  1. For each fold's backbone model:")
    print(f"     - Extract logits from backbone")
    print(f"     - Apply temperature scaling: logit_scaled = logit / T")
    print(f"     - Convert to probability: P = softmax(logit_scaled)")
    print(f"     - Apply fold-specific threshold: pred = (P >= threshold)")
    print(f"  2. Ensemble: Aggregate predictions from all 5 folds")


if __name__ == '__main__':
    main()
