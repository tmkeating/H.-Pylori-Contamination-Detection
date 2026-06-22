#!/usr/bin/env python3
"""
Apply Calibrated Thresholds + Temperature Scaling to DeepHP Predictions

EXPERIMENTAL: Tests temperature scaling alongside threshold calibration.

This script applies temperature scaling to probabilities, then applies fold-specific
thresholds before generating corrected predictions.

Usage:
    python3 apply_calibrated_thresholds_with_temperature.py --run 02_34.3
"""

import json
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, cohen_kappa_score
)


def apply_temperature_scaling(probabilities, temperature):
    """
    Apply temperature scaling to probabilities.
    
    For probabilities already in [0, 1], we approximate by:
    logit ≈ log(P / (1-P))
    Then: P_scaled = 1 / (1 + exp(-logit/T))
    """
    probs_clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logits = np.log(probs_clipped / (1 - probs_clipped))
    scaled_logits = logits / temperature
    scaled_probs = 1.0 / (1.0 + np.exp(-scaled_logits))
    return scaled_probs


def apply_thresholds_to_fold(fold_idx, probabilities_json_path, thresholds_data, results_dir, run_id="", iteration=""):
    """
    Apply temperature-scaled thresholds to fold predictions.
    """
    with open(probabilities_json_path, 'r') as f:
        data = json.load(f)
    
    labels = np.array(data['labels'], dtype=int)
    probabilities = np.array(data['probabilities'], dtype=float)
    
    fold_key = str(fold_idx)
    if fold_key not in thresholds_data['fold_thresholds']:
        print(f"ERROR: No threshold data for fold {fold_idx}")
        return None
    
    fold_config = thresholds_data['fold_thresholds'][fold_key]
    temperature = fold_config.get('temperature', 1.0)
    threshold = fold_config['threshold']
    
    # Apply temperature scaling
    if temperature != 1.0:
        probabilities_scaled = apply_temperature_scaling(probabilities, temperature)
    else:
        probabilities_scaled = probabilities.copy()
    
    # Apply threshold
    predictions = (probabilities_scaled >= threshold).astype(int)
    
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (recall + specificity) / 2
    
    print(f"\nFold {fold_idx} (T={temperature:.1f}, threshold={threshold:.3f}):")
    print(f"  Accuracy:   {accuracy:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  F1 Score:   {f1:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"  MCC: {matthews_corrcoef(labels, predictions):.4f}, Kappa: {cohen_kappa_score(labels, predictions):.4f}")
    print(f"  Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    
    # Save corrected predictions
    output_filename = f"{run_id}_{iteration}_f{fold_idx}_predictions_corrected_temp.json"
    output_path = Path(results_dir) / output_filename
    
    corrected_data = {
        'fold_idx': fold_idx,
        'temperature': float(temperature),
        'threshold_applied': float(threshold),
        'labels': labels.tolist(),
        'probabilities': probabilities.tolist(),
        'probabilities_temperature_scaled': probabilities_scaled.tolist(),
        'predictions_at_threshold': predictions.tolist(),
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'balanced_accuracy': float(balanced_accuracy),
            'specificity': float(specificity),
            'mcc': float(matthews_corrcoef(labels, predictions)),
            'kappa': float(cohen_kappa_score(labels, predictions)),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(corrected_data, f)
    
    print(f"  ✓ Saved to {output_filename}")
    
    return corrected_data['metrics']


def main():
    parser = argparse.ArgumentParser(description='Apply calibrated thresholds + temperatures to DeepHP predictions')
    parser.add_argument('--run', default='02_34.3', help='Run ID (e.g., 02_34.3)')
    parser.add_argument('--model', default='convnext_tiny', help='Model name')
    args = parser.parse_args()
    
    # Parse run ID and iteration
    run_parts = args.run.split('_', 1)
    run_id = run_parts[0] if len(run_parts) > 0 else ""
    iteration = run_parts[1] if len(run_parts) > 1 else ""
    
    # Load calibrated thresholds + temperatures
    threshold_file = f'/home/tkeating/model/H.-Pylori-Contamination-Detection/results/{args.run}_calibrated_thresholds_temp.json'
    with open(threshold_file) as f:
        thresholds_data = json.load(f)
    
    results_dir = '/home/tkeating/model/H.-Pylori-Contamination-Detection/results'
    
    print("="*80)
    print(f"Applying Calibrated Thresholds + Temperature Scaling to {args.run}")
    print(f"Run ID: {run_id}, Iteration: {iteration}")
    print("="*80)
    
    # Process all folds
    all_metrics = []
    for fold_idx in range(5):
        prob_files = glob.glob(f'{results_dir}/{args.run}_*_f{fold_idx}_{args.model}_probabilities.json')
        if prob_files:
            metrics = apply_thresholds_to_fold(fold_idx, prob_files[0], thresholds_data, results_dir, 
                                              run_id=run_id, iteration=iteration)
            if metrics:
                all_metrics.append(metrics)
    
    # Summary
    if all_metrics:
        print(f"\n{'='*80}")
        print("SUMMARY: Mean Metrics Across All Folds (with Temperature Scaling)")
        print(f"{'='*80}")
        mean_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
        print(f"Accuracy:          {mean_metrics['accuracy']:.4f}")
        print(f"Precision:         {mean_metrics['precision']:.4f}")
        print(f"Recall:            {mean_metrics['recall']:.4f}")
        print(f"F1 Score:          {mean_metrics['f1_score']:.4f}")
        print(f"Balanced Accuracy: {mean_metrics['balanced_accuracy']:.4f}")


if __name__ == '__main__':
    main()
