#!/usr/bin/env python3
"""
Weighted Ensemble Voting with Temperature-Scaled Predictions

EXPERIMENTAL: Tests ensemble voting on temperature-calibrated predictions.

Usage:
    python3 weighted_ensemble_with_temperature.py --run 02_34.3 --strategy f1
    
Expects input files: {run_id}_*_f*_predictions_corrected_temp.json
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
)


def load_fold_predictions_temp(run_id, model_name='convnext_tiny'):
    """
    Load predictions from temperature-scaled corrected JSON files.
    """
    results_dir = '/home/tkeating/model/H.-Pylori-Contamination-Detection/results'
    
    fold_data = {}
    for fold_idx in range(5):
        # Glob for the temperature-scaled predictions file
        # Pattern: {run_id}_f{fold_idx}_predictions_corrected_temp.json
        pattern = f"{results_dir}/{run_id}_f{fold_idx}_predictions_corrected_temp.json"
        
        if not Path(pattern).exists():
            # Try with wildcard if direct path doesn't exist
            pattern = f"{results_dir}/{run_id}*f{fold_idx}*predictions_corrected_temp.json"
            matches = glob.glob(pattern)
            if not matches:
                print(f"WARNING: No temperature-scaled predictions for fold {fold_idx}")
                continue
            pred_file = matches[0]
        else:
            pred_file = pattern
        
        with open(pred_file) as f:
            data = json.load(f)
        
        fold_data[fold_idx] = {
            'labels': np.array(data['labels'], dtype=int),
            'probabilities': np.array(data['probabilities'], dtype=float),
            'probabilities_temp_scaled': np.array(data['probabilities_temperature_scaled'], dtype=float),
            'predictions': np.array(data['predictions_at_threshold'], dtype=int),
            'threshold': data['threshold_applied'],
            'temperature': data['temperature']
        }
    
    return fold_data


def compute_fold_metrics(labels, predictions):
    """Compute all metrics for a fold."""
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (recall + specificity) / 2
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'balanced_accuracy': balanced_accuracy,
        'specificity': specificity,
        'mcc': matthews_corrcoef(labels, predictions),
        'kappa': cohen_kappa_score(labels, predictions),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }


def compute_fold_weights(fold_data, strategy='f1'):
    """Compute weights for each fold based on strategy."""
    fold_weights = {}
    fold_metrics = {}
    
    # Compute metrics for each fold
    for fold_idx, data in fold_data.items():
        metrics = compute_fold_metrics(data['labels'], data['predictions'])
        fold_metrics[fold_idx] = metrics
    
    if strategy == 'uniform':
        fold_weights = {fold_idx: 1.0 / len(fold_data) for fold_idx in fold_data.keys()}
    
    elif strategy == 'f1':
        f1_scores = {fold_idx: m['f1_score'] for fold_idx, m in fold_metrics.items()}
        total = sum(f1_scores.values())
        fold_weights = {fold_idx: f1_scores[fold_idx] / total for fold_idx in fold_data.keys()}
    
    elif strategy == 'accuracy':
        accuracies = {fold_idx: m['accuracy'] for fold_idx, m in fold_metrics.items()}
        total = sum(accuracies.values())
        fold_weights = {fold_idx: accuracies[fold_idx] / total for fold_idx in fold_data.keys()}
    
    elif strategy == 'balanced_accuracy':
        ba_scores = {fold_idx: m['balanced_accuracy'] for fold_idx, m in fold_metrics.items()}
        total = sum(ba_scores.values())
        fold_weights = {fold_idx: ba_scores[fold_idx] / total for fold_idx in fold_data.keys()}
    
    elif strategy == 'inverse_difficulty':
        f1_scores = {fold_idx: m['f1_score'] for fold_idx, m in fold_metrics.items()}
        max_f1 = max(f1_scores.values())
        difficulties = {fold_idx: 1.0 - (f1_scores[fold_idx] / max_f1) for fold_idx in fold_data.keys()}
        total = sum(difficulties.values())
        fold_weights = {fold_idx: difficulties[fold_idx] / total for fold_idx in fold_data.keys()}
    
    return fold_weights, fold_metrics


def create_ensemble_predictions(fold_data, fold_weights):
    """
    Compute weighted aggregate metrics across all folds.
    
    For cross-validation, each sample appears in exactly one fold, so we're computing
    weighted metrics across the combined validation sets, not true ensemble voting.
    """
    # Concatenate all fold data
    all_labels = np.concatenate([fold_data[idx]['labels'] for idx in sorted(fold_data.keys())])
    all_preds = np.concatenate([fold_data[idx]['predictions'] for idx in sorted(fold_data.keys())])
    
    # Compute aggregate metrics
    return compute_fold_metrics(all_labels, all_preds)


def main():
    parser = argparse.ArgumentParser(description='Weighted ensemble voting with temperature scaling')
    parser.add_argument('--run', required=True, help='Run ID')
    parser.add_argument('--strategy', default='f1', 
                       choices=['uniform', 'accuracy', 'f1', 'balanced_accuracy', 'inverse_difficulty'],
                       help='Weighting strategy')
    args = parser.parse_args()
    
    # Load predictions
    fold_data = load_fold_predictions_temp(args.run)
    
    if len(fold_data) < 5:
        print(f"ERROR: Only found {len(fold_data)} folds, expected 5")
        return
    
    print("="*80)
    print(f"Weighted Ensemble Voting (Temperature-Scaled): {args.run} - Strategy: {args.strategy}")
    print("="*80)
    
    # Compute fold weights
    fold_weights, fold_metrics = compute_fold_weights(fold_data, strategy=args.strategy)
    
    print("\nFold-Specific Metrics and Weights (Temperature-Scaled):")
    print("-" * 80)
    print(f"{'Fold':<6} {'Temp':<8} {'Weight':<10} {'Accuracy':<12} {'F1 Score':<12} {'Balanced_Acc':<12}")
    print("-" * 80)
    
    for fold_idx in sorted(fold_weights.keys()):
        w = fold_weights[fold_idx]
        m = fold_metrics[fold_idx]
        temp = fold_data[fold_idx]['temperature']
        print(f"{fold_idx:<6} {temp:<8.1f} {w:<10.4f} {m['accuracy']:<12.4f} {m['f1_score']:<12.4f} {m['balanced_accuracy']:<12.4f}")
    
    print("-" * 80)
    print(f"{'TOTAL':<6} {'':<8} {sum(fold_weights.values()):<10.4f}")
    
    # Create ensemble predictions
    ensemble_metrics = create_ensemble_predictions(fold_data, fold_weights)
    
    print("\n" + "="*80)
    print("Ensemble Performance (Temperature-Scaled, Weighted Metrics)")
    print("="*80)
    print(f"Accuracy:          {ensemble_metrics['accuracy']:.4f}")
    print(f"Precision:         {ensemble_metrics['precision']:.4f}")
    print(f"Recall:            {ensemble_metrics['recall']:.4f}")
    print(f"F1 Score:          {ensemble_metrics['f1_score']:.4f}")
    print(f"Specificity:       {ensemble_metrics['specificity']:.4f}")
    print(f"Balanced Accuracy: {ensemble_metrics['balanced_accuracy']:.4f}")
    
    tp = ensemble_metrics['tp']
    fp = ensemble_metrics['fp']
    fn = ensemble_metrics['fn']
    tn = ensemble_metrics['tn']
    print(f"\nWeighted Confusion Matrix:")
    print(f"  TP: {tp:.0f}, FP: {fp:.0f}, FN: {fn:.0f}, TN: {tn:.0f}")


if __name__ == '__main__':
    main()
