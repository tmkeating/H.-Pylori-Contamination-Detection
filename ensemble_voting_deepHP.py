#!/usr/bin/env python3
"""
Weighted Ensemble Voting for DeepHP 5-Fold Cross-Validation

Combines predictions from all 5 folds with fold-specific weights based on
fold difficulty/performance. This accounts for the fact that some folds
(1 and 4) are easier than others (0 and 3).

Weighting Strategies:
1. Accuracy-weighted: Weight by validation accuracy
2. F1-weighted: Weight by validation F1 score
3. Balanced-accuracy-weighted: Weight by validation balanced accuracy
4. Inverse-difficulty: Weight by (1 - relative_difficulty)
5. Uniform: Equal weight (baseline)

Usage:
    python3 ensemble_voting_deepHP.py --run 01_34.2 --strategy f1
    
Expects input files: {run_id}_f0_predictions_corrected.json, {run_id}_f1_predictions_corrected.json, etc.
Output files: {run_id}_ensemble_metrics_{strategy}.csv, {run_id}_ensemble_weights_{strategy}.json
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
    roc_auc_score, classification_report
)


def load_fold_predictions(run_id, model_name='convnext_tiny'):
    """
    Load predictions and labels for all folds from corrected JSON files.
    Uses the new naming convention: {run_id}_f{fold_idx}_predictions_corrected.json
    """
    results_dir = '/home/tkeating/model/H.-Pylori-Contamination-Detection/results'
    
    fold_data = {}
    for fold_idx in range(5):
        pred_file = Path(results_dir) / f"{run_id}_f{fold_idx}_predictions_corrected.json"
        if not pred_file.exists():
            print(f"WARNING: {pred_file} not found, skipping fold {fold_idx}")
            continue
        
        with open(pred_file) as f:
            data = json.load(f)
        
        fold_data[fold_idx] = {
            'labels': np.array(data['labels'], dtype=int),
            'probabilities': np.array(data['probabilities'], dtype=float),
            'predictions': np.array(data['predictions_at_threshold'], dtype=int),
            'threshold': data['threshold_applied']
        }
    
    return fold_data


def compute_fold_metrics(labels, predictions):
    """
    Compute all metrics for a fold.
    """
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
        'kappa': cohen_kappa_score(labels, predictions)
    }


def compute_fold_weights(fold_data, strategy='f1'):
    """
    Compute weights for each fold based on strategy.
    
    Args:
        fold_data: Dict with fold predictions
        strategy: 'accuracy', 'f1', 'balanced_accuracy', 'inverse_difficulty', 'uniform'
    
    Returns:
        weights: Dict with fold_idx -> weight
    """
    fold_weights = {}
    fold_metrics = {}
    
    # Compute metrics for each fold
    for fold_idx, data in fold_data.items():
        metrics = compute_fold_metrics(data['labels'], data['predictions'])
        fold_metrics[fold_idx] = metrics
    
    # Compute weights based on strategy
    if strategy == 'uniform':
        for fold_idx in fold_data.keys():
            fold_weights[fold_idx] = 1.0
    
    elif strategy == 'accuracy':
        accuracies = {i: fold_metrics[i]['accuracy'] for i in fold_data.keys()}
        total = sum(accuracies.values())
        fold_weights = {i: accuracies[i] / total for i in fold_data.keys()}
    
    elif strategy == 'f1':
        f1_scores = {i: fold_metrics[i]['f1_score'] for i in fold_data.keys()}
        # Handle zero scores
        f1_scores = {i: max(f1, 0.1) for i, f1 in f1_scores.items()}
        total = sum(f1_scores.values())
        fold_weights = {i: f1_scores[i] / total for i in fold_data.keys()}
    
    elif strategy == 'balanced_accuracy':
        ba_scores = {i: fold_metrics[i]['balanced_accuracy'] for i in fold_data.keys()}
        ba_scores = {i: max(ba, 0.1) for i, ba in ba_scores.items()}
        total = sum(ba_scores.values())
        fold_weights = {i: ba_scores[i] / total for i in fold_data.keys()}
    
    elif strategy == 'inverse_difficulty':
        # Difficulty = 1 - balanced_accuracy
        difficulties = {i: 1 - fold_metrics[i]['balanced_accuracy'] for i in fold_data.keys()}
        # Inverse: weight by 1 - difficulty
        weights_raw = {i: 1 - difficulties[i] for i in fold_data.keys()}
        weights_raw = {i: max(w, 0.1) for i, w in weights_raw.items()}
        total = sum(weights_raw.values())
        fold_weights = {i: weights_raw[i] / total for i in fold_data.keys()}
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return fold_weights, fold_metrics


def create_ensemble_predictions(fold_data, fold_weights):
    """
    Create ensemble metrics by weighting each fold's performance.
    
    NOTE: For proper 5-fold CV ensembling where validation sets don't overlap:
    - Average metrics from each fold weighted by fold_weight * num_samples
    - This gives overall ensemble performance estimate
    - For actual test predictions, apply all 5 fold models and vote
    """
    weighted_metrics = {
        'tp': 0,
        'fp': 0,
        'fn': 0,
        'tn': 0,
        'total_samples': 0
    }
    
    # Weight confusion matrices by sample count
    for fold_idx in sorted(fold_data.keys()):
        data = fold_data[fold_idx]
        labels = data['labels']
        predictions = data['predictions']
        weight = fold_weights[fold_idx]
        
        num_samples = len(labels)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        
        # Weight by fold importance and sample count
        sample_weight = weight * num_samples
        weighted_metrics['tp'] += tp * weight
        weighted_metrics['fp'] += fp * weight
        weighted_metrics['fn'] += fn * weight
        weighted_metrics['tn'] += tn * weight
        weighted_metrics['total_samples'] += sample_weight
    
    # Normalize to get weighted metrics
    tp_w = weighted_metrics['tp']
    fp_w = weighted_metrics['fp']
    fn_w = weighted_metrics['fn']
    tn_w = weighted_metrics['tn']
    total_w = tp_w + fp_w + fn_w + tn_w
    
    # Compute metrics from weighted confusion matrix
    accuracy = (tp_w + tn_w) / total_w if total_w > 0 else 0
    precision = tp_w / (tp_w + fp_w) if (tp_w + fp_w) > 0 else 0
    recall = tp_w / (tp_w + fn_w) if (tp_w + fn_w) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    specificity = tn_w / (tn_w + fp_w) if (tn_w + fp_w) > 0 else 0
    balanced_accuracy = (recall + specificity) / 2
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'balanced_accuracy': balanced_accuracy,
        'specificity': specificity,
        'tp': tp_w,
        'fp': fp_w,
        'fn': fn_w,
        'tn': tn_w,
        'total_samples': total_w
    }


def main():
    parser = argparse.ArgumentParser(description='Weighted ensemble voting for DeepHP')
    parser.add_argument('--run', default='01_34.0', help='Run ID')
    parser.add_argument('--strategy', default='f1', 
                       choices=['uniform', 'accuracy', 'f1', 'balanced_accuracy', 'inverse_difficulty'],
                       help='Weighting strategy')
    args = parser.parse_args()
    
    # Load predictions
    fold_data = load_fold_predictions(args.run)
    
    if len(fold_data) < 5:
        print(f"ERROR: Only found {len(fold_data)} folds, expected 5")
        return
    
    print("="*80)
    print(f"Weighted Ensemble Voting: {args.run} - Strategy: {args.strategy}")
    print("="*80)
    
    # Compute fold weights
    fold_weights, fold_metrics = compute_fold_weights(fold_data, strategy=args.strategy)
    
    print("\nFold-Specific Metrics and Weights:")
    print("-" * 80)
    print(f"{'Fold':<6} {'Weight':<10} {'Accuracy':<12} {'F1 Score':<12} {'Balanced_Acc':<12} {'Specificity':<12}")
    print("-" * 80)
    
    for fold_idx in sorted(fold_weights.keys()):
        w = fold_weights[fold_idx]
        m = fold_metrics[fold_idx]
        print(f"{fold_idx:<6} {w:.4f}      {m['accuracy']:.4f}       {m['f1_score']:.4f}       "
              f"{m['balanced_accuracy']:.4f}       {m['specificity']:.4f}")
    
    print("-" * 80)
    print(f"{'TOTAL':<6} {sum(fold_weights.values()):.4f}")
    
    # Create ensemble predictions
    ensemble_metrics = create_ensemble_predictions(fold_data, fold_weights)
    
    print("\n" + "="*80)
    print("Ensemble Performance (Weighted Metrics from All Folds)")
    print("="*80)
    print(f"Accuracy:          {ensemble_metrics['accuracy']:.4f}")
    print(f"Precision:         {ensemble_metrics['precision']:.4f}")
    print(f"Recall:            {ensemble_metrics['recall']:.4f}")
    print(f"F1 Score:          {ensemble_metrics['f1_score']:.4f}")
    print(f"Specificity:       {ensemble_metrics['specificity']:.4f}")
    print(f"Balanced Accuracy: {ensemble_metrics['balanced_accuracy']:.4f}")
    
    # Confusion matrix
    tp = ensemble_metrics['tp']
    fp = ensemble_metrics['fp']
    fn = ensemble_metrics['fn']
    tn = ensemble_metrics['tn']
    print(f"\nWeighted Confusion Matrix:")
    print(f"  TP: {tp:.0f}, FP: {fp:.0f}, FN: {fn:.0f}, TN: {tn:.0f}")
    
    # Save ensemble metrics
    output_csv = Path('/home/tkeating/model/H.-Pylori-Contamination-Detection/results') / \
                 f"{args.run}_ensemble_metrics_{args.strategy}.csv"
    
    ensemble_df = pd.DataFrame([{
        'strategy': args.strategy,
        'accuracy': ensemble_metrics['accuracy'],
        'precision': ensemble_metrics['precision'],
        'recall': ensemble_metrics['recall'],
        'f1_score': ensemble_metrics['f1_score'],
        'balanced_accuracy': ensemble_metrics['balanced_accuracy'],
        'specificity': ensemble_metrics['specificity'],
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }])
    
    ensemble_df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved ensemble metrics to {output_csv}")
    
    # Save weights
    weights_file = Path('/home/tkeating/model/H.-Pylori-Contamination-Detection/results') / \
                   f"{args.run}_ensemble_weights_{args.strategy}.json"
    
    weights_data = {
        'run_id': args.run,
        'strategy': args.strategy,
        'fold_weights': {str(k): v for k, v in fold_weights.items()},
        'fold_metrics': {str(k): v for k, v in fold_metrics.items()},
        'ensemble_metrics': ensemble_metrics
    }
    
    with open(weights_file, 'w') as f:
        json.dump(weights_data, f, indent=2)
    print(f"✓ Saved ensemble weights to {weights_file}")
    
    # Compare strategies
    print(f"\n{'='*80}")
    print("Comparison: All Weighting Strategies")
    print(f"{'='*80}")
    
    all_strategies = ['uniform', 'accuracy', 'f1', 'balanced_accuracy', 'inverse_difficulty']
    comparison_results = []
    
    for strat in all_strategies:
        w, m = compute_fold_weights(fold_data, strategy=strat)
        metrics = create_ensemble_predictions(fold_data, w)
        comparison_results.append({
            'strategy': strat,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'balanced_accuracy': metrics['balanced_accuracy']
        })
    
    comp_df = pd.DataFrame(comparison_results)
    print("\n" + comp_df.to_string(index=False))
    
    # Save comparison
    comp_csv = Path('/home/tkeating/model/H.-Pylori-Contamination-Detection/results') / \
               f"{args.run}_ensemble_strategy_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    print(f"\n✓ Saved strategy comparison to {comp_csv}")


if __name__ == '__main__':
    main()
