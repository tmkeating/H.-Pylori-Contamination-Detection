#!/usr/bin/env python3
"""
Regenerate backbone with weighted ensemble averaging from existing fold checkpoints.

This script loads fold checkpoints and ensemble weights from a completed DeepHP run
and regenerates the backbone using F1-based (or other strategy) weighted averaging.

Usage:
    python3 regenerate_weighted_backbone.py --run 01_34.4 --model convnext_tiny
"""

import json
import argparse
import glob
from pathlib import Path
from load_pretrained_backbone import weighted_average_backbone_weights


def main():
    parser = argparse.ArgumentParser(description='Regenerate backbone with weighted ensemble averaging')
    parser.add_argument('--run', type=str, required=True, help='Run ID (e.g., 01_34.4)')
    parser.add_argument('--model', type=str, default='convnext_tiny', help='Model name (default: convnext_tiny)')
    parser.add_argument('--strategy', type=str, default='f1', help='Ensemble strategy to use (default: f1)')
    args = parser.parse_args()
    
    run_id = args.run
    model_name = args.model
    strategy = args.strategy
    
    results_dir = Path('results')
    
    print("="*80)
    print(f"Regenerating Backbone with Weighted Ensemble Averaging")
    print(f"Run ID: {run_id}")
    print(f"Model: {model_name}")
    print(f"Strategy: {strategy}")
    print("="*80)
    print()
    
    # Load ensemble weights
    weights_file = results_dir / f"{run_id}_ensemble_weights_{strategy}.json"
    
    if not weights_file.exists():
        print(f"ERROR: Ensemble weights file not found:")
        print(f"  Expected: {weights_file}")
        print(f"\nPlease run ensemble_voting_deepHP.py first:")
        print(f"  python3 ensemble_voting_deepHP.py --run {run_id} --strategy {strategy}")
        return False
    
    print(f"Loading ensemble weights from: {weights_file}")
    with open(weights_file) as f:
        ensemble_data = json.load(f)
        fold_weights = ensemble_data.get("fold_weights", {})
    
    print(f"\nEnsemble Weights ({strategy}-based):")
    for fold_idx in sorted(fold_weights.keys()):
        weight = float(fold_weights[fold_idx])
        print(f"  Fold {fold_idx}: {weight:.4f}")
    print()
    
    # Find fold checkpoints
    # Pattern: {run_id}_{slurm_id}_f{fold}_{model}_model_brain.pth
    print(f"Searching for fold checkpoints...")
    fold_paths = []
    for fold_idx in range(5):
        # Search for pattern: 01_34.4_*_f{fold}_convnext_tiny_model_brain.pth
        fold_files = list(results_dir.glob(f"{run_id}_*_f{fold_idx}_{model_name}_model_brain.pth"))
        if fold_files:
            # Get the most recent one (in case multiple exist)
            fold_path = str(sorted(fold_files, key=lambda x: x.stat().st_mtime, reverse=True)[0])
            fold_paths.append(fold_path)
            print(f"  ✓ Fold {fold_idx}: {Path(fold_path).name}")
        else:
            print(f"  ✗ Fold {fold_idx}: NOT FOUND")
    
    if len(fold_paths) < 5:
        print(f"\nERROR: Expected 5 folds, found {len(fold_paths)}")
        print("Please ensure all 5 fold training jobs completed successfully.")
        return False
    
    print()
    
    # Regenerate backbone with weighted averaging
    # Parse run_id to get RUN_ID and ITER separately
    # Format of run_id: "{RUN_ID}_{ITER}" (e.g., "01_34.4")
    parts = run_id.rsplit('_', 1)
    if len(parts) == 2:
        run_part, iter_part = parts
        output_path = f"results/deephp_backbone_final_{run_part}_{model_name}_{iter_part}.pth"
    else:
        # Fallback if parsing fails
        output_path = f"results/deephp_backbone_final_{run_id}_{model_name}.pth"
    
    print(f"Regenerating backbone with weighted averaging...")
    weighted_average_backbone_weights(fold_paths, fold_weights, output_path)
    
    print(f"✓ Successfully regenerated weighted backbone!")
    print(f"  Output: {output_path}")
    print()
    
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
