#!/usr/bin/env python3
"""
Test Temperature Scaling: Comparison Script

Runs calibration and applies thresholds both with and without temperature scaling,
then compares the results to evaluate if temperature scaling helps.

Usage:
    python3 test_temperature_scaling.py --run 02_34.3
"""

import subprocess
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    result = subprocess.run(cmd, shell=True, capture_output=False, executable='/bin/bash')
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description='Test temperature scaling vs baseline')
    parser.add_argument('--run', required=True, help='Run ID (e.g., 02_34.3)')
    args = parser.parse_args()
    
    results_dir = Path('/home/tkeating/model/H.-Pylori-Contamination-Detection/results')
    base_cmd = 'cd /home/tkeating/model/H.-Pylori-Contamination-Detection && source venv/bin/activate &&'
    
    print("\n" + "="*80)
    print("TEMPERATURE SCALING TEST")
    print("="*80)
    print(f"Run ID: {args.run}")
    
    # Step 1: Run baseline calibration
    print(f"\n[STEP 1] Baseline Calibration (no temperature scaling)")
    cmd1 = f"{base_cmd} python3 calibrate_per_fold_thresholds.py --run {args.run}"
    if not run_command(cmd1, "Running baseline calibration"):
        print("ERROR: Baseline calibration failed")
        return
    
    # Step 2: Run temperature-scaled calibration
    print(f"\n[STEP 2] Temperature-Scaled Calibration")
    cmd2 = f"{base_cmd} python3 calibrate_per_fold_thresholds_with_temperature.py --run {args.run}"
    if not run_command(cmd2, "Running temperature-scaled calibration"):
        print("ERROR: Temperature-scaled calibration failed")
        return
    
    # Step 3: Load and compare calibration results
    print(f"\n[STEP 3] Comparing Calibration Results")
    print("="*80)
    
    baseline_file = results_dir / f"{args.run}_calibrated_thresholds.json"
    temp_file = results_dir / f"{args.run}_calibrated_thresholds_temp.json"
    
    with open(baseline_file) as f:
        baseline_config = json.load(f)
    
    with open(temp_file) as f:
        temp_config = json.load(f)
    
    print(f"\n{'Fold':<6} {'Baseline T':<12} {'Baseline Th':<14} {'Baseline F1':<12} "
          f"{'Temp T':<8} {'Temp Th':<12} {'Temp F1':<12} {'Improvement':<12}")
    print("-" * 100)
    
    total_improvement = 0
    for fold_idx in range(5):
        fold_key = str(fold_idx)
        
        baseline = baseline_config['fold_thresholds'][fold_key]
        temp = temp_config['fold_thresholds'][fold_key]
        
        baseline_f1 = baseline['f1_score']
        temp_f1 = temp['f1_score']
        improvement = ((temp_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
        total_improvement += improvement
        
        baseline_temp = baseline.get('temperature', 1.0)
        temp_temp = temp.get('temperature', 1.0)
        
        print(f"{fold_idx:<6} {baseline_temp:<12.1f} {baseline['threshold']:<14.3f} {baseline_f1:<12.4f} "
              f"{temp_temp:<8.1f} {temp['threshold']:<12.3f} {temp_f1:<12.4f} {improvement:<12.1f}%")
    
    print("-" * 100)
    print(f"Average Improvement: {total_improvement/5:.1f}%")
    
    # Step 4: Apply thresholds (baseline)
    print(f"\n[STEP 4] Applying Baseline Thresholds")
    cmd3 = f"{base_cmd} python3 apply_calibrated_thresholds.py --run {args.run}"
    if not run_command(cmd3, "Applying baseline thresholds"):
        print("WARNING: Baseline threshold application had issues")
    
    # Step 5: Apply thresholds (temperature-scaled)
    print(f"\n[STEP 5] Applying Temperature-Scaled Thresholds")
    cmd4 = f"{base_cmd} python3 apply_calibrated_thresholds_with_temperature.py --run {args.run}"
    if not run_command(cmd4, "Applying temperature-scaled thresholds"):
        print("WARNING: Temperature-scaled threshold application had issues")
    
    # Step 6: Summary
    print(f"\n[SUMMARY]")
    print("="*80)
    print("\nTo evaluate temperature scaling results:")
    print(f"  Baseline files: {args.run}_calibrated_thresholds.json")
    print(f"                  {args.run}_f*_predictions_corrected.json")
    print(f"  Temperature files: {args.run}_calibrated_thresholds_temp.json")
    print(f"                     {args.run}_*_f*_predictions_corrected_temp.json")
    print("\nNext steps:")
    print("  1. Run weighted_ensemble.py on baseline predictions")
    print("  2. Run weighted_ensemble_with_temperature.py on temperature-scaled predictions")
    print("  3. Compare ensemble F1 scores to evaluate temperature scaling effectiveness")

if __name__ == '__main__':
    main()
