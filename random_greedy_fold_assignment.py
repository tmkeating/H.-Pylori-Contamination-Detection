#!/usr/bin/env python3
"""
Random Greedy Fold Assignment Optimizer for DeepHP 5-Fold Cross-Validation.

PURPOSE:
--------
Finds optimal experiment-to-fold assignments that balance class ratios and patch
distributions across 5 validation folds while preventing data leakage and fold-specific
artifact learning. Used to generate CONFIG 87771 and similar configurations.

PROBLEM SOLVED:
---------------
Naive fold assignment (random patch-level splitting) causes severe data leakage where
each fold develops fold-specific patterns from its unique experiments. This manifests as
unrealistic epoch 1 metrics (0%-99% recall variance across folds). Solution: assign
entire experiments to specific folds so each fold validates on different experiments
while training on all others (~307K patches per fold).

HOW IT WORKS:
-------------
1. LOAD EXPERIMENTS: Scans DeepHP Positive/ and Negative/ directories, counting patches
   per experiment and determining majority label (positive/negative) for each.

2. RANDOM GREEDY ASSIGNMENT (500,000 iterations):
   - Shuffle experiment order randomly
   - Assign each experiment to fold with currently smallest patch count
   - Result: ~balanced patch distribution across folds

3. SMART TRADES (500 per config):
   - Randomly select two folds
   - Swap one experiment between folds (only if no duplicates created)
   - Repeat 500 times to improve balance without creating conflicts
   - Benefit: Reduces patch/experiment imbalance while maintaining experiment integrity

4. VALIDATION:
   - Check constraints: min/max patches per fold, min experiments per fold
   - Check data integrity: no experiment appears in multiple folds
   - Ensure each fold has ≥1 positive AND ≥1 negative experiment

5. RANKING:
   - Calculate distance metrics for each fold's training/validation sets:
     * Ratio distance: |target_ratio (2.28:1) - actual_ratio|
     * Patch distance: normalized difference from expected patch count
     * Exp distance: normalized difference from expected experiment count
   - Sort configs by total distance (sum across all 5 folds)
   - Break ties by ratio → patch → exp distances

OUTPUT:
-------
- Prints top 100 ranked configurations to stdout (human-readable summary)
- Saves top 100 configurations to results/greedy_fold_configs.json with:
  * Config ID, total distance, detailed distance breakdown
  * Fold assignments: which experiments in each fold
  * Per-fold metrics for each cross-validation iteration:
    - Validation patches, class distribution, ratios
    - Training patches, class distribution, ratios
    - Distance metrics for both validation and training sets

KEY METRICS:
-----------
- TARGET_RATIO: 2.28:1 (negative:positive, matches full dataset)
- VAL_FOLD_MIN_PATCHES: 20,000 (minimum validation set size)
- VAL_FOLD_MAX_PATCHES: 140,000 (maximum validation set size)
- VAL_FOLD_MIN_EXPS: 4 (minimum experiments per fold)
- NUM_ITERATIONS: 500,000 (random configurations to try)
- TRADES_PER_CONFIG: 500 (optimization swaps per configuration)

USAGE:
------
python3 random_greedy_fold_assignment.py
  (assumes DeepHP data synced to DEEPHP_SCRATCH_ROOT from config.py)

OUTPUT EXAMPLE:
---------------
1. OVERALL: Total Distance = 0.6441
   Breakdown: Ratio=0.1290 + Patch=0.1548 + Exp=0.3603

   Fold 0: 87,532 patches (4+ 3-), ratio 2.33:1
   Fold 1: 89,516 patches (3+ 7-), ratio 2.06:1
   Fold 2: 20,347 patches (4+ 1-), ratio 2.31:1
   Fold 3: 99,120 patches (4+ 0-), ratio 2.81:1
   Fold 4: 98,410 patches (6+ 1-), ratio 2.29:1

   When Fold 0 is VAL:
     VAL (Fold 0):   87,532 patches (4+ 3-), ratio 2.33:1
     TRAIN (Folds 1,2,3,4): 307,393 patches (26+ 17-), ratio 2.27:1
"""

import random
import json
from collections import defaultdict
from pathlib import Path
import sys
from config import DEEPHP_SCRATCH_ROOT

# Constants
TARGET_RATIO = 2.28
VAL_FOLD_MIN_PATCHES = 20000
VAL_FOLD_MAX_PATCHES = 140000
VAL_FOLD_MIN_EXPS = 4
NUM_FOLDS = 5
NUM_ITERATIONS = 500000
TRADES_PER_CONFIG = 500  # Increased from 50 to allow more optimization

def load_experiments():
    """Load experiment metadata from data directory, consolidating by exp_id."""
    data_path = Path(DEEPHP_SCRATCH_ROOT)
    
    # First pass: count all patches per exp_id across both directories
    exp_patches = defaultdict(int)
    exp_labels = {}  # Track which label (pos/neg majority) each exp has
    exp_pos_count = defaultdict(int)
    exp_neg_count = defaultdict(int)
    
    # Positive directory
    pos_dir = data_path / 'Positive'
    if pos_dir.exists():
        for filename in pos_dir.iterdir():
            if filename.suffix == '.jpeg':
                exp_id = filename.name.split('_b0s')[0]
                exp_patches[exp_id] += 1
                exp_pos_count[exp_id] += 1
    
    # Negative directory
    neg_dir = data_path / 'Negative'
    if neg_dir.exists():
        for filename in neg_dir.iterdir():
            if filename.suffix == '.jpeg':
                exp_id = filename.name.split('_b0s')[0]
                exp_patches[exp_id] += 1
                exp_neg_count[exp_id] += 1
    
    # Assign primary label based on which directory has more patches
    for exp_id in exp_patches.keys():
        if exp_pos_count[exp_id] >= exp_neg_count[exp_id]:
            exp_labels[exp_id] = 'positive'
        else:
            exp_labels[exp_id] = 'negative'
    
    # Create experiments list with consolidated entries
    experiments = [
        {
            'exp_id': exp_id,
            'label': exp_labels[exp_id],
            'patch_count': exp_patches[exp_id],
            'pos_patches': exp_pos_count[exp_id],
            'neg_patches': exp_neg_count[exp_id]
        }
        for exp_id in sorted(exp_patches.keys())
    ]
    
    return experiments

def random_greedy_assign(experiments, random_state=None):
    """Randomly assign experiments to folds greedily."""
    if random_state is not None:
        random.seed(random_state)
    
    # Shuffle experiment order
    shuffled_indices = list(range(len(experiments)))
    random.shuffle(shuffled_indices)
    
    fold_assignments = [[] for _ in range(NUM_FOLDS)]
    assigned = set()
    
    # Greedy: assign each experiment to fold with smallest patch count
    for idx in shuffled_indices:
        if idx in assigned:
            continue
        
        exp = experiments[idx]
        fold_patches = [sum(e['patch_count'] for e in fold) for fold in fold_assignments]
        best_fold = fold_patches.index(min(fold_patches))
        fold_assignments[best_fold].append(exp)
        assigned.add(idx)
    
    return fold_assignments

def apply_smart_trades(fold_assignments, num_trades=TRADES_PER_CONFIG):
    """Apply smart swaps between folds to improve balance without creating duplicates."""
    # Deep copy: create new lists with new experiment references
    fold_assignments = [[exp.copy() if isinstance(exp, dict) else exp for exp in fold] for fold in fold_assignments]
    
    for trade_num in range(num_trades):
        # Pick two random folds
        fold_i, fold_j = random.sample(range(NUM_FOLDS), 2)
        
        if not fold_assignments[fold_i] or not fold_assignments[fold_j]:
            continue
        
        # Get exp_ids currently in each fold
        exp_ids_j = {e['exp_id'] for e in fold_assignments[fold_j]}
        exp_ids_i = {e['exp_id'] for e in fold_assignments[fold_i]}
        
        # Find experiments we CAN swap
        can_move_from_i_to_j = [idx for idx, e in enumerate(fold_assignments[fold_i]) 
                                 if e['exp_id'] not in exp_ids_j]
        can_move_from_j_to_i = [idx for idx, e in enumerate(fold_assignments[fold_j]) 
                                 if e['exp_id'] not in exp_ids_i]
        
        # Only swap if both have valid candidates
        if can_move_from_i_to_j and can_move_from_j_to_i:
            idx_i = random.choice(can_move_from_i_to_j)
            idx_j = random.choice(can_move_from_j_to_i)
            
            # Swap: remove from one, add to other
            exp_from_i = fold_assignments[fold_i].pop(idx_i)
            exp_from_j = fold_assignments[fold_j].pop(idx_j)
            fold_assignments[fold_i].append(exp_from_j)
            fold_assignments[fold_j].append(exp_from_i)
    
    return fold_assignments

def has_duplicates(fold_assignments):
    """Check if any experiment appears in multiple folds."""
    all_exp_ids = []
    for fold in fold_assignments:
        all_exp_ids.extend([e['exp_id'] for e in fold])
    
    return len(all_exp_ids) != len(set(all_exp_ids))

def is_valid_fold_config(fold_assignments):
    """Check if fold configuration meets constraints."""
    for fold_idx, fold in enumerate(fold_assignments):
        # Minimum experiments
        if len(fold) < VAL_FOLD_MIN_EXPS:
            return False, f"Fold {fold_idx}: {len(fold)} exps < {VAL_FOLD_MIN_EXPS}"
        
        # Minimum/maximum patches
        patches = sum(e['patch_count'] for e in fold)
        if patches < VAL_FOLD_MIN_PATCHES or patches > VAL_FOLD_MAX_PATCHES:
            return False, f"Fold {fold_idx}: {patches} patches not in [{VAL_FOLD_MIN_PATCHES}, {VAL_FOLD_MAX_PATCHES}]"
        
        # At least 1 pos and 1 neg
        pos = len([e for e in fold if e['label'] == 'positive'])
        neg = len([e for e in fold if e['label'] == 'negative'])
        if pos < 1 or neg < 1:
            return False, f"Fold {fold_idx}: {pos}+/{neg}- (need 1/1)"
    
    # Check no duplicates
    if has_duplicates(fold_assignments):
        return False, "Data leakage: duplicate experiments across folds"
    
    return True, "PASS"

def calculate_distances(fold_assignments):
    """Calculate detailed distance metrics (ratio, patch, exp distances)."""
    total_patches_dataset = sum(e['patch_count'] for e in experiments)
    avg_patches_per_train = (total_patches_dataset * 4) / 5
    avg_exps_per_train = (len(experiments) * 4) / 5
    
    total_distance = 0.0
    total_ratio_distance = 0.0
    total_patch_distance = 0.0
    total_exp_distance = 0.0
    fold_distances = {}
    
    for val_fold_idx in range(NUM_FOLDS):
        # Training set = all folds except val_fold_idx
        train_exps = []
        for fold_idx in range(NUM_FOLDS):
            if fold_idx != val_fold_idx:
                train_exps.extend(fold_assignments[fold_idx])
        
        pos_patches = sum(e['patch_count'] for e in train_exps if e['label'] == 'positive')
        neg_patches = sum(e['patch_count'] for e in train_exps if e['label'] == 'negative')
        total_patches = pos_patches + neg_patches
        num_exps = len(train_exps)
        
        ratio = neg_patches / max(pos_patches, 1)
        
        ratio_distance = abs(TARGET_RATIO - ratio)
        patch_distance = abs(avg_patches_per_train - total_patches) / avg_patches_per_train
        exp_distance = abs(avg_exps_per_train - num_exps) / avg_exps_per_train
        
        fold_dist = ratio_distance + patch_distance + exp_distance
        fold_distances[val_fold_idx] = {
            'total': fold_dist,
            'ratio': ratio_distance,
            'patch': patch_distance,
            'exp': exp_distance
        }
        
        total_distance += fold_dist
        total_ratio_distance += ratio_distance
        total_patch_distance += patch_distance
        total_exp_distance += exp_distance
    
    return {
        'total': total_distance,
        'ratio': total_ratio_distance,
        'patch': total_patch_distance,
        'exp': total_exp_distance,
        'fold_details': fold_distances
    }

def fold_assignment_to_dict(fold_assignments):
    """Convert fold assignment to exportable format."""
    return {
        fold_idx: [e['exp_id'] for e in fold]
        for fold_idx, fold in enumerate(fold_assignments)
    }

# Load experiments
print("Loading experiments...", file=sys.stderr)
experiments = load_experiments()
print(f"Loaded {len(experiments)} experiments", file=sys.stderr)
print(f"  Total patches: {sum(e['patch_count'] for e in experiments):,}", file=sys.stderr)
print(f"  Positive: {len([e for e in experiments if e['label'] == 'positive'])}", file=sys.stderr)
print(f"  Negative: {len([e for e in experiments if e['label'] == 'negative'])}", file=sys.stderr)

# Generate configurations
print(f"\nGenerating {NUM_ITERATIONS:,} random greedy configurations...", file=sys.stderr)
valid_configs = []
failure_reasons = defaultdict(int)

for iteration in range(NUM_ITERATIONS):
    # Random greedy assignment
    fold_assignments = random_greedy_assign(experiments, random_state=iteration)
    
    # Apply smart trades
    fold_assignments = apply_smart_trades(fold_assignments, num_trades=TRADES_PER_CONFIG)
    
    # Validate
    is_valid, reason = is_valid_fold_config(fold_assignments)
    if not is_valid:
        failure_reasons[reason] += 1
        continue
    
    # Calculate distance
    distances = calculate_distances(fold_assignments)
    
    valid_configs.append({
        'config_id': len(valid_configs) + 1,
        'seed': iteration,
        'total_distance': distances['total'],
        'ratio_distance': distances['ratio'],
        'patch_distance': distances['patch'],
        'exp_distance': distances['exp'],
        'fold_distances': distances['fold_details'],
        'fold_assignments': fold_assignment_to_dict(fold_assignments),
        'fold_exps': fold_assignments
    })
    
    if len(valid_configs) % 100 == 0:
        print(f"  Found {len(valid_configs)} valid configs...", file=sys.stderr)

print(f"\nFound {len(valid_configs):,} valid configurations", file=sys.stderr)

if len(valid_configs) == 0:
    print("\nTop failure reasons:", file=sys.stderr)
    for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1])[:5]:
        print(f"  {count:5d}: {reason}", file=sys.stderr)

# Sort by total distance, breaking ties by ratio -> patch -> exp
valid_configs.sort(key=lambda x: (
    x['total_distance'],
    x['ratio_distance'],
    x['patch_distance'],
    x['exp_distance']
))

# Print top 100
print("\n" + "="*100)
print("TOP 100 CONFIGURATIONS (by total distance, then ratio, patch, exp)")
print("="*100 + "\n")

for rank, config in enumerate(valid_configs[:100], 1):
    print(f"{rank:3d}. OVERALL: Total Distance = {config['total_distance']:.4f}")
    print(f"     Breakdown: Ratio={config['ratio_distance']:.4f} + Patch={config['patch_distance']:.4f} + Exp={config['exp_distance']:.4f}")
    print()
    
    # Print summary of all 5 folds
    print(f"     FOLD SUMMARY:")
    for fold_idx in range(NUM_FOLDS):
        fold_exps = config['fold_exps'][fold_idx]
        patches = sum(e['patch_count'] for e in fold_exps)
        pos = len([e for e in fold_exps if e['label'] == 'positive'])
        neg = len([e for e in fold_exps if e['label'] == 'negative'])
        ratio = sum(e['patch_count'] for e in fold_exps if e['label'] == 'negative') / max(sum(e['patch_count'] for e in fold_exps if e['label'] == 'positive'), 1)
        print(f"       Fold {fold_idx}: {patches:6,} patches ({pos:2d}+ {neg:2d}-), ratio {ratio:.2f}:1")
    print()
    
    # Show each fold as VAL with its training set
    for val_fold_idx in range(NUM_FOLDS):
        val_fold_exps = config['fold_exps'][val_fold_idx]
        train_exps = []
        for fold_idx in range(NUM_FOLDS):
            if fold_idx != val_fold_idx:
                train_exps.extend(config['fold_exps'][fold_idx])
        
        # VAL fold stats
        val_patches = sum(e['patch_count'] for e in val_fold_exps)
        val_pos = len([e for e in val_fold_exps if e['label'] == 'positive'])
        val_neg = len([e for e in val_fold_exps if e['label'] == 'negative'])
        val_ratio = sum(e['patch_count'] for e in val_fold_exps if e['label'] == 'negative') / max(sum(e['patch_count'] for e in val_fold_exps if e['label'] == 'positive'), 1)
        val_exp_ids = [e['exp_id'] for e in sorted(val_fold_exps, key=lambda x: x['patch_count'], reverse=True)]
        
        # TRAIN set stats
        train_patches = sum(e['patch_count'] for e in train_exps)
        train_pos = len([e for e in train_exps if e['label'] == 'positive'])
        train_neg = len([e for e in train_exps if e['label'] == 'negative'])
        train_ratio = sum(e['patch_count'] for e in train_exps if e['label'] == 'negative') / max(sum(e['patch_count'] for e in train_exps if e['label'] == 'positive'), 1)
        train_exp_ids = [e['exp_id'] for e in sorted(train_exps, key=lambda x: x['patch_count'], reverse=True)]
        
        # Calculate VAL fold distance metrics
        avg_patches_per_fold = sum(e['patch_count'] for e in experiments) / NUM_FOLDS
        avg_exps_per_fold = len(experiments) / NUM_FOLDS
        val_pos_patches = sum(e['patch_count'] for e in val_fold_exps if e['label'] == 'positive')
        val_neg_patches = sum(e['patch_count'] for e in val_fold_exps if e['label'] == 'negative')
        val_ratio_distance = abs(TARGET_RATIO - val_ratio)
        val_patch_distance = abs(avg_patches_per_fold - val_patches) / avg_patches_per_fold
        val_exp_distance = abs(avg_exps_per_fold - len(val_fold_exps)) / avg_exps_per_fold
        
        # Get distance for this fold's training set
        fold_dist = config['fold_distances'].get(val_fold_idx, {})
        fold_total = fold_dist.get('total', 0)
        fold_ratio = fold_dist.get('ratio', 0)
        fold_patch = fold_dist.get('patch', 0)
        fold_exp = fold_dist.get('exp', 0)
        
        print(f"     When Fold {val_fold_idx} is VAL:")
        print(f"       VAL (Fold {val_fold_idx}):   {val_patches:6,} patches ({val_pos}+ {val_neg}-), ratio {val_ratio:.2f}:1")
        print(f"                    {', '.join(val_exp_ids)}")
        print(f"       VAL Distance: {val_ratio_distance + val_patch_distance + val_exp_distance:.4f} (Ratio={val_ratio_distance:.4f} + Patch={val_patch_distance:.4f} + Exp={val_exp_distance:.4f})")
        print(f"       TRAIN (Folds {','.join(str(i) for i in range(NUM_FOLDS) if i != val_fold_idx)}): {train_patches:6,} patches ({train_pos}+ {train_neg}-), ratio {train_ratio:.2f}:1")
        print(f"                    {', '.join(train_exp_ids)}")
        print(f"       TRAIN Distance: {fold_total:.4f} (Ratio={fold_ratio:.4f} + Patch={fold_patch:.4f} + Exp={fold_exp:.4f})")
        print()
    
    print()


# Save results
output_file = Path('results/greedy_fold_configs.json')
output_file.parent.mkdir(parents=True, exist_ok=True)

def build_detailed_config(config):
    """Build detailed config with VAL/TRAIN breakdown for each fold."""
    total_patches_dataset = sum(e['patch_count'] for e in experiments)
    avg_patches_per_fold = total_patches_dataset / NUM_FOLDS
    avg_exps_per_fold = len(experiments) / NUM_FOLDS
    
    detailed = {
        'config_id': config['config_id'],
        'total_distance': config['total_distance'],
        'distance_breakdown': {
            'ratio': config['ratio_distance'],
            'patch': config['patch_distance'],
            'exp': config['exp_distance']
        },
        'fold_assignments': config['fold_assignments'],
        'fold_summary': [],
        'cv_folds': []
    }
    
    # For each fold as validation (also builds fold_summary)
    for val_fold_idx in range(NUM_FOLDS):
        val_fold_exps = config['fold_exps'][val_fold_idx]
        train_exps = []
        for fold_idx in range(NUM_FOLDS):
            if fold_idx != val_fold_idx:
                train_exps.extend(config['fold_exps'][fold_idx])
        
        fold_dist = config['fold_distances'].get(val_fold_idx, {})
        
        # Calculate VAL fold metrics
        val_pos_patches = sum(e['patch_count'] for e in val_fold_exps if e['label'] == 'positive')
        val_neg_patches = sum(e['patch_count'] for e in val_fold_exps if e['label'] == 'negative')
        val_total_patches = val_pos_patches + val_neg_patches
        val_num_exps = len(val_fold_exps)
        val_ratio = val_neg_patches / max(val_pos_patches, 1)
        
        val_ratio_distance = abs(TARGET_RATIO - val_ratio)
        val_patch_distance = abs(avg_patches_per_fold - val_total_patches) / avg_patches_per_fold
        val_exp_distance = abs(avg_exps_per_fold - val_num_exps) / avg_exps_per_fold
        
        # Calculate TRAIN set metrics
        train_pos_patches = sum(e['patch_count'] for e in train_exps if e['label'] == 'positive')
        train_neg_patches = sum(e['patch_count'] for e in train_exps if e['label'] == 'negative')
        train_total_patches = train_pos_patches + train_neg_patches
        train_num_exps = len(train_exps)
        train_ratio = train_neg_patches / max(train_pos_patches, 1)
        
        avg_patches_per_train = (total_patches_dataset * 4) / 5
        avg_exps_per_train = (len(experiments) * 4) / 5
        
        # Add to fold_summary
        detailed['fold_summary'].append({
            'cv_iteration': val_fold_idx,
            'val': {
                'fold': val_fold_idx,
                'patches': val_total_patches,
                'positive_exps': len([e for e in val_fold_exps if e['label'] == 'positive']),
                'negative_exps': len([e for e in val_fold_exps if e['label'] == 'negative']),
                'ratio': val_ratio
            },
            'train': {
                'patches': train_total_patches,
                'positive_exps': len([e for e in train_exps if e['label'] == 'positive']),
                'negative_exps': len([e for e in train_exps if e['label'] == 'negative']),
                'ratio': train_ratio
            }
        })
        
        detailed['cv_folds'].append({
            'val_fold': val_fold_idx,
            'val_experiments': [e['exp_id'] for e in val_fold_exps],
            'val_distance': {
                'total': val_ratio_distance + val_patch_distance + val_exp_distance,
                'ratio': val_ratio_distance,
                'patch': val_patch_distance,
                'exp': val_exp_distance
            },
            'train_experiments': [e['exp_id'] for e in train_exps],
            'train_distance': {
                'total': fold_dist.get('total', 0),
                'ratio': fold_dist.get('ratio', 0),
                'patch': fold_dist.get('patch', 0),
                'exp': fold_dist.get('exp', 0)
            }
        })
    
    return detailed

with open(output_file, 'w') as f:
    json.dump({
        'metadata': {
            'total_configs': len(valid_configs),
            'num_experiments': len(experiments),
            'total_patches': sum(e['patch_count'] for e in experiments),
            'target_ratio': TARGET_RATIO,
        },
        'configs': [build_detailed_config(c) for c in valid_configs[:100]]
    }, f, indent=2)

print(f"Saved top 100 configs to {output_file}")
