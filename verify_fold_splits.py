#!/usr/bin/env python3
"""
Verify that fold splits are actually different across folds.
This checks if the stratification is working correctly.
"""

import os
from dataset_deepHP import DeepHPDataset
from config import DEEPHP_DATASET_ROOT

def main():
    num_folds = 5
    
    print("="*80)
    print("VERIFYING FOLD SPLITS ACROSS ALL 5 FOLDS")
    print("="*80)
    
    all_fold_val_indices = {}
    
    for fold_idx in range(num_folds):
        print(f"\n[FOLD {fold_idx}]")
        print("-" * 80)
        
        # Create both train and val datasets for this fold
        train_ds = DeepHPDataset(
            root_dir=DEEPHP_DATASET_ROOT,
            fold=fold_idx,
            num_folds=num_folds,
            train=True
        )
        
        val_ds = DeepHPDataset(
            root_dir=DEEPHP_DATASET_ROOT,
            fold=fold_idx,
            num_folds=num_folds,
            train=False
        )
        
        train_indices = set(train_ds.indices)
        val_indices = set(val_ds.indices)
        
        # Check for leakage within this fold
        overlap = train_indices & val_indices
        if overlap:
            print(f"❌ ERROR: {len(overlap)} patches in BOTH train and val for fold {fold_idx}!")
        else:
            print(f"✓ No overlap within fold {fold_idx}")
        
        # Check sizes
        print(f"  Train: {len(train_indices):,} patches")
        print(f"  Val:   {len(val_indices):,} patches")
        print(f"  Total: {len(train_indices) + len(val_indices):,} patches")
        
        # Store val indices for cross-fold comparison
        all_fold_val_indices[fold_idx] = val_indices
        
        # Extract and report experiment composition
        val_experiments = set()
        for idx in val_indices:
            path, _ = val_ds.samples[idx]
            filename = os.path.basename(path)
            exp_id = filename.split('_b0s')[0]
            val_experiments.add(exp_id)
        
        print(f"  Val experiments: {sorted(val_experiments)}")
    
    print("\n" + "="*80)
    print("CROSS-FOLD VALIDATION SET COMPARISON")
    print("="*80)
    
    # Check that each fold's validation set is UNIQUE and doesn't overlap with others
    for fold_a in range(num_folds):
        for fold_b in range(fold_a + 1, num_folds):
            val_a = all_fold_val_indices[fold_a]
            val_b = all_fold_val_indices[fold_b]
            overlap = val_a & val_b
            
            if overlap:
                print(f"❌ ERROR: Folds {fold_a} and {fold_b} have {len(overlap)} overlapping validation patches!")
            else:
                print(f"✓ Fold {fold_a} val and Fold {fold_b} val are disjoint")
    
    # Check that validation sets together cover all unique patches
    total_unique_patches = set()
    for fold_idx in range(num_folds):
        total_unique_patches.update(all_fold_val_indices[fold_idx])
    
    print(f"\nTotal unique patches across all validation sets: {len(total_unique_patches):,}")
    
    # The total should be NUM_PATCHES / NUM_FOLDS (approximately)
    # For 5 folds of ~79k patches each = ~395k total
    expected_per_fold = len(total_unique_patches) // num_folds
    print(f"Expected per fold: ~{expected_per_fold:,}")
    
    # Check fold statistics
    print("\n" + "="*80)
    print("VALIDATION SET STATISTICS")
    print("="*80)
    
    for fold_idx in range(num_folds):
        val_ds = DeepHPDataset(
            root_dir=DEEPHP_DATASET_ROOT,
            fold=fold_idx,
            num_folds=num_folds,
            train=False
        )
        
        print(f"\nFold {fold_idx}:")
        print(f"  Total patches: {len(val_ds)}")
        print(f"  Positive patches: {val_ds.statistics['positive']}")
        print(f"  Negative patches: {val_ds.statistics['negative']}")
        print(f"  Ratio (Neg:Pos): {val_ds.statistics['imbalance_ratio']:.2f}:1")

if __name__ == "__main__":
    main()
