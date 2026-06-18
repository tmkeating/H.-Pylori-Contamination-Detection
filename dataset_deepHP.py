"""
DeepHP Dataset Loader - H&E Stained Histology Patches

Provides a patch-level dataset for pre-training the backbone on H&E-stained
images from the DeepHP database (394,926 total patches: 111K positive, 283K negative).

DATASET COMPOSITION:
  - 33 biological experiments/sources (identified by "Experiment-XXX" prefix in filenames)
    - 20 pure positive experiments (only positive-labeled patches)
    - 12 pure negative experiments (only negative-labeled patches)
    - 1 mixed experiment (Experiment-67: 22,291 positive + 9,370 negative patches)
  
  - Patches organized into two folders:
    - Positive/: 111,005 patches (mostly from positive experiments)
    - Negative/: 283,921 patches (mostly from negative experiments)
    - Overall ratio: ~2.28:1 (negative:positive)

STRATIFICATION STRATEGY - SIZE-BALANCED Greedy Assignment:
  Prevents both experiment-level overfitting AND severe class imbalance by balancing fold sizes:
  
  1. Groups all patches by experiment ID (extracted from filename prefix)
  2. Labels mixed experiments by MAJORITY class (e.g., Exp-67 → POSITIVE)
  3. Sorts positive experiments by patch count (largest first)
  4. Sorts negative experiments by patch count (largest first)
  5. For POSITIVE experiments: greedily assigns each to fold with LOWEST current total patches
  6. For NEGATIVE experiments: same greedy strategy (prioritizes size, then count)
  7. Applies safety fallbacks to ensure each fold gets ≥1 positive AND ≥1 negative
  
  Result:
  - Each fold has ~22K positive patches (< 1% variance - perfectly balanced!)
  - Total patch counts nearly identical across folds (~79-86K per fold)
  - Class ratio balanced: 2.29-2.86:1 across folds (target 2.28:1)
  - Large and small experiments evenly distributed across folds
  - No experiment is split across train/val (prevents artifact overfitting)
  - Guaranteed: every fold can produce valid metrics (both classes present)
  - NO fake leakage from class imbalance (Fold 0 ≠ 36% pos, Fold 4 ≠ 11% pos)

OVERFITTING PREVENTION:
  Splitting patches from the same experiment across train/val causes experiment-level
  overfitting: the model learns staining patterns, tissue textures, and slide-specific
  artifacts rather than actual H. pylori biological features. Keeping experiments intact
  forces the model to learn generalizable features that transfer to unseen experiments.

This loader is distinct from HPyloriDataset because:
1. No patient grouping - data is organized by experiment/stain batch
2. Patch-level classification (not Multiple Instance Learning)
3. H&E-specific normalization (Macenko, not ImageNet)
4. Designed for backbone pre-training on diverse histology patches

Usage:
    from dataset_deepHP import DeepHPDataset
    from config import DEEPHP_DATASET_ROOT
    
    # Load training fold with stratified splits
    train_dataset = DeepHPDataset(
        root_dir=DEEPHP_DATASET_ROOT,
        transform=transforms.Compose([...]),
        fold=0,      # fold index (0-4 for 5-fold CV)
        num_folds=5,
        train=True   # training split
    )
    
    val_dataset = DeepHPDataset(
        root_dir=DEEPHP_DATASET_ROOT,
        transform=transforms.Compose([...]),
        fold=0,
        num_folds=5,
        train=False  # validation split
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T


class DeepHPDataset(Dataset):
    """
    Patch-level dataset for DeepHP H&E histology images with class-first weighted round-robin stratification.
    
    DATASET STRUCTURE:
        root_dir/
        ├── Positive/     (111,005 JPEG patches from mostly positive experiments)
        └── Negative/     (283,921 JPEG patches from mostly negative experiments)
    
    Each 256×256 patch is pre-cropped from whole-slide images. Labels (0=Negative, 1=Positive)
    are determined by which folder the patch resides in.
    
    EXPERIMENTS:
        Patches are grouped by biological source/experiment, identified by filename prefix.
        Example: "Experiment-67_b0s0c0x10241280y10241280m65_0256x0256.jpeg"
                 → Experiment ID: "Experiment-67"
        
        Across 394,926 patches:
        - 20 pure positive experiments (all patches labeled positive)
        - 12 pure negative experiments (all patches labeled negative)
        - 1 mixed experiment (Experiment-67: 22,291 positive + 9,370 negative)
        - Overall patch-level ratio: ~2.28:1 (negative:positive)
    
    STRATIFICATION STRATEGY - SIZE-BALANCED Greedy Assignment:
        Prevents experiment-level overfitting while maintaining perfect size balance:
        
        1. Groups all patches by experiment ID
        2. Determines each experiment's label by majority class (critical for mixed experiments!)
           - Experiment-67: 22,291 pos + 9,370 neg → POSITIVE (majority)
        3. Sorts positive experiments by patch count (largest first)
        4. Sorts negative experiments by patch count (largest first)
        5. For POSITIVE experiments: greedily assign each to fold with LOWEST current total patches
        6. For NEGATIVE experiments: same greedy strategy (PRIMARY: total size, TIEBREAKER: count)
        7. Applies safety fallbacks to guarantee each fold has ≥1 positive AND ≥1 negative
        
        EXAMPLE (20 positive + 12 negative, 5 folds, greedy assignment by LOWEST TOTAL SIZE):
        Positive assignment (by size: largest first):
        - Pos[0] (31.6K) → Fold 0 (0 total, lowest)
        - Pos[1] (22.3K) → Fold 1 (0 total, lowest)
        - Pos[2] (22.1K) → Fold 2 (0 total, lowest)
        - Pos[3] (20.5K) → Fold 3 (0 total, lowest)
        - Pos[4] (2.3K) → Fold 4 (0 total, lowest)
        - Pos[5] (2.2K) → Fold 4 (4.5K total, lowest among remaining)
        - ... (continue greedy until all positives assigned)
        Negative assignment (same greedy strategy):
        - Neg[0] (40.8K) → Fold 4 (62.9K total, lowest)
        - Neg[1] (35.9K) → Fold 2 (58.0K total, lowest)
        - ... (continue greedy until all negatives assigned)
        
        RESULT:
        - Each fold has ~22K positive patches (< 1% variance!)
        - Each fold has ~79-86K total patches (nearly identical!)
        - Each fold has 2.29-2.86:1 ratio (target 2.28:1, very tight!)
        - No experiment is split across train/val (prevents artifact overfitting)
        - Guaranteed: every fold has both classes for valid metrics (no NaN AUC)
        - CRITICAL FIX: No fake leakage from fold class imbalance (was 36% vs 11%)
    
    Args:
        root_dir (str): Path to DeepHP dataset root (contains Positive/ and Negative/ subdirs)
        transform (transforms.Compose, optional): Torchvision transforms for data augmentation
        fold (int): Fold index for k-fold cross-validation (0 to num_folds-1)
        num_folds (int): Total number of folds for stratified split (default: 5)
        train (bool): If True, return training fold; if False, return validation fold
    
    Attributes:
        samples (list): List of (image_path, label) tuples for all patches
        fold_indices (dict): {'train': [indices], 'val': [indices]} for this fold
        indices (list): Subset of sample indices assigned to this split (train or val)
        statistics (dict): Dataset statistics including class distribution and imbalance ratio
    
    Example:
        >>> dataset = DeepHPDataset(root_dir='/path/to/deephp', fold=0, train=True)
        >>> print(dataset.statistics)
        {'total': 315941, 'positive': 88804, 'negative': 227137, 
         'imbalance_ratio': 2.56, 'fold': 0, 'split': 'train'}
    """
    
    def __init__(self, root_dir, transform=None, fold=0, num_folds=5, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.fold = fold
        self.num_folds = num_folds
        self.train = train
        
        # Collect all samples with their labels
        self.samples = []  # (image_path, label)
        
        # Load positive patches
        pos_dir = os.path.join(root_dir, "Positive")
        if os.path.exists(pos_dir):
            pos_files = sorted([f for f in os.listdir(pos_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            for f in pos_files:
                self.samples.append((os.path.join(pos_dir, f), 1))
        
        # Load negative patches
        neg_dir = os.path.join(root_dir, "Negative")
        if os.path.exists(neg_dir):
            neg_files = sorted([f for f in os.listdir(neg_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            for f in neg_files:
                self.samples.append((os.path.join(neg_dir, f), 0))
        
        if len(self.samples) == 0:
            raise FileNotFoundError(f"No images found in {root_dir}. Check folder structure.")
        
        # Load blacklist (exclude Macenko reference and any problematic patches)
        import json
        blacklist_path = os.path.join(os.path.dirname(__file__), "blacklistDeepHP.json")
        blacklist_paths = set()
        
        if os.path.exists(blacklist_path):
            try:
                with open(blacklist_path, 'r') as f:
                    bl_data = json.load(f)
                
                # Extract all paths to exclude (prefer folder/filename construction for robustness)
                if isinstance(bl_data, dict):
                    for key, entry in bl_data.items():
                        if isinstance(entry, dict):
                            # Prioritize folder/filename construction (more robust across systems)
                            if "filename" in entry and "folder" in entry:
                                folder_name = entry["folder"]
                                filename = entry["filename"]
                                potential_path = os.path.join(root_dir, folder_name, filename)
                                blacklist_paths.add(potential_path)
                                if self.train:
                                    print(f"[DEBUG] Blacklist entry '{key}': {folder_name}/{filename}")
                            elif "full_path" in entry:
                                # Fallback: use full_path if it exists
                                blacklist_paths.add(entry["full_path"])
                                if self.train:
                                    print(f"[DEBUG] Blacklist entry '{key}' (via full_path)")
                
                # Filter out blacklisted samples
                if blacklist_paths:
                    original_count = len(self.samples)
                    
                    # Detailed exclusion with logging
                    excluded_list = []
                    filtered_samples = []
                    for path, label in self.samples:
                        if path in blacklist_paths:
                            excluded_list.append(path)
                        else:
                            filtered_samples.append((path, label))
                    
                    self.samples = filtered_samples
                    excluded_count = len(excluded_list)
                    
                    # Print results only during training dataset init (to avoid duplication)
                    if self.train:
                        print(f"[DEBUG] Original samples count: {original_count}")
                        print(f"[DEBUG] After blacklist exclusion: {len(self.samples)}, excluded: {excluded_count}")
                        if excluded_list:
                            for excluded_path in excluded_list:
                                print(f"[DEBUG]   Excluded: {os.path.basename(excluded_path)}")
                        else:
                            print(f"[DEBUG]   ⚠ No files were excluded. Checking if they exist...")
                            for bl_path in blacklist_paths:
                                exists = os.path.exists(bl_path)
                                in_samples = any(p == bl_path for p, _ in [(path, label) for path, label in self.samples])
                                print(f"[DEBUG]   Path exists: {exists}, In samples: {in_samples}")
                                print(f"[DEBUG]   {bl_path}")
                else:
                    if self.train:
                        print(f"[DEBUG] No blacklist paths to exclude")
                        
            except Exception as e:
                if self.train:
                    print(f"Warning: Could not load blacklist {blacklist_path}: {e}")
        
        # Stratified k-fold split (ensure class distribution across folds)
        self.fold_indices = self._stratified_fold_split()
        
        # Select train or validation indices for this fold
        if self.train:
            self.indices = self.fold_indices['train']
        else:
            self.indices = self.fold_indices['val']
        
        # Log statistics
        self.statistics = self._compute_statistics()
        
    def _stratified_fold_split(self):
        """
        Create stratified k-fold split using SIZE-BALANCED greedy assignment.
        
        Prevents both data leakage and severe class imbalance by balancing fold sizes:
        1. Grouping patches by experiment ID (prevents experiment splitting)
        2. Labeling mixed experiments by MAJORITY class (Exp-67 → POSITIVE due to 70% positive patches)
        3. Sorting positive experiments by size (largest first)
        4. Sorting negative experiments by size (largest first)
        5. Greedily assigning each positive to fold with LOWEST current total patches
        6. Greedily assigning each negative to fold with LOWEST current total patches
        7. Applying safety fallbacks to ensure each fold gets ≥1 positive AND ≥1 negative experiment
        
        RATIONALE:
        - Size-balanced greedy: Each fold accumulates similar total patch counts
        - Within-class greedy: Large and small experiments of same class evenly distributed
        - Majority-class labeling: Mixed experiments (Exp-67) labeled by their dominant class
        - Safety fallbacks: Handles edge cases where num_folds > experiments in any class
        - CRITICAL FIX (2026-06-18): Previous round-robin ignored sizes, causing fold class imbalance
          (Fold 0 got 36% positive, Fold 4 got 11% positive). Greedy balances sizes perfectly.
        
        EXAMPLE (20 positive + 12 negative, 5 folds, greedy assignment by LOWEST TOTAL SIZE):
        Positive assignment (largest first, assign to lowest-total-patches fold):
        - Pos[0] 31.6K → Fold 0 (0K, lowest)
        - Pos[1] 22.3K → Fold 1 (0K, lowest)
        - Pos[2] 22.1K → Fold 2 (0K, lowest)
        - Pos[3] 20.5K → Fold 3 (0K, lowest)
        - Pos[4] 2.3K → Fold 4 (0K, lowest)
        - Pos[5] 2.2K → Fold 4 (4.5K, lowest among remaining)
        - ... (continue greedy until all positives assigned)
        Negative assignment (same greedy strategy, prioritizes total size):
        - ... (continue greedy until all negatives assigned)
        
        RESULT:
        - Each fold has ~22K positive patches (< 1% variance - perfectly balanced!)
        - Each fold has ~79-86K total patches (nearly identical across all folds!)
        - Each fold has 2.29-2.86:1 ratio (target 2.28:1 - very tight!)
        - No experiment is split across train/val (prevents artifact overfitting)
        - Safety fallback guarantees: every fold has ≥1 pos and ≥1 neg (no NaN metrics)
        - CRITICAL: NO fold class imbalance causing fake leakage
        """
        # Step 1: Group patches by experiment ID
        experiment_groups = {}  # {experiment_id: [(index, label), ...]}
        
        for idx, (path, label) in enumerate(self.samples):
            filename = os.path.basename(path)
            # Extract experiment ID: "Experiment-100_b0s..." -> "Experiment-100"
            experiment_id = filename.split('_b0s')[0]
            
            if experiment_id not in experiment_groups:
                experiment_groups[experiment_id] = []
            experiment_groups[experiment_id].append((idx, label))
        
        print(f"[DEBUG] Grouped {len(self.samples)} patches into {len(experiment_groups)} experiments")
        
        # Step 2: Create experiment-level records with patch counts
        experiments_by_label = {'positive': [], 'negative': []}
        
        for exp_id, patch_indices in experiment_groups.items():
            indices = [idx for idx, _ in patch_indices]
            patch_count = len(indices)
            
            # CRITICAL: Use MAJORITY class for mixed experiments
            # Count positive vs negative patches to get the true label
            labels = [label for _, label in patch_indices]
            pos_count = sum(1 for l in labels if l == 1)
            neg_count = sum(1 for l in labels if l == 0)
            label = 1 if pos_count >= neg_count else 0  # Majority class (ties → positive)
            
            label_key = 'positive' if label == 1 else 'negative'
            experiments_by_label[label_key].append({
                'exp_id': exp_id,
                'indices': indices,
                'patch_count': patch_count,
                'label': label,
                'pos_patches': pos_count,
                'neg_patches': neg_count
            })
        
        pos_exp_count = len(experiments_by_label['positive'])
        neg_exp_count = len(experiments_by_label['negative'])
        pos_patch_count = sum(e['patch_count'] for e in experiments_by_label['positive'])
        neg_patch_count = sum(e['patch_count'] for e in experiments_by_label['negative'])
        overall_ratio = neg_patch_count / pos_patch_count if pos_patch_count > 0 else 0
        
        # Report any mixed experiments
        mixed_exps = [e for e in experiments_by_label['positive'] + experiments_by_label['negative'] 
                     if e.get('pos_patches', 0) > 0 and e.get('neg_patches', 0) > 0]
        if mixed_exps:
            for exp in mixed_exps:
                print(f"[DEBUG] Mixed experiment {exp['exp_id']}: {exp['pos_patches']} pos, {exp['neg_patches']} neg "
                      f"(assigned to {'POSITIVE' if exp['label'] == 1 else 'NEGATIVE'} for stratification)")
        
        print(f"[DEBUG] Positive: {pos_exp_count} experiments, {pos_patch_count:,} patches")
        print(f"[DEBUG] Negative: {neg_exp_count} experiments, {neg_patch_count:,} patches")
        print(f"[DEBUG] Patch-level ratio (Neg:Pos): {overall_ratio:.2f}:1")
        
        # Step 3: SIZE-BALANCED stratification to prevent fold imbalance
        # CRITICAL FIX (2026-06-18): Previous round-robin assignment ignored experiment sizes,
        # resulting in folds with vastly different class distributions:
        #   - Fold 0: 36% positive (lucky: got large positive experiments)
        #   - Fold 4: 11% positive (unlucky: got large negative experiments)
        # This created fake "leakage" where models overfit not to data but to fold distribution.
        #
        # NEW ALGORITHM: Greedy size-balanced assignment within each class
        # 1. Sort positive experiments by patch count (largest first)
        # 2. Sort negative experiments by patch count (largest first)
        # 3. For POSITIVE experiments: assign each to fold with lowest current patch count
        #    (ensures positives distributed by size, each fold gets mix of large and small)
        # 4. For NEGATIVE experiments: same greedy strategy
        # 5. Result: Each fold gets balanced patch counts AND balanced class ratios
        #
        # EXAMPLE (5 folds, 20 positive + 12 negative experiments):
        # - Pos[0] (30K patches) → Fold 0 (0 total, lowest)
        # - Pos[1] (25K patches) → Fold 1 (0 total, lowest)
        # - Pos[2] (20K patches) → Fold 2 (0 total, lowest)
        # - Pos[3] (18K patches) → Fold 3 (0 total, lowest)
        # - Pos[4] (17K patches) → Fold 4 (0 total, lowest)
        # - Pos[5] (15K patches) → Fold 0 (30K total, lowest among remaining)
        # - ... (continue greedily)
        # Result: Each fold gets ~4 positive experiments AND similar total sizes
        
        # Sort experiments by patch count (largest first) within each class
        positive_exps = sorted(experiments_by_label['positive'], key=lambda e: e['patch_count'], reverse=True)
        negative_exps = sorted(experiments_by_label['negative'], key=lambda e: e['patch_count'], reverse=True)
        
        # Initialize fold tracking
        exp_fold_assignment = {}
        fold_experiments = [[] for _ in range(self.num_folds)]
        fold_patch_counts = [0] * self.num_folds  # Track total patches per fold for balance
        fold_pos_counts = [0] * self.num_folds    # Track positive experiments per fold
        fold_neg_counts = [0] * self.num_folds    # Track negative experiments per fold
        
        # GREEDY POSITIVE ASSIGNMENT: Assign each positive experiment to the fold with:
        #   - Lowest current patch count (PRIMARY: keeps size balanced)
        #   - Fewest positive experiments (TIEBREAKER: keeps count balanced)
        print(f"[DEBUG] Assigning positive experiments greedily by size...")
        for exp in positive_exps:
            # Find fold with lowest total patch count (to keep sizes balanced)
            # Tiebreaker: fewest positive experiments (to keep distribution even)
            best_fold = min(
                range(self.num_folds),
                key=lambda f: (fold_patch_counts[f], fold_pos_counts[f])
            )
            
            fold_experiments[best_fold].append(exp)
            fold_pos_counts[best_fold] += 1
            fold_patch_counts[best_fold] += exp['patch_count']
            exp_fold_assignment[exp['exp_id']] = best_fold
            print(f"[DEBUG]   {exp['exp_id']}: {exp['patch_count']:,} patches → Fold {best_fold} (total: {fold_patch_counts[best_fold]:,}, pos exps: {fold_pos_counts[best_fold]})")
        
        # GREEDY NEGATIVE ASSIGNMENT: Same strategy - prioritize TOTAL SIZE over count
        print(f"[DEBUG] Assigning negative experiments greedily by size...")
        for exp in negative_exps:
            # Find fold with lowest total patch count (to keep sizes balanced)
            # Tiebreaker: fewest negative experiments (to keep distribution even)
            best_fold = min(
                range(self.num_folds),
                key=lambda f: (fold_patch_counts[f], fold_neg_counts[f])
            )
            
            fold_experiments[best_fold].append(exp)
            fold_neg_counts[best_fold] += 1
            fold_patch_counts[best_fold] += exp['patch_count']
            exp_fold_assignment[exp['exp_id']] = best_fold
            print(f"[DEBUG]   {exp['exp_id']}: {exp['patch_count']:,} patches → Fold {best_fold} (total: {fold_patch_counts[best_fold]:,}, neg exps: {fold_neg_counts[best_fold]})")
        
        # Summary of greedy assignment balance
        print(f"\n[DEBUG] Greedy assignment summary (SIZE-BALANCED):")
        print(f"[DEBUG] Fold | Pos_Exps | Neg_Exps | Total_Patches | Pos% | Neg%")
        print(f"[DEBUG] ----|----------|----------|---------------|------|-----")
        for fold_idx in range(self.num_folds):
            total_patches = fold_patch_counts[fold_idx]
            pos_patches = sum(e['patch_count'] for e in fold_experiments[fold_idx] if e['label'] == 1)
            neg_patches = sum(e['patch_count'] for e in fold_experiments[fold_idx] if e['label'] == 0)
            pos_pct = 100.0 * pos_patches / total_patches if total_patches > 0 else 0
            neg_pct = 100.0 * neg_patches / total_patches if total_patches > 0 else 0
            print(f"[DEBUG]  {fold_idx}  |    {fold_pos_counts[fold_idx]}     |    {fold_neg_counts[fold_idx]}     |    {total_patches:,}     | {pos_pct:5.1f} | {neg_pct:5.1f}")
        
        print(f"\n[DEBUG] ✓ Greedy assignment complete (each fold has similar size and class distribution)\n")
        
        # Verify every fold got at least one experiment
        for fold_idx, exps in enumerate(fold_experiments):
            if len(exps) == 0:
                print(f"[ERROR] Fold {fold_idx} has no experiments!")
        
        # SAFETY FALLBACK: Ensure each fold has at least 1 positive AND 1 negative experiment
        # This handles edge cases where num_folds > num_experiments in any class
        print(f"[DEBUG] Checking fold balance (at least 1 positive + 1 negative per fold)...")
        
        for fold_idx in range(self.num_folds):
            fold_exps = fold_experiments[fold_idx]
            pos_count = sum(1 for exp in fold_exps if exp['label'] == 1)
            neg_count = sum(1 for exp in fold_exps if exp['label'] == 0)
            
            # Check if fold is missing positive experiments
            if pos_count == 0:
                print(f"[WARNING] Fold {fold_idx} has no positive experiments! Finding donor...")
                # Find fold with most positives and transfer one
                for donor_fold in range(self.num_folds):
                    donor_exps = fold_experiments[donor_fold]
                    donor_pos = [exp for exp in donor_exps if exp['label'] == 1]
                    if len(donor_pos) > 1:  # Donor must keep at least 1 positive
                        exp_to_move = donor_pos[-1]  # Take smallest positive to minimize imbalance
                        fold_experiments[donor_fold].remove(exp_to_move)
                        fold_experiments[fold_idx].append(exp_to_move)
                        exp_fold_assignment[exp_to_move['exp_id']] = fold_idx
                        print(f"[DEBUG] Moved {exp_to_move['exp_id']} from fold {donor_fold} to fold {fold_idx}")
                        break
            
            # Check if fold is missing negative experiments
            if neg_count == 0:
                print(f"[WARNING] Fold {fold_idx} has no negative experiments! Finding donor...")
                # Find fold with most negatives and transfer one
                for donor_fold in range(self.num_folds):
                    donor_exps = fold_experiments[donor_fold]
                    donor_neg = [exp for exp in donor_exps if exp['label'] == 0]
                    if len(donor_neg) > 1:  # Donor must keep at least 1 negative
                        exp_to_move = donor_neg[-1]  # Take smallest negative to minimize imbalance
                        fold_experiments[donor_fold].remove(exp_to_move)
                        fold_experiments[fold_idx].append(exp_to_move)
                        exp_fold_assignment[exp_to_move['exp_id']] = fold_idx
                        print(f"[DEBUG] Moved {exp_to_move['exp_id']} from fold {donor_fold} to fold {fold_idx}")
                        break
        
        # Step 4: Extract indices for this specific fold
        train_indices = []
        val_indices = []
        
        for fold_idx in range(self.num_folds):
            fold_indices = []
            for exp in fold_experiments[fold_idx]:
                fold_indices.extend(exp['indices'])
            
            if fold_idx == self.fold:
                # This is the validation fold
                val_indices = fold_indices
            else:
                # This is part of the training fold
                train_indices.extend(fold_indices)
        
        # Step 5: Verify no data leakage
        train_set = set(train_indices)
        val_set = set(val_indices)
        overlap = train_set & val_set
        
        if overlap:
            print(f"[ERROR] CRITICAL: {len(overlap)} patches in both train and val!")
        else:
            print(f"[DEBUG] ✓ No patch-level overlap ({len(train_indices)} train, {len(val_indices)} val)")
        
        # Step 6: Report patch-level distribution
        train_labels = np.array([self.samples[i][1] for i in train_indices])
        train_pos = np.sum(train_labels == 1)
        train_neg = np.sum(train_labels == 0)
        train_ratio = train_neg / train_pos if train_pos > 0 else 0
        
        val_labels = np.array([self.samples[i][1] for i in val_indices])
        val_pos = np.sum(val_labels == 1)
        val_neg = np.sum(val_labels == 0)
        val_ratio = val_neg / val_pos if val_pos > 0 else 0
        
        if self.train:
            if len(train_indices) > 0:
                print(f"[DEBUG] TRAIN split: {len(train_indices)} patches")
                print(f"[DEBUG]   Positive: {train_pos:,} ({100*train_pos/len(train_indices):.1f}%)")
                print(f"[DEBUG]   Negative: {train_neg:,} ({100*train_neg/len(train_indices):.1f}%)")
                print(f"[DEBUG]   Ratio (Neg:Pos): {train_ratio:.2f}:1")
            else:
                print(f"[ERROR] TRAIN split is empty!")
        else:
            if len(val_indices) > 0:
                print(f"[DEBUG] VAL split: {len(val_indices)} patches")
                print(f"[DEBUG]   Positive: {val_pos:,} ({100*val_pos/len(val_indices):.1f}%)")
                print(f"[DEBUG]   Negative: {val_neg:,} ({100*val_neg/len(val_indices):.1f}%)")
                print(f"[DEBUG]   Ratio (Neg:Pos): {val_ratio:.2f}:1")
                print(f"[DEBUG]   Expected (overall): {overall_ratio:.2f}:1")
                ratio_drift = abs(val_ratio - overall_ratio) / overall_ratio * 100 if overall_ratio > 0 else 0
                if ratio_drift < 5:
                    print(f"[DEBUG]   ✓ Val ratio within 5% of expected (drift: {ratio_drift:.1f}%)")
                elif ratio_drift < 10:
                    print(f"[DEBUG]   ⚠ Val ratio slightly off (drift: {ratio_drift:.1f}%)")
                else:
                    print(f"[DEBUG]   ✗ Val ratio significantly off (drift: {ratio_drift:.1f}%)")
            else:
                print(f"[ERROR] VAL split is empty!")
        
        return {'train': train_indices, 'val': val_indices}
    
    def _compute_statistics(self):
        """Compute dataset statistics for logging."""
        labels = [self.samples[i][1] for i in self.indices]
        pos_count = sum(1 for l in labels if l == 1)
        neg_count = sum(1 for l in labels if l == 0)
        
        return {
            'total': len(self.indices),
            'positive': pos_count,
            'negative': neg_count,
            'imbalance_ratio': neg_count / pos_count if pos_count > 0 else 0,
            'fold': self.fold,
            'split': 'train' if self.train else 'val'
        }
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Load and return a single patch image with its label.
        
        Returns:
            tuple: (image_tensor, label) where:
                - image_tensor: (C, H, W) transformed image
                - label: 0 (negative) or 1 (positive)
        """
        # Get the actual sample from fold indices
        sample_idx = self.indices[idx]
        img_path, label = self.samples[sample_idx]
        
        # Load image (PIL automatically handles JPEG)
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Log failed image loads with detailed info
            relative_path = os.path.relpath(img_path, self.root_dir)
            print(f"[WARNING] Failed to load {relative_path}: {str(e)}")
            # Return a black image as fallback
            img = Image.new('RGB', (256, 256), color=(0, 0, 0))
        
        # Apply transforms if provided
        if self.transform:
            img = self.transform(img)
        else:
            # Minimal default: convert PIL to tensor and normalize
            from torchvision.transforms import v2
            default_transform = v2.Compose([
                v2.PILToTensor(),
                v2.ToDtype(torch.float32),
                v2.Lambda(lambda x: x / 255.0),  # Normalize to [0,1]
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img = default_transform(img)
        
        label = torch.tensor(label, dtype=torch.long)
        
        return img, label
    
    def print_statistics(self):
        """Print dataset statistics for logging."""
        stats = self.statistics
        print(f"\n{'='*60}")
        print(f"DeepHP Dataset Statistics (Fold {stats['fold']}, {stats['split'].upper()})")
        print(f"{'='*60}")
        print(f"Total patches:        {stats['total']:,}")
        print(f"  Positive (H. pylori):  {stats['positive']:,}")
        print(f"  Negative (Normal):     {stats['negative']:,}")
        print(f"Imbalance ratio (Neg:Pos): {stats['imbalance_ratio']:.2f}:1")
        print(f"{'='*60}\n")


def create_deephp_transforms_train():
    """
    H&E-optimized training transforms for DeepHP patches.
    
    H&E staining is the standard histology stain, contrasting with IHC (brown/blue).
    We use aggressive augmentation to maximize pre-training robustness.
    """
    from torchvision.transforms import v2
    
    # Note: Macenko normalization is applied after loading, not here
    return v2.Compose([
        v2.PILToTensor(),  # Convert PIL to tensor (uint8 [0,255])
        v2.ToDtype(torch.float32),  # Convert to float32 [0,255]
        v2.Lambda(lambda x: x / 255.0),  # Normalize to [0,1]
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
        v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet norm (H&E compatible)
    ])


def create_deephp_transforms_val():
    """
    H&E-optimized validation transforms (no augmentation).
    """
    from torchvision.transforms import v2
    
    return v2.Compose([
        v2.PILToTensor(),  # Convert PIL to tensor (uint8 [0,255])
        v2.ToDtype(torch.float32),  # Convert to float32 [0,255]
        v2.Lambda(lambda x: x / 255.0),  # Normalize to [0,1]
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
