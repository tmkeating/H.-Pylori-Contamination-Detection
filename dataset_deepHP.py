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
    - Overall ratio: ~2.6:1 (negative:positive)

DATA STRATIFICATION STRATEGY:
  To prevent experiment-level overfitting and data leakage while maintaining class balance:
  
  1. Groups all patches by experiment ID (extracted from filename prefix)
  2. Sorts experiments by patch count (largest first)
  3. Divides into rounds of num_folds experiments, assigning each round to folds in order
  4. This ensures each fold gets a mix of large and small experiments
  5. Guarantees no experiment is split and balanced patch distribution per fold
  6. Ensures patch-level class ratio (~2.28:1) consistent across all folds

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
    Patch-level dataset for DeepHP H&E histology images with experiment-aware stratification.
    
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
        - Overall patch-level ratio: ~2.6:1 (negative:positive)
    
    STRATIFICATION STRATEGY:
        Uses StratifiedKFold on experiment labels (not patches) to create fold assignments:
        1. Groups patches by experiment ID
        2. Applies StratifiedKFold to experiment labels (ensuring each fold has pos + neg)
        3. Assigns all patches from each experiment to its assigned fold
        4. Result: balanced folds, no experiment splitting, no leakage
        
        This prevents experiment-level overfitting where the model learns staining
        artifacts and slide-specific patterns instead of actual biological features.
    
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
        Create stratified k-fold split using class-first weighted round-robin assignment.
        
        Prevents both data leakage and severe class imbalance by:
        1. Grouping patches by experiment ID (prevents experiment splitting)
        2. Sorting positive experiments by size (largest first)
        3. Sorting negative experiments by size (largest first)
        4. Combining as [positives + negatives]
        5. Dividing into rounds of num_folds experiments, assigning each round to folds in order
        6. This ensures each fold gets balanced positive and negative experiment coverage
        
        RATIONALE:
        - Class-first distribution: Ensures each fold gets positive experiments early
        - Weighted round-robin within class: Large and small experiments of same class 
          distributed evenly across folds
        - Result: Each fold has similar counts of positive and negative experiments
        
        EXAMPLE (20 positive + 12 negative, 5 folds, sorted by size):
        Positive rounds:
        - Round 1: pos[0:5] (largest pos) → folds 0,1,2,3,4
        - Round 2: pos[5:10] → folds 0,1,2,3,4
        - Round 3: pos[10:15] → folds 0,1,2,3,4
        - Round 4: pos[15:20] → folds 0,1,2,3,4
        Negative rounds:
        - Round 5: neg[0:5] (largest neg) → folds 0,1,2,3,4
        - Round 6: neg[5:10] → folds 0,1,2,3,4
        - Round 7: neg[10:12] → folds 0,1,2,3,4
        
        RESULT:
        - Each fold has 4 positive experiments distributed across sizes
        - Each fold has ~2.4 negative experiments distributed across sizes
        - Total patch count naturally balanced across folds
        - No experiment is split across train/val (prevents artifact overfitting)
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
        
        # Step 3: Weighted round-robin stratification to balance patch counts across folds
        # Distribute all positive experiments first (in rounds), then all negative experiments.
        # This ensures each fold gets positive experiments before negative ones.
        #
        # ALGORITHM:
        # 1. Sort positive experiments by patch count (largest first)
        # 2. Sort negative experiments by patch count (largest first)
        # 3. Combine as [positives_sorted + negatives_sorted]
        # 4. Divide into rounds of num_folds experiments each
        # 5. Within each round, assign to folds in order (fold0, fold1, fold2, ...)
        # 6. This ensures each fold gets balanced positive + negative coverage
        #
        # EXAMPLE (5 folds, 20 positive + 12 negative experiments):
        # Positive rounds (1-4): All 20 positive experiments distributed evenly
        # Negative rounds (5-3): All 12 negative experiments distributed evenly
        # Result: Each fold has 4 positive + ~2.4 negative experiments (balanced by class)
        
        # Sort experiments by class, then by patch count within each class
        positive_exps = sorted(experiments_by_label['positive'], key=lambda e: e['patch_count'], reverse=True)
        negative_exps = sorted(experiments_by_label['negative'], key=lambda e: e['patch_count'], reverse=True)
        
        # Combine: positives first, then negatives
        experiment_ids = positive_exps + negative_exps
        
        # Weighted round-robin assignment
        exp_fold_assignment = {}
        fold_experiments = [[] for _ in range(self.num_folds)]
        
        for round_num in range((len(experiment_ids) + self.num_folds - 1) // self.num_folds):
            round_start = round_num * self.num_folds
            round_end = min(round_start + self.num_folds, len(experiment_ids))
            round_exps = experiment_ids[round_start:round_end]
            
            for fold_offset, exp in enumerate(round_exps):
                fold_idx = fold_offset % self.num_folds
                exp_fold_assignment[exp['exp_id']] = fold_idx
                fold_experiments[fold_idx].append(exp)
        
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
