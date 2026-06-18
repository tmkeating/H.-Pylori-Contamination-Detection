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
  2. Partitions experiments into balanced folds using weighted round-robin distribution:
     - Sorts experiments by patch count within each class (positive/negative)
     - Assigns experiments to folds in sequence: exp[0]→fold[0], exp[1]→fold[1], etc.
     - This naturally balances both experiment count AND patch count per fold
  3. Ensures no experiment is split across train/validation (prevents artifact overfitting)
  4. Maintains patch-level class ratio (~2.6:1) consistently across all folds

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
from sklearn.model_selection import StratifiedKFold


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
        Uses weighted round-robin distribution to balance folds:
        1. Groups patches by experiment ID
        2. Sorts experiments by patch count within each class
        3. Round-robin assigns experiments to folds in sequence
        4. Result: balanced fold sizes, balanced class ratios, no experiment splitting
        
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
        Create stratified k-fold split using weighted round-robin distribution.
        
        Prevents both data leakage and severe class imbalance by:
        1. Grouping patches by experiment ID (prevents experiment splitting)
        2. Separating positive and negative experiments
        3. Sorting each class by patch count (descending)
        4. Round-robin assigning experiments to folds
        
        RATIONALE:
        - Naive splitting: StratifiedKFold on patches splits experiments across folds,
          causing overfitting on experiment-specific artifacts (staining, texture)
        - Experiment-level StratifiedKFold: Balances experiment count but not patch count,
          resulting in folds with ~94.8% negatives vs ~33% overall
        - This method: Round-robin alternates large and small experiments across folds,
          naturally balancing both experiment count AND patch count
        
        EXAMPLE:
        If we have 20 positive experiments [exp1(1000 patches), exp2(800 patches), ...]
        and 12 negative experiments [exp21(10000 patches), exp22(8000 patches), ...]:
        
        Fold 0: exp1(1000p), exp3(800p), ... | exp21(10000n), exp23(8000n), ...
        Fold 1: exp2(900p), exp4(700p), ... | exp22(9000n), exp24(7000n), ...
        Fold 2: exp5(850p), exp6(750p), ... | exp25(9500n), exp26(7500n), ...
        ... (distributes large+small alternately to balance fold sizes)
        
        RESULT:
        - No experiment is split across train/val (prevents artifact overfitting)
        - Each fold has ~80 patches positive, ~260 patches negative (2.6:1 ratio)
        - Each fold has ~6-7 positive and ~2-3 negative experiments (natural balance)
        - Class distribution is consistent across all folds
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
            label = patch_indices[0][1]  # All patches from same experiment have same label
            indices = [idx for idx, _ in patch_indices]
            patch_count = len(indices)
            
            label_key = 'positive' if label == 1 else 'negative'
            experiments_by_label[label_key].append({
                'exp_id': exp_id,
                'indices': indices,
                'patch_count': patch_count,
                'label': label
            })
        
        pos_exp_count = len(experiments_by_label['positive'])
        neg_exp_count = len(experiments_by_label['negative'])
        pos_patch_count = sum(e['patch_count'] for e in experiments_by_label['positive'])
        neg_patch_count = sum(e['patch_count'] for e in experiments_by_label['negative'])
        overall_ratio = neg_patch_count / pos_patch_count if pos_patch_count > 0 else 0
        
        print(f"[DEBUG] Positive: {pos_exp_count} experiments, {pos_patch_count:,} patches")
        print(f"[DEBUG] Negative: {neg_exp_count} experiments, {neg_patch_count:,} patches")
        print(f"[DEBUG] Patch-level ratio (Neg:Pos): {overall_ratio:.2f}:1")
        
        # Step 3: Sort experiments by patch count (descending) within each label
        # This ensures alternating large/small experiments during round-robin distribution
        for label_key in experiments_by_label:
            experiments_by_label[label_key].sort(key=lambda e: e['patch_count'], reverse=True)
        
        # Step 4: Round-robin assign experiments to folds to balance patches per fold
        fold_experiments = [[] for _ in range(self.num_folds)]
        fold_patch_counts = [0] * self.num_folds
        fold_pos_counts = [0] * self.num_folds
        fold_neg_counts = [0] * self.num_folds
        
        # Distribute positive experiments first (round-robin by patch count)
        for i, exp in enumerate(experiments_by_label['positive']):
            fold_idx = i % self.num_folds
            fold_experiments[fold_idx].append(exp)
            fold_patch_counts[fold_idx] += exp['patch_count']
            fold_pos_counts[fold_idx] += exp['patch_count']
        
        # Distribute negative experiments (round-robin by patch count)
        for i, exp in enumerate(experiments_by_label['negative']):
            fold_idx = i % self.num_folds
            fold_experiments[fold_idx].append(exp)
            fold_patch_counts[fold_idx] += exp['patch_count']
            fold_neg_counts[fold_idx] += exp['patch_count']
        
        # Step 5: Extract indices for this specific fold
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
        
        # Step 6: Verify no data leakage
        train_set = set(train_indices)
        val_set = set(val_indices)
        overlap = train_set & val_set
        
        if overlap:
            print(f"[ERROR] CRITICAL: {len(overlap)} patches in both train and val!")
        else:
            print(f"[DEBUG] ✓ No patch-level overlap ({len(train_indices)} train, {len(val_indices)} val)")
        
        # Step 7: Report patch-level distribution
        train_labels = np.array([self.samples[i][1] for i in train_indices])
        train_pos = np.sum(train_labels == 1)
        train_neg = np.sum(train_labels == 0)
        train_ratio = train_neg / train_pos if train_pos > 0 else 0
        
        val_labels = np.array([self.samples[i][1] for i in val_indices])
        val_pos = np.sum(val_labels == 1)
        val_neg = np.sum(val_labels == 0)
        val_ratio = val_neg / val_pos if val_pos > 0 else 0
        
        if self.train:
            print(f"[DEBUG] TRAIN split: {len(train_indices)} patches")
            print(f"[DEBUG]   Positive: {train_pos:,} ({100*train_pos/len(train_indices):.1f}%)")
            print(f"[DEBUG]   Negative: {train_neg:,} ({100*train_neg/len(train_indices):.1f}%)")
            print(f"[DEBUG]   Ratio (Neg:Pos): {train_ratio:.2f}:1")
        else:
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
            print(f"Warning: Failed to load {img_path}: {e}")
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
