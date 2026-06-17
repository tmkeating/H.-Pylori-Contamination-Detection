"""
DeepHP Dataset Loader - H&E Stained Histology Patches

Provides a simple patch-level dataset for pre-training the backbone on H&E-stained
images from the DeepHP database (394,926 total patches: 111K positive, 283K negative).

This loader is distinct from HPyloriDataset because:
1. No patient grouping (flat directory structure: Positive/ and Negative/)
2. Patch-level classification (not Multiple Instance Learning)
3. H&E-specific normalization (Macenko, not ImageNet)
4. Designed for backbone pre-training, not clinical patient-level inference

Usage:
    from dataset_deepHP import DeepHPDataset
    from config import DEEPHP_DATASET_ROOT
    
    # Load dataset with H&E normalization
    dataset = DeepHPDataset(
        root_dir=DEEPHP_DATASET_ROOT,
        transform=transforms.Compose([...]),
        fold=0,  # for cross-validation stratification
        num_folds=5
    )
    
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8)
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
    Patch-level dataset for DeepHP H&E histology images.
    
    Structure:
        root_dir/
        ├── Positive/     (111,005 JPEG patches)
        └── Negative/     (283,921 JPEG patches)
    
    Each image is a 256×256 patch (pre-cropped) from WSIs.
    Labels are derived from folder membership (0=Negative, 1=Positive).
    
    Args:
        root_dir (str): Path to DeepHP dataset root (contains Positive/ and Negative/ folders)
        transform (transforms.Compose, optional): Torchvision transforms to apply
        fold (int): Fold index for k-fold cross-validation (0 to num_folds-1)
        num_folds (int): Total number of folds for stratified split
        train (bool): If True, return training fold; if False, return validation fold
    
    Attributes:
        samples (list): List of (image_path, label) tuples
        fold_indices (dict): {'train': [...], 'val': [...]} indices for this fold
        statistics (dict): Dataset statistics (total count, class distribution)
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
        Create stratified k-fold split at the EXPERIMENT level to prevent data leakage.
        Groups patches by experiment ID, then splits experiments (not patches) into folds.
        This ensures patches from the same patient/experiment don't leak between train/val.
        
        Experiment ID is extracted from filename: "Experiment-108_b0s..." -> "Experiment-108"
        All patches from the same experiment stay together in the same fold.
        """
        # Group samples by experiment ID (extracted from filename before _b0s)
        experiment_groups = {}  # {experiment_id: [(index, label), ...]}
        
        for idx, (path, label) in enumerate(self.samples):
            # Extract experiment ID: "Experiment-108_b0s..." -> "Experiment-108"
            filename = os.path.basename(path)
            experiment_id = filename.split('_b0s')[0]
            
            if experiment_id not in experiment_groups:
                experiment_groups[experiment_id] = []
            experiment_groups[experiment_id].append((idx, label))
        
        print(f"[DEBUG] Grouped {len(self.samples)} patches into {len(experiment_groups)} experiments")
        
        # Separate experiments by label
        pos_experiments = []  # List of (experiment_id, patch_indices)
        neg_experiments = []
        
        for exp_id, patch_indices in experiment_groups.items():
            # Get the label from first patch (all patches from same experiment have same label)
            label = patch_indices[0][1]
            indices = [idx for idx, _ in patch_indices]
            
            if label == 1:
                pos_experiments.append((exp_id, indices))
            else:
                neg_experiments.append((exp_id, indices))
        
        print(f"[DEBUG] Separated into {len(pos_experiments)} positive and {len(neg_experiments)} negative experiments")
        print(f"[DEBUG] Fold {self.fold}/{self.num_folds}: Splitting experiments to prevent data leakage")
        
        # Shuffle experiments deterministically (per-fold seed for reproducibility)
        rng = np.random.RandomState(42 + self.fold)
        rng.shuffle(pos_experiments)
        rng.shuffle(neg_experiments)
        
        # Create stratified folds at experiment level (NOT patch level)
        pos_fold_size = len(pos_experiments) // self.num_folds
        neg_fold_size = len(neg_experiments) // self.num_folds
        
        pos_val_start = self.fold * pos_fold_size
        pos_val_end = pos_val_start + pos_fold_size if self.fold < self.num_folds - 1 else len(pos_experiments)
        
        neg_val_start = self.fold * neg_fold_size
        neg_val_end = neg_val_start + neg_fold_size if self.fold < self.num_folds - 1 else len(neg_experiments)
        
        # Collect all patch indices for validation experiments
        val_indices = []
        val_experiment_ids = []
        for exp_id, indices in pos_experiments[pos_val_start:pos_val_end]:
            val_indices.extend(indices)
            val_experiment_ids.append(exp_id)
        for exp_id, indices in neg_experiments[neg_val_start:neg_val_end]:
            val_indices.extend(indices)
            val_experiment_ids.append(exp_id)
        
        # Collect all patch indices for training experiments
        train_indices = []
        train_experiment_ids = []
        for exp_id, indices in pos_experiments[:pos_val_start] + pos_experiments[pos_val_end:]:
            train_indices.extend(indices)
            train_experiment_ids.append(exp_id)
        for exp_id, indices in neg_experiments[:neg_val_start] + neg_experiments[neg_val_end:]:
            train_indices.extend(indices)
            train_experiment_ids.append(exp_id)
        
        # Verify no data leakage at experiment level
        train_set = set(train_indices)
        val_set = set(val_indices)
        overlap = train_set & val_set
        
        if overlap:
            print(f"[ERROR] CRITICAL: Train and validation indices overlap! {len(overlap)} samples in both!")
            print(f"[ERROR] This indicates experiment-level separation failed!")
        else:
            print(f"[DEBUG] ✓ Train/Val indices are disjoint ({len(train_indices)} train, {len(val_indices)} val)")
            print(f"[DEBUG] ✓ Validation fold has {len(val_experiment_ids)} experiments, training has {len(train_experiment_ids)}")
            
            if self.train:
                print(f"[DEBUG] TRAIN split: {len(train_experiment_ids)} experiments, {len(train_indices)} patches")
            else:
                print(f"[DEBUG] VAL split: {len(val_experiment_ids)} experiments, {len(val_indices)} patches")
                print(f"[DEBUG] Validation experiments: {sorted(val_experiment_ids)}")
        
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
