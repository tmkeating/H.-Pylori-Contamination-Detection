#!/usr/bin/env python3
"""
Create a pre-computed Macenko normalization reference from DeepHP dataset.

This script:
1. Loads high-quality patches from the DeepHP dataset
2. Selects one with optimal H&E staining characteristics
3. Saves it as a reference image for all future training runs

This eliminates the need for fragile per-run reference fitting and ensures
consistency across all pre-training experiments.
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from dataset_deepHP import DeepHPDataset
import torchvision.transforms as T
from config import DEEPHP_SCRATCH_ROOT


def assess_patch_quality(image_tensor):
    """
    Score a patch for H&E staining quality.
    Higher score = better reference candidate.
    
    Criteria:
    - Not too bright (white background)
    - Not too dark (oversaturated)
    - HIGH color variation (good H&E signal)
    - BALANCED H&E colors (mix of Hematoxylin and Eosin, not one dominating)
    """
    # Convert [0, 1] to [0, 255] for analysis
    img = (image_tensor * 255).numpy().astype(np.uint8)
    
    # Mean brightness (should be ~100-150 for good staining)
    brightness = img.mean()
    brightness_score = 1.0 - abs(brightness - 120) / 120
    
    # Avoid mostly white patches (low staining)
    if brightness > 200:
        return -1.0
    
    # Avoid mostly black patches (oversaturation)
    if brightness < 30:
        return -1.0
    
    # Color variation - need high variance for eigenvalue stability
    variation = img.std()
    variation_score = min(variation / 40, 1.0)
    
    # Heavily penalize low variation patches
    if variation < 20:
        return -0.5
    
    # CRITICAL: Check for BALANCED H&E colors
    # Hematoxylin = blue/purple (high B channel, lower R)
    # Eosin = red/pink (high R channel, lower B)
    # Good reference needs BOTH visible
    
    r_mean = img[0].mean()
    g_mean = img[1].mean()
    b_mean = img[2].mean()
    
    r_std = img[0].std()
    g_std = img[1].std()
    b_std = img[2].std()
    
    # Prefer patches where BOTH red and blue channels have significant presence
    # Avoid heavily one-color-dominated patches
    # Red and blue should both be well-represented (not one >> other)
    rb_ratio = min(r_mean, b_mean) / max(r_mean, b_mean, 1)  # 1.0 = perfectly balanced
    
    # Also check variance in red and blue - both should contribute
    rb_variance_balance = min(r_std, b_std) / max(r_std, b_std, 1)
    
    # Penalize if one channel completely dominates
    if rb_ratio < 0.3:  # One color is way more than the other
        return -0.5
    
    # Combined score:
    # - Brightness (0.15): ensure good staining intensity
    # - Variation (0.35): high color variation for eigenvalue stability
    # - RB color balance (0.35): both Hematoxylin and Eosin visible
    # - RB variance balance (0.15): both colors contribute to variation
    quality_score = (brightness_score * 0.15 + 
                     variation_score * 0.35 + 
                     rb_ratio * 0.35 +
                     rb_variance_balance * 0.15)
    
    return quality_score


def create_reference():
    """Create and save Macenko reference from DeepHP dataset."""
    
    print("Creating Macenko reference from DeepHP dataset...")
    print("=" * 60)
    
    # Path to dataset (use scratch for faster I/O)
    deephp_root = DEEPHP_SCRATCH_ROOT
    output_path = os.path.join(os.getcwd(), "macenko_reference.png")
    
    if not os.path.exists(deephp_root):
        print(f"ERROR: DeepHP dataset not found at {deephp_root}")
        return False
    
    # Create dataset with minimal transforms (raw patches)
    transform = T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float32),
        # No normalization - we want raw H&E colors
    ])
    
    # Scan patches from training set (will prioritize positives for better H&E staining)
    dataset = DeepHPDataset(
        root_dir=deephp_root,
        transform=transform,
        fold=0,
        num_folds=5,
        train=True  # Use training fold
    )
    
    # Create loader to sample patches
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=False
    )
    
    print(f"Dataset loaded: {len(dataset)} patches (POSITIVE class only)")
    print("Scanning patches for optimal H&E staining quality...")
    print("(High color variation is KEY for eigenvalue stability)\n")
    
    best_patch = None
    best_score = -1.0
    best_idx = -1
    patch_count = 0
    positive_count = 0
    top_candidates = []  # Track top 5 candidates
    
    # Scan up to 100 batches for much better sample coverage
    # At batch_size=64, this is 6,400 patches from ~315K total (2% sample)
    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx >= 100:  # Increased from 5 to 100 batches
            break
        
        for idx in range(len(images)):
            patch_count += 1
            img = images[idx]  # [C, H, W], values in [0, 1]
            label = labels[idx].item() if hasattr(labels[idx], 'item') else labels[idx]
            
            score = assess_patch_quality(img)
            
            # Bonus for positive class patches (better H&E staining)
            if label == 1:
                positive_count += 1
                score = score * 1.2  # 20% bonus for positive patches
            
            if score > best_score:
                best_score = score
                best_patch = img
                best_idx = patch_count
                top_candidates.append((score, patch_count, label))
                if len(top_candidates) > 5:
                    top_candidates.pop(0)
                if len(top_candidates) <= 3:  # Print first few
                    print(f"  Batch {batch_idx+1}, Patch {idx+1}: Score={score:.3f} ← NEW BEST")
            elif batch_idx % 10 == 0 and idx == 0:
                print(f"  Batch {batch_idx+1}: Scanning... (best so far: {best_score:.3f})")
    
    print(f"\n✓ Scanned {patch_count} total patches ({positive_count} positive class)")
    print(f"✓ Best patch score: {best_score:.3f} (patch #{best_idx})")
    if top_candidates:
        labels_str = ['P' if l==1 else 'N' for _,_,l in sorted(top_candidates, reverse=True)]
        scores_str = ', '.join([f'{s:.3f}({l})' for s,_,l in sorted(top_candidates, reverse=True)])
        print(f"✓ Top 5 scores: {scores_str}")
    
    if best_patch is None:
        print("\nERROR: No suitable reference patch found!")
        return False
    
    # Convert reference patch back to uint8 PIL image for saving
    ref_uint8 = (best_patch * 255).byte().permute(1, 2, 0).numpy()
    ref_pil = Image.fromarray(ref_uint8, mode='RGB')
    
    # Save reference image
    ref_pil.save(output_path)
    print(f"\n✓ Reference saved: {output_path}")
    print(f"  Size: {ref_pil.size}")
    print(f"  Format: RGB JPEG")
    
    # Verify we can load it back
    test_load = Image.open(output_path)
    print(f"✓ Verified: Successfully loaded from disk ({test_load.size})")
    
    print("\n" + "=" * 60)
    print("Reference creation complete!")
    print("\nNext steps:")
    print("1. The reference image is now available for training")
    print("2. train_deepHP_patches.py will automatically load it")
    print("3. All pre-training runs will use this consistent reference")
    print("4. This eliminates Macenko fitting failures from per-run variance")
    
    return True


if __name__ == "__main__":
    success = create_reference()
    exit(0 if success else 1)
