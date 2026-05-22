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


def assess_patch_quality(image_tensor):
    """
    Score a patch for H&E staining quality.
    Higher score = better reference candidate.
    
    Criteria:
    - Not too bright (white background)
    - Not too dark (oversaturated)
    - High color variation (good H&E signal)
    - Balanced RGB channels (typical H&E)
    """
    # Convert [0, 1] to [0, 255] for analysis
    img = (image_tensor * 255).numpy().astype(np.uint8)
    
    # Mean brightness (should be ~100-150 for good staining)
    brightness = img.mean()
    brightness_score = 1.0 - abs(brightness - 120) / 120
    
    # Avoid mostly white patches
    if brightness > 200:
        return -1.0  # Disqualify
    
    # Avoid mostly black patches
    if brightness < 30:
        return -1.0  # Disqualify
    
    # Color variation (std across all channels)
    variation = img.std()
    variation_score = min(variation / 50, 1.0)  # Normalize to [0, 1]
    
    # H&E specific: Hematoxylin (blue) and Eosin (red/pink)
    # Good H&E patches have moderate-to-high variation in Red and Blue channels
    r_std = img[0].std()
    g_std = img[1].std()
    b_std = img[2].std()
    
    # Prefer balanced color variation
    channel_balance = 1.0 - (abs(r_std - b_std) / max(r_std, b_std, 1))
    
    # Combined score
    quality_score = (brightness_score * 0.3 + 
                     variation_score * 0.4 + 
                     channel_balance * 0.3)
    
    return quality_score


def create_reference():
    """Create and save Macenko reference from DeepHP dataset."""
    
    print("Creating Macenko reference from DeepHP dataset...")
    print("=" * 60)
    
    # Path to dataset
    deephp_root = "/export/hhome/tkeating/datasets/DeepHP"
    output_path = "./macenko_reference.png"
    
    if not os.path.exists(deephp_root):
        print(f"ERROR: DeepHP dataset not found at {deephp_root}")
        return False
    
    # Create dataset with minimal transforms (raw patches)
    transform = T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float32),
        # No normalization - we want raw H&E colors
    ])
    
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
    
    print(f"Dataset loaded: {len(dataset)} patches")
    print("Scanning patches for optimal H&E staining quality...\n")
    
    best_patch = None
    best_score = -1.0
    best_idx = -1
    patch_count = 0
    
    # Scan first 5 batches for quality assessment
    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx >= 5:  # Only scan first 5 batches
            break
        
        for idx in range(len(images)):
            patch_count += 1
            img = images[idx]  # [C, H, W], values in [0, 1]
            
            score = assess_patch_quality(img)
            
            if score > best_score:
                best_score = score
                best_patch = img
                best_idx = patch_count
                print(f"  Batch {batch_idx+1}, Patch {idx+1}: Score={score:.3f} ← NEW BEST")
            elif patch_count % 16 == 0:
                print(f"  Batch {batch_idx+1}, Patch {idx+1}: Score={score:.3f}")
    
    print(f"\n✓ Scanned {patch_count} patches")
    print(f"✓ Best patch score: {best_score:.3f} (patch #{best_idx})")
    
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
