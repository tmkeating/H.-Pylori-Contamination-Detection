"""
load_pretrained_backbone.py - Utility for Loading Pre-trained DeepHP Backbone

Provides utilities to load pre-trained backbone weights from DeepHP pre-training
into the full model used for patient-level MIL training on HelicoDataSet.

This enables transfer learning where the backbone learns general H. pylori 
morphology from 400K H&E patches, then fine-tunes on 114 patient-level IHC samples.

Usage:
    from load_pretrained_backbone import load_pretrained_backbone
    
    model = get_model(model_name="convnext_tiny", num_classes=2, pretrained=True)
    model = load_pretrained_backbone(model, "/path/to/deephp_backbone_pretrained_convnext_tiny_f0.pth")
    
    # Now model's backbone is initialized from DeepHP pre-training
    # Fine-tune on HelicoDataSet patient-level MIL
"""

import os
import torch
import torch.nn as nn
from collections import OrderedDict


def load_pretrained_backbone(model, pretrained_backbone_path, freeze_backbone=False):
    """
    Load pre-trained backbone weights into a model.
    
    This function extracts the backbone weights from a DeepHP pre-trained checkpoint
    and loads them into the provided model. It handles:
    1. Weight naming conventions (DeepHP vs HelicoDataSet models may differ)
    2. Optional backbone freezing (for some transfer learning scenarios)
    3. Partial loading (only backbone, not MIL head)
    
    Args:
        model (nn.Module): Target model (typically HPyNet) to load weights into
        pretrained_backbone_path (str): Path to pre-trained checkpoint (.pth file)
        freeze_backbone (bool): If True, freeze backbone layers during training
    
    Returns:
        model (nn.Module): Model with loaded backbone weights
    
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        KeyError: If checkpoint format is unexpected
    """
    
    if not os.path.exists(pretrained_backbone_path):
        raise FileNotFoundError(f"Pre-trained backbone not found: {pretrained_backbone_path}")
    
    print(f"\n{'='*80}")
    print(f"Loading Pre-trained Backbone from: {pretrained_backbone_path}")
    print(f"{'='*80}")
    
    # Load checkpoint
    checkpoint = torch.load(pretrained_backbone_path, map_location='cpu')
    
    # Extract backbone weights
    # The checkpoint may contain:
    # - Full model state_dict with keys like "backbone.features.0...."
    # - Or may be the backbone weights directly
    
    backbone_weights = OrderedDict()
    
    for key, value in checkpoint.items():
        # If the checkpoint is from a full model (has "backbone.features" prefix)
        if key.startswith('backbone.'):
            # Extract just the backbone part
            backbone_key = key  # Keep as is - backbone.* pattern
            backbone_weights[key] = value
        # If checkpoint is backbone-only weights
        elif key.startswith('features.'):
            # Add "backbone." prefix for consistency
            backbone_weights[f'backbone.{key}'] = value
        else:
            # Could be root-level backbone keys, add prefix
            backbone_weights[f'backbone.{key}'] = value
    
    if not backbone_weights:
        # Fallback: all keys in checkpoint might be backbone weights
        print("Warning: Could not identify backbone-specific keys. Loading entire checkpoint as backbone...")
        backbone_weights = {f'backbone.{k}' if not k.startswith('backbone.') else k: v 
                          for k, v in checkpoint.items()}
    
    # Load into model
    try:
        # Try to load only backbone weights, ignore missing keys (MIL head)
        missing_keys, unexpected_keys = model.load_state_dict(backbone_weights, strict=False)
        
        print(f"\nLoaded backbone weights successfully!")
        print(f"Missing keys (expected - these are MIL head): {len(missing_keys)}")
        if len(missing_keys) <= 5:
            for key in missing_keys[:5]:
                print(f"  - {key}")
        
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")
            if len(unexpected_keys) <= 5:
                for key in unexpected_keys[:5]:
                    print(f"  - {key}")
    
    except Exception as e:
        print(f"Warning: Could not load all weights. Error: {e}")
        print("Attempting partial loading...")
        
        # Try loading with strict=False to skip incompatible weights
        model.load_state_dict(backbone_weights, strict=False)
        print("Partial loading successful (some weights skipped)")
    
    # Optionally freeze backbone
    if freeze_backbone:
        print(f"\nFreezing backbone layers (no gradient updates during fine-tuning)...")
        for name, param in model.named_parameters():
            if name.startswith('backbone.'):
                param.requires_grad = False
        
        # Count frozen vs trainable parameters
        frozen_count = sum(1 for name, p in model.named_parameters() 
                          if name.startswith('backbone.') and not p.requires_grad)
        trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"Frozen backbone parameters: {frozen_count}")
        print(f"Trainable parameters (MIL head): {trainable_count}")
    else:
        print(f"\nBackbone is trainable (will fine-tune during HelicoDataSet training)")
        print(f"Total trainable parameters: {sum(1 for p in model.parameters() if p.requires_grad)}")
    
    print(f"{'='*80}\n")
    
    return model


def average_backbone_weights(fold_paths, output_path):
    """
    Average backbone weights across multiple folds for a unified pre-trained model.
    
    After training 5-fold DeepHP pre-training, this function combines the learned
    backbones into a single averaged model for deployment.
    
    Args:
        fold_paths (list): List of paths to fold checkpoints (e.g., f0.pth, f1.pth, ...)
        output_path (str): Where to save the averaged weights
    
    Returns:
        None (saves averaged weights to output_path)
    """
    
    print(f"\n{'='*80}")
    print(f"Averaging Backbone Weights Across {len(fold_paths)} Folds")
    print(f"{'='*80}\n")
    
    averaged_weights = None
    
    for fold_idx, fold_path in enumerate(fold_paths):
        if not os.path.exists(fold_path):
            print(f"Warning: Fold {fold_idx} checkpoint not found: {fold_path}")
            continue
        
        print(f"Loading fold {fold_idx + 1}/{len(fold_paths)}: {fold_path}")
        
        checkpoint = torch.load(fold_path, map_location='cpu')
        fold_weights = {k: v.float() for k, v in checkpoint.items()}
        
        if averaged_weights is None:
            # Initialize average with first fold
            averaged_weights = {k: v.clone() for k, v in fold_weights.items()}
        else:
            # Running average
            for key in averaged_weights.keys():
                if key in fold_weights:
                    averaged_weights[key] = (averaged_weights[key] * fold_idx + fold_weights[key]) / (fold_idx + 1)
    
    if averaged_weights is None:
        print("ERROR: No valid checkpoints found to average")
        return
    
    # Save averaged weights
    torch.save(averaged_weights, output_path)
    print(f"\n✓ Saved averaged backbone weights to: {output_path}")
    print(f"{'='*80}\n")


def weighted_average_backbone_weights(fold_paths, weights_dict, output_path):
    """
    Average backbone weights across folds using performance-based weights.
    
    Folds with better validation metrics (higher F1, accuracy, etc.) get
    proportionally more influence on the final averaged backbone.
    
    Args:
        fold_paths (list): List of paths to fold checkpoints (f0.pth, f1.pth, ...)
        weights_dict (dict): Mapping fold_idx -> weight (e.g., {"0": 0.07, "1": 0.33, ...})
                            Weights should sum to 1.0 (already normalized from ensemble voting)
        output_path (str): Where to save the weighted averaged backbone
    
    Returns:
        None (saves weighted averaged weights to output_path)
    """
    
    print(f"\n{'='*80}")
    print(f"Weighted Averaging Backbone Across {len(fold_paths)} Folds")
    print(f"Using F1-Based Ensemble Weights")
    print(f"{'='*80}\n")
    
    # Print weight distribution
    print("Fold Weights (from ensemble voting):")
    for fold_idx in range(len(fold_paths)):
        weight = float(weights_dict.get(str(fold_idx), 0.0))
        print(f"  Fold {fold_idx}: {weight:.4f}")
    print()
    
    averaged_weights = None
    total_weight = sum(float(weights_dict.get(str(i), 0.0)) for i in range(len(fold_paths)))
    
    if total_weight == 0:
        print("ERROR: No valid weights provided")
        return
    
    for fold_idx, fold_path in enumerate(fold_paths):
        if not os.path.exists(fold_path):
            print(f"Warning: Fold {fold_idx} checkpoint not found: {fold_path}")
            continue
        
        weight = float(weights_dict.get(str(fold_idx), 0.0))
        print(f"Loading fold {fold_idx + 1}/{len(fold_paths)}: {fold_path} (weight: {weight:.4f})")
        
        checkpoint = torch.load(fold_path, map_location='cpu')
        fold_weights = {k: v.float() for k, v in checkpoint.items()}
        
        if averaged_weights is None:
            # Initialize with weighted first fold
            averaged_weights = {k: (v.clone() * weight) for k, v in fold_weights.items()}
        else:
            # Add weighted fold contributions
            for key in averaged_weights.keys():
                if key in fold_weights:
                    averaged_weights[key] += fold_weights[key] * weight
    
    if averaged_weights is None:
        print("ERROR: No valid checkpoints found to average")
        return
    
    # Normalize by total weight (in case not all folds exist)
    for key in averaged_weights.keys():
        averaged_weights[key] = averaged_weights[key] / total_weight
    
    # Save weighted averaged weights
    torch.save(averaged_weights, output_path)
    print(f"\n✓ Saved weighted average backbone to: {output_path}")
    print(f"  Weight normalization factor: {1.0/total_weight:.4f}")
    print(f"{'='*80}\n")


# Example usage
if __name__ == "__main__":
    import os
    
    # Example: Average 5-fold checkpoints
    fold_paths = [
        f"results/deephp_backbone_pretrained_convnext_tiny_f{i}.pth" 
        for i in range(5)
    ]
    
    output_path = "results/deephp_backbone_final_convnext_tiny.pth"
    
    average_backbone_weights(fold_paths, output_path)
