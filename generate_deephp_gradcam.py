#!/usr/bin/env python3
"""
Generate Grad-CAM visualizations for DeepHP pre-trained backbones post-training.

This script loads a trained DeepHP backbone model and generates Grad-CAM visualizations
for validation set predictions. Generates individual side-by-side images (original + overlay)
for each sample, organized by prediction category (TP/FP/FN/TN).

Usage:
    python3 generate_deephp_gradcam.py --run 01_34.4 --fold 0 --model convnext_tiny
    python3 generate_deephp_gradcam.py --run 01_34.4 --fold 0-4 --model convnext_tiny  # All folds
    python3 generate_deephp_gradcam.py --run 01_34.4 --fold 0,1,3 --model convnext_tiny
    python3 generate_deephp_gradcam.py --backbone_path results/deephp_backbone_final_01_convnext_tiny_34.4.pth --fold 0-4 --model convnext_tiny

Output:
    - Saves individual Grad-CAM images to: results/{backbone_name}_f{fold}_gradcam/ (if --backbone_path provided)
      or results/{run}_f{fold}_{model}_gradcam/ (if using auto-detected backbone)
    - Files named: f{fold}_{category}_{index}.png (e.g., f0_TP_0000.png)
"""

import argparse
import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEEPHP_DATASET_ROOT
from dataset_deepHP import DeepHPDataset, create_deephp_transforms_val
from model import get_model


def load_validation_set(fold_idx, model_name="convnext_tiny"):
    """
    Load DeepHP validation set for a specific fold.
    
    Args:
        fold_idx: Fold index (0-4)
        model_name: Model architecture name
        
    Returns:
        dataset: DeepHPDataset in validation mode
        val_loader: DataLoader for validation set
    """
    print(f"Loading DeepHP validation set for fold {fold_idx}...")
    
    val_transform = create_deephp_transforms_val()
    dataset = DeepHPDataset(
        root_dir=DEEPHP_DATASET_ROOT,
        fold=fold_idx,
        num_folds=5,
        train=False,
        transform=val_transform
    )
    
    # Adjust DataLoader parameters based on device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 2 if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()
    
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"  ✓ Loaded {len(dataset)} validation patches")
    return dataset, val_loader


def load_backbone(backbone_path, model_name="convnext_tiny", device="cuda"):
    """
    Load pre-trained backbone weights.
    
    Args:
        backbone_path: Path to backbone checkpoint
        model_name: Model architecture name
        device: torch device
        
    Returns:
        model: Model with loaded backbone weights
    """
    print(f"Loading backbone from: {backbone_path}")
    
    if not os.path.exists(backbone_path):
        print(f"ERROR: Backbone not found at {backbone_path}")
        sys.exit(1)
    
    # Create model
    model = get_model(
        model_name=model_name,
        num_classes=2,
        pretrained=False,
        pool_type="attention",
        dropout=0.25
    )
    
    # Load backbone weights
    checkpoint = torch.load(backbone_path, map_location=device)
    
    # Extract backbone state dict (handle wrapper models)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    # Load with strict=False to ignore missing MIL head keys
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"  ✓ Backbone loaded successfully")
    return model


def generate_gradcam_batch(model, images, labels, device):
    """
    Generate Grad-CAM for a batch of images using the backbone.
    
    Args:
        model: Backbone model
        images: Input tensor (B, C, H, W)
        labels: Ground truth labels (B,)
        device: torch device
        
    Returns:
        heatmaps: (B, H, W) Grad-CAM heatmaps in [0, 1]
        predictions: (B,) Model predictions
        probabilities: (B,) Softmax probabilities for positive class
    """
    model.eval()
    heatmaps = []
    predictions = []
    probabilities = []
    
    for i in range(images.shape[0]):
        img_input = images[i:i+1].clone().detach().requires_grad_(True)
        
        try:
            with torch.enable_grad():
                logits = model(img_input)
                
                # Flatten if needed
                if len(logits.shape) > 2:
                    logits = torch.flatten(logits, 1)
                
                # Create scalar loss from all features
                loss = logits.sum()
            
            # Backward to compute gradients
            model.zero_grad()
            loss.backward()
            
            # Get gradients
            if img_input.grad is not None:
                # Compute absolute gradients, average across channels
                abs_grads = torch.abs(img_input.grad)  # (1, C, H, W)
                saliency = torch.sum(abs_grads, dim=1, keepdim=True)  # (1, 1, H, W)
                hmap = saliency[0, 0].detach().cpu().numpy()  # (H, W)
                
                # Normalize [0, 1]
                hmap_min = hmap.min()
                hmap = hmap - hmap_min
                hmap_max = hmap.max()
                if hmap_max > 0:
                    hmap = hmap / hmap_max
                
                # Apply Gaussian smoothing
                hmap = gaussian_filter(hmap, sigma=1.5)
                
                # Final normalization
                hmap = np.clip(hmap, 0, 1)
                hmap_min = hmap.min()
                hmap = hmap - hmap_min
                hmap_max = hmap.max()
                if hmap_max > 0:
                    hmap = hmap / hmap_max
                
                # Check if meaningful
                if np.std(hmap) < 1e-8:
                    hmap = np.zeros_like(hmap)
                
                heatmaps.append(hmap)
            
            # Get predictions
            with torch.no_grad():
                probs = F.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                prob = probs[0, 1].item()  # Probability of positive class
                
                predictions.append(pred)
                probabilities.append(prob)
        
        except Exception as e:
            print(f"  Warning: Error computing Grad-CAM for sample {i}: {e}")
            heatmaps.append(np.zeros((images.shape[2], images.shape[3])))
            predictions.append(0)
            probabilities.append(0.5)
        
        finally:
            del img_input
            torch.cuda.empty_cache()
    
    return np.array(heatmaps), np.array(predictions), np.array(probabilities)


def save_gradcam_image(patch_img, heatmap, label, prediction, prob, output_dir, 
                       fold_idx, img_idx, category):
    """
    Save individual Grad-CAM visualization as side-by-side image.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Denormalize image
    orig_img = patch_img.cpu().permute(1, 2, 0).numpy()
    orig_img = orig_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    orig_img = np.clip(orig_img, 0, 1)
    
    # Create side-by-side figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Left: Original image
    axes[0].imshow(orig_img)
    label_str = "POS" if label == 1 else "NEG"
    pred_str = "POS" if prediction == 1 else "NEG"
    axes[0].set_title(f"Original\nTrue: {label_str}, Pred: {pred_str}")
    axes[0].axis('off')
    
    # Right: Grad-CAM overlay
    axes[1].imshow(orig_img)
    if heatmap is not None and np.std(heatmap) > 1e-8:
        hmap_enhanced = np.power(heatmap, 0.2)
        axes[1].imshow(hmap_enhanced, cmap='YlOrRd', alpha=0.85, vmin=0, vmax=1)
    axes[1].set_title(f"Grad-CAM\nProb: {prob:.2f}")
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save with descriptive filename
    output_path = os.path.join(output_dir, f"f{fold_idx}_{category}_{img_idx:04d}.png")
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_and_save_gradcams(images, labels, predictions, heatmaps, probabilities, 
                               output_dir, fold_idx):
    """
    Generate and save individual Grad-CAM images for each sample.
    """
    # Categorize predictions
    tp_mask = (predictions == 1) & (labels == 1)
    fp_mask = (predictions == 1) & (labels == 0)
    fn_mask = (predictions == 0) & (labels == 1)
    tn_mask = (predictions == 0) & (labels == 0)
    
    categories = {
        'TP': tp_mask,
        'FP': fp_mask,
        'FN': fn_mask,
        'TN': tn_mask
    }
    
    saved_count = 0
    for category, mask in categories.items():
        indices = np.where(mask)[0]
        
        for count, idx in enumerate(indices):
            idx = int(idx)
            
            # Save individual image
            save_gradcam_image(
                images[idx],
                heatmaps[idx],
                labels[idx],
                predictions[idx],
                probabilities[idx],
                output_dir,
                fold_idx,
                count,
                category
            )
            saved_count += 1
            
            # Limit to prevent excessive output
            if count >= 9:  # Max 10 per category
                break
    
    print(f"  ✓ Saved {saved_count} individual Grad-CAM images to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualizations for DeepHP pre-trained backbones"
    )
    parser.add_argument("--run", type=str, default=None, 
                       help="Run ID with iteration (e.g., '01_34.4'). Required if --backbone_path not provided.")
    parser.add_argument("--fold", type=str, default="0",
                       help="Fold index or range (e.g., '0', '0-4', '0,2,4')")
    parser.add_argument("--model", type=str, default="convnext_tiny",
                       choices=["convnext_tiny", "convnext_small", "resnet50"],
                       help="Model architecture")
    parser.add_argument("--backbone_path", type=str, default=None,
                       help="Custom backbone path (default: auto-search results/deephp_backbone_final_*_{model}_{iter}.pth)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Torch device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Validate that either --run or --backbone_path is provided
    if not args.run and not args.backbone_path:
        parser.error("Either --run or --backbone_path must be provided")
    
    # Parse fold argument
    folds = []
    if "-" in args.fold:
        start, end = map(int, args.fold.split("-"))
        folds = list(range(start, end + 1))
    elif "," in args.fold:
        folds = [int(x.strip()) for x in args.fold.split(",")]
    else:
        folds = [int(args.fold)]
    
    # Extract iteration from run_id (only needed if backbone_path not provided)
    run_id = None
    iter_name = None
    if args.run:
        try:
            run_id, iter_name = args.run.rsplit("_", 1)
        except ValueError:
            print(f"ERROR: Invalid run ID format. Expected format: '01_34.4', got '{args.run}'")
            sys.exit(1)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Process each fold
    for fold_idx in folds:
        print(f"\n{'='*80}")
        print(f"Generating Grad-CAM for Fold {fold_idx}")
        print(f"{'='*80}")
        
        # Find or use provided backbone
        if args.backbone_path:
            backbone_path = args.backbone_path
        else:
            # Search for backbone with pattern
            backbone_pattern = f"results/deephp_backbone_final_{run_id}_{args.model}_{iter_name}.pth"
            if os.path.exists(backbone_pattern):
                backbone_path = backbone_pattern
            else:
                # Try to find with glob
                from glob import glob
                matches = glob(f"results/deephp_backbone_final_*_{args.model}_{iter_name}.pth")
                if matches:
                    backbone_path = sorted(matches, key=lambda x: os.path.getmtime(x), reverse=True)[0]
                else:
                    print(f"ERROR: Could not find backbone for model={args.model}, iter={iter_name}")
                    print(f"  Searched pattern: results/deephp_backbone_final_*_{args.model}_{iter_name}.pth")
                    continue
        
        # Load data and model
        try:
            dataset, val_loader = load_validation_set(fold_idx, args.model)
            model = load_backbone(backbone_path, args.model, device)
        except Exception as e:
            print(f"ERROR loading data/model: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Generate predictions and Grad-CAM
        print(f"Generating Grad-CAM visualizations...")
        all_images = []
        all_labels = []
        all_predictions = []
        all_heatmaps = []
        all_probabilities = []
        
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Processing batches")):
            # Handle DANN datasets that return 3-tuples with exp_indices
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            
            images = images.to(device)
            labels = labels.to(device)
            
            heatmaps, predictions, probabilities = generate_gradcam_batch(
                model, images, labels, device
            )
            
            all_images.append(images.cpu())
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions)
            all_heatmaps.append(heatmaps)
            all_probabilities.append(probabilities)
        
        # Concatenate results
        all_images = torch.cat(all_images)
        all_labels = np.concatenate(all_labels)
        all_predictions = np.concatenate(all_predictions)
        all_heatmaps = np.concatenate(all_heatmaps)
        all_probabilities = np.concatenate(all_probabilities)
        
        # Generate individual image visualizations
        # Use backbone filename if custom backbone provided, otherwise use run_id
        if args.backbone_path:
            backbone_name = os.path.splitext(os.path.basename(args.backbone_path))[0]
            output_dir = f"results/{backbone_name}_f{fold_idx}_gradcam"
        else:
            output_dir = f"results/{args.run}_f{fold_idx}_{args.model}_gradcam"
        
        generate_and_save_gradcams(
            all_images, all_labels, all_predictions, all_heatmaps, 
            all_probabilities, output_dir, fold_idx
        )
        
        # Print statistics
        accuracy = (all_predictions == all_labels).mean()
        print(f"  Validation Accuracy: {accuracy:.2%}")
        print(f"  Pos predictions: {(all_predictions == 1).sum()}")
        print(f"  Neg predictions: {(all_predictions == 0).sum()}")
    
    print(f"\n✓ Grad-CAM generation complete!")


if __name__ == "__main__":
    main()
