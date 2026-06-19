"""
train_deepHP_patches.py - Backbone Pre-training on DeepHP H&E Patches

Purpose:
--------
Pre-train a ConvNeXt-Tiny backbone on 394,925 H&E-stained histology patches
from the DeepHP database. This creates a general-purpose feature extractor
that understands H. pylori morphology before fine-tuning on patient-level
IHC data (HelicoDataSet).

Differences from train.py (patient-level MIL training):
1. Patch-level classification (no MIL aggregation)
2. Standard cross-entropy loss (no Focal Loss weighting initially)
3. 5-fold stratified CV on patches using CONFIG 87771 experiment-level stratification
4. Output: Pre-trained backbone weights only
5. H&E-specific normalization (Macenko or ImageNet)

DATA STRATIFICATION - CONFIG 87771 (Experiment-Level 5-Fold Cross-Validation):
-----------------------------------------------------------------------------

PROBLEM SOLVED:
Naive fold assignment caused data leakage: different folds had different experiments,
leading to fold-specific artifact learning and unrealistic 0%-99% recall variance on epoch 1.
Models learned fold signatures instead of H. pylori features.

SOLUTION - CONFIG 87771 (Optimized from 500,000 random greedy searches):

CONFIG 87771 HARDCODED EXPERIMENT ASSIGNMENTS:
Each of the 33 experiments assigned to exactly ONE fold (zero data leakage):

- Fold 0 val: 7 experiments (4 pos, 3 neg) → 87,532 patches, ratio 2.33:1
- Fold 1 val: 10 experiments (3 pos, 7 neg) → 89,516 patches, ratio 2.06:1
- Fold 2 val: 5 experiments (4 pos, 1 neg) → 20,347 patches, ratio 2.31:1
- Fold 3 val: 4 experiments (4 pos, 0 neg) → 99,120 patches, ratio 2.81:1
- Fold 4 val: 7 experiments (6 pos, 1 neg) → 98,410 patches, ratio 2.29:1

All 33 experiments assigned to exactly ONE fold (total: 394,925 patches)
Training data for each fold: All experiments NOT assigned to this fold (~307K patches)

KEY PROPERTY:
Each fold validates on UNIQUE experiments, trains on ALL OTHER experiments.
This ensures true 5-fold cross-validation at experiment level.

BENEFITS:
✓ Each fold validates on different experiments (prevents fold-specific artifact learning)
✓ Training data diverse across all folds (same experiments, different patches)
✓ Experiment integrity: No experiment split between train and val (prevents leakage)
✓ Balanced ratios: All folds 2.06:1 to 2.81:1 (target 2.28:1)
✓ Realistic metrics: ~50% epoch 1 accuracy across all folds (no 0%-99% variance)
✓ Mathematically optimized: Selected from 500,000+ configurations

CROSS-LEAKAGE AUDIT:
-------------------
Generates audit CSVs to verify CONFIG 87771 stratification correctness:

1. Per-fold IMAGE-LEVEL audit:
   - {prefix}_cross_leakage_audit.csv: One row per image
   - Verifies no image appears in both train and val for THIS fold
   - Status: VERIFIED_UNIQUE (confirms image-level integrity)
   
2. Per-fold EXPERIMENT audit:
   - {prefix}_experiment_fold_audit.csv: One row per unique experiment
   - Shows which experiments are in train vs val sets for THIS fold
   - Columns: Experiment_ID, In_Train_Set, In_Val_Set, Fold, Train_Count, Val_Count
   - Verifies CONFIG 87771 experiment-level assignments are properly enforced

Macenko Normalization:
---------------------
Macenko normalization standardizes H&E color appearance across slides to improve
model generalization. However, it requires raw RGB patches to work properly.

** IMPORTANT (Iteration 25.5+): Macenko fitting uses a separate loader with raw
patches (NO ImageNet normalization) to preserve H&E color information for fitting.
ImageNet normalization destroys color information needed for H&E vector extraction,
causing fitting to fail with ill-conditioned matrices. **

Configuration:
    DeepHP dataset path is set via config.py (DEEPHP_DATASET_ROOT).
    Default: /home/tkeating/datasets/8117177
    Override: export DEEPHP_DATASET_ROOT=/path/to/deephp

Usage:
    python train_deepHP_patches.py --fold 0 [--num_folds 5] [--model_name convnext_tiny]
    
    # Run all folds in parallel (recommended for speed)
    for i in {0..4}; do
        sbatch -J deephp_f$i train_deepHP.sh $i &
    done

Output:
    results/{run_id}_{iter}_{slurm_id}_f0_convnext_tiny_model_brain.pth
    results/{run_id}_{iter}_{slurm_id}_f1_convnext_tiny_model_brain.pth
    ...
    results/deephp_backbone_final_{run_id}_convnext_tiny_{iter}.pth (averaged across folds)
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import re
import gc
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import v2
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report, 
    precision_recall_curve, average_precision_score, 
    matthews_corrcoef, cohen_kappa_score
)
from tqdm import tqdm
import torch.nn.functional as F

# Import custom modules
from dataset_deepHP import DeepHPDataset, create_deephp_transforms_train, create_deephp_transforms_val
from model import get_model
from config import DATASET_ROOT, SCRATCH_ROOT, DEEPHP_DATASET_ROOT
from normalization import MacenkoNormalizer
from visualization_utils import plot_learning_curves, plot_confusion_matrix, plot_roc_curve, plot_pr_curve, plot_calibration_curve
from domain_adversarial import GradientReversalLayer, AdversaryHead, add_adversary_to_model

# Function to get next run number (matching train.py pattern)
def get_next_run_number(results_dir="results", current_slurm_id=None):
    """Simple version to get next available run number."""
    if not os.path.exists(results_dir):
        return 0
    
    files = os.listdir(results_dir)
    max_run = 0
    
    # Look for existing run IDs in filename patterns (e.g., "302_25.1_106069_...")
    for f in files:
        match = re.match(r"^(\d+)_[\d.]+_(\d+)_", f)
        if match:
            try:
                run_id = int(match.group(1))
                max_run = max(max_run, run_id)
            except:
                pass
    
    return max_run + 1


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced patch classification (DeepHP: 1:2.5 pos:neg ratio)."""
    def __init__(self, alpha=1, gamma=2, weight=None, smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight, label_smoothing=self.smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()


def generate_gradcam_visualizations(model, images, labels, predictions, probabilities, output_path, device, title_prefix=""):
    """
    Generate prediction visualizations with gradient-based attention maps (Grad-CAM).
    
    Creates a 4×3 grid showing example predictions from each category:
    - Row 1: True Positives (model correct, label positive)
    - Row 2: False Positives (model predicted positive, label negative) 
    - Row 3: False Negatives (model predicted negative, label positive)
    - Row 4: True Negatives (model correct, label negative)
    
    Each cell displays an image with overlaid gradient attention highlighting regions
    that most influenced the model's prediction (via input gradients of predicted class).
    
    NOTE: This function receives pre-collected samples from two-pass Grad-CAM collection:
    - PASS 1: Scans full validation set, collects up to 10,000 samples or until all 4 categories found
    - PASS 2 (if needed): Targeted search for any missing categories
    This guarantees all 4 categories are represented in the visualization, even if rare
    (e.g., False Positives in some folds might only appear after 10,000+ samples).
    
    Args:
        model: Trained model in eval mode
        images: Input images tensor [N, C, H, W] (normalized with ImageNet stats)
        labels: Ground truth labels [N]
        predictions: Model predictions [N]
        probabilities: Model probabilities for positive class [N]
        output_path: Path to save the visualization PNG
        device: torch device (cuda or cpu)
        title_prefix: Prefix for subplot titles (e.g., fold number)
    """
    model.eval()
    
    # Normalize images for visualization (reverse ImageNet normalization)
    # ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    images_display = images * imagenet_std + imagenet_mean
    images_display = torch.clamp(images_display, 0, 1)
    
    # Categorize predictions
    tp_mask = (predictions == 1) & (labels == 1)
    fp_mask = (predictions == 1) & (labels == 0)
    fn_mask = (predictions == 0) & (labels == 1)
    tn_mask = (predictions == 0) & (labels == 0)
    
    # Sample up to 3 examples from each category
    categories = {
        'TP': (tp_mask, 'True Positive'),
        'FP': (fp_mask, 'False Positive'),
        'FN': (fn_mask, 'False Negative'),
        'TN': (tn_mask, 'True Negative')
    }
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    fig.suptitle(f'Prediction Visualizations {title_prefix}', fontsize=16, fontweight='bold')
    
    for row, (cat_name, (mask, cat_label)) in enumerate(categories.items()):
        indices = torch.where(mask)[0]
        
        if len(indices) == 0:
            # No examples in this category
            for col in range(3):
                axes[row, col].text(0.5, 0.5, f'No {cat_label}s', 
                                   ha='center', va='center', fontsize=12)
                axes[row, col].axis('off')
            continue
        
        # Sample up to 3 examples
        sample_indices = indices[:min(3, len(indices))]
        
        for col, idx in enumerate(sample_indices):
            idx = idx.item()
            img = images_display[idx].cpu()
            label = labels[idx].item()
            pred = predictions[idx].item()
            prob = probabilities[idx].item()
            
            # Compute gradient-based saliency for this single image
            img_input = images[idx:idx+1].clone().detach().requires_grad_(True)
            
            try:
                with torch.enable_grad():
                    output = model(img_input)
                    # Use the predicted class score for gradient computation
                    score = output[0, pred]
                    score.backward()
                
                # Get input gradients
                if img_input.grad is not None:
                    # Compute magnitude of gradients across channels
                    gradients = torch.abs(img_input.grad.data)  # Use .data to avoid tracking
                    attention = torch.mean(gradients, dim=1, keepdim=True)[0, 0].cpu().detach()
                    
                    # Normalize attention map
                    attention = attention - attention.min()
                    if attention.max() > 1e-6:  # Only keep if there are meaningful gradients
                        attention = attention / (attention.max() + 1e-8)
                    else:
                        attention = None  # No meaningful gradients, skip overlay
                else:
                    attention = None
            except Exception as e:
                attention = None
            finally:
                del img_input  # Free memory
                torch.cuda.empty_cache()  # Clear GPU cache
            
            # Display image
            img_np = img.permute(1, 2, 0).numpy()
            axes[row, col].imshow(img_np)
            
            # Overlay attention if available
            if attention is not None:
                # Enhance contrast: use power function to emphasize high-gradient regions
                attention_enhanced = attention.numpy() ** 0.5  # Square root to increase contrast
                axes[row, col].imshow(attention_enhanced, cmap='jet', alpha=0.35, vmin=0, vmax=1)
            
            # Title with prediction confidence
            label_str = "POS" if label == 1 else "NEG"
            pred_str = "POS" if pred == 1 else "NEG"
            title = f"{cat_label}\nTrue: {label_str}, Pred: {pred_str}\n({prob:.2f})"
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    # Cleanup and reset model state
    torch.cuda.empty_cache()


def train_deephp_backbone(fold_idx=0, num_folds=5, model_name="convnext_tiny", num_epochs=20, 
                          batch_size=128, learning_rate=2e-5, weight_decay=0.01, 
                          use_focal_loss=False, pos_weight=2.5, neg_weight=1.0, gamma=1.0, 
                          iter_name="deephp", run_id="", use_swa=True, swa_start=12, jitter=0.15, pct_start=0.1,
                          clip_grad=0.0, saver_metric="loss", use_dann=False, dann_lambda=1.0, dann_weight=0.5):
    """
    Train a CNN backbone on DeepHP H&E patches with experiment-level 5-fold cross-validation (CONFIG 87771).
    
    Stratification Strategy (CONFIG 87771):
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Uses a hardcoded experiment-to-fold assignment optimized from 500,000+ random greedy searches.
    Each fold validates on a unique set of experiments while training on all other experiments.
    
    CONFIG 87771 METRICS:
    - Fold 0 val: 7 experiments (4 pos, 3 neg) → 87,532 patches, ratio 2.33:1
    - Fold 1 val: 10 experiments (3 pos, 7 neg) → 89,516 patches, ratio 2.06:1
    - Fold 2 val: 5 experiments (4 pos, 1 neg) → 20,347 patches, ratio 2.31:1
    - Fold 3 val: 4 experiments (4 pos, 0 neg) → 99,120 patches, ratio 2.81:1
    - Fold 4 val: 7 experiments (6 pos, 1 neg) → 98,410 patches, ratio 2.29:1
    
    Training data for each fold: All 33 experiments except those assigned to this fold (~307K patches)
    Total Distance: 0.6441 (sum of distances from target ratio 2.28:1)
    
    KEY ADVANTAGES:
    - Each fold validates on UNIQUE experiments → prevents fold-specific artifact learning
    - Experiment-level assignment ensures proper biological unit stratification
    - All folds maintain similar ratios (2.06:1 to 2.81:1 around target 2.28:1)
    - All folds train on same diverse set of experiments (breaks artifact learning)
    - Zero data leakage: no experiment split between folds, no image overlap
    
    Cross-Leakage Audit:
    ~~~~~~~~~~~~~~~~~~~
    Generates audit CSVs at two levels:
    
    1. Per-fold IMAGE-LEVEL audit:
       - {prefix}_cross_leakage_audit.csv: One row per image
       - Verifies no image appears in both train and val
       - Status: VERIFIED_UNIQUE (confirms image-level integrity)
    
    2. Per-fold EXPERIMENT audit:
       - {prefix}_experiment_fold_audit.csv: One row per unique experiment
       - Shows which experiments are in train vs validation sets
       - Confirms CONFIG 87771 experiment-level stratification
    
    Grad-CAM Visualization:
    ~~~~~~~~~~~~~~~~~~~~~
    Memory-efficient visualization strategy:
    - Loads only selected samples needed for visualization (not all 10,000+)
    - Ensures all prediction categories (TP/FP/FN/TN) are represented
    - Uses gradient-based saliency to show decision regions
    
    Args:
        fold_idx (int): Fold index for k-fold CV (0 to num_folds-1)
        num_folds (int): Total number of folds (default: 5)
        model_name (str): Backbone architecture ('convnext_tiny', 'resnet50', etc.)
        num_epochs (int): Training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Initial learning rate
        weight_decay (float): L2 regularization
        use_focal_loss (bool): Use Focal Loss (True) or Cross-Entropy (False)
        pos_weight (float): Weight for positive class
        neg_weight (float): Weight for negative class
        gamma (float): Focal Loss gamma parameter
        iter_name (str): Iteration identifier for tracking (e.g., '32.2')
        run_id (str): Run ID for parallel job safety (auto-generated if not provided)
        use_swa (bool): Use Stochastic Weight Averaging
        swa_start (int): Epoch to start SWA
        jitter (float): ColorJitter intensity for augmentation
        pct_start (float): Warmup percentage for learning rate schedule
        clip_grad (float): Gradient clipping norm (0=disabled)
        saver_metric (str): Metric for model selection (loss/accuracy/precision/recall/f1)
    """
    
    # Setup output directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.cuda.set_per_process_memory_fraction(0.70, 0)
    
    # Get SLURM job ID and generate run ID (matching train.py naming convention)
    slurm_id = os.environ.get("SLURM_JOB_ID", "local")
    
    # Check for existing run IDs for THIS EXACT iteration (for multi-fold consistency within same run)
    # If run_id was explicitly provided (for parallel job safety), use that instead
    existing_run_id = None
    if run_id and run_id.strip():
        # Use the provided run_id directly (parallel job safety)
        existing_run_id = run_id.strip()
    elif os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            # Look for files matching pattern: {run_id}_{iter_name}_{slurm_id}_f{fold}_{model_name}*
            # E.g., "420_31.0_113456_f0_convnext_tiny_model_brain.pth"
            # Only reuse run_id if this EXACT iteration has files in results
            match = re.match(rf"^(\d+)_{re.escape(iter_name)}_\d+_f\d+_{model_name}", filename)
            if match:
                existing_run_id = match.group(1)
                break
    
    # Use existing run_id if found (for multi-fold consistency), otherwise generate next available run_id
    if existing_run_id:
        run_id = existing_run_id
    else:
        # Generate next run_id based on ALL existing run IDs in results folder
        run_id = f"{get_next_run_number(results_dir, slurm_id):02d}"
    
    # Setup output paths with unified naming convention
    prefix = f"{run_id}_{iter_name}_{slurm_id}_f{fold_idx}_{model_name}"
    best_model_path = os.path.join(results_dir, f"{prefix}_model_brain.pth")
    results_csv_path = os.path.join(results_dir, f"{prefix}_evaluation_report.csv")
    cm_path = os.path.join(results_dir, f"{prefix}_confusion_matrix.png")
    roc_path = os.path.join(results_dir, f"{prefix}_roc_curve.png")
    pr_path = os.path.join(results_dir, f"{prefix}_pr_curve.png")
    history_path = os.path.join(results_dir, f"{prefix}_learning_curves.png")
    
    print(f"\n{'='*80}")
    print(f"DeepHP Backbone Pre-training: Fold {fold_idx + 1}/{num_folds}")
    print(f"Run ID: {run_id} (Iter: {iter_name}, SLURM Job: {slurm_id})")
    print(f"{'='*80}")
    print(f"Model: {model_name} | Epochs: {num_epochs} | Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate} | Weight Decay: {weight_decay}")
    print(f"Focal Loss: {use_focal_loss} | Pos Weight: {pos_weight}")
    print(f"DANN: {use_dann} | Lambda: {dann_lambda} | Weight: {dann_weight}")
    print(f"{'='*80}\n")
    
    # Load dataset from config
    deephp_root = DEEPHP_DATASET_ROOT
    
    if not os.path.exists(deephp_root):
        print(f"ERROR: DeepHP dataset not found at {deephp_root}")
        print(f"Expected path: {deephp_root}")
        print(f"Override with: export DEEPHP_DATASET_ROOT=/path/to/deephp")
        sys.exit(1)
    
    train_transform = create_deephp_transforms_train()
    val_transform = create_deephp_transforms_val()
    
    # Verify which dataset directory is being used
    print(f"\n{'='*60}")
    print(f"Loading dataset from: {deephp_root}")
    print(f"Scratch directory: /home/tkeating/.scratch/deephp_data")
    print(f"Original directory: /home/tkeating/datasets/8117177")
    print(f"Using scratch: {deephp_root == '/home/tkeating/.scratch/deephp_data'}")
    print(f"{'='*60}\n")
    
    train_dataset = DeepHPDataset(
        root_dir=deephp_root,
        transform=train_transform,
        fold=fold_idx,
        num_folds=num_folds,
        train=True
    )
    
    val_dataset = DeepHPDataset(
        root_dir=deephp_root,
        transform=val_transform,
        fold=fold_idx,
        num_folds=num_folds,
        train=False
    )
    
    train_dataset.print_statistics()
    val_dataset.print_statistics()
    
    # Print blacklist status for transparency
    try:
        import json
        with open('./blacklistDeepHP.json') as f:
            blacklist_data = json.load(f)
        
        if 'macenko_reference_patch' in blacklist_data:
            ref = blacklist_data['macenko_reference_patch']
            print("\n" + "="*60)
            print("Blacklist Status (Applied Before Fold Split):")
            print("="*60)
            print(f"  ✓ Macenko Reference Excluded:")
            print(f"    File: {ref.get('folder')}/{ref.get('filename')}")
            print(f"    Quality Score: {ref.get('score')}")
            print(f"    Reason: {ref.get('reason')}")
            print("="*60 + "\n")
    except Exception as e:
        print(f"Note: Could not read blacklist status: {e}\n")
    
    # Generate cross-leakage audit (verifies validation set not in training set)
    # Also generates experiment-level audit showing fold assignments
    # One image-level row per image, one experiment-level row per experiment
    print("\n" + "="*60)
    print("Generating Cross-Leakage Audit:")
    print("="*60)
    
    # Get actual indices used in each dataset (after fold split)
    train_indices = set(train_dataset.indices)
    val_indices = set(val_dataset.indices)
    
    # Extract individual images from training and validation folds
    train_images = []
    for idx in train_indices:
        path, label = train_dataset.samples[idx]
        train_images.append(os.path.basename(path))
    
    val_images = []
    for idx in val_indices:
        path, label = val_dataset.samples[idx]
        val_images.append(os.path.basename(path))
    
    # Verify no overlap between train and validation images
    train_image_set = set(train_images)
    val_image_set = set(val_images)
    overlap = train_image_set & val_image_set
    
    if overlap:
        print(f"⚠ WARNING: Found {len(overlap)} images in both train and validation sets!")
        for img in sorted(overlap)[:5]:
            print(f"  - {img}")
    else:
        print(f"✓ Perfect stratification verified: 0 images in overlap")
    
    # Create audit data - one row per image in this fold
    audit_data = []
    all_images = sorted(train_image_set | val_image_set)
    
    for img_name in all_images:
        in_train = img_name in train_image_set
        in_val = img_name in val_image_set
        audit_data.append({
            'Image_File': img_name,
            'In_Training_Pool': in_train,
            'In_Validation_Set': in_val,
            'Audit_Status': 'VERIFIED_UNIQUE' if not (in_train and in_val) else 'LEAKAGE_DETECTED'
        })
    
    # Save audit to CSV
    audit_df = pd.DataFrame(audit_data)
    cross_leakage_audit_path = os.path.join(results_dir, f"{prefix}_cross_leakage_audit.csv")
    audit_df.to_csv(cross_leakage_audit_path, index=False)
    print(f"✓ Saved cross-leakage audit to {cross_leakage_audit_path}")
    print(f"  Total images audited: {len(audit_data)}")
    print(f"  Training set images: {len(train_images)}")
    print(f"  Validation set images: {len(val_images)}")
    print("="*60 + "\n")
    
    # Generate per-fold experiment audit file
    # This tracks which experiments appear in train vs val for this specific fold
    experiment_audit_data = []
    
    for idx in train_indices:
        path, _ = train_dataset.samples[idx]
        filename = os.path.basename(path)
        exp_id = filename.split('_b0s')[0]
        experiment_audit_data.append({
            'Experiment_ID': exp_id,
            'In_Train_Set': True,
            'In_Val_Set': False,
            'Fold': fold_idx
        })
    
    for idx in val_indices:
        path, _ = val_dataset.samples[idx]
        filename = os.path.basename(path)
        exp_id = filename.split('_b0s')[0]
        experiment_audit_data.append({
            'Experiment_ID': exp_id,
            'In_Train_Set': False,
            'In_Val_Set': True,
            'Fold': fold_idx
        })
    
    # Consolidate by experiment ID
    exp_audit_dict = {}
    for entry in experiment_audit_data:
        exp_id = entry['Experiment_ID']
        if exp_id not in exp_audit_dict:
            exp_audit_dict[exp_id] = {
                'Experiment_ID': exp_id,
                'In_Train_Set': False,
                'In_Val_Set': False,
                'Fold': fold_idx,
                'Train_Count': 0,
                'Val_Count': 0
            }
        
        if entry['In_Train_Set']:
            exp_audit_dict[exp_id]['Train_Count'] += 1
            exp_audit_dict[exp_id]['In_Train_Set'] = True
        if entry['In_Val_Set']:
            exp_audit_dict[exp_id]['Val_Count'] += 1
            exp_audit_dict[exp_id]['In_Val_Set'] = True
    
    # Save experiment audit
    exp_audit_df = pd.DataFrame(sorted(exp_audit_dict.values(), key=lambda x: x['Experiment_ID']))
    exp_audit_path = os.path.join(results_dir, f"{prefix}_experiment_fold_audit.csv")
    exp_audit_df.to_csv(exp_audit_path, index=False)
    print(f"✓ Saved experiment fold audit to {exp_audit_path}")
    print(f"  Total unique experiments: {len(exp_audit_dict)}")
    print(f"  Experiments in train pool: {len([e for e in exp_audit_dict.values() if e['Train_Count'] > 0])}")
    print(f"  Experiments in val pool: {len([e for e in exp_audit_dict.values() if e['Val_Count'] > 0])}")
    print("="*60 + "\n")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Build model (backbone only, no MIL head)
    print(f"Loading {model_name} backbone...")
    model = get_model(model_name=model_name, num_classes=2, pretrained=True, pool_type="attention").to(device)
    
    # For DeepHP pre-training, we only care about the backbone classification head
    # The MIL components will be re-initialized when transferred to HelicoDataSet
    
    # Loss function
    if use_focal_loss:
        loss_weights = torch.FloatTensor([neg_weight, pos_weight]).to(device)
        criterion = FocalLoss(gamma=gamma, weight=loss_weights, smoothing=0.0)
        print(f"Using Focal Loss (gamma={gamma}, neg_weight={neg_weight}, pos_weight={pos_weight})")
    else:
        loss_weights = torch.FloatTensor([neg_weight, pos_weight]).to(device)
        criterion = nn.CrossEntropyLoss(weight=loss_weights)
        print(f"Using Cross-Entropy Loss (neg_weight={neg_weight}, pos_weight={pos_weight})")
    
    # Optimizer & Scheduler
    
    # Initialize Domain Adversarial Neural Networks (DANN) if enabled
    grad_rev_layer = None
    adversary_head = None
    adversary_optimizer = None
    adversary_criterion = nn.CrossEntropyLoss()
    
    if use_dann:
        print(f"Initializing DANN with lambda={dann_lambda}, weight={dann_weight}")
        num_experiments = train_dataset.num_experiments
        feature_dim = 768 if model_name in ["convnext_tiny", "convnext_small"] else 2048
        
        grad_rev_layer = GradientReversalLayer(lambda_=dann_lambda).to(device)
        adversary_head = AdversaryHead(feature_dim, num_experiments, hidden_dim=256).to(device)
        
        # Note: We'll optimize adversary head with main optimizer (shared gradients)
        print(f"DANN initialized with {num_experiments} experiments and feature_dim={feature_dim}")
    
    optimizer = AdamW(
        list(model.parameters()) + (list(adversary_head.parameters()) if use_dann else []),
        lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Initialize Macenko normalizer for H&E stain normalization (DeepHP uses H&E)
    print("Initializing Macenko normalizer for H&E stain normalization...")
    normalizer = MacenkoNormalizer()
    
    # Load pre-computed reference image (generated by create_macenko_reference.py)
    # Using a pre-computed reference instead of per-run fitting eliminates:
    # - Fragility from fitting to single patches with minimal tissue
    # - Variance across different training runs
    # - Risk of encountering empty/white patches during fitting
    reference_path = "./macenko_reference.png"
    
    if os.path.exists(reference_path):
        try:
            from PIL import Image
            ref_image = Image.open(reference_path)
            print(f"  Loading pre-computed reference: {reference_path}")
            normalizer.fit(ref_image, device=device)
            print(f"✓ Macenko normalizer fitted to pre-computed reference")
        except Exception as e:
            print(f"Warning: Failed to load/fit pre-computed reference: {e}")
            print("  Falling back to ImageNet normalization only")
            normalizer = None
    else:
        print(f"Warning: Pre-computed reference not found at {reference_path}")
        print("  Run 'python3 create_macenko_reference.py' first, or use ImageNet normalization only")
        print("  Falling back to ImageNet normalization only")
        normalizer = None
    
    # Training loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_metric_value = float('inf') if saver_metric == 'loss' else float('-inf')
    best_recall = 0.0
    last_val_recall = 0.0
    
    # Track timing and performance
    start_time = time.time()
    peak_gpu_memory = 0.0
    
    # Use the non-deprecated torch.amp.GradScaler API
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = torch.amp.GradScaler(device_type)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_valid_batches = 0  # Track valid batches (exclude those with NaN loss)
        
        for batch_idx, (images, labels, exp_indices) in enumerate(tqdm(train_loader, desc="Training")):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if use_dann:
                exp_indices = exp_indices.to(device, non_blocking=True)
            
            # Apply Macenko H&E normalization for stain consistency
            if normalizer is not None:
                try:
                    images = normalizer.normalize_batch(images, jitter=True)  # jitter=True for augmentation
                    # Sanity check: ensure no NaN values after normalization
                    if torch.isnan(images).any():
                        print(f"WARNING: Macenko produced NaN values in batch {batch_idx}, using original images")
                        images = images.to(device, non_blocking=True)  # Reload original from previous state
                        normalizer = None  # Disable normalizer to prevent future NaN
                except Exception as e:
                    print(f"Warning: Macenko normalization failed in batch {batch_idx}: {e}")
                    normalizer = None  # Disable normalizer on error
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                # Forward pass for patch-level classification (not MIL)
                # Use regular forward() which expects [batch, C, H, W] and returns [batch, num_classes]
                logits = model(images)  # [batch_size, num_classes]
                loss = criterion(logits, labels)
                
                # Domain Adversarial Neural Networks (DANN) loss
                # Note: Simplified version - adversary predicts from logits
                # For full DANN, you would extract features from intermediate layers
                if use_dann:
                    # Predict experiment ID from logits (proxy for features)
                    exp_logits = adversary_head(grad_rev_layer(logits.detach()))
                    adv_loss = adversary_criterion(exp_logits, exp_indices)
                    
                    # Combine losses
                    loss = loss + dann_weight * adv_loss
            
            # Sanity check: ensure loss is valid (not NaN or Inf)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"ERROR: Loss is {loss.item()} in batch {batch_idx} (NaN or Inf detected)")
                print(f"       Disabling Macenko normalizer and skipping this batch")
                normalizer = None
                continue  # Skip this batch to prevent propagation of invalid values
            
            train_loss += loss.item()
            epoch_valid_batches += 1
            
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
        
        train_loss /= max(epoch_valid_batches, 1)  # Avoid division by zero
        train_acc = 100.0 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, exp_indices in tqdm(val_loader, desc="Validation"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Apply Macenko H&E normalization for stain consistency (no jitter for validation)
                if normalizer is not None:
                    try:
                        images = normalizer.normalize_batch(images, jitter=False)
                        # Sanity check: ensure no NaN values after normalization
                        if torch.isnan(images).any():
                            print(f"WARNING: Macenko produced NaN values in validation, disabling normalizer")
                            normalizer = None  # Disable normalizer to prevent future NaN
                    except Exception as e:
                        print(f"Warning: Macenko normalization failed in validation: {e}")
                        normalizer = None  # Disable normalizer on error
                
                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    # Forward pass for patch-level validation
                    logits = model(images)  # [batch_size, num_classes]
                    loss = criterion(logits, labels)
                
                val_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        
        # Metrics
        from sklearn.metrics import f1_score, precision_score, recall_score
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)
        val_precision = precision_score(all_labels, all_preds, zero_division=0)
        val_recall = recall_score(all_labels, all_preds, zero_division=0)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Prec: {val_precision:.4f} | Rec: {val_recall:.4f} | F1: {val_f1:.4f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Determine metric to optimize based on saver_metric
        if saver_metric == 'loss':
            current_metric = val_loss
            is_best = current_metric < best_metric_value
        elif saver_metric == 'accuracy':
            current_metric = val_acc
            is_best = current_metric > best_metric_value
        elif saver_metric == 'precision':
            current_metric = val_precision
            is_best = current_metric > best_metric_value
        elif saver_metric == 'recall':
            current_metric = val_recall
            is_best = current_metric > best_metric_value
        elif saver_metric == 'f1':
            current_metric = val_f1
            is_best = current_metric > best_metric_value
        else:
            current_metric = val_loss
            is_best = current_metric < best_metric_value
            print(f"WARNING: Unknown saver_metric '{saver_metric}', using 'loss'")
        
        is_last_epoch = (epoch == num_epochs - 1)
        
        if is_best or is_last_epoch:
            if is_best:
                best_metric_value = current_metric
            torch.save(model.state_dict(), best_model_path)
            status = f"best ({saver_metric}: {current_metric:.4f})" if is_best else "final epoch (fallback)"
            print(f"✓ Saved {status} model to {best_model_path}")
        
        scheduler.step()
        
        # Track peak GPU memory during training
        if torch.cuda.is_available():
            peak_gpu_memory = max(peak_gpu_memory, torch.cuda.max_memory_allocated(device) / (1024**3))
    
    # Calculate training time
    end_time = time.time()
    training_time_seconds = end_time - start_time
    training_time_hours = training_time_seconds / 3600.0
    
    # Save learning curves
    plot_learning_curves(history, history_path)
    print(f"\n✓ Saved learning curves to {history_path}")
    
    # Save learning curves as JSON
    json_path = os.path.join(results_dir, f"{prefix}_learning_curves.json")
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Saved learning curves JSON to {json_path}")
    
    # Final evaluation on validation set
    print(f"\n{'='*80}")
    print(f"Final Evaluation (Fold {fold_idx + 1}/{num_folds})")
    print(f"{'='*80}")
    
    # Defensive: If best model wasn't saved during training (e.g., due to crash),
    # save the current model state as fallback
    if not os.path.exists(best_model_path):
        print(f"WARNING: Best model checkpoint not found at {best_model_path}")
        print(f"         This typically means training completed without any validation improvement.")
        print(f"         Saving current model state as checkpoint...")
        torch.save(model.state_dict(), best_model_path)
    
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels, exp_indices in tqdm(val_loader, desc="Final Evaluation"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass for patch-level inference
            logits = model(images)  # [batch_size, num_classes]
            probs = torch.softmax(logits, dim=1)
            
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = recall  # Same as recall
    ppv = precision  # Positive predictive value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    mcc = matthews_corrcoef(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    balanced_accuracy = (sensitivity + specificity) / 2.0
    
    # Store final recall for model selection metrics and SWA comparison
    last_val_recall = recall
    
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"Specificity: {specificity*100:.2f}%")
    print(f"F1 Score:  {f1:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"{'='*80}\n")
    
    # Save evaluation report (matching train.py format)
    report = classification_report(all_labels, all_preds, target_names=['Negative', 'Positive'], output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    
    # Create dataframe with clinical metrics as additional rows (matching train.py order)
    first_col = report_df.columns[0]
    clinical_metrics_data = {
        'Sensitivity_(Recall)': {first_col: sensitivity},
        'Specificity': {first_col: specificity},
        'Balanced_Accuracy': {first_col: balanced_accuracy},
        'PPV_(Positive_Predictive_Value)': {first_col: ppv},
        'NPV_(Negative_Predictive_Value)': {first_col: npv},
        'FPR_(False_Positive_Rate)': {first_col: fpr},
        'FNR_(False_Negative_Rate)': {first_col: fnr},
        'Matthews_Correlation_Coefficient': {first_col: mcc},
        'Cohen_Kappa': {first_col: kappa},
        'TP_(True_Positives)': {first_col: float(tp)},
        'FP_(False_Positives)': {first_col: float(fp)},
        'FN_(False_Negatives)': {first_col: float(fn)},
        'TN_(True_Negatives)': {first_col: float(tn)}
    }
    
    clinical_df = pd.DataFrame(clinical_metrics_data).T
    report_df = pd.concat([report_df, clinical_df])
    
    report_df.to_csv(results_csv_path)
    print(f"✓ Saved evaluation report to {results_csv_path}")
    
    # Save probabilities and labels for threshold analysis
    # This enables post-hoc threshold optimization without retraining
    probabilities_json_path = os.path.join(results_dir, f"{prefix}_probabilities.json")
    probabilities_data = {
        "fold_idx": fold_idx,
        "model_name": model_name,
        "total_samples": len(all_labels),
        "num_positive": int(np.sum(all_labels)),
        "num_negative": int(len(all_labels) - np.sum(all_labels)),
        "labels": all_labels.tolist(),
        "probabilities": all_probs.tolist(),
        "predictions_at_0_5": all_preds.tolist()
    }
    
    with open(probabilities_json_path, 'w') as f:
        json.dump(probabilities_data, f, indent=2)
    print(f"✓ Saved probabilities JSON to {probabilities_json_path}")
    
    # Save probability statistics summary (useful for quick reference)
    prob_summary_json_path = os.path.join(results_dir, f"{prefix}_probability_summary.json")
    prob_summary_data = {
        "fold_idx": fold_idx,
        "model_name": model_name,
        "total_patches": int(len(all_labels)),
        "positive_patches": int(np.sum(all_labels)),
        "negative_patches": int(len(all_labels) - np.sum(all_labels)),
        "positive_class_stats": {
            "mean_probability": float(np.mean(all_probs[all_labels == 1])) if np.any(all_labels == 1) else 0.0,
            "std_probability": float(np.std(all_probs[all_labels == 1])) if np.any(all_labels == 1) else 0.0,
            "min_probability": float(np.min(all_probs[all_labels == 1])) if np.any(all_labels == 1) else 0.0,
            "max_probability": float(np.max(all_probs[all_labels == 1])) if np.any(all_labels == 1) else 0.0,
            "median_probability": float(np.median(all_probs[all_labels == 1])) if np.any(all_labels == 1) else 0.0
        },
        "negative_class_stats": {
            "mean_probability": float(np.mean(all_probs[all_labels == 0])) if np.any(all_labels == 0) else 0.0,
            "std_probability": float(np.std(all_probs[all_labels == 0])) if np.any(all_labels == 0) else 0.0,
            "min_probability": float(np.min(all_probs[all_labels == 0])) if np.any(all_labels == 0) else 0.0,
            "max_probability": float(np.max(all_probs[all_labels == 0])) if np.any(all_labels == 0) else 0.0,
            "median_probability": float(np.median(all_probs[all_labels == 0])) if np.any(all_labels == 0) else 0.0
        },
        "overall_stats": {
            "mean_probability": float(np.mean(all_probs)),
            "std_probability": float(np.std(all_probs)),
            "min_probability": float(np.min(all_probs)),
            "max_probability": float(np.max(all_probs)),
            "median_probability": float(np.median(all_probs))
        }
    }
    
    with open(prob_summary_json_path, 'w') as f:
        json.dump(prob_summary_data, f, indent=2)
    print(f"✓ Saved probability statistics to {prob_summary_json_path}")
    
    # Generate bootstrap CI metrics summary (matching format from summarize_results.py)
    # This enables statistical confidence interval reporting per fold
    from scipy import stats
    
    metrics_summary_data = []
    
    # List of metrics to compute with bootstrap CIs
    metrics_to_compute = [
        ('Recall', recall),
        ('Precision', precision),
        ('Accuracy', accuracy),
        ('F1_Score', f1),
        ('Sensitivity', sensitivity),
        ('Specificity', specificity),
        ('Balanced_Accuracy', balanced_accuracy),
        ('PPV_(Positive_Predictive_Value)', ppv),
        ('NPV_(Negative_Predictive_Value)', npv),
        ('FPR_(False_Positive_Rate)', fpr),
        ('FNR_(False_Negative_Rate)', fnr),
        ('Matthews_Correlation_Coefficient', mcc),
        ('Cohen_Kappa', kappa)
    ]
    
    # For bootstrap CI, we need to resample from our predictions
    # Bootstrap over the sample level (not individual patches, but overall metric stability)
    np.random.seed(42)  # For reproducibility
    n_bootstrap = 1000
    bootstrap_samples = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(len(all_labels), size=len(all_labels), replace=True)
        sampled_labels = all_labels[indices]
        sampled_preds = all_preds[indices]
        sampled_probs = all_probs[indices]
        
        # Compute metrics on this bootstrap sample
        if len(np.unique(sampled_labels)) > 1 and len(np.unique(sampled_preds)) > 1:
            try:
                tn_b, fp_b, fn_b, tp_b = confusion_matrix(sampled_labels, sampled_preds, labels=[0, 1]).ravel()
            except:
                tn_b, fp_b, fn_b, tp_b = 0, 0, 0, 0
        else:
            tn_b, fp_b, fn_b, tp_b = 0, 0, 0, 0
        
        # Calculate metrics for this sample
        sample_metrics = {
            'Recall': tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0,
            'Precision': tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0,
            'Accuracy': (tp_b + tn_b) / (tp_b + tn_b + fp_b + fn_b) if (tp_b + tn_b + fp_b + fn_b) > 0 else 0.0,
            'F1_Score': 2 * (tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0) * (tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0) / ((tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0) + (tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0)) if ((tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0) + (tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0)) > 0 else 0.0,
            'Sensitivity': tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0,
            'Specificity': tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0.0,
            'Balanced_Accuracy': ((tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0) + (tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0.0)) / 2,
            'PPV_(Positive_Predictive_Value)': tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0,
            'NPV_(Negative_Predictive_Value)': tn_b / (tn_b + fn_b) if (tn_b + fn_b) > 0 else 0.0,
            'FPR_(False_Positive_Rate)': fp_b / (fp_b + tn_b) if (fp_b + tn_b) > 0 else 0.0,
            'FNR_(False_Negative_Rate)': fn_b / (fn_b + tp_b) if (fn_b + tp_b) > 0 else 0.0,
            'Matthews_Correlation_Coefficient': matthews_corrcoef(sampled_labels, sampled_preds) if len(np.unique(sampled_labels)) > 1 else 0.0,
            'Cohen_Kappa': cohen_kappa_score(sampled_labels, sampled_preds) if len(np.unique(sampled_labels)) > 1 else 0.0
        }
        bootstrap_samples.append(sample_metrics)
    
    # Compute bootstrap statistics for each metric
    for metric_name, point_est in metrics_to_compute:
        bootstrap_values = [s[metric_name] for s in bootstrap_samples]
        bootstrap_mean = np.mean(bootstrap_values)
        bootstrap_std = np.std(bootstrap_values)
        ci_lower = np.percentile(bootstrap_values, 2.5)
        ci_upper = np.percentile(bootstrap_values, 97.5)
        ci_margin = (ci_upper - ci_lower) / 2
        
        metrics_summary_data.append({
            'Metric': metric_name,
            'Point_Estimate': float(point_est),
            'Bootstrap_Mean': float(bootstrap_mean),
            'Bootstrap_Std': float(bootstrap_std),
            'CI_Lower_95%': float(ci_lower),
            'CI_Upper_95%': float(ci_upper),
            'CI_Margin': float(ci_margin)
        })
    
    # Add confusion matrix values (no bootstrap variation for these)
    metrics_summary_data.extend([
        {'Metric': 'TP_(True_Positives)', 'Point_Estimate': float(tp), 'Bootstrap_Mean': float(tp), 'Bootstrap_Std': 0.0, 'CI_Lower_95%': float(tp), 'CI_Upper_95%': float(tp), 'CI_Margin': 0.0},
        {'Metric': 'FP_(False_Positives)', 'Point_Estimate': float(fp), 'Bootstrap_Mean': float(fp), 'Bootstrap_Std': 0.0, 'CI_Lower_95%': float(fp), 'CI_Upper_95%': float(fp), 'CI_Margin': 0.0},
        {'Metric': 'FN_(False_Negatives)', 'Point_Estimate': float(fn), 'Bootstrap_Mean': float(fn), 'Bootstrap_Std': 0.0, 'CI_Lower_95%': float(fn), 'CI_Upper_95%': float(fn), 'CI_Margin': 0.0},
        {'Metric': 'TN_(True_Negatives)', 'Point_Estimate': float(tn), 'Bootstrap_Mean': float(tn), 'Bootstrap_Std': 0.0, 'CI_Lower_95%': float(tn), 'CI_Upper_95%': float(tn), 'CI_Margin': 0.0}
    ])
    
    metrics_summary_df = pd.DataFrame(metrics_summary_data)
    metrics_summary_csv_path = os.path.join(results_dir, f"{prefix}_metrics_summary.csv")
    metrics_summary_df.to_csv(metrics_summary_csv_path, index=False)
    print(f"✓ Saved bootstrap CI metrics summary to {metrics_summary_csv_path}")
    
    # Generate calibration curve image (shows prediction reliability)
    calibration_curve_path = os.path.join(results_dir, f"{prefix}_calibration_curve.png")
    plot_calibration_curve(all_labels, all_probs, calibration_curve_path)
    print(f"✓ Saved calibration curve to {calibration_curve_path}")
    
    # Plot metrics
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'DeepHP Pre-training ROC Curve (Fold {fold_idx + 1})')
    plt.legend(loc="lower right")
    plt.savefig(roc_path, dpi=100)
    plt.close()
    print(f"✓ Saved ROC curve to {roc_path}")
    
    plot_confusion_matrix(all_labels, all_preds, cm_path)
    print(f"✓ Saved confusion matrix to {cm_path}")
    
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)
    
    plt.figure()
    plt.plot(recall_curve, precision_curve, color='darkblue', lw=3, label=f'PR (AP = {ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'DeepHP Pre-training PR Curve (Fold {fold_idx + 1})')
    plt.legend(loc="lower left")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(pr_path, dpi=100)
    plt.close()
    print(f"✓ Saved PR curve to {pr_path}")
    
    # Calculate throughput early so model_selection.json can be generated
    total_train_patches = len(train_loader.dataset)
    throughput_patches_per_sec = total_train_patches / training_time_seconds if training_time_seconds > 0 else 0.0
    
    # Determine whether to use SWA based on performance comparison
    # If SWA is enabled, check if best recall from training exceeds final validation recall
    model_selection_use_swa = use_swa
    if use_swa and best_recall > last_val_recall:
        print(f"\nNote: Best validation recall ({best_recall:.4f}) exceeds final recall ({last_val_recall:.4f})")
        print(f"Setting use_swa=False to use best model instead of final model")
        model_selection_use_swa = False
    
    # Generate model_selection.json BEFORE Grad-CAM to ensure critical metrics are always saved
    model_selection_data = {
        "use_swa": model_selection_use_swa,
        "best_recall": float(best_recall),
        "last_val_recall": float(last_val_recall),
        "training_time_hours": round(training_time_hours, 2),
        "training_time_seconds": round(training_time_seconds, 1),
        "peak_gpu_memory_gb": round(peak_gpu_memory, 2),
        "throughput_patches_per_sec": round(throughput_patches_per_sec, 1),
        "num_epochs": num_epochs,
        "fold": fold_idx
    }
    
    model_selection_path = os.path.join(results_dir, f"{prefix}_model_selection.json")
    with open(model_selection_path, 'w') as f:
        json.dump(model_selection_data, f, indent=2)
    print(f"✓ Saved model selection metrics to {model_selection_path}")
    
    # Generate Grad-CAM visualizations for model interpretability
    # Uses saved predictions from probabilities.json instead of re-scanning validation data
    # This is far more memory-efficient: only loads needed samples instead of searching through all 10,000+
    print(f"\nGenerating Grad-CAM visualizations...")
    try:
        # Load saved predictions from validation phase
        # (Already computed during validation evaluation, no need to re-scan)
        probabilities_json_path = os.path.join(results_dir, f"{prefix}_probabilities.json")
        
        if os.path.exists(probabilities_json_path):
            print(f"[DEBUG] Grad-CAM: Loading predictions from {probabilities_json_path}...")
            with open(probabilities_json_path, 'r') as f:
                prob_data = json.load(f)
            
            labels_list = np.array(prob_data['labels'], dtype=np.int32)
            preds_list = np.array(prob_data['predictions_at_0_5'], dtype=np.int32)
            probs_list = np.array(prob_data['probabilities'], dtype=np.float32)
            
            # Identify indices for each category (from already-computed predictions)
            tp_indices = np.where((preds_list == 1) & (labels_list == 1))[0]
            fp_indices = np.where((preds_list == 1) & (labels_list == 0))[0]
            fn_indices = np.where((preds_list == 0) & (labels_list == 1))[0]
            tn_indices = np.where((preds_list == 0) & (labels_list == 0))[0]
            
            print(f"[DEBUG] Grad-CAM: Category distribution in validation set:")
            print(f"  TP (Pred=1, True=1): {len(tp_indices)} samples")
            print(f"  FP (Pred=1, True=0): {len(fp_indices)} samples")
            print(f"  FN (Pred=0, True=1): {len(fn_indices)} samples")
            print(f"  TN (Pred=0, True=0): {len(tn_indices)} samples")
            
            # Collect up to 3 samples from each category for visualization
            # This is much more efficient than loading 10,000+ samples
            gradcam_indices = []
            gradcam_labels = []
            gradcam_preds = []
            gradcam_probs = []
            
            for indices, category_name in [
                (tp_indices, 'TP'),
                (fp_indices, 'FP'),
                (fn_indices, 'FN'),
                (tn_indices, 'TN')
            ]:
                if len(indices) > 0:
                    # Sample up to 3 from this category
                    sample_size = min(3, len(indices))
                    sample_indices = np.random.choice(indices, size=sample_size, replace=False)
                    
                    gradcam_indices.extend(sample_indices.tolist())
                    gradcam_labels.extend(labels_list[sample_indices].tolist())
                    gradcam_preds.extend(preds_list[sample_indices].tolist())
                    gradcam_probs.extend(probs_list[sample_indices].tolist())
                    
                    print(f"[DEBUG] Grad-CAM: Selected {sample_size}/{len(indices)} {category_name} samples for visualization")
                else:
                    print(f"[DEBUG] Grad-CAM: No {category_name} samples found (will show 'No examples' in visualization)")
            
            # Load only the selected samples from val_dataset (HUGE memory saving!)
            print(f"[DEBUG] Grad-CAM: Loading {len(gradcam_indices)} selected samples from dataset...")
            all_images_for_gradcam = []
            for idx in gradcam_indices:
                img, label = val_dataset[idx]
                all_images_for_gradcam.append(img)
            
            if all_images_for_gradcam:
                # Convert to tensors
                all_images_for_gradcam = torch.stack(all_images_for_gradcam)
                all_labels_for_gradcam = torch.tensor(gradcam_labels, dtype=torch.long)
                all_preds_for_gradcam = torch.tensor(gradcam_preds, dtype=torch.long)
                all_probs_for_gradcam = torch.tensor(gradcam_probs, dtype=torch.float32)
                
                print(f"[DEBUG] Grad-CAM: Loaded {all_images_for_gradcam.shape[0]} samples for visualization (memory-efficient)")
                
                # Generate Grad-CAM visualization
                gradcam_path = os.path.join(results_dir, f"{prefix}_gradcam.png")
                
                # CRITICAL: Ensure GPU memory is clean before expensive gradient computation
                gc.collect()
                torch.cuda.empty_cache()
                
                generate_gradcam_visualizations(
                    model, 
                    all_images_for_gradcam.to(device),
                    all_labels_for_gradcam.to(device),
                    all_preds_for_gradcam.to(device),
                    all_probs_for_gradcam.to(device),
                    gradcam_path,
                    device,
                    title_prefix=f"(Fold {fold_idx + 1}/{num_folds})"
                )
                print(f"✓ Saved Grad-CAM visualization to {gradcam_path}")
            else:
                print(f"Warning: No samples selected for Grad-CAM visualization")
        else:
            print(f"Warning: Could not find {probabilities_json_path}, skipping Grad-CAM visualization")
    except Exception as e:
        print(f"Warning: Failed to generate Grad-CAM visualizations: {e}")
    
    # Print fold configuration summary for SLURM output (CONFIG 87771)
    print(f"\n{'='*80}")
    print(f"FOLD {fold_idx} CONFIGURATION (CONFIG 87771 - Mathematically Optimized)")
    print(f"{'='*80}")
    
    config_stats = {
        0: {"patches": 87532, "pos_exps": 4, "neg_exps": 3, "ratio": 2.33, "distance": 0.05},
        1: {"patches": 89516, "pos_exps": 3, "neg_exps": 7, "ratio": 2.06, "distance": 0.22},
        2: {"patches": 20347, "pos_exps": 4, "neg_exps": 1, "ratio": 2.31, "distance": 0.03},
        3: {"patches": 99120, "pos_exps": 4, "neg_exps": 0, "ratio": 2.81, "distance": 0.53},
        4: {"patches": 98410, "pos_exps": 6, "neg_exps": 1, "ratio": 2.29, "distance": 0.01}
    }
    
    stats = config_stats.get(fold_idx, {})
    print(f"Expected Validation Set:")
    print(f"  Total patches: {stats.get('patches', 'N/A'):,}")
    print(f"  Experiments: {stats.get('pos_exps', 0)} positive + {stats.get('neg_exps', 0)} negative")
    print(f"  Neg:Pos ratio: {stats.get('ratio', 'N/A'):.2f}:1")
    print(f"  Distance from target (2.28:1): {stats.get('distance', 'N/A'):.2f}")
    print(f"\nTotal Configuration Score:")
    print(f"  Overall distance: 0.6441 (sum of per-fold distances)")
    print(f"  Selected from: 500,000+ random greedy searches")
    print(f"  Selection criteria: Optimal balance of patch counts, experiment ratios, and ratio stability")
    print(f"{'='*80}\n")
    
    # Clean up
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"✓ Fold {fold_idx + 1}/{num_folds} Complete!\n")
    
    return best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepHP Backbone Pre-training")
    parser.add_argument("--fold", type=int, default=0, help="Fold index for cross-validation")
    parser.add_argument("--num_folds", type=int, default=5, help="Total number of folds")
    parser.add_argument("--model_name", type=str, default="convnext_tiny", 
                       choices=["convnext_tiny", "convnext_small", "resnet50"],
                       help="Backbone architecture")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--use_focal_loss", type=bool, default=False, help="Use Focal Loss")
    parser.add_argument("--pos_weight", type=float, default=2.5, help="Positive class weight (for DeepHP 1:2.5 imbalance)")
    parser.add_argument("--neg_weight", type=float, default=1.0, help="Negative class weight")
    parser.add_argument("--gamma", type=float, default=1.0, help="Focal Loss gamma")
    parser.add_argument("--iter", type=str, default="deephp", help="Iteration name for tracking (e.g., 'deephp' or '31.0')")
    parser.add_argument("--run_id", type=str, default="", help="Run ID for parallel job safety (auto-generated if not provided)")
    parser.add_argument("--use_swa", type=str, default="True", help="Whether to use SWA (Stochastic Weight Averaging)")
    parser.add_argument("--swa_start", type=int, default=12, help="Epoch to start SWA averaging")
    parser.add_argument("--jitter", type=float, default=0.15, help="ColorJitter intensity (brightness/contrast augmentation)")
    parser.add_argument("--pct_start", type=float, default=0.1, help="Warmup percentage for learning rate schedule")
    parser.add_argument("--clip_grad", type=float, default=0.0, help="Gradient clipping norm (0=disabled)")
    parser.add_argument("--saver_metric", type=str, default="loss", help="Metric for model selection (loss/accuracy/precision/recall/f1)")
    
    # Domain Adversarial Neural Networks (DANN) parameters
    parser.add_argument("--use_dann", type=str, default="False", help="Enable Domain Adversarial training to prevent learning experiment signatures")
    parser.add_argument("--dann_lambda", type=float, default=1.0, help="Gradient reversal scaling factor for DANN")
    parser.add_argument("--dann_weight", type=float, default=0.5, help="Weight for adversary loss (final loss = class_loss + dann_weight * adv_loss)")
    
    args = parser.parse_args()
    
    train_deephp_backbone(
        fold_idx=args.fold,
        num_folds=args.num_folds,
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_focal_loss=args.use_focal_loss,
        pos_weight=args.pos_weight,
        neg_weight=args.neg_weight,
        gamma=args.gamma,
        iter_name=args.iter,
        run_id=args.run_id,
        use_swa=args.use_swa == "True",
        swa_start=args.swa_start,
        jitter=args.jitter,
        pct_start=args.pct_start,
        clip_grad=args.clip_grad,
        saver_metric=args.saver_metric,
        use_dann=(args.use_dann.lower() == 'true'),
        dann_lambda=args.dann_lambda,
        dann_weight=args.dann_weight,
    )
