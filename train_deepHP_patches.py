"""
train_deepHP_patches.py - Backbone Pre-training on DeepHP H&E Patches

Purpose:
--------
Pre-train a ConvNeXt-Tiny backbone on 394,926 H&E-stained histology patches
from the DeepHP database. This creates a general-purpose feature extractor
that understands H. pylori morphology before fine-tuning on patient-level
IHC data (HelicoDataSet).

Differences from train.py (patient-level MIL training):
1. Patch-level classification (no MIL aggregation)
2. Standard cross-entropy loss (no Focal Loss weighting initially)
3. 5-fold stratified CV on patches (not patients)
4. Output: Pre-trained backbone weights only
5. H&E-specific normalization (Macenko or ImageNet)

Configuration:
    DeepHP dataset path is set via config.py (DEEPHP_DATASET_ROOT).
    Default: /export/hhome/ricse03/8117177
    Override: export DEEPHP_DATASET_ROOT=/path/to/deephp

Usage:
    python train_deepHP_patches.py --fold 0 [--num_folds 5] [--model_name convnext_tiny]
    
    # Run all folds in parallel (recommended for speed)
    for i in {0..4}; do
        sbatch -J deephp_f$i train_deepHP.sh $i &
    done

Output:
    results/deephp_backbone_pretrained_convnext_tiny_f0.pth
    results/deephp_backbone_pretrained_convnext_tiny_f1.pth
    ...
    results/deephp_backbone_final_convnext_tiny.pth (averaged across folds)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
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
from visualization_utils import plot_learning_curves, plot_confusion_matrix, plot_roc_curve, plot_pr_curve


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


def train_deephp_backbone(fold_idx=0, num_folds=5, model_name="convnext_tiny", num_epochs=20, 
                          batch_size=128, learning_rate=2e-5, weight_decay=0.01, 
                          use_focal_loss=False, pos_weight=2.5, gamma=1.0):
    """
    Train a CNN backbone on DeepHP H&E patches for pre-training.
    
    Args:
        fold_idx (int): Fold index for k-fold CV (0 to num_folds-1)
        num_folds (int): Total number of folds
        model_name (str): Backbone architecture ('convnext_tiny', 'resnet50', etc.)
        num_epochs (int): Training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Initial learning rate
        weight_decay (float): L2 regularization
        use_focal_loss (bool): Use Focal Loss (True) or Cross-Entropy (False)
        pos_weight (float): Weight for positive class (if use_focal_loss=False)
        gamma (float): Focal Loss gamma parameter
    """
    
    # Setup output directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.cuda.set_per_process_memory_fraction(0.70, 0)
    
    # Setup output paths
    prefix = f"deephp_backbone_pretrained_{model_name}_f{fold_idx}"
    best_model_path = os.path.join(results_dir, f"{prefix}.pth")
    results_csv_path = os.path.join(results_dir, f"{prefix}_evaluation.csv")
    cm_path = os.path.join(results_dir, f"{prefix}_confusion_matrix.png")
    roc_path = os.path.join(results_dir, f"{prefix}_roc_curve.png")
    pr_path = os.path.join(results_dir, f"{prefix}_pr_curve.png")
    history_path = os.path.join(results_dir, f"{prefix}_learning_curves.png")
    
    print(f"\n{'='*80}")
    print(f"DeepHP Backbone Pre-training: Fold {fold_idx + 1}/{num_folds}")
    print(f"{'='*80}")
    print(f"Model: {model_name} | Epochs: {num_epochs} | Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate} | Weight Decay: {weight_decay}")
    print(f"Focal Loss: {use_focal_loss} | Pos Weight: {pos_weight}")
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    # Build model (backbone only, no MIL head)
    print(f"Loading {model_name} backbone...")
    model = get_model(model_name=model_name, num_classes=2, pretrained=True, pool_type="attention").to(device)
    
    # For DeepHP pre-training, we only care about the backbone classification head
    # The MIL components will be re-initialized when transferred to HelicoDataSet
    
    # Loss function
    if use_focal_loss:
        loss_weights = torch.FloatTensor([1.0, pos_weight]).to(device)
        criterion = FocalLoss(gamma=gamma, weight=loss_weights, smoothing=0.0)
        print(f"Using Focal Loss (gamma={gamma}, pos_weight={pos_weight})")
    else:
        loss_weights = torch.FloatTensor([1.0, pos_weight]).to(device)
        criterion = nn.CrossEntropyLoss(weight=loss_weights)
        print(f"Using Cross-Entropy Loss (pos_weight={pos_weight})")
    
    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_loss = float('inf')
    
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                # Forward pass for patch-level classification (not MIL)
                # Use regular forward() which expects [batch, C, H, W] and returns [batch, num_classes]
                logits = model(images)  # [batch_size, num_classes]
                loss = criterion(logits, labels)
            
            train_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        
        train_loss /= len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
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
        from sklearn.metrics import f1_score
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | F1: {val_f1:.4f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ Saved best model to {best_model_path}")
        
        scheduler.step()
    
    # Save learning curves
    plot_learning_curves(history, history_path)
    print(f"\n✓ Saved learning curves to {history_path}")
    
    # Final evaluation on validation set
    print(f"\n{'='*80}")
    print(f"Final Evaluation (Fold {fold_idx + 1}/{num_folds})")
    print(f"{'='*80}")
    
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Final Evaluation"):
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
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"Specificity: {specificity*100:.2f}%")
    print(f"F1 Score:  {f1:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"{'='*80}\n")
    
    # Save evaluation report
    report = classification_report(all_labels, all_preds, target_names=['Negative', 'Positive'], output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    
    clinical_metrics = {
        'Sensitivity_(Recall)': recall,
        'Specificity': specificity,
        'Precision': precision,
        'F1_Score': f1,
        'Accuracy': accuracy,
        'Matthews_Correlation_Coefficient': mcc,
        'TP': float(tp),
        'FP': float(fp),
        'FN': float(fn),
        'TN': float(tn)
    }
    
    for metric_name, metric_value in clinical_metrics.items():
        report_df.loc[metric_name] = metric_value
    
    report_df.to_csv(results_csv_path)
    print(f"✓ Saved evaluation report to {results_csv_path}")
    
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
    parser.add_argument("--gamma", type=float, default=1.0, help="Focal Loss gamma")
    
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
        gamma=args.gamma
    )
