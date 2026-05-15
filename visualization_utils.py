"""
H. Pylori Contamination Detection - Visualization Utilities Module
==================================================================

OVERVIEW
--------
This module consolidates all visualization and interpretability functions for the
H. Pylori contamination detection system. It provides a unified API for:
  - Gradient-based attribution (Grad-CAM) for model interpretability
  - Training metrics visualization (learning curves, confusion matrices)
  - Diagnostic metrics (ROC curves, Precision-Recall curves, probability distributions)
  - Interpretable patch-level heatmap generation with attention visualization

PURPOSE
-------
Eliminates code duplication between train.py and generate_visuals.py by providing
a single source of truth for all PNG image generation. Can be used:
  - During training to track convergence and model behavior
  - Post-training for comprehensive evaluation reporting
  - For clinical interpretation and model debugging
  - In custom analysis pipelines requiring specific visualizations

ARCHITECTURE
------------
This is a STATELESS UTILITY MODULE: All functions are independent, take explicit
inputs, and produce explicit outputs. No class instantiation or module state required.

Functions are organized into three categories:

1. CORE GRADIENT ATTRIBUTION
   - generate_gradcam(): Input-level gradient saliency (model-agnostic)
   - Works with any PyTorch backbone architecture

2. METRIC VISUALIZATIONS (Training pipeline metrics)
   - plot_learning_curves(): Training/validation loss and accuracy over epochs
   - plot_confusion_matrix(): Patient-level 2x2 confusion matrix
   - plot_probability_histogram(): Distribution of predicted probabilities
   - plot_roc_curve(): Receiver Operating Characteristic with AUC
   - plot_pr_curve(): Precision-Recall curve with Average Precision

3. INTERPRETABILITY VISUALIZATION
   - plot_gradcam_pair(): Side-by-side original patch + heatmap overlay for
     top-ranked predictions and false negatives

HOW IT WORKS
------------

GRADIENT-BASED ATTRIBUTION (generate_gradcam):
  1. Forward pass: Input batch → backbone → logits
  2. Loss computation: Sum logits (proxy for feature signal magnitude)
  3. Backward pass: Compute ∇(loss)/∇(input)
  4. Attribution: Absolute gradients summed across channels
  5. Smoothing: Apply Gaussian blur (σ=1.5) to reduce noise
  6. Normalization: Scale to [0, 1] per-sample range

METRIC VISUALIZATIONS:
  - All functions use matplotlib for consistent styling
  - Patient-level aggregation (not patch-level)
  - Auto-creates output directory if needed
  - Closes figures after saving (prevents memory accumulation)

INTERPRETABILITY:
  - Combines original patch image with jet-colormap heatmap overlay
  - Includes attention score and predicted probability in titles
  - Different naming convention for false negatives (FN_ prefix)
  - Denormalizes to ImageNet statistics for visual inspection

USAGE
-----

IMPORT STATEMENT:
  from visualization_utils import (
      generate_gradcam, plot_learning_curves, plot_confusion_matrix,
      plot_probability_histogram, plot_roc_curve, plot_pr_curve, plot_gradcam_pair
  )

BASIC USAGE EXAMPLES:

  # Compute Grad-CAM for a batch of images
  heatmaps, probs = generate_gradcam(model.backbone, img_batch)

  # Plot training curves
  history = {'train_loss': [...], 'val_loss': [...], 
             'train_acc': [...], 'val_acc': [...]}
  plot_learning_curves(history, 'results/learning_curves.png')

  # Plot confusion matrix
  plot_confusion_matrix(all_labels, all_preds, 'results/confusion_matrix.png')

  # Plot probability distribution
  plot_probability_histogram(all_probs, all_labels, 'results/histogram.png')

  # Plot ROC and PR curves
  plot_roc_curve(all_labels, all_probs, 'results/roc.png')
  plot_pr_curve(all_labels, all_probs, 'results/pr.png')

  # Visualize top prediction with Grad-CAM
  plot_gradcam_pair(
      patch_img=patch_tensor,           # (1, C, H, W) or (C, H, W)
      heatmap=heatmap_array,             # (H, W) normalized to [0,1]
      patient_id='patient_123',
      rank=0,                            # Rank among top patches
      patch_idx=42,
      attention_score=0.8234,
      prob=0.95,
      is_false_negative=False,
      output_dir='results/gradcam_samples'
  )

FUNCTION REFERENCE
------------------

generate_gradcam(backbone, input_batch, target_layer=None)
  Generates interpretable saliency heatmap using input-level gradients.
  Args:
    backbone: Neural network (ConvNeXt-Tiny or ResNet50)
    input_batch: (B, C, H, W) tensor on GPU/CPU
    target_layer: Deprecated (kept for backwards compatibility)
  Returns:
    heatmap_np: (B, 1, H, W) normalized to [0, 1]
    probs: (B, 2) softmax probabilities for [negative, positive]

plot_learning_curves(history, output_path, figsize=(12, 5))
  Args:
    history: Dict with keys ['train_loss', 'val_loss', 'train_acc', 'val_acc']
    output_path: PNG save location
    figsize: Matplotlib figure size
  Output: 2-panel plot (loss on left, accuracy on right)

plot_confusion_matrix(all_labels, all_preds, output_path, figsize=(8, 6))
  Args:
    all_labels: (N,) binary true labels
    all_preds: (N,) binary predicted labels
    output_path: PNG save location
    figsize: Matplotlib figure size
  Output: 2x2 confusion matrix heatmap

plot_probability_histogram(all_probs, all_labels, output_path, figsize=(8, 6))
  Args:
    all_probs: (N,) predicted probabilities [0, 1]
    all_labels: (N,) binary true labels
    output_path: PNG save location
    figsize: Matplotlib figure size
  Output: Histogram with negative/positive overlay + 0.5 threshold line

plot_roc_curve(all_labels, all_probs, output_path)
  Args:
    all_labels: (N,) binary true labels
    all_probs: (N,) predicted probabilities [0, 1]
    output_path: PNG save location
  Output: ROC curve with AUC score in legend

plot_pr_curve(all_labels, all_probs, output_path)
  Args:
    all_labels: (N,) binary true labels
    all_probs: (N,) predicted probabilities [0, 1]
    output_path: PNG save location
  Output: Precision-Recall curve with Average Precision in legend

plot_gradcam_pair(patch_img, heatmap, patient_id, rank, patch_idx,
                   attention_score, prob, is_false_negative, output_dir)
  Args:
    patch_img: (1, C, H, W) or (C, H, W) tensor
    heatmap: (H, W) saliency array in [0, 1]
    patient_id: String identifier
    rank: Integer rank (0 for top positive)
    patch_idx: Patch index within patient
    attention_score: MIL attention weight [0, 1]
    prob: Positive class probability [0, 1]
    is_false_negative: Boolean (affects filename/title)
    output_dir: Directory to save PNG
  Returns: Path to saved PNG file
  Output: Side-by-side (original | heatmap overlay)

INTEGRATION POINTS
------------------
Called from:
  - train.py: During model evaluation after each epoch
  - generate_visuals.py: After loading trained checkpoint for reporting
  - Custom analysis scripts: For post-hoc model interpretation

DEPENDENCIES
------------
  - PyTorch: tensor operations, gradient computation
  - NumPy: array manipulation, statistics
  - Matplotlib: figure generation and styling
  - Scipy: gaussian_filter for smoothing
  - Scikit-learn: metrics (confusion matrix, ROC/PR curves)

NOTES
-----
  - All functions assume inputs are properly preprocessed (tensors on correct device)
  - Grad-CAM uses model.eval() internally to disable stochastic components
  - Probability histograms use patient-level aggregations (not per-patch)
  - Heatmap normalization uses per-sample min-max (not global statistics)
  - Gaussian smoothing (σ=1.5) removes interpolation artifacts while preserving edges
  - Output figures automatically close after saving (prevents matplotlib state accumulation)
  - All visualizations use patient-level metrics, not patch-level aggregations
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

# ============================================================================
# CORE GRADIENT ATTRIBUTION (used by both train.py and generate_visuals.py)
# ============================================================================

def generate_gradcam(backbone, input_batch, target_layer=None):
    """
    Generates interpretable heatmap using input-level gradient saliency.
    This is more robust than layer-specific Grad-CAM and works for any architecture.
    
    Approach: Compute gradient of output with respect to input.
    Shows which pixels matter most for the backbone's feature extraction.
    
    Args:
        backbone: Neural network backbone (ConvNeXt or ResNet)
        input_batch: Image tensor (B, C, H, W) on DEVICE
        target_layer: Deprecated parameter (kept for backwards compatibility)
    
    Returns:
        heatmap_np: Normalized saliency heatmap (B, 1, H, W) in [0, 1]
        probs: Softmax probabilities (B, num_classes)
    """
    backbone.eval()
    
    # Create input with requires_grad to compute gradients
    input_batch.requires_grad_(True)
    
    # Forward pass
    with torch.enable_grad():
        logits = backbone(input_batch)
        
        # Flatten if needed
        if len(logits.shape) > 2:
            logits = torch.flatten(logits, 1)
        
        # Create a scalar loss: sum of features (positive class signal)
        # For clinical safety: higher feature magnitude = more signal
        loss = logits.sum()
    
    # Backward to compute gradients at input
    backbone.zero_grad()
    loss.backward()
    
    # Get gradients
    gradients = input_batch.grad
    if gradients is None:
        batch_size = input_batch.shape[0]
        return np.zeros((batch_size, 1, input_batch.shape[2], input_batch.shape[3])), np.zeros((batch_size, 2))
    
    # Compute absolute gradients, average across channels
    abs_grads = torch.abs(gradients)  # (B, C, H, W)
    saliency = torch.sum(abs_grads, dim=1, keepdim=True)  # (B, 1, H, W)
    
    # Convert to numpy
    heatmap_np = saliency.detach().cpu().numpy()
    
    # Process each sample in batch
    for b in range(heatmap_np.shape[0]):
        hmap = heatmap_np[b, 0]  # (H, W)
        
        # Normalize [0, 1]
        hmap_min = hmap.min()
        hmap = hmap - hmap_min
        hmap_max = hmap.max()
        if hmap_max > 0:
            hmap = hmap / hmap_max
        
        # Apply Gaussian smoothing to reduce noise while preserving structure
        hmap = gaussian_filter(hmap, sigma=1.5)
        
        # Final normalization after smoothing
        hmap = np.clip(hmap, 0, 1)
        hmap_min = hmap.min()
        hmap = hmap - hmap_min
        hmap_max = hmap.max()
        if hmap_max > 0:
            hmap = hmap / hmap_max
        
        heatmap_np[b, 0] = hmap
    
    # Get probabilities
    with torch.no_grad():
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    
    # Detach input
    input_batch.requires_grad_(False)
    
    return heatmap_np, probs

# ============================================================================
# METRIC VISUALIZATIONS
# ============================================================================

def plot_learning_curves(history, output_path, figsize=(12, 5)):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dict with keys ['train_loss', 'val_loss', 'train_acc', 'val_acc']
        output_path: Path to save PNG file
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='tab:blue', linestyle='--')
    plt.plot(history['val_loss'], label='Val Loss', color='tab:blue')
    plt.title('Patient-Level Loss Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('Focal Loss')
    plt.legend()
    
    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc', color='tab:orange', linestyle='--')
    plt.plot(history['val_acc'], label='Val Acc', color='tab:orange')
    plt.title('Patient-Level Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(all_labels, all_preds, output_path, figsize=(8, 6)):
    """
    Plot patient-level confusion matrix.
    
    Args:
        all_labels: True labels (binary)
        all_preds: Predicted labels (binary)
        output_path: Path to save PNG file
        figsize: Figure size tuple
    """
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=figsize)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap='Blues')
    plt.title('Patient-Level Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_probability_histogram(all_probs, all_labels, output_path, figsize=(8, 6)):
    """
    Plot predicted probability distribution.
    
    Args:
        all_probs: Predicted probabilities (patient-level)
        all_labels: True labels (binary)
        output_path: Path to save PNG file
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    plt.hist(all_probs[all_labels == 0], bins=20, alpha=0.5, label='Actual Negative', color='blue')
    plt.hist(all_probs[all_labels == 1], bins=20, alpha=0.5, label='Actual Positive', color='red')
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    plt.xlabel('Predicted Probability (Positive Class)')
    plt.ylabel('Patient Count')
    plt.title('Patient-Level Probability Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_roc_curve(all_labels, all_probs, output_path):
    """
    Plot ROC curve with AUC score.
    
    Args:
        all_labels: True labels (binary)
        all_probs: Predicted probabilities
        output_path: Path to save PNG file
    """
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Patient-Level ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_pr_curve(all_labels, all_probs, output_path):
    """
    Plot Precision-Recall curve with Average Precision.
    
    Args:
        all_labels: True labels (binary)
        all_probs: Predicted probabilities
        output_path: Path to save PNG file
    """
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    avg_prec = average_precision_score(all_labels, all_probs)
    
    plt.figure()
    plt.plot(recall, precision, color='green', lw=2, label=f'PR (AP = {avg_prec:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Patient-Level Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_threshold_analysis(all_labels, all_probs, output_path, figsize=(14, 8)):
    """
    Plot performance metrics across different decision thresholds.
    
    Shows how Sensitivity, Specificity, Precision, Recall, Accuracy, and F1
    vary as the decision threshold changes from 0 to 1. Helps identify optimal
    thresholds for different clinical criteria (high sensitivity vs specificity).
    
    Args:
        all_labels: True labels (binary)
        all_probs: Predicted probabilities
        output_path: Path to save PNG file
        figsize: Figure size tuple
    """
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, accuracy_score,
        confusion_matrix
    )
    
    # Generate thresholds from 0 to 1
    thresholds = np.linspace(0, 1, 101)
    metrics_by_threshold = {
        'Sensitivity': [],
        'Specificity': [],
        'Precision': [],
        'Recall': [],
        'Accuracy': [],
        'F1_Score': []
    }
    
    for threshold in thresholds:
        # Convert probabilities to binary predictions using this threshold
        preds = (np.array(all_probs) >= threshold).astype(int)
        
        # Handle edge cases (all predictions same class)
        if len(np.unique(preds)) == 1:
            # If all predictions are same, metrics become undefined
            tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel() if len(np.unique(all_labels)) > 1 else (0, 0, 0, 0)
        else:
            tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
        
        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = precision_score(all_labels, preds, zero_division=0)
        recall = recall_score(all_labels, preds, zero_division=0)
        accuracy = accuracy_score(all_labels, preds)
        f1 = f1_score(all_labels, preds, zero_division=0)
        
        metrics_by_threshold['Sensitivity'].append(sensitivity)
        metrics_by_threshold['Specificity'].append(specificity)
        metrics_by_threshold['Precision'].append(precision)
        metrics_by_threshold['Recall'].append(recall)
        metrics_by_threshold['Accuracy'].append(accuracy)
        metrics_by_threshold['F1_Score'].append(f1)
    
    # Find optimal thresholds for different objectives
    optimal_f1_idx = np.argmax(metrics_by_threshold['F1_Score'])
    optimal_f1_threshold = thresholds[optimal_f1_idx]
    optimal_f1_value = metrics_by_threshold['F1_Score'][optimal_f1_idx]
    
    # Youden's J statistic (maximizes Sensitivity + Specificity - 1)
    youden_j = [s + sp - 1 for s, sp in zip(metrics_by_threshold['Sensitivity'], metrics_by_threshold['Specificity'])]
    optimal_j_idx = np.argmax(youden_j)
    optimal_j_threshold = thresholds[optimal_j_idx]
    optimal_j_value = youden_j[optimal_j_idx]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Panel 1: Core metrics
    ax1.plot(thresholds, metrics_by_threshold['Sensitivity'], label='Sensitivity (Recall)', linewidth=2.5, color='#2E86AB')
    ax1.plot(thresholds, metrics_by_threshold['Specificity'], label='Specificity', linewidth=2.5, color='#A23B72')
    ax1.plot(thresholds, metrics_by_threshold['Precision'], label='Precision', linewidth=2.5, color='#F18F01')
    ax1.plot(thresholds, metrics_by_threshold['Accuracy'], label='Accuracy', linewidth=2.5, color='#C73E1D')
    
    # Mark optimal F1 threshold
    ax1.axvline(optimal_f1_threshold, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal F1 (threshold={optimal_f1_threshold:.2f})')
    
    ax1.set_xlabel('Decision Threshold', fontsize=11)
    ax1.set_ylabel('Metric Value', fontsize=11)
    ax1.set_title('Performance Metrics Across Decision Thresholds', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.05])
    
    # Panel 2: F1 Score and Youden's J
    ax2.plot(thresholds, metrics_by_threshold['F1_Score'], label='F1 Score', linewidth=2.5, color='#06A77D')
    ax2.plot(thresholds, youden_j, label="Youden's J (Sensitivity + Specificity - 1)", linewidth=2.5, color='#D62828')
    
    # Mark optimal points
    ax2.scatter([optimal_f1_threshold], [optimal_f1_value], color='green', s=100, zorder=5, edgecolors='darkgreen', linewidth=2, label=f'Max F1: {optimal_f1_value:.4f}')
    ax2.scatter([optimal_j_threshold], [optimal_j_value], color='red', s=100, zorder=5, edgecolors='darkred', linewidth=2, label=f"Max J: {optimal_j_value:.4f}")
    
    # Mark default 0.5 threshold
    ax2.axvline(0.5, color='gray', linestyle=':', linewidth=2, alpha=0.6, label='Default threshold (0.5)')
    
    ax2.set_xlabel('Decision Threshold', fontsize=11)
    ax2.set_ylabel('Metric Value', fontsize=11)
    ax2.set_title('Optimization Metrics: F1 Score and Youden\'s J Statistic', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([-0.1, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Threshold analysis saved: {output_path}")
    print(f"    - Optimal F1 threshold: {optimal_f1_threshold:.3f} (F1={optimal_f1_value:.4f})")
    print(f"    - Optimal Youden threshold: {optimal_j_threshold:.3f} (J={optimal_j_value:.4f})")


def plot_ensemble_roc_pr_curves(all_labels, ensemble_mean_prob, ensemble_max_prob, output_path, figsize=(16, 6)):
    """
    Plot ensemble voting ROC and PR curves with multiple probability aggregation methods.
    
    Shows model performance across all decision thresholds using:
    - ROC Curve: Plots TPR vs FPR (sensitivity vs false positive rate)
    - PR Curve: Plots Precision vs Recall (positive predictive value vs sensitivity)
    
    Compares two ensemble probability aggregation methods:
    - Mean Ensemble Probability: Average prediction confidence across 5 folds
    - Max Ensemble Probability: Maximum prediction confidence across 5 folds
    
    Args:
        all_labels: True labels (binary, 0/1)
        ensemble_mean_prob: Mean probability from ensemble (for each patient)
        ensemble_max_prob: Max probability from ensemble (for each patient)
        output_path: Path to save PNG file
        figsize: Figure size tuple (width, height)
    """
    # Calculate metrics for both probability aggregation methods
    fpr_mean, tpr_mean, _ = roc_curve(all_labels, ensemble_mean_prob)
    roc_auc_mean = auc(fpr_mean, tpr_mean)
    
    fpr_max, tpr_max, _ = roc_curve(all_labels, ensemble_max_prob)
    roc_auc_max = auc(fpr_max, tpr_max)
    
    precision_mean, recall_mean, _ = precision_recall_curve(all_labels, ensemble_mean_prob)
    pr_auc_mean = average_precision_score(all_labels, ensemble_mean_prob)
    
    precision_max, recall_max, _ = precision_recall_curve(all_labels, ensemble_max_prob)
    pr_auc_max = average_precision_score(all_labels, ensemble_max_prob)
    
    # Create side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # ========== ROC CURVE PANEL ==========
    # ROC: TPR (sensitivity) vs FPR (1-specificity)
    ax1.plot(fpr_mean, tpr_mean, color='#2E86AB', lw=3, label=f'Mean Prob (AUC = {roc_auc_mean:.4f})')
    ax1.plot(fpr_max, tpr_max, color='#A23B72', lw=3, linestyle='--', label=f'Max Prob (AUC = {roc_auc_max:.4f})')
    ax1.plot([0, 1], [0, 1], color='red', lw=2, linestyle=':', alpha=0.6, label='Random Classifier')
    
    ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
    ax1.set_title('Ensemble ROC Curves\n(Probability aggregation comparison)', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    
    # Add diagonal reference
    ax1.fill_between([0, 1], 0, 1, alpha=0.1, color='gray')
    
    # ========== PRECISION-RECALL CURVE PANEL ==========
    # PR: Precision (PPV) vs Recall (Sensitivity)
    ax2.plot(recall_mean, precision_mean, color='#06A77D', lw=3, label=f'Mean Prob (AP = {pr_auc_mean:.4f})')
    ax2.plot(recall_max, precision_max, color='#F18F01', lw=3, linestyle='--', label=f'Max Prob (AP = {pr_auc_max:.4f})')
    
    # Add reference: no-skill classifier (proportion of positives)
    baseline = np.sum(all_labels == 1) / len(all_labels)
    ax2.axhline(y=baseline, color='red', lw=2, linestyle=':', alpha=0.6, label=f'Random Classifier (P={baseline:.3f})')
    
    ax2.set_xlabel('Recall (Sensitivity = TP/(TP+FN))', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Precision (PPV = TP/(TP+FP))', fontsize=12, fontweight='bold')
    ax2.set_title('Ensemble Precision-Recall Curves\n(Probability aggregation comparison)', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    
    # Add shaded region for ideal performance
    ax2.fill_between([0, 1], 1, 0, alpha=0.05, color='green')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Ensemble ROC/PR curves saved: {output_path}")
    print(f"    - ROC-AUC (Mean Prob):  {roc_auc_mean:.4f}")
    print(f"    - ROC-AUC (Max Prob):   {roc_auc_max:.4f}")
    print(f"    - PR-AUC (Mean Prob):   {pr_auc_mean:.4f}")
    print(f"    - PR-AUC (Max Prob):    {pr_auc_max:.4f}")


# ============================================================================
# GRAD-CAM VISUALIZATION
# ============================================================================

def plot_gradcam_pair(patch_img, heatmap, patient_id, rank, patch_idx, 
                      attention_score, prob, is_false_negative=False, 
                      output_dir=None):
    """
    Create side-by-side visualization of patch and Grad-CAM heatmap.
    
    Args:
        patch_img: Original patch tensor (1, C, H, W) or (C, H, W)
        heatmap: Normalized saliency heatmap (H, W) in [0, 1]
        patient_id: Patient identifier string
        rank: Rank among top patches (0, 1, 2, ...)
        patch_idx: Patch index within patient bag
        attention_score: Attention weight for this patch
        prob: Model's positive class probability
        is_false_negative: Whether this is a false negative (ghost patient)
        output_dir: Directory to save PNG file
    
    Returns:
        output_path: Path to saved PNG file
    """
    # Handle tensor reshaping
    if len(patch_img.shape) == 4:
        patch_img = patch_img[0]  # (C, H, W)
    
    # Convert to numpy and denormalize (ImageNet stats)
    orig_img = patch_img.cpu().permute(1, 2, 0).numpy()
    orig_img = orig_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    orig_img = np.clip(orig_img, 0, 1)
    
    # Create side-by-side figure
    plt.figure(figsize=(10, 5))
    
    # Left: Original image
    plt.subplot(1, 2, 1)
    plt.imshow(orig_img)
    plt.title(f"Patch {patch_idx} (Attn: {attention_score:.4f})")
    plt.axis('off')
    
    # Right: Heatmap overlay
    plt.subplot(1, 2, 2)
    plt.imshow(orig_img)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    prefix = "FN_" if is_false_negative else ""
    plt.title(f"{prefix}Grad-CAM (Prob: {prob:.4f})")
    plt.axis('off')
    
    # Save
    if output_dir is None:
        output_dir = "results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    if is_false_negative:
        out_path = os.path.join(output_dir, f"FN_{patient_id}_rank{rank}_patch{patch_idx}.png")
    else:
        out_path = os.path.join(output_dir, f"{patient_id}_rank{rank}_patch{patch_idx}.png")
    
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    
    return out_path
