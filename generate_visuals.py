
"""
H. Pylori Contamination Detection - Visual Report Generation
============================================================

OVERVIEW
--------
This script generates comprehensive visual reports for trained H. Pylori detection models.
It loads a trained model checkpoint, runs inference on a validation fold, and produces:
  - Patient-level performance metrics (confusion matrix, ROC curve, PR curve)
  - Grad-CAM visualizations showing which image patches triggered positive predictions
  - Ranking of top-positive and false-negative predictions for interpretability

PURPOSE
-------
After model training (train.py), this standalone utility enables post-hoc analysis of model
predictions without re-running the full training pipeline. It's useful for:
  - Visualizing which regions the model learned to detect H. Pylori
  - Debugging misclassifications (false positives/negatives)
  - Generating clean visualizations for reports and presentations
  - Validating model behavior on specific dataset folds

HOW IT WORKS
------------
1. Loads Dataset: Fetches validation patches and patient metadata from the H. Pylori dataset
2. Loads Model: Retrieves trained checkpoint (attention-based MIL architecture)
3. Inference: Runs forward pass on validation fold, aggregating patch-level predictions to patient level
4. Grad-CAM: Computes input-level gradient saliency maps showing attention regions
5. Metrics: Generates confusion matrix, ROC curve, precision-recall curve at patient level
6. Visualization: Outputs side-by-side (image, heatmap) pairs for top predictions and errors
7. Output: Saves all visualizations as PNG images in results/{RUN_ID}_gradcam_samples/

USAGE
-----
Run from command line with required arguments:

  python generate_visuals.py --run_id <RUN_ID> [--fold <FOLD>] [--num_folds <NUM_FOLDS>] [--model_name <MODEL>]

ARGUMENTS
---------
  --run_id (required)
    The experiment run identifier (e.g., "62_102498")
    Used to locate model checkpoint in results/ directory
    
  --fold (optional, default: 0)
    Which cross-validation fold to visualize (0 to num_folds-1)
    Must match the fold used during training
    
  --num_folds (optional, default: 5)
    Total number of cross-validation folds
    Used to determine which patients belong to validation set
    
  --model_name (optional, default: "convnext_tiny")
    Backbone architecture: "convnext_tiny" or "resnet50"
    Must match the model used during training

EXAMPLES
--------
  # Default settings (fold 0, convnext_tiny, 5-fold CV)
  python generate_visuals.py --run_id 62_102498
  
  # Visualize fold 2 with ResNet50 backbone
  python generate_visuals.py --run_id 62_102498 --fold 2 --model_name resnet50
  
  # Custom fold split (10-fold cross-validation)
  python generate_visuals.py --run_id 62_102498 --fold 1 --num_folds 10

OUTPUT
------
All visualizations are saved in: results/{RUN_ID}_gradcam_samples/
  - confusion_matrix.png - Patient-level 2x2 confusion matrix
  - roc_curve.png - Receiver Operating Characteristic curve with AUC
  - top_patients_i.png - Top-ranked positive predictions with Grad-CAM
  - false_negatives_i.png - Misclassified negative cases with attention maps

REQUIREMENTS
------------
  - PyTorch with GPU support (CUDA recommended)
  - Trained model checkpoint at: results/{RUN_ID}_f{FOLD}_{MODEL_NAME}_model_brain.pth
  - H. Pylori dataset at: /export/hhome/tkeating/HelicoDataSet or ../HelicoDataSet

NOTES
-----
  - Script uses model.eval() mode (no dropout, batch norm statistics fixed)
  - Inference runs in deterministic mode (set seed in torch/numpy)
  - Grad-CAM uses input-level gradient saliency (model-agnostic approach)
  - Visualization pairs show original patch + normalized heatmap side-by-side
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse
import gc
from tqdm import tqdm
from config import DATASET_ROOT, DEEPHP_DATASET_ROOT, PATIENT_CSV, PATCH_XLSX, CV_ANNOTATED, HOLDOUT
from dataset import HPyloriDataset
try:
    from dataset_deepHP import DeepHPDataset
except ImportError:
    DeepHPDataset = None
from model import get_model
import torch.nn.functional as F
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, 
    precision_recall_curve, average_precision_score, classification_report
)
from torchvision import transforms
from PIL import Image
from visualization_utils import (
    generate_gradcam, plot_roc_curve, plot_confusion_matrix, plot_gradcam_pair,
    plot_pr_curve, plot_probability_histogram, plot_threshold_analysis
)

# --- Config ---
# These paths are set through config.py for consistency across all scripts
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing (Deterministic for validation)
def det_preprocess_batch(batch, training=False):
    """
    Standardize the preprocessing for evaluation.
    Args:
        batch: (B, C, H, W) tensor
    """
    # Simple normalization to [0,1] and then ImageNet stats
    # No heavy augmentations or TTA here for clean visualization
    return batch

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def full_visual_report(RUN_ID, MODEL_PATH, MODEL_NAME="convnext_tiny", fold_idx=0, num_folds=5, dataset_type="helicodataset"):
    # Extract full prefix from model filename to include iteration name and SLURM job ID
    # e.g., 317_IntegrityRunV7_108024_f4_convnext_tiny from 317_IntegrityRunV7_108024_f4_convnext_tiny_model_brain.pth
    model_filename = os.path.basename(MODEL_PATH)
    # Remove _model_brain.pth or _swa_model_brain.pth suffix
    full_prefix = model_filename.replace("_swa_model_brain.pth", "").replace("_model_brain.pth", "")
    
    print(f"\n{'='*80}")
    print(f"Generating Visual Report for {full_prefix}")
    print(f"Dataset: {dataset_type.upper()} | Model: {MODEL_NAME} | Fold: {fold_idx}/{num_folds}")
    print(f"{'='*80}\n")
    OUTPUT_DIR = os.path.join("results", f"{full_prefix}_gradcam_samples")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize Dataset (Hold-out / Unseen Test Set) with manageable bag size
    if dataset_type.lower() == "helicodataset":
        print(f"Loading HelicoDataSet (HoldOut set from {HOLDOUT})...")
        full_dataset = HPyloriDataset(
            HOLDOUT, PATIENT_CSV, PATCH_XLSX, 
            transform=VAL_TRANSFORM, bag_mode=True, 
            max_bag_size=1000, train=False  # Reduced from 10000 to save memory
        )
    elif dataset_type.lower() == "deephp":
        if DeepHPDataset is None:
            print("ERROR: DeepHP dataset module not available. Install dataset_deepHP.py")
            sys.exit(1)
        print(f"Loading DeepHP Dataset (fold {fold_idx}/{num_folds} from {DEEPHP_DATASET_ROOT})...")
        full_dataset = DeepHPDataset(
            DEEPHP_DATASET_ROOT, fold_idx=fold_idx, num_folds=num_folds,
            train=False, transform=VAL_TRANSFORM, bag_mode=True,
            max_bag_size=1000
        )
    else:
        print(f"ERROR: Unknown dataset_type '{dataset_type}'. Use 'helicodataset' or 'deephp'")
        sys.exit(1)
    
    print(f"✓ Dataset loaded: {len(full_dataset)} patients/samples")
    
    # Create DataLoader: one patient (bag) per batch
    val_loader = DataLoader(
        full_dataset, batch_size=1, shuffle=False, 
        num_workers=0, pin_memory=False  # Reduced VRAM usage
    )
    
    # Load Model (Attention-MIL Architecture)
    model = get_model(model_name=MODEL_NAME, num_classes=2).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Handle both checkpoint formats (entire state_dict or dict wrapping it)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # Remove 'module.' prefix if it exists (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    # Filter out SWA-specific keys that don't exist in the model
    # (e.g., "n_averaged" from AveragedModel checkpoints)
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in model_state_dict}
    
    model.load_state_dict(filtered_state_dict)
    model.eval()

    # Determine Target Layer (no longer used, but keeping for reference)
    # The new Grad-CAM implementation uses input-level gradients instead
    target_layer = None

    # --- Step 1: Run Inference and Collect Patient-Level Results ---
    # Store patient metrics and dataset indices (NOT full bags to save memory)
    all_probs = []
    all_labels = []
    patient_ids = []
    pat_to_dataset_idx = {}  # Map patient_id to dataset index for reloading
    
    patient_performance = []
    vram_bag_limit = 250  # Reduced from 500 to save VRAM, process smaller chunks
    
    print(f"Running Inference on {len(full_dataset)} Validation Patients...")
    print(f"(Using smaller chunks: max_bag_size=1000, vram_limit={vram_bag_limit}, max_patches_for_gradcam=100)")
    
    with torch.no_grad():
        # Also need to track the index in the dataset
        for dataset_idx, (bags, labels, p_ids) in enumerate(tqdm(val_loader, desc="Patient Inference")):
            # bags: (1, bag_size, C, H, W), labels: (1,), p_ids: (1,)
            bags = bags.squeeze(0)  # (bag_size, C, H, W)
            label = labels.item()
            p_id = p_ids[0]
            
            # Store dataset index for later reloading (not the bags themselves)
            pat_to_dataset_idx[p_id] = dataset_idx
            
            # Divide bag into chunks for VRAM
            bag_size = bags.size(0)
            bag_probs_list = []
            
            if bag_size <= vram_bag_limit:
                chunk_ranges = [(0, bag_size)]
            else:
                chunk_ranges = []
                for s in range(0, bag_size - vram_bag_limit + 1, 250):
                    chunk_ranges.append((s, s + vram_bag_limit))
                if chunk_ranges[-1][1] < bag_size:
                    chunk_ranges.append((bag_size - vram_bag_limit, bag_size))
            
            for start_idx, end_idx in chunk_ranges:
                chunk = bags[start_idx:end_idx].to(DEVICE)
                logits, _ = model.forward_bag(chunk)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
                bag_probs_list.append(prob)
                
                # Free chunk memory immediately and aggressively clear cache
                del chunk
                torch.cuda.empty_cache()
                gc.collect()
            
            # Average across chunks
            prob = np.mean(bag_probs_list)
            
            all_probs.append(prob)
            all_labels.append(label)
            patient_ids.append(p_id)
            
            patient_performance.append({
                "Patient": p_id,
                "Label": label,
                "Prob": prob,
                "Pred": 1 if prob >= 0.5 else 0
            })
            
            # Free bag memory after processing
            del bags
            torch.cuda.empty_cache()
    all_labels_bin = [1 if l != 0 else 0 for l in all_labels]
    
    # Create performance dataframe for Grad-CAM selection
    perf_df = pd.DataFrame(patient_performance)
    
    # --- Step 1.5: Compute Per-Fold Metrics and Bootstrap CIs ---
    print(f"\n--- Computing Per-Fold Metrics and Bootstrap Confidence Intervals ---")
    
    # Calculate standard metrics
    from sklearn.metrics import matthews_corrcoef, cohen_kappa_score
    
    all_preds_bin = [1 if p >= 0.5 else 0 for p in all_probs]
    all_labels_array = np.array(all_labels_bin)
    all_preds_array = np.array(all_preds_bin)
    all_probs_array = np.array(all_probs)
    
    cm = confusion_matrix(all_labels_array, all_preds_array)
    tn, fp, fn, tp = cm.flatten()
    
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    sensitivity = rec
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2
    ppv = prec
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    mcc = matthews_corrcoef(all_labels_array, all_preds_array)
    kappa = cohen_kappa_score(all_labels_array, all_preds_array)
    
    fold_metrics = {
        'recall': rec, 'precision': prec, 'accuracy': acc, 'f1': f1,
        'sensitivity': sensitivity, 'specificity': specificity,
        'balanced_accuracy': balanced_accuracy,
        'ppv': ppv, 'npv': npv, 'fpr': fpr, 'fnr': fnr,
        'mcc': mcc, 'kappa': kappa,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }
    
    # Bootstrap resampling for CIs (500 iterations for speed)
    def bootstrap_fold_metrics(y_true, y_pred, y_prob, n_bootstrap=500):
        """Bootstrap confidence intervals for fold metrics."""
        n_patients = len(y_true)
        bootstrap_metrics = {
            'recall': [], 'precision': [], 'accuracy': [], 'f1': [],
            'sensitivity': [], 'specificity': [], 'balanced_accuracy': [],
            'ppv': [], 'npv': [], 'fpr': [], 'fnr': [],
            'mcc': [], 'kappa': []
        }
        
        print(f"  Running {n_bootstrap} bootstrap resamples...", end='', flush=True)
        for b in range(n_bootstrap):
            indices = np.random.choice(n_patients, size=n_patients, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            cm_boot = confusion_matrix(y_true_boot, y_pred_boot, labels=[0, 1])
            if cm_boot.shape == (2, 2):
                tn_b, fp_b, fn_b, tp_b = cm_boot.flatten()
            else:
                tn_b = fp_b = fn_b = tp_b = 0
                if len(cm_boot) == 2:
                    if cm_boot.shape[0] == 2:
                        tn_b, fp_b = cm_boot[0]
                        fn_b, tp_b = cm_boot[1]
            
            rec_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
            prec_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0
            acc_b = (tp_b + tn_b) / (tp_b + tn_b + fp_b + fn_b) if (tp_b + tn_b + fp_b + fn_b) > 0 else 0
            f1_b = 2 * (prec_b * rec_b) / (prec_b + rec_b) if (prec_b + rec_b) > 0 else 0
            
            bootstrap_metrics['recall'].append(rec_b)
            bootstrap_metrics['precision'].append(prec_b)
            bootstrap_metrics['accuracy'].append(acc_b)
            bootstrap_metrics['f1'].append(f1_b)
            bootstrap_metrics['sensitivity'].append(rec_b)
            bootstrap_metrics['specificity'].append(tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0)
            bootstrap_metrics['balanced_accuracy'].append((rec_b + bootstrap_metrics['specificity'][-1]) / 2)
            bootstrap_metrics['ppv'].append(prec_b)
            bootstrap_metrics['npv'].append(tn_b / (tn_b + fn_b) if (tn_b + fn_b) > 0 else 0)
            bootstrap_metrics['fpr'].append(fp_b / (fp_b + tn_b) if (fp_b + tn_b) > 0 else 0)
            bootstrap_metrics['fnr'].append(fn_b / (fn_b + tp_b) if (fn_b + tp_b) > 0 else 0)
            bootstrap_metrics['mcc'].append(matthews_corrcoef(y_true_boot, y_pred_boot))
            bootstrap_metrics['kappa'].append(cohen_kappa_score(y_true_boot, y_pred_boot))
            
            if (b + 1) % 100 == 0:
                print(f" {b+1}", end='', flush=True)
        
        print(" ✓")
        
        # Compute CI statistics
        ci_results = {}
        for metric_name, values in bootstrap_metrics.items():
            values_array = np.array(values)
            ci_results[metric_name] = {
                'mean': np.mean(values_array),
                'std': np.std(values_array),
                'ci_lower': np.percentile(values_array, 2.5),
                'ci_upper': np.percentile(values_array, 97.5),
                'ci_margin': (np.percentile(values_array, 97.5) - np.percentile(values_array, 2.5)) / 2
            }
        
        return ci_results
    
    bootstrap_ci = bootstrap_fold_metrics(all_labels_array, all_preds_array, all_probs_array)
    
    # Save per-fold metrics summary
    metrics_summary_path = os.path.join("results", f"{full_prefix}_metrics_summary.csv")
    metrics_data = {
        "Metric": [
            "Recall", "Precision", "Accuracy", "F1_Score",
            "Sensitivity", "Specificity", "Balanced_Accuracy",
            "PPV_(Positive_Predictive_Value)", "NPV_(Negative_Predictive_Value)", 
            "FPR_(False_Positive_Rate)", "FNR_(False_Negative_Rate)",
            "Matthews_Correlation_Coefficient", "Cohen_Kappa",
            "TP_(True_Positives)", "FP_(False_Positives)", "FN_(False_Negatives)", 
            "TN_(True_Negatives)"
        ],
        "Point_Estimate": [
            fold_metrics['recall'], fold_metrics['precision'], fold_metrics['accuracy'], fold_metrics['f1'],
            fold_metrics['sensitivity'], fold_metrics['specificity'], fold_metrics['balanced_accuracy'],
            fold_metrics['ppv'], fold_metrics['npv'], fold_metrics['fpr'], fold_metrics['fnr'],
            fold_metrics['mcc'], fold_metrics['kappa'],
            fold_metrics['tp'], fold_metrics['fp'], fold_metrics['fn'], fold_metrics['tn']
        ],
        "Bootstrap_Mean": [
            bootstrap_ci['recall']['mean'], bootstrap_ci['precision']['mean'], 
            bootstrap_ci['accuracy']['mean'], bootstrap_ci['f1']['mean'],
            bootstrap_ci['sensitivity']['mean'], bootstrap_ci['specificity']['mean'], 
            bootstrap_ci['balanced_accuracy']['mean'],
            bootstrap_ci['ppv']['mean'], bootstrap_ci['npv']['mean'], 
            bootstrap_ci['fpr']['mean'], bootstrap_ci['fnr']['mean'],
            bootstrap_ci['mcc']['mean'], bootstrap_ci['kappa']['mean'],
            fold_metrics['tp'], fold_metrics['fp'], fold_metrics['fn'], fold_metrics['tn']
        ],
        "Bootstrap_Std": [
            bootstrap_ci['recall']['std'], bootstrap_ci['precision']['std'], 
            bootstrap_ci['accuracy']['std'], bootstrap_ci['f1']['std'],
            bootstrap_ci['sensitivity']['std'], bootstrap_ci['specificity']['std'], 
            bootstrap_ci['balanced_accuracy']['std'],
            bootstrap_ci['ppv']['std'], bootstrap_ci['npv']['std'], 
            bootstrap_ci['fpr']['std'], bootstrap_ci['fnr']['std'],
            bootstrap_ci['mcc']['std'], bootstrap_ci['kappa']['std'],
            0, 0, 0, 0
        ],
        "CI_Lower_95%": [
            bootstrap_ci['recall']['ci_lower'], bootstrap_ci['precision']['ci_lower'], 
            bootstrap_ci['accuracy']['ci_lower'], bootstrap_ci['f1']['ci_lower'],
            bootstrap_ci['sensitivity']['ci_lower'], bootstrap_ci['specificity']['ci_lower'], 
            bootstrap_ci['balanced_accuracy']['ci_lower'],
            bootstrap_ci['ppv']['ci_lower'], bootstrap_ci['npv']['ci_lower'], 
            bootstrap_ci['fpr']['ci_lower'], bootstrap_ci['fnr']['ci_lower'],
            bootstrap_ci['mcc']['ci_lower'], bootstrap_ci['kappa']['ci_lower'],
            fold_metrics['tp'], fold_metrics['fp'], fold_metrics['fn'], fold_metrics['tn']
        ],
        "CI_Upper_95%": [
            bootstrap_ci['recall']['ci_upper'], bootstrap_ci['precision']['ci_upper'], 
            bootstrap_ci['accuracy']['ci_upper'], bootstrap_ci['f1']['ci_upper'],
            bootstrap_ci['sensitivity']['ci_upper'], bootstrap_ci['specificity']['ci_upper'], 
            bootstrap_ci['balanced_accuracy']['ci_upper'],
            bootstrap_ci['ppv']['ci_upper'], bootstrap_ci['npv']['ci_upper'], 
            bootstrap_ci['fpr']['ci_upper'], bootstrap_ci['fnr']['ci_upper'],
            bootstrap_ci['mcc']['ci_upper'], bootstrap_ci['kappa']['ci_upper'],
            fold_metrics['tp'], fold_metrics['fp'], fold_metrics['fn'], fold_metrics['tn']
        ],
        "CI_Margin": [
            bootstrap_ci['recall']['ci_margin'], bootstrap_ci['precision']['ci_margin'], 
            bootstrap_ci['accuracy']['ci_margin'], bootstrap_ci['f1']['ci_margin'],
            bootstrap_ci['sensitivity']['ci_margin'], bootstrap_ci['specificity']['ci_margin'], 
            bootstrap_ci['balanced_accuracy']['ci_margin'],
            bootstrap_ci['ppv']['ci_margin'], bootstrap_ci['npv']['ci_margin'], 
            bootstrap_ci['fpr']['ci_margin'], bootstrap_ci['fnr']['ci_margin'],
            bootstrap_ci['mcc']['ci_margin'], bootstrap_ci['kappa']['ci_margin'],
            0, 0, 0, 0
        ]
    }
    
    pd.DataFrame(metrics_data).to_csv(metrics_summary_path, index=False)
    print(f"✓ Per-fold metrics summary saved to [{metrics_summary_path}]")
    
    # Also save patient-level predictions for detailed analysis
    perf_df.to_csv(os.path.join("results", f"{full_prefix}_predictions.csv"), index=False)
    print(f"✓ Patient-level predictions saved to [results/{full_prefix}_predictions.csv]")
    
    # --- Step 2: Visualization Plots ---
    # 1. Confusion Matrix
    all_preds_bin = [1 if p >= 0.5 else 0 for p in all_probs]
    plot_confusion_matrix(all_labels_bin, all_preds_bin, os.path.join("results", f"{full_prefix}_confusion_matrix.png"))

    # 2. Patient-Level ROC (also compute AUC for reporting)
    fpr, tpr, _ = roc_curve(all_labels_bin, all_probs)
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(all_labels_bin, all_probs, os.path.join("results", f"{full_prefix}_roc_curve.png"))

    # 3. Precision-Recall Curve (compute AP for reporting)
    pr_auc = average_precision_score(all_labels_bin, all_probs)
    plot_pr_curve(all_labels_bin, all_probs, os.path.join("results", f"{full_prefix}_pr_curve.png"))

    # 4. Probability Histogram
    plot_probability_histogram(np.array(all_probs), np.array(all_labels_bin), os.path.join("results", f"{full_prefix}_probability_histogram.png"))

    # 5. Threshold Analysis (Performance across decision boundaries)
    plot_threshold_analysis(all_labels_bin, all_probs, os.path.join("results", f"{full_prefix}_threshold_analysis.png"))
    
    # Print per-fold metrics summary
    print(f"\n{'='*50}")
    print(f"PER-FOLD METRICS SUMMARY: {full_prefix}")
    print(f"{'='*50}")
    print(f"Recall:       {fold_metrics['recall']:.4f} [CI: {bootstrap_ci['recall']['ci_lower']:.4f} - {bootstrap_ci['recall']['ci_upper']:.4f}]")
    print(f"Precision:    {fold_metrics['precision']:.4f} [CI: {bootstrap_ci['precision']['ci_lower']:.4f} - {bootstrap_ci['precision']['ci_upper']:.4f}]")
    print(f"Accuracy:     {fold_metrics['accuracy']:.4f} [CI: {bootstrap_ci['accuracy']['ci_lower']:.4f} - {bootstrap_ci['accuracy']['ci_upper']:.4f}]")
    print(f"F1 Score:     {fold_metrics['f1']:.4f} [CI: {bootstrap_ci['f1']['ci_lower']:.4f} - {bootstrap_ci['f1']['ci_upper']:.4f}]")
    print(f"Specificity:  {fold_metrics['specificity']:.4f} [CI: {bootstrap_ci['specificity']['ci_lower']:.4f} - {bootstrap_ci['specificity']['ci_upper']:.4f}]")
    print(f"ROC-AUC:      {roc_auc:.4f}")
    print(f"PR-AUC:       {pr_auc:.4f}")
    print(f"TP: {fold_metrics['tp']} | FP: {fold_metrics['fp']} | FN: {fold_metrics['fn']} | TN: {fold_metrics['tn']}")
    print(f"{'='*50}\n")

    # --- Step 3: Grad-CAM for Top Suspicious Patients ---
    # Pick Top 3 Positives and Top 3 False Negatives (if any)
    top_positives = perf_df[perf_df['Label'] == 1].sort_values('Prob', ascending=False).head(3)
    ghosts = perf_df[(perf_df['Label'] == 1) & (perf_df['Prob'] < 0.5)].sort_values('Prob', ascending=False).head(3)
    
    targets = pd.concat([top_positives, ghosts])

    print(f"Generating Grad-CAM for {len(targets)} patients...")
    for _, row in targets.iterrows():
        p_id = row['Patient']
        is_fn = row['Prob'] < 0.5
        
        # Reload bags from dataset instead of keeping in memory
        if p_id not in pat_to_dataset_idx:
            continue
        
        dataset_idx = pat_to_dataset_idx[p_id]
        bags_tensor, _, _ = full_dataset[dataset_idx]
        bags_tensor = bags_tensor.squeeze(0)  # (bag_size, C, H, W)
        
        # Get attention weights to pick the most important patches
        all_attns = []
        max_patches_to_check = min(bags_tensor.size(0), 100)  # Reduced from 500 for speed
        
        with torch.no_grad():
            for i in range(max_patches_to_check):
                img = bags_tensor[i]
                img_t = img.unsqueeze(0).to(DEVICE)
                _, attn = model.forward_bag(img_t)
                all_attns.append((i, attn.item()))
        
        # Pick top 2 patches by attention
        all_attns.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in all_attns[:2]]
        
        for rank, idx in enumerate(top_indices):
            patch_img = bags_tensor[idx]
            patch_t = patch_img.unsqueeze(0).to(DEVICE)
            
            with torch.enable_grad():
                heatmap_batch, _ = generate_gradcam(model.backbone, patch_t)
            
            # Plot side-by-side visualization (original + heatmap overlay)
            plot_gradcam_pair(
                patch_img, heatmap_batch[0, 0], p_id, rank, idx,
                all_attns[rank][1], row['Prob'],
                is_false_negative=is_fn, output_dir=OUTPUT_DIR
            )
        
        # Free bag memory after processing
        del bags_tensor
        torch.cuda.empty_cache()

    print(f"Visual report finished. Results in results/{full_prefix}_*")

def get_latest_run_id():
    """Find the latest run ID from results directory by parsing model filenames."""
    import re
    if not os.path.exists("results"):
        return None
    
    run_ids = set()
    for filename in os.listdir("results"):
        # Match pattern: {run_id}_{anything}_model_brain.pth
        # Run ID is just the leading digits
        match = re.match(r"^(\d+)_.*_model_brain\.pth$", filename)
        if match:
            run_ids.add(match.group(1))
    
    if run_ids:
        # Sort numerically to get the latest
        sorted_ids = sorted(run_ids, key=lambda x: int(x))
        return sorted_ids[-1]
    return None

def find_model_path(run_id, fold, model_name):
    """Find model file for given run_id and fold. Uses metadata to pick the correct model.
    If specified fold doesn't exist, searches for any available fold."""
    import re
    import json
    results_dir = "results"
    
    # Pattern: {run_id}_{anything}_f{fold}_{model_name}_model_brain.pth  
    swa_pattern = re.compile(rf"^{run_id}_.*_f{fold}_{model_name}_swa_model_brain\.pth$")
    model_pattern = re.compile(rf"^{run_id}_.*_f{fold}_{model_name}_model_brain\.pth$")
    metadata_pattern = re.compile(rf"^{run_id}_.*_f{fold}_{model_name}_model_selection\.json$")
    
    swa_models = []
    regular_models = []
    metadata_files = []
    
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            if swa_pattern.match(filename):
                swa_models.append(os.path.join(results_dir, filename))
            elif model_pattern.match(filename):
                regular_models.append(os.path.join(results_dir, filename))
            if metadata_pattern.match(filename):
                metadata_files.append(os.path.join(results_dir, filename))
    
    # Check metadata to see which model was used during training
    if metadata_files:
        try:
            with open(metadata_files[0], 'r') as f:
                metadata = json.load(f)
                if metadata.get("use_swa") and swa_models:
                    return swa_models[0], fold
                elif not metadata.get("use_swa") and regular_models:
                    return regular_models[0], fold
        except (json.JSONDecodeError, IOError):
            pass
    
    # Models found for specified fold
    if swa_models:
        return swa_models[0], fold
    elif regular_models:
        return regular_models[0], fold
    
    # If specified fold not found, search for any available fold for this run_id
    fold_pattern = re.compile(rf"^{run_id}_.*_f(\d+)_{model_name}_swa_model_brain\.pth$")
    fold_pattern_regular = re.compile(rf"^{run_id}_.*_f(\d+)_{model_name}_model_brain\.pth$")
    
    available_folds = set()
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            match = fold_pattern.match(filename)
            if match:
                available_folds.add(int(match.group(1)))
            match = fold_pattern_regular.match(filename)
            if match:
                available_folds.add(int(match.group(1)))
    
    if available_folds:
        first_fold = min(available_folds)
        print(f"Fold {fold} not found for run_id={run_id}. Available folds: {sorted(available_folds)}. Using fold {first_fold}.")
        return find_model_path(run_id, first_fold, model_name)
    
    return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H. Pylori Visual Generation")
    parser.add_argument("--run_id", type=str, default=None, help="Run ID (e.g., 313). Defaults to latest run.")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (default: 0)")
    parser.add_argument("--num_folds", type=int, default=5, help="Total number of folds")
    parser.add_argument("--model_name", type=str, default="convnext_tiny", choices=["resnet50", "convnext_tiny"],
                         help="Backbone architecture")
    parser.add_argument("--dataset", type=str, default="helicodataset", 
                       choices=["helicodataset", "deephp", "both"],
                       help="Which dataset to visualize: 'helicodataset', 'deephp', or 'both'")
    args = parser.parse_args()
    
    # If no run_id provided, use the latest one
    run_id = args.run_id
    if run_id is None:
        run_id = get_latest_run_id()
        if run_id is None:
            print("Error: No run ID provided and no models found in results directory")
            sys.exit(1)
        print(f"Using latest run: {run_id}")
    
    # Find model path (prefers SWA model, falls back to any available fold if needed)
    model_path, actual_fold = find_model_path(run_id, args.fold, args.model_name)
    
    if model_path is None:
        print(f"Error: Model not found for run_id={run_id}, fold={args.fold}, model={args.model_name}")
        sys.exit(1)
    
    if actual_fold is None:
        sys.exit(1)
        
    if os.path.exists(model_path):
        print(f"Using model: {os.path.basename(model_path)}")
        
        # Process selected dataset(s)
        datasets_to_process = []
        if args.dataset.lower() == "both":
            datasets_to_process = ["helicodataset", "deephp"]
        else:
            datasets_to_process = [args.dataset.lower()]
        
        for ds_type in datasets_to_process:
            full_visual_report(run_id, model_path, args.model_name, actual_fold, args.num_folds, ds_type)
    else:
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
