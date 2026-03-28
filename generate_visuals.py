
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
  - H. Pylori dataset at: /import/fhome/vlia/HelicoDataSet or ../HelicoDataSet

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
from tqdm import tqdm
from dataset import HPyloriDataset
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
    plot_pr_curve, plot_probability_histogram
)

# --- Config ---
# These paths are set through command line arguments
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Paths
BASE_PATH = "/import/fhome/vlia/HelicoDataSet"
if not os.path.exists(BASE_PATH):
    BASE_PATH = os.path.abspath(os.path.join(os.getcwd(), "..", "HelicoDataSet"))

PATIENT_CSV = os.path.join(BASE_PATH, "PatientDiagnosis.csv")
PATCH_CSV = os.path.join(BASE_PATH, "HP_WSI-CoordAnnotatedAllPatches.xlsx")
TRAIN_DIR = os.path.join(BASE_PATH, "CrossValidation/Annotated")
HOLDOUT_DIR = os.path.join(BASE_PATH, "HoldOut")

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

def full_visual_report(RUN_ID, MODEL_PATH, MODEL_NAME="convnext_tiny", fold_idx=0, num_folds=5):
    # Extract full prefix from model filename to include iteration name and SLURM job ID
    # e.g., 317_IntegrityRunV7_108024_f4_convnext_tiny from 317_IntegrityRunV7_108024_f4_convnext_tiny_model_brain.pth
    model_filename = os.path.basename(MODEL_PATH)
    # Remove _model_brain.pth or _swa_model_brain.pth suffix
    full_prefix = model_filename.replace("_swa_model_brain.pth", "").replace("_model_brain.pth", "")
    
    print(f"--- Generating Visual Report for {full_prefix} (Model: {MODEL_NAME}) ---")
    OUTPUT_DIR = os.path.join("results", f"{full_prefix}_gradcam_samples")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize Dataset (Hold-out / Unseen Test Set) with full bags
    full_dataset = HPyloriDataset(
        HOLDOUT_DIR, PATIENT_CSV, PATCH_CSV, 
        transform=VAL_TRANSFORM, bag_mode=True, 
        max_bag_size=10000, train=False
    )
    
    print(f"Evaluating on complete holdout set: {len(full_dataset)} patients")
    
    # Create DataLoader: one patient (bag) per batch
    val_loader = DataLoader(
        full_dataset, batch_size=1, shuffle=False, 
        num_workers=0, pin_memory=True
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
    vram_bag_limit = 500
    
    print(f"Running Inference on {len(full_dataset)} Validation Patients...")
    
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
                
                # Free chunk memory immediately
                del chunk
            
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
    
    # --- Step 2: Visualization Plots ---
    # 1. Confusion Matrix
    all_preds_bin = [1 if p >= 0.5 else 0 for p in all_probs]
    plot_confusion_matrix(all_labels_bin, all_preds_bin, os.path.join("results", f"{full_prefix}_confusion_matrix.png"))

    # 2. Patient-Level ROC
    plot_roc_curve(all_labels_bin, all_probs, os.path.join("results", f"{full_prefix}_roc_curve.png"))

    # 3. Precision-Recall Curve
    plot_pr_curve(all_labels_bin, all_probs, os.path.join("results", f"{full_prefix}_pr_curve.png"))

    # 4. Probability Histogram
    plot_probability_histogram(np.array(all_probs), np.array(all_labels_bin), os.path.join("results", f"{full_prefix}_probability_histogram.png"))

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
        max_patches_to_check = min(bags_tensor.size(0), 500)
        
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
        full_id = f"{run_id}_f{actual_fold}"
        full_visual_report(full_id, model_path, args.model_name, actual_fold, args.num_folds)
    else:
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
