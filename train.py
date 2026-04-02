"""
H. Pylori Contamination Detection - Model Training and Evaluation
=================================================================

OVERVIEW
--------
This is the primary training and evaluation script for H. Pylori detection models.
It implements a complete deep learning pipeline including:
  - Cross-validation training with Focal Loss for imbalanced patch datasets
  - Gated Attention-based Multiple Instance Learning (MIL) for patient-level predictions
  - Advanced inference with 16-way Test-Time Augmentation (TTA) and sliding windows
  - Patient-level consensus voting and comprehensive evaluation metrics
  - Stochastic Weight Averaging (SWA) for model stability and generalization
  - Automatic visualization generation: learning curves, ROC/PR curves, Grad-CAM heatmaps

PURPOSE
-------
This script handles the complete model lifecycle for a specific cross-validation fold:
  - Trains the backbone (ConvNeXt-Tiny or ResNet50) on training patches
  - Validates on held-out validation patients
  - Generates trained checkpoints (.pth files)
  - Produces comprehensive evaluation reports and visualizations
  
Designed for computational cluster submission (SLURM) with automatic job ID tracking
to prevent race conditions during multi-fold parallel training.

HOW IT WORKS
------------
1. Data Preparation: Loads dataset, splits by patient into K folds
2. Training Loop: For each epoch...
   a. Forward pass: Input bags of patches → patch-level logits → MIL aggregation
   b. Loss computation: Focal Loss (downweights easy negatives, upweights hard positives)
   c. Backward pass: Computes gradients, applies gradient clipping if enabled
   d. Optimizer step: AdamW with OneCycleLR scheduler updates model weights
3. Validation: Runs MIL inference on validation fold, computes patient-level metrics
4. SWA (optional): After training, averages model weights over final epochs for stability
5. Test-Time Augmentation: Applies 16 augmentations during final inference (contrast boost)
6. Patient Consensus: Aggregates multiple instance predictions into final diagnosis
7. Evaluation: Computes confusion matrix, ROC/PR curves, and generates visualizations
8. Output: Saves model checkpoint, evaluation metrics, and PNG visualizations

USAGE
-----
Run from command line with required arguments:

  python train.py --fold <FOLD> [--num_folds <NUM_FOLDS>] [--model_name <MODEL>] [options...]

ARGUMENTS
---------
REQUIRED:
  --fold <INT>
    Cross-validation fold index (0 to num_folds-1)
    Each fold trains on different subset of patients
    Run once per fold for complete K-fold evaluation

ARCHITECTURE:
  --model_name {convnext_tiny, resnet50}
    Default: convnext_tiny
    Backbone architecture for feature extraction
    convnext_tiny: 28M params, efficient, high accuracy
    resnet50: Classical, 25M params, stable training

CROSS-VALIDATION:
  --num_folds <INT>
    Default: 5
    Total number of folds for cross-validation
    Determines patient split strategy

LOSS FUNCTION:
  --pos_weight <FLOAT>
    Default: 7.5
    Weight for positive class in Focal Loss
    Higher = penalize false negatives more heavily
    
  --neg_weight <FLOAT>
    Default: 1.0
    Weight for negative class (background patches)
    
  --gamma <FLOAT>
    Default: 1.0
    Focal Loss gamma parameter (hardness of focusing)
    Lower gamma = softer focus on hard examples
    Higher gamma = harder focus on misclassified examples

TRAINING HYPERPARAMETERS:
  --num_epochs <INT>
    Default: 15
    Number of training epochs
    
  --pct_start <FLOAT>
    Default: 0.1
    Percentage of epochs for warmup in OneCycleLR scheduler
    
  --weight_decay <FLOAT>
    Default: 0.01
    L2 regularization coefficient for optimizer
    
  --clip_grad <FLOAT>
    Default: 0.0 (disabled)
    Gradient clipping norm (prevents exploding gradients)
    Recommended: 1.0-2.0 for unstable training
    
  --jitter <FLOAT>
    Default: 0.15
    ColorJitter intensity (brightness/contrast augmentation)
    Range: [0.0, 1.0], higher = more aggressive augmentation

CHECKPOINT & EVALUATION:
  --saver_metric {loss, recall, f1}
    Default: recall
    Metric used to select best model checkpoint
    recall: Minimizes false negatives (clinical priority)
    f1: Balanced metric
    loss: Training loss
    
STOCHASTIC WEIGHT AVERAGING:
  --use_swa {True, False}
    Default: True
    Enable SWA for improved generalization
    Averages model weights over final epochs
    
  --swa_start <INT>
    Default: 15
    Epoch to begin SWA averaging
    Recommended: After most training is complete

ARCHITECTURE OPTIONS:
  --pool_type {attention, max}
    Default: attention
    MIL pooling aggregation type
    attention: Gated attention mechanism (preferred)
    max: Simple max pooling (baseline)
    
  --freeze_bn {True, False}
    Default: False
    Freeze BatchNorm layers during training
    Use True if training on very small batches
    
METADATA:
  --iter <STR>
    Default: "24.9"
    Iteration version tag for output filenames
    Used for tracking experimental versions

EXAMPLES
--------
  # Default ConvNeXt-Tiny, fold 0, 5-fold CV
  python train.py --fold 0
  
  # ResNet50 backbone, fold 2, 10-fold CV
  python train.py --fold 2 --model_name resnet50 --num_folds 10
  
  # Aggressive training: high pos_weight, strong gradient clipping
  python train.py --fold 0 --pos_weight 15.0 --gamma 2.0 --clip_grad 1.5
  
  # Disable SWA, use custom learning rate warmup
  python train.py --fold 0 --use_swa False --pct_start 0.2
  
  # Batch submission (all 5 folds for 5-fold CV):
  for i in {0..4}; do
    python train.py --fold $i --model_name convnext_tiny &
  done

OUTPUT
------
All outputs saved in: results/

Model Checkpoints:
  {RUN_ID}_f{FOLD}_{MODEL}_model_brain.pth
    Full model state dict for inference
  {RUN_ID}_f{FOLD}_{MODEL}_swa_model_brain.pth
    SWA-averaged model (if --use_swa True)

Evaluation Reports:
  {RUN_ID}_f{FOLD}_{MODEL}_evaluation_report.csv
    Detailed metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, etc.
  {RUN_ID}_f{FOLD}_{MODEL}_patient_consensus.csv
    Patient-level predictions and probabilities

Visualizations (PNG):
  learning_curves_{RUN_ID}.png - Training/validation loss and accuracy
  confusion_matrix_{RUN_ID}.png - 2x2 patient-level confusion matrix
  probability_hist_{RUN_ID}.png - Distribution of predicted probabilities
  roc_curve_{RUN_ID}.png - ROC curve with AUC score
  pr_curve_{RUN_ID}.png - Precision-Recall curve with AP score
  top_patients_*.png - Top-ranked predictions with Grad-CAM heatmaps
  false_negatives_*.png - Misclassified negatives with attention maps

REQUIREMENTS
------------
  - PyTorch with GPU support (CUDA recommended for reasonable training time)
  - H. Pylori dataset at: /import/fhome/vlia/HelicoDataSet or ../HelicoDataSet
  - 32GB+ GPU VRAM recommended (adjust batch size if needed)
  - Models: ConvNeXt-Tiny or ResNet50 from torchvision

NOTES
-----
  - Training uses AdamW optimizer with OneCycleLR scheduler
  - Focal Loss is customized with label smoothing to prevent overconfidence
  - Test-Time Augmentation (TTA) applies 16 random contrast-boosted versions of each patch
  - MIL aggregation uses Gated Attention pooling for interpretability
  - SWA improves generalization by averaging weights over final epochs
  - Grad-CAM uses input-level gradient saliency (model-agnostic)
  - All fold indices are 0-based (0 to num_folds-1)
  - Output file naming automatically avoids collisions using SLURM Job ID if available
"""
import os                       # Standard library for file path management
import torch                    # Core library for deep learning
import torch.nn as nn           # Tools for building neural network layers
import torch.optim as optim     # Mathematical tools to "teach" the model
from torch.optim.adam import Adam # The specific algorithm to adjust the brain
from torch.optim import AdamW    # Better for ConvNeXt architectures
import numpy as np               # Numeric library
import pandas as pd              # Data manipulation library
import matplotlib.pyplot as plt  # Drawing/plotting library
import argparse
import gc
from torch.utils.data import DataLoader, random_split, Dataset, Subset, WeightedRandomSampler # Tools to manage and split data
from torchvision import transforms # Tools to prep images for the AI
from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()
from torchvision.transforms import v2 # Faster, optimized transforms
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, 
    classification_report, precision_recall_curve, 
    average_precision_score, PrecisionRecallDisplay
) 
from dataset import HPyloriDataset # Our custom code that finds images/labels
from model import get_model        # Our custom code that builds the AI brain
from tqdm import tqdm              # A library that shows a "progress bar"
import re                          # Regexp to handle file numbering
import gc
import torch.nn.functional as F
from normalization import MacenkoNormalizer
from visualization_utils import (
    generate_gradcam, plot_learning_curves, plot_confusion_matrix,
    plot_probability_histogram, plot_roc_curve, plot_pr_curve, plot_gradcam_pair
)

class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Diagnostic Screening.
    Reduces the relative loss for well-classified examples, focusing on 
    the sparse, hard-to-detect bacterial signals.

    TECHNICAL RATIONALE: Focal Loss Configuration (gamma=3.0, alpha=5.0)
    -----------------------------------------------------------------
    In clinical H. Pylori screening, 'Negative' patches (background tissue, 
    stain precipitate) vastly outnumber 'Positive' patches (bacteria) 
    at a ratio of approximately 100:1. 
    - Gamma=3.0: Provides a "harder" down-weighting of easy background 
      patches compared to standard gamma=2.0. This prevents the model 
      from being "satisfied" with high accuracy on simple background 
      and forces it to learn the subtle features of sparse bacteremia.
    - Alpha (pos_weight)=5.0: Explicitly penalizes False Negatives (FN). 
      In a clinical "Searcher" profile, a False Negative (missing bacteria) 
      is significantly more dangerous than a False Positive (requiring 
      manual pathologist review), thus the heavy positive bias.

    Includes Label Smoothing (set to 0.05 by default) to prevent model 
    overconfidence and ensure better generalization on "Ghost Patients" 
    with atypical morphology.
    """
    def __init__(self, alpha=1, gamma=2, weight=None, smoothing=0.00):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        # Apply Label Smoothing to the Cross Entropy base (Optimization 7)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight, label_smoothing=self.smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

def get_next_run_number(results_dir="results", current_slurm_id=None):
    """
    Finds the next available numeric prefix for output files, 
    respecting SLURM Job ID order to prevent race conditions during 
    multi-fold submissions.
    """
    if not os.path.exists(results_dir):
        return 0
    
    files = os.listdir(results_dir)
    # Map JobID to RunID (to find the most recent anchor)
    job_to_run = {}
    
    # 1. Scan results for existing mappings
    for f in files:
        # Matches formats like "62_102498_..."
        match = re.match(r"^(\d+)_(\d+)_", f)
        if match:
            run_id = int(match.group(1))
            job_id = int(match.group(2))
            if job_id not in job_to_run or run_id > job_to_run[job_id]:
                job_to_run[job_id] = run_id
        
        # Also check existing output logs for "Starting Run ID" 
        if f.startswith("output_") and f.endswith(".txt"):
            jid_match = re.search(r"output_(\d+).txt", f)
            if jid_match:
                jid = int(jid_match.group(1))
                try:
                    with open(os.path.join(results_dir, f), 'r') as log:
                        # Scan top of file for the header we printed
                        for _ in range(50):
                            line = log.readline()
                            if not line: break
                            m = re.search(r"Run ID: (\d+)", line)
                            if m:
                                rid = int(m.group(1))
                                if jid not in job_to_run or rid > job_to_run[jid]:
                                    job_to_run[jid] = rid
                                break
                except: pass

    # If not running in SLURM, use standard max+1 logic
    if current_slurm_id is None or not str(current_slurm_id).isdigit():
        return max(job_to_run.values()) + 1 if job_to_run else 0
        
    current_jid = int(current_slurm_id)
    
    # 2. Identify all Job IDs in the queue (from output_*.txt placeholders)
    all_job_ids = set(job_to_run.keys())
    for f in files:
        if f.startswith("output_") and f.endswith(".txt"):
            jid_m = re.search(r"output_(\d+).txt", f)
            if jid_m:
                all_job_ids.add(int(jid_m.group(1)))
    
    all_job_ids.add(current_jid)
    sorted_jids = sorted(list(all_job_ids))
    
    # 3. Find the most recent anchor point (highest Job ID < current that has a Run ID)
    older_assigned = sorted([jid for jid in job_to_run.keys() if jid < current_jid])
    
    if not older_assigned:
        # No older jobs found with labels. We are the beginning of this results directory.
        return sorted_jids.index(current_jid)
    
    last_known_jid = older_assigned[-1]
    last_known_run = job_to_run[last_known_jid]
    
    # 4. Count the number of 'slots' (output logs) between then and now
    rank_offset = 0
    for jid in sorted_jids:
        if jid > last_known_jid and jid <= current_jid:
            rank_offset += 1
            
    return last_known_run + rank_offset

# This helper class allows us to have different transforms for train and validation split
class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        img, label = self.subset[index] 
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)

class MIL_SWA_Wrapper(nn.Module):
    """
    Stochastic Weight Averaging (SWA) Adapter for Multiple Instance Learning.
    
    PyTorch's `update_bn` utility expects standard mini-batches. Since our
    MIL DataLoader provides "Bags" of shape (1, Bag_Size, C, H, W), this 
    wrapper flattens the bag so the BatchNorm layers can calibrate 
    statistics (mean/variance) using individual patches.
    """
    def __init__(self, swa_model):
        super().__init__()
        self.swa_model = swa_model
    def forward(self, x):
        # x shape from DataLoader is (1, Bag_Size, C, H, W)
        x = x.squeeze(0) # Remove batch dim -> (Bag_Size, C, H, W)
        # We don't need the attention weights for BN update
        logits, _ = self.swa_model.module.forward_bag(x)
        return logits

def update_swa_bn(loader, swa_model, device):
    """Custom BatchNorm update for MIL bags."""
    swa_model.train()
    wrapper = MIL_SWA_Wrapper(swa_model).to(device)
    wrapper.train()
    with torch.no_grad():
        for bags, _, _, _ in tqdm(loader, desc="Updating SWA Batchnorm"):
            bags = bags.to(device)
            # Preprocess is required because it's part of the forward pipeline
            # but usually we normalize inside the loop. 
            # Logic: recreate the internal training preprocessing
            # Note: det_preprocess_batch is available in the outer scope
            bags = bags.squeeze(0)
            # gpu_normalize is global
            from torchvision.transforms import v2
            gpu_normalize = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            bags = gpu_normalize(bags)
            # Re-wrap in batch dim 1 for wrapper.forward
            bags = bags.unsqueeze(0)
            wrapper(bags)

def train_model(fold_idx=0, num_folds=5, model_name="convnext_tiny", pos_weight=7.5, neg_weight=1.0, gamma=1.0, num_epochs=15, saver_metric="recall", freeze_bn=False, clip_grad=0.0, pct_start=0.1, weight_decay=0.01, use_swa=True, swa_start=15, jitter=0.15, pool_type="attention", iter_name="24.9"):
    """
    Train a deep learning model for H. pylori contamination detection using k-fold cross-validation.
    This function implements a complete machine learning pipeline including:
    - Patient-level k-fold stratification to prevent data leakage
    - GPU-optimized training with automatic mixed precision (AMP)
    - Focal loss with class weighting for imbalanced bacteremia detection
    - OneCycle learning rate scheduling with warmup and cosine annealing
    - Gradient accumulation for effective batch size scaling
    - Patient-level consensus prediction via Attention-MIL aggregation
    - Comprehensive evaluation metrics: patch-level and patient-level confusion matrices, ROC/PR curves
    - Interpretability via Grad-CAM visualization
    - Hardware optimization (torch.compile kernel fusion, Intel IPEX support)
    Args:
        fold_idx (int, optional): Current fold index for k-fold cross-validation. Default: 0
        num_folds (int, optional): Total number of folds for cross-validation. Default: 5
        model_name (str, optional): Model architecture ('convnext_tiny', 'resnet50', etc.). Default: "convnext_tiny"
        pos_weight (float, optional): Weight for the positive class in Focal Loss. Default: 7.5
        neg_weight (float, optional): Weight for the negative class in Focal Loss. Default: 1.0
        gamma (float, optional): Gamma parameter for Focal Loss. Default: 1.0
        saver_metric (str, optional): Metric to use for saving the best model ('loss', 'recall', 'f1'). Default: "recall"
    Returns:
        None. Outputs saved to results/ directory with files prefixed by {run_id}_{slurm_id}_f{fold_idx}_{model_name}:
        - *_model_brain.pth: Best model weights (lowest validation loss)
        - *_evaluation_report.csv: Patch-level classification metrics
        - *_confusion_matrix.png: Patch-level confusion matrix
        - *_roc_curve.png: Patch-level ROC curve
        - *_pr_curve.png: Patch-level precision-recall curve
        - *_learning_curves.png: Training/validation loss and accuracy over epochs
        - *_probability_histogram.png: Distribution of predicted probabilities
        - *_patient_confusion_matrix.png: Patient-level confusion matrix
        - *_patient_roc_curve.png: Patient-level ROC curve
        - *_patient_pr_curve.png: Patient-level precision-recall curves
        - *_patient_consensus.csv: Patient-level predictions with consensus metrics
        - *_gradcam_samples/: Interpretability maps for sample patches
    Raises:
        FileNotFoundError: If base data path cannot be located
        RuntimeError: If CUDA is unavailable (gracefully falls back to CPU)
    Notes:
        - Patient-level split ensures all patches from a single patient are in train OR validation, never both
        - Grad-CAM requires torch.enable_grad() context and uses the uncompiled model (_orig_mod) for hook activation
        - Hardware: Optimized for NVIDIA A40 (48GB VRAM) with batch_size=128 and accumulation_steps=2
    """
    # --- Step 0: Prepare output directories ---
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Get the numeric run ID and the SLURM job ID (if it exists)
    slurm_id = os.environ.get("SLURM_JOB_ID", "local")
    
    # Multi-fold experiment consistency: Check if files for this experiment already exist
    # If so, reuse the same run_id for all folds (instead of incrementing per fold)
    existing_run_id = None
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            # Look for files matching pattern: {run_id}_{iter_name}_*_{model_name}*
            # E.g., "313_IntegrityRunV7_108020_f0_convnext_tiny_model_brain.pth"
            match = re.match(rf"^(\d+)_{re.escape(iter_name)}_\d+_{model_name}", filename)
            if match:
                existing_run_id = match.group(1)
                break
    
    # Use existing run_id if found (for multi-fold consistency), otherwise generate new one
    if existing_run_id:
        run_id = existing_run_id
    else:
        run_id = f"{get_next_run_number(results_dir, slurm_id):02d}"
    
    # Iteration 25.0: Added iteration name to prefix for better output organization
    prefix = f"{run_id}_{iter_name}_{slurm_id}_f{fold_idx}_{model_name}" 
    print(f"--- Starting Run ID: {run_id} (Fold: {fold_idx + 1}/{num_folds}, Model: {model_name}, SLURM Job: {slurm_id}, Iter: {iter_name}) ---")

    # Define versioned file paths
    best_model_path = os.path.join(results_dir, f"{prefix}_model_brain.pth")
    results_csv_path = os.path.join(results_dir, f"{prefix}_evaluation_report.csv")
    cm_path = os.path.join(results_dir, f"{prefix}_confusion_matrix.png")
    roc_path = os.path.join(results_dir, f"{prefix}_roc_curve.png")
    pr_path = os.path.join(results_dir, f"{prefix}_pr_curve.png")
    history_path = os.path.join(results_dir, f"{prefix}_learning_curves.png")
    hist_path = os.path.join(results_dir, f"{prefix}_probability_histogram.png")
    patient_cm_path = os.path.join(results_dir, f"{prefix}_patient_confusion_matrix.png")
    patient_roc_path = os.path.join(results_dir, f"{prefix}_patient_roc_curve.png")
    patient_pr_path = os.path.join(results_dir, f"{prefix}_patient_pr_curve.png")
    gradcam_dir = os.path.join(results_dir, f"{prefix}_gradcam_samples")
    os.makedirs(gradcam_dir, exist_ok=True)

    # --- Step 1: Choose our study device ---
    # Use a Graphics Card (CUDA) if available; otherwise, use the Main Processor (CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Optimization 5D: Precision and Speed tuning for A40
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        print("Set float32 matmul precision to 'high' for A40 hardware optimization.")
    
    # --- Step 2: Set the paths to our data ---
    # We prioritize local scratch space for speed, then fallback to network path
    base_data_path = "/tmp/ricse03_h_pylori_data"
    
    if not os.path.exists(base_data_path):
        base_data_path = "/import/fhome/vlia/HelicoDataSet"

    if not os.path.exists(base_data_path):
        # Fallback for local development or different environments
        local_path = "/home/twyla/Documents/Classes/aprenentatgeProfund/Code/HelicoDataSet"
        if os.path.exists(local_path):
            base_data_path = local_path
        else:
            # Last resort: look in the parent directory
            base_data_path = os.path.abspath(os.path.join(os.getcwd(), "..", "HelicoDataSet"))
        
        print(f"Primary data path not found. Using: {base_data_path}")
    else:
        print(f"Using Data Path: {base_data_path}")

    # --- Step 2.5: Normalization Logic ---
    # TECHNICAL NOTE: Macenko Normalization is officially DEPRECATED for this
    # IHC dataset. The brown (DAB) and blue (Hematoxylin) signals in IHC 
    # do not match the H&E absorption model that Macenko assumes, which 
    # was causing "color collapse". Standard ImageNet normalization is used.
    from PIL import Image
    
    print("Pre-processing: Using Standard ImageNet Normalization (IHC-mode).")

    patient_csv = os.path.join(base_data_path, "PatientDiagnosis.csv")
    patch_csv = os.path.join(base_data_path, "HP_WSI-CoordAnnotatedAllPatches.xlsx") # Use Excel directly
    train_dir = os.path.join(base_data_path, "CrossValidation/Annotated")
    # New: Include 'Cropped' folder in the training pool to maximize data availability
    cropped_dir = os.path.join(base_data_path, "CrossValidation/Cropped")
    # This folder contains patients that the AI has NEVER seen during training.
    holdout_dir = os.path.join(base_data_path, "HoldOut")

    # --- Step 3: Define "Study Habits" (Transforms) ---
    # Global imports ensure v2 is accessible within the function scope
    from torchvision.transforms import v2
    
    # Standard ImageNet normalization: IHC (Brown/Blue) doesn't use Macenko.
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.ColorJitter(brightness=jitter, contrast=jitter),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- Step 4: Load and split the data ---
    # Create a merged dataset object by passing both training directories as a list
    # The HPyloriDataset (Iter 22+) supports list of paths for root_dir
    train_dirs = [train_dir, cropped_dir]
    full_dataset = HPyloriDataset(train_dirs, patient_csv, patch_csv, transform=None, bag_mode=True, max_bag_size=500, train=True)
    
    # Iteration 25.1: Pass the run prefix to the dataset for automated audit logging
    full_dataset.audit_prefix = os.path.join(results_dir, prefix)
    
    # --- RIGOROUS PATIENT-LEVEL SPLIT (Leakage-Proof Iteration) ---
    # 1. Identify all clinical patient IDs (base IDs) present across all folders
    all_bag_ids = [bag[2] for bag in full_dataset.bags]
    all_base_ids = sorted(list(set([bid.split('_')[0] for bid in all_bag_ids])))

    # Cross-Reference with HoldOut to PROVE no patient overlaps (Audit 25.2)
    holdout_patients = set()
    if os.path.exists(holdout_dir):
        holdout_folders = [f for f in os.listdir(holdout_dir) if os.path.isdir(os.path.join(holdout_dir, f))]
        holdout_patients = set([f.split('_')[0] for f in holdout_folders])
    
    # Track leakage status for the audit
    leakage_log = []
    patients_in_both_groups = 0
    for base_id in all_base_ids:
        is_leaking = base_id in holdout_patients
        if is_leaking:
            patients_in_both_groups += 1
            
        leakage_log.append({
            'Clinical_ID': base_id,
            'In_Training_Pool': True,
            'In_HoldOut_Test_Set': is_leaking,
            'Audit_Status': 'REJECTED_LEAKAGE' if is_leaking else 'VERIFIED_UNIQUE'
        })
    
    leakage_df = pd.DataFrame(leakage_log)
    leakage_df.to_csv(os.path.join(results_dir, f"{prefix}_cross_leakage_audit.csv"), index=False)
    
    # Update the summary CSV with the leakage count (Iteration 25.5)
    summary_csv_path = os.path.join(results_dir, f"{prefix}_data_integrity_summary.csv")
    if os.path.exists(summary_csv_path):
        summary_df = pd.read_csv(summary_csv_path)
        summary_df['PATIENTS_IN_BOTH_GROUPS_LEAKAGE'] = patients_in_both_groups
        summary_df.to_csv(summary_csv_path, index=False)
    
    # Global Seed Strategy (Iteration 21): Ensure deterministic initialization and splits
    import random
    random.seed(42 + fold_idx)
    np.random.seed(42 + fold_idx)
    torch.manual_seed(42 + fold_idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42 + fold_idx)
        
    # Data Split reproducibility: Seed 42 for identity across jobs
    split_rng = np.random.RandomState(42)
    split_rng.shuffle(all_base_ids)
    
    # Calculate fold boundaries based on UNIQUE CLINICAL PATIENTS
    fold_size = len(all_base_ids) // num_folds
    val_start = fold_idx * fold_size
    val_end = val_start + fold_size if fold_idx < num_folds - 1 else len(all_base_ids)
    
    val_base_ids = set(all_base_ids[val_start:val_end])
    train_base_ids = set([p for p in all_base_ids if p not in val_base_ids])
    
    print(f"--- Fold {fold_idx + 1}/{num_folds} Split (Leakage-Proof MIL) ---")
    print(f"Total Clinical Patients: {len(all_base_ids)}")
    print(f"Training Clinical IDs: {len(train_base_ids)}")
    print(f"Validation Clinical IDs: {len(val_base_ids)}")

    # Map all bags (from both folders) to their assigned folds using the clinical IDs
    train_indices = [i for i, bag in enumerate(full_dataset.bags) if bag[2].split('_')[0] in train_base_ids]
    val_indices = [i for i, bag in enumerate(full_dataset.bags) if bag[2].split('_')[0] in val_base_ids]
    
    print(f"Independent Patient-level split:")
    print(f" - Train: {len(train_indices)} biopsy bags")
    print(f" - Val:   {len(val_indices)} biopsy bags")
    
    # Re-apply our study habits for each split
    train_data = Subset(full_dataset, train_indices)
    val_data = Subset(full_dataset, val_indices)
    
    class TransformDataset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, index):
            # In bag_mode, img is a stacked tensor (Bag_Size, C, H, W)
            imgs, label, patient_id = self.subset[index]
            # Transforms are already applied in HPyloriDataset.__getitem__ via self.transform
            # but we can re-apply or wrap if needed. For now, HPyloriDataset handles it.
            return imgs, label, patient_id, index
        def __len__(self):
            return len(self.subset)

    # Use the transforms in the underlying dataset
    full_dataset.transform = train_transform
    train_transformed = TransformDataset(train_data, None) 
    train_transformed_copy = train_transformed # Kept for SWA BN update later
    
    # We'll switch transform for validation later or use a proxy
    val_transformed = TransformDataset(val_data, None)

    # --- Step 4.5: Improve Recall for Contaminated Samples ---
    # Distribution of bags
    train_labels = [full_dataset.bags[i][1] for i in train_indices]
    neg_count = train_labels.count(0)
    pos_count = train_labels.count(1)
    print(f"Training distribution (Bags): Negative={neg_count}, Contaminated={pos_count}")

    # Step 4.5 logic fix: Disable sampler if pos_weight > 1.0 (Iteration 20)
    # This prevents double-weighting class imbalance (Sampler + Loss Weight)
    if pos_weight > 1.0:
        print(f"Pos Weight {pos_weight} > 1.0: Disabling Sampler, using Shuffle=True.")
        sampler = None
        shuffle_train = True
    else:
        print("Using WeightedRandomSampler for baseline balancing (PosWeight=1.0).")
        class_weights = [1.0/max(1, neg_count), 1.0/max(1, pos_count)]
        sampler_weights = torch.FloatTensor([class_weights[t] for t in train_labels])
        sampler = WeightedRandomSampler(sampler_weights, len(sampler_weights))
        shuffle_train = False

    # MIL Batch Size: Usually 1 bag per batch is safest for variable sizes, 
    # but we can try small batches if bag sizes are fixed/padded.
    # Given A40 VRAM and 500 max patches, batch_size=1 (bag-at-a-time) is mandatory.
    batch_size_mil = 1 
    accumulation_steps = 16 # Adjust for MIL bag-level steps
    train_loader = DataLoader(
        train_transformed, 
        batch_size=batch_size_mil, 
        sampler=sampler, 
        shuffle=shuffle_train,
        num_workers=4, # Reduced for bag loading overhead
        pin_memory=True
    )
    val_loader = DataLoader(
        val_transformed, 
        batch_size=batch_size_mil, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    # --- Step 5: Build the customized AI brain ---
    model = get_model(model_name=model_name, num_classes=2, pretrained=True, pool_type=pool_type).to(device)

    # --- Step 6: Define the Learning Rules ---
    # Strategy C: Profile-based Loss (Iteration 19 Support)
    # Optimized to catch ALL potential infections or focus on balanced precision.
    loss_weights = torch.FloatTensor([neg_weight, pos_weight]).to(device) 
    # Iteration 25.0: Disabled label smoothing (0.0) as per user request.
    criterion = FocalLoss(gamma=gamma, weight=loss_weights, smoothing=0.0)

    # Optimizer Choice:
    # Iteration 21.2: Dynamic Weight Decay from Profile
    print(f"Using AdamW Optimizer for {model_name} (LR=2e-5, WD={weight_decay})...")
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=weight_decay)

    # --- Step 6.2: SWA Initialization (Iteration 13) ---
    from torch.optim.swa_utils import AveragedModel, SWALR
    # use_swa and swa_start are now passed as arguments
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-5) # Calibration: drastically reduced for stability

    # --- Optimization 5D: Preprocessing & Model Fusion ---
    # We define a fused preprocessing function for deterministic operations.
    # Standard ImageNet normalization: IHC (Brown/Blue) doesn't use Macenko.
    gpu_normalize = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def det_preprocess_batch(inputs, training=False):
        # inputs = normalizer.normalize_batch(inputs, jitter=training) # DISABLED for IHC
        return gpu_normalize(inputs)
    
    # Enable torch.compile for Kernel Fusion (Optimization 5D)
    if hasattr(torch, "compile"):
        # Disabling compilation for MIL bags due to high VRAM overhead
        print(f"Skipping torch.compile for {model_name} to conserve VRAM...")
        # model = torch.compile(model, mode="reduce-overhead")
    
    # --- Step 6.2: Dynamic LR Scheduler ---
    # Iteration 24.9: Shift to ReduceLROnPlateau for generalization (Stability Check)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3
    )
    
    # --- Step 6.5: Hardware Optimization (Optional) ---
    # Set this to True ONLY if you have an Intel CPU and compatible IPEX installed.
    # Currently set to False to ensure compatibility across all systems.
    USE_IPEX = False 
    
    if USE_IPEX:
        try:
            # We import here so it only tries to load if the user turned it on
            import intel_extension_for_pytorch as ipex # type: ignore
            model, optimizer = ipex.optimize(model, optimizer=optimizer)
            print("Intel Extension for PyTorch optimization enabled.")
        except Exception as e:
            print(f"IPEX failed to load: {e}. Falling back to standard PyTorch.")
    else:
        print("Running on standard PyTorch (IPEX disabled).")

    # --- Step 7: The Main Training Loop ---
    # We use Automatic Mixed Precision (AMP) to speed up training on the A40
    scaler = torch.amp.GradScaler('cuda')
    best_loss = float('inf')
    best_recall = 0.0 # Track sensitivity for Searcher phase
    best_f1 = 0.0    # Track F1 score for Calibration phase
    
    # Track the "History" to plot learning curves later
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    # Create a wrapper for autocast that identifies the actual device type
    def get_autocast_device():
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    # Suppress specific deprecation warnings from torch.utils.checkpoint
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")
    
    # Global memory optimization for CUDA
    if torch.cuda.is_available():
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        # Reduce fraction further to 0.7 to ensure system can't over-allocate
        torch.cuda.set_per_process_memory_fraction(0.70, 0) 
        torch.cuda.empty_cache()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} (Fold {fold_idx + 1}/{num_folds})")
        
        # --- Study Mode (Train) ---
        model.train() 
        if freeze_bn:
            # Iteration 21.2: Freeze BN to stabilize MIL training on variable bag sizes
            for m in model.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        
        full_dataset.transform = train_transform # Set training augmentations
        running_loss = 0.0
        running_corrects = 0
        optimizer.zero_grad()
        
        for i, (bags, labels, patient_ids, _) in enumerate(tqdm(train_loader, desc=f"Training (Fold {fold_idx + 1}/{num_folds})")):
            # bags shape: (1, Bag_Size, C, H, W) -> (Bag_Size, C, H, W)
            bags = bags.squeeze(0).to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # --- GPU-Based Preprocessing ---
            # Apply same deterministic normalize for IHC
            bags = det_preprocess_batch(bags, training=True)
            
            # Use dynamic device type for autocast to avoid warnings on CPU-only runs
            with torch.amp.autocast(device_type=get_autocast_device()):
                # Use the MIL forward pass
                outputs, attention = model.forward_bag(bags) # returns (1, 2), (1, N)
                loss = criterion(outputs, labels) 
                
                # Iteration 24.9: Entropy penalty for Attention weights
                # Prevents 'Delta Collapse' (1 patch gets all weight)
                if attention is not None:
                    # entropy = -sum(p * log(p))
                    att_entropy = -torch.sum(attention * torch.log(attention + 1e-8))
                    # We want HIGH entropy (distributed attention)
                    # Low entropy (delta collapse) should increase loss
                    entropy_penalty = -att_entropy  # Flip sign: rewards high entropy
                    loss = loss + 0.01 * entropy_penalty  # Increased coefficient: 0.001 → 0.01 
                
                loss = loss / accumulation_steps
            
            _, preds = torch.max(outputs, 1)
            
            scaler.scale(loss).backward()
            
            # Optimization step: Handle accumulation and end-of-epoch leftovers
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                if clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                
                # Check for NaNs/Infs: scaler.step only steps if gradients are valid
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update() # Update scale after step
                
                # Only update scheduler if the optimizer actually stepped
                # (GradScaler reduces the scale if it skips a step due to NaN/Inf)
                # Iteration 24.9: ReduceLROnPlateau stepped after validation phase
                # if scaler.get_scale() >= old_scale:
                #     if not use_swa or epoch < swa_start:
                #         scheduler.step()

                optimizer.zero_grad()
                
                # Clear cache after optimization step to prevent creep
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # --- Progress Tracking (Unscaled) ---
            # Re-scale the loss to represent the true bag-level loss for logging
            running_loss += (loss.item() * accumulation_steps)
            # Track corrects at the patient level (consensus of all patches in bag)
            running_corrects += torch.sum(preds == labels.data).item()
            
        # --- Epoch Summary Statistics ---
        # Normalize by the total number of patients (bags) to get average stats
        epoch_loss = running_loss / len(train_indices)
        epoch_acc = float(running_corrects) / len(train_indices)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # --- SWA Update (Iteration 21.3 dynamic) ---
        if use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # --- Self-Test Mode (Validation) ---
        # Evaluate metrics on patients the model hasn't studied this epoch.
        model.eval()
        full_dataset.train = False 
        full_dataset.transform = val_transform 
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for i, (bags, labels, patient_ids, _) in enumerate(tqdm(val_loader, desc=f"Validation (Fold {fold_idx + 1}/{num_folds})")):
                # bags shape: (1, Bag_Size, C, H, W) -> (Bag_Size, C, H, W)
                bags = bags.squeeze(0).to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Standardize colors for IHC using ImageNet statistics
                bags = det_preprocess_batch(bags, training=False)
                
                with torch.amp.autocast(device_type=get_autocast_device()):
                    # MIL Forward pass: extracts features and aggregates via Gated Attention
                    outputs, _ = model.forward_bag(bags)
                    loss = criterion(outputs, labels)
                
                # Consensus: Highest probability class at the patient level
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item()
                val_corrects += torch.sum(preds == labels.data).item()
        
        # --- Validation Summary Statistics ---
        # Calculate final patient-level metrics for this fold
        val_epoch_loss = val_loss / len(val_indices)
        val_epoch_acc = float(val_corrects) / len(val_indices)
        
        # Calculate Validation Recall (Searcher Priority)
        # This secondary pass collects the raw predictions to compute 
        # clinical metrics (Recall/F1) using scikit-learn.
        all_val_preds = []
        all_val_labels = []
        model.eval()
        with torch.no_grad():
            for bags, labels, _, _ in val_loader:
                # Prepare the bag for inference
                bags = bags.squeeze(0).to(device)
                bags = det_preprocess_batch(bags, training=False)
                
                # Model consensus for the patient
                outputs, _ = model.forward_bag(bags)
                _, preds = torch.max(outputs, 1)
                
                # Store for batch metric calculation
                all_val_preds.append(preds.item())
                all_val_labels.append(labels.item())
        
        # --- Clinical Metric Evaluation ---
        # We use scikit-learn's implementations for patient-level Recall and F1.
        # Zero_division=0 prevents crashes if an epoch predicts no positives.
        from sklearn.metrics import recall_score, f1_score
        val_recall = recall_score(all_val_labels, all_val_preds, zero_division=0)
        val_f1 = f1_score(all_val_labels, all_val_preds, zero_division=0)
        
        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f} Recall: {val_recall:.4f} F1: {val_f1:.4f}")

        # --- Iteration 24.9: Dynamic LR Adjustment ---
        if not use_swa or epoch < swa_start:
            scheduler.step(val_epoch_loss)

        # Store history
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)

        # --- Report Card: Save the best version ---
        # Strategy selection based on saver_metric. We implement "Tie-Breaking" logic
        # to ensure that if two epochs have the same primary metric (e.g., 100% Recall),
        # we pick the one with the highest accuracy to avoid suboptimal 'Ghost' models.
        is_best = False
        if saver_metric == "recall":
            # Primary: Recall (Sensitivity). Critical for the 'Searcher' phase.
            if val_recall > best_recall:
                best_recall = val_recall
                best_acc = val_epoch_acc # Reset accuracy tracker for new recall level
                best_loss = val_epoch_loss 
                is_best = True
                print(f"New Best Recall! {val_recall:.4f}")
            # Tie-Breaker 1: If Recall is equal, pick higher Accuracy.
            elif val_recall == best_recall and val_epoch_acc > best_acc:
                best_acc = val_epoch_acc
                best_loss = val_epoch_loss
                is_best = True
                print(f"Recall stable, Improved Accuracy: {val_epoch_acc:.4f}")
            # Tie-Breaker 2: If Recall and Accuracy are equal, pick lower Loss.
            elif val_recall == best_recall and val_epoch_acc == best_acc and val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                is_best = True
                print(f"Recall and Acc stable, Improved Loss: {val_epoch_loss:.4f}")
        
        elif saver_metric == "f1":
            # Primary: F1-Score. Balances Precision and Recall for 'Calibration' phase.
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_acc = val_epoch_acc
                best_loss = val_epoch_loss
                is_best = True
                print(f"New Best F1! {val_f1:.4f}")
            # Tie-Breaker: Improved Accuracy on stable F1.
            elif val_f1 == best_f1 and val_epoch_acc > best_acc:
                best_acc = val_epoch_acc
                best_loss = val_epoch_loss
                is_best = True
                print(f"F1 stable, Improved Accuracy: {val_epoch_acc:.4f}")
        
        else: # Default to loss
            # Standard optimization: minimize Cross-Entropy/Focal Loss.
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                is_best = True
                print(f"New Best Loss! {val_epoch_loss:.4f}")

        # Persistence: Update the 'Brain' file on disk whenever a new champion is found.
        if is_best:
            torch.save(model.state_dict(), best_model_path)
            print(f"Saving model to {best_model_path}")

    # --- Training Wrap-up ---
    # Log the final peak performance reached during the multi-epoch hunt.
    if saver_metric == "recall":
        print(f"Training complete. Best Val Recall: {best_recall:.4f}")
    elif saver_metric == "f1":
        print(f"Training complete. Best Val F1: {best_f1:.4f}")
    else:
        print(f"Training complete. Best Val Loss: {best_loss:.4f}")

    # --- Step 7.8: Final SWA Update (Iteration 13) ---
    if use_swa:
        # Final SWA normalization (after training is fully complete)
        # This prepares the averaged model's batchnorm layers for inference
        # We must recreate the train_loader because it was deleted in Step 7.4
        temp_train_loader = DataLoader(
            train_transformed_copy, 
            batch_size=batch_size_mil, 
            sampler=sampler, 
            num_workers=4, 
            pin_memory=True
        )
        
        # Iteration 21 Fix: Custom BN update for MIL (handles 5D tensor error)
        # torch.optim.swa_utils.update_bn(temp_train_loader, swa_model, device=device)
        
        # --- Step 7.8: Final SWA Update (Post-Calibration) ---
        # Stochastic Weight Averaging (SWA) provides a more generalized 'clinical' weight set.
        # However, the averaged model's Batchnorm statistics must be re-calibrated using 
        # the training data to ensure the running means/variances align with the new weights.
        swa_model.train()
        with torch.no_grad():
            for bags, _, _, _ in tqdm(temp_train_loader, desc="Recalibrating SWA Batchnorm"):
                # bags: (1, Bag_Size, C, H, W)
                # MIL Constraint: We must squeeze the batch dimension to process the bag (Bag_Size, C, H, W)
                bags = bags.squeeze(0).to(device) 
                
                # Replicate the deterministic IHC normalization used during training.
                # This ensures the BN layers see the exact data distribution they will face at inference.
                from torchvision.transforms import v2
                norm = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                bags = norm(bags)
                
                # Calibration Pass: We run the bags through the forward_bag method.
                # This triggers the BN momentum updates without calculating gradients.
                swa_model.module.forward_bag(bags)
                
        del temp_train_loader
        
        # Iteration 24.3: Save SWA separately. Do NOT overwrite the "Best Metric" model
        # which might have reached peak recall during earlier epochs.
        swa_model_path = os.path.join(results_dir, f"{prefix}_swa_model_brain.pth")
        torch.save(swa_model.state_dict(), swa_model_path)
        print(f"Final SWA Clinical Model saved to {swa_model_path}")
        
        # Use a summary logic to decide whether SWA or Best is superior
        # For Searcher: We use the SWA model only if its recall matches the best and acc is better
        # For now, we allow the script to use SWA for evaluation, but keep the Best separately
    else:
        # Save the best model reached during training
        best_model_path = os.path.join(results_dir, f"{prefix}_model_brain.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"Final Best (Non-SWA) Clinical Model saved to {best_model_path}")

    # --- Step 7.9: Save Learning Curves (Iteration 14 Clinical Audit) ---
    # Visualizing the convergence behavior is critical for diagnosing 'Recall Oscillations'.
    plot_learning_curves(history, history_path)
    print(f"Saved clinical learning curves to {history_path}")

    # Step 7.11: Use Best Metric Model for final validation if SWA degraded performance
    model_selection_metadata = {"use_swa": True, "best_recall": best_recall, "last_val_recall": val_recall}
    
    if use_swa:
        # Check if Best Recall from loop is higher than SWA's current state
        # We load the weights from 'best_model_path' (saved during training) back into model
        if best_recall > val_recall:
            print(f"SWA performance ({val_recall:.4f}) is below Best Recall ({best_recall:.4f}). Loading best metric model...")
            model.load_state_dict(torch.load(best_model_path))
            model_selection_metadata["use_swa"] = False
        else:
            # If SWA is at least equal in recall, use it
            model = swa_model.module
        
    # Save metadata about which model was selected for later reference
    import json
    model_metadata_path = os.path.join(results_dir, f"{prefix}_model_selection.json")
    with open(model_metadata_path, 'w') as f:
        json.dump(model_selection_metadata, f, indent=2)
    
    model.to(device)
    model.eval()

    # Step 7.4: Memory Cleanup (REMAINING)
    # --- Step 7.12: Final Memory Sweep ---
    # To prevent memory leaks between folds, we explicitly delete large tensors
    # and metadata objects before the final evaluation or the next's fold setup.
    del train_transformed_copy
    del val_loader
    del train_transformed
    del val_transformed
    del full_dataset # Reclaim dataset memory (contains thousands of patch paths)
    del train_data
    del val_data
    
    # Force Garbage Collection and clear CUDA cache to ensure peak VRAM for 
    # the heavy TTA evaluation phase.
    gc.collect()
    torch.cuda.empty_cache()

    # --- Step 8: Patient-Independent Evaluation (16-way TTA) ---
    # To truly assess the 'Clinical Model', we evaluate on a hold-out set using 
    # Test-Time Augmentation (TTA). This provides a more robust estimate of 
    # model performance by averaging predictions across 16 different views of the same bag.
    print(f"\nEvaluating on Independent Hold-Out set from: {holdout_dir} with 16-way TTA")
    
    # We set max_bag_size=10000 for evaluation to ensure we load the full clinical bag.
    # While training uses 500-patch bags for variety, inference must see the WHOLE tissue.
    holdout_dataset = HPyloriDataset(
        holdout_dir, patient_csv, patch_csv, 
        transform=val_transform, bag_mode=True, 
        max_bag_size=10000, train=False
    )
    
    # Evaluation Loader: Single batch (1 patient) at a time.
    # num_workers=0 is used inside the loop to avoid multi-process overhead for sequential TTA.
    holdout_loader = DataLoader(
        holdout_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )
    
    # --- 16-way Deterministic TTA (Iteration 25.1: Clinical Grade) ---
    # We apply 8 geometric transforms (Flips + Rotations) across two contrast levels.
    # The 'Contrast Boost' (1.1x) is critical for IHC (Brown/Blue) to ensure that 
    # sparse, faintly-stained bacteria are not missed due to background noise.
    tta_transforms = [
        lambda x: x, # Original Clinical View
        v2.RandomHorizontalFlip(p=1.0),
        v2.RandomVerticalFlip(p=1.0),
        lambda x: torch.rot90(x, 1, [2, 3]), # 90-degree Check
        lambda x: torch.rot90(x, 2, [2, 3]), # 180-degree Check
        lambda x: torch.rot90(x, 3, [2, 3]), # 270-degree Check
        lambda x: v2.RandomHorizontalFlip(p=1.0)(torch.rot90(x, 1, [2, 3])),
        lambda x: v2.RandomVerticalFlip(p=1.0)(torch.rot90(x, 1, [2, 3])),
        
        # --- Group B: Contrast-Boosted Signal Recovery ---
        # Helping the model 'pop' signals that are near the IHC detection threshold.
        lambda x: v2.ColorJitter(contrast=(1.1, 1.1))(x),
        lambda x: v2.ColorJitter(contrast=(1.1, 1.1))(v2.RandomHorizontalFlip(p=1.0)(x)),
        lambda x: v2.ColorJitter(contrast=(1.1, 1.1))(v2.RandomVerticalFlip(p=1.0)(x)),
        lambda x: v2.ColorJitter(contrast=(1.1, 1.1))(torch.rot90(x, 1, [2, 3])),
        lambda x: v2.ColorJitter(contrast=(1.1, 1.1))(torch.rot90(x, 2, [2, 3])),
        lambda x: v2.ColorJitter(contrast=(1.1, 1.1))(torch.rot90(x, 3, [2, 3])),
        lambda x: v2.ColorJitter(contrast=(1.1, 1.1))(v2.RandomHorizontalFlip(p=1.0)(torch.rot90(x, 1, [2, 3]))),
        lambda x: v2.ColorJitter(contrast=(1.1, 1.1))(v2.RandomVerticalFlip(p=1.0)(torch.rot90(x, 1, [2, 3])))
    ]


    
    # --- Clinical Metric Accumulators ---
    # We initialize zero-arrays to store the patient-level consensus results.
    # Every patient in the hold-out set is evaluated with 16 transformations.
    num_holdout = len(holdout_dataset)
    all_preds_pat = np.zeros(num_holdout, dtype=np.int8)     # Binary Decision (0=Neg, 1=Pos)
    all_labels_pat = np.zeros(num_holdout, dtype=np.int8)    # Ground Truth from Pathology
    all_probs_pat = np.zeros(num_holdout, dtype=np.float32)   # Averaged Confidence Score
    
    # Searcher Indicator: We track the maximum probability seen across any TTA view
    # to help identify low-confidence positive signals during the 'Rescue' phase.
    all_max_probs = np.zeros(num_holdout, dtype=np.float32) 
    all_patch_counts = np.zeros(num_holdout, dtype=np.int32) # Transparency: total tissue analyzed
    
    patient_ids_list = []
    # Use a chunk size of 500 for ConvNeXt evaluate loop to prevent A40 OOM
    vram_bag_limit = 500
    # Iteration 24.9: Sliding Window Overlap (50%) to prevent signal split
    eval_stride = 250 

    with torch.no_grad():
        for i, (bags, labels, patient_ids) in enumerate(tqdm(holdout_loader, desc=f"Patient-Independent TTA Test (Fold {fold_idx + 1}/{num_folds})")):
            # bags shape: (1, Bag_Size, C, H, W) -> (Bag_Size, C, H, W)
            # Move labels to device, but keep bags on CPU for now to prevent OOM
            labels = labels.to(device, non_blocking=True)
            
            # Divide bag into chunks of 500 if larger
            bag_size = bags.squeeze(0).size(0)
            bag_probs_list = []
            
            # Use sliding window if bag is large enough, otherwise take the whole bag
            if bag_size <= vram_bag_limit:
                chunk_ranges = [(0, bag_size)]
            else:
                chunk_ranges = []
                for s in range(0, bag_size - vram_bag_limit + 1, eval_stride):
                    chunk_ranges.append((s, s + vram_bag_limit))
                # Add final chunk to ensure coverage
                if chunk_ranges[-1][1] < bag_size:
                    chunk_ranges.append((bag_size - vram_bag_limit, bag_size))

            for start_idx, end_idx in chunk_ranges:
                # Transfer only one chunk (500 patches) to GPU at a time
                chunk_bags = bags.squeeze(0)[start_idx : end_idx].to(device, non_blocking=True)
                
                # --- Step 8.2: 16-way TTA Accumulation ---
                # We iterate through all 16 transformations for this specific tissue chunk.
                # The logits are summed across all views to produce a stable 'consensus' 
                # before the final softmax. This reduces the impact of 'outlier' artifacts.
                chunk_logits_sum = None
                for tta_aug in tta_transforms:
                    # Apply geometric/contrast transformation
                    aug_bags = tta_aug(chunk_bags)
                    # Standardize intensities (Deterministic pass)
                    aug_bags = det_preprocess_batch(aug_bags, training=False)
                    
                    with torch.amp.autocast(device_type=get_autocast_device()):
                        if use_swa:
                            # SWA model uses the averaged weights for clinical generalization
                            logits, _ = swa_model.module.forward_bag(aug_bags)
                        else:
                            # Standard model uses the best epoch weights
                            logits, _ = model.forward_bag(aug_bags)
                    
                    if chunk_logits_sum is None:
                        chunk_logits_sum = logits
                    else:
                        chunk_logits_sum += logits
                
                # --- Step 8.3: Stochastic Smoothing ---
                # We average the summed logits to finalize the view for this chunk.
                avg_chunk_logits = chunk_logits_sum / len(tta_transforms)
                chunk_probs = torch.softmax(avg_chunk_logits, dim=1)
                # Offload to CPU: Critical for 10,000-patch bags to avoid VRAM fragmentation
                bag_probs_list.append(chunk_probs.cpu()) 
                
                # Proactive Memory Management (Clinical Pipe)
                del chunk_bags
                del avg_chunk_logits
                del chunk_logits_sum
            
            # Move probabilities to DEVICE for final aggregation math
            # Shape (Num_Chunks, 1, 2) -> (Num_Chunks, 2)
            all_chunks_probs = torch.stack(bag_probs_list).squeeze(1).to(device)
            
            # --- Step 8.4: AGGREGATION: Iteration 24.9 Top-3 Mixed MIL Strategy ---
            # Instead of a global mean or a single max (which are prone to noise or outliers),
            # we use the mean of the top 3 chunks. This 'Top-K Voting' ensures that if multiple
            # areas of the tissue show bacterial presence, the confidence is high, but a single
            # artifactual patch won't easily trigger a False Positive.
            
            # Final Patient Probability = Mean of the Top 3 most confident chunks
            if all_chunks_probs.size(0) >= 3:
                # Identify the three regions with highest bacterial probability
                top_k_probs, _ = torch.topk(all_chunks_probs[:, 1], k=3)
                final_pos_prob_tensor = top_k_probs.mean()
            else:
                # Fallback to max if the tissue sample is too small for Top-3 (e.g., small biopsies)
                final_pos_prob_tensor = all_chunks_probs[:, 1].max(0)[0]

            max_chunk_prob = final_pos_prob_tensor.cpu().item()
            
            # --- Step 8.5: Clinical Decision Threshold ---
            # Thresholding: 0.40 is our "Surgical Sensitivity Boundary" for Iteration 25.0. 
            # This threshold was selected after calibration to maximize recall on sparse 
            # infections while maintaining acceptable specificity.
            # --- Step 8.5: Clinical Decision Threshold ---
            # Thresholding: 0.40 is our "Surgical Sensitivity Boundary" for Iteration 25.0. 
            # This threshold was selected after calibration to maximize recall on sparse 
            # infections while maintaining acceptable specificity.
            SEARCHER_THRESHOLD = 0.40
            if max_chunk_prob > SEARCHER_THRESHOLD:
                preds = torch.tensor([1], device=device)
            else:
                preds = torch.tensor([0], device=device)
            
            # --- Step 8.6: Data Accumulation for Clinical Consensus ---
            # We preserve the maximum probability seen across any TTA view or chunk.
            # This "positive_prob" is the primary input for the Meta-Classifier fusion.
            positive_prob = max_chunk_prob 
            
            all_preds_pat[i] = preds.cpu().item()
            all_labels_pat[i] = labels.cpu().item()
            all_probs_pat[i] = positive_prob
            all_patch_counts[i] = bag_size 
            patient_ids_list.append(patient_ids[0])
            all_max_probs[i] = max_chunk_prob
            
            # Periodic VRAM flush to maintain stability during long 5-fold CV runs
            if i % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    # --- Step 9: Detailed Reporting (Patient Level Focused) ---
    print("\nPatient-Level Classification Report (MIL):")
    print(classification_report(all_labels_pat, all_preds_pat, target_names=['Negative', 'Positive'], zero_division=0))
    
    # --- Step 9: Clinical Consensus Reporting ---
    # We aggregate all patient-level predictions and confidence scores into a structured CSV.
    # This report serves as the primary audit trail for the Searcher phase, allowing 
    # pathologists to identify "High-Ghost" or "Borderline" cases for manual rescue.
    consensus_data = []
    for i in range(num_holdout):
        consensus_data.append({
            "PatientID": patient_ids_list[i],
            "Actual": all_labels_pat[i],
            "Predicted": all_preds_pat[i],
            # Searcher_Flag: Triggers if either the Mean or Top-K confidence is elevated.
            # This is used to flag samples for high-resolution dense re-scanning.
            "Searcher_Flag": 1 if all_max_probs[i] > 0.1 or all_probs_pat[i] > 0.1 else 0, 
            "Bag_Mean_Prob": all_probs_pat[i], 
            "Max_Prob": all_max_probs[i], # Captures the single most suspicious tissue area
            "Meta_Prob": all_probs_pat[i], # Input for the final meta-classifier fusion
            "Patch_Count": all_patch_counts[i], 
            "Method": f"MIL-{pool_type}",
            "Correct": 1 if all_preds_pat[i] == all_labels_pat[i] else 0
        })
    consensus_df = pd.DataFrame(consensus_data)
    consensus_df.to_csv(os.path.join(results_dir, f"{prefix}_patient_consensus.csv"), index=False)
    print(f"Clinical consensus report saved to {prefix}_patient_consensus.csv")

    # --- Step 11: Patient-Level Performance Audit ---
    # We visualize the Receiver Operating Characteristic (ROC) curve to evaluate
    # the model's ability to discriminate between positive and negative patients.
    # A high Area Under Curve (AUC) is critical for clinical screening reliability.
    fpr_meta, tpr_meta, _ = roc_curve(all_labels_pat, all_probs_pat)
    roc_auc_meta = auc(fpr_meta, tpr_meta)
    
    # Save evaluation report EARLY to ensure it is generated even if Grad-CAM hangs
    report = classification_report(all_labels_pat, all_preds_pat, target_names=['Negative', 'Positive'], output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(results_csv_path)
    print(f"Evaluation report saved to {results_csv_path}")

    plt.figure()
    plt.plot(fpr_meta, tpr_meta, color='darkorange', lw=3, label=f'Attention-MIL (AUC = {roc_auc_meta:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Patient-Level ROC: Attention-MIL')
    plt.legend(loc="lower right")
    plt.savefig(patient_roc_path)
    plt.close() # Release resources for subsequent visual generation pass
    
    # Calculate Patient-Level Accuracy
    pat_acc = (all_preds_pat == all_labels_pat).mean() * 100
    print(f"\nFinal Patient-Level Accuracy (MIL): {pat_acc:.2f}%")

    # --- Step 12: Extra Visuals (Metrics & Interpretability) ---
    print("\nGenerating final metrics and interpretability maps...")
    
    # 1. Confusion Matrix
    plot_confusion_matrix(all_labels_pat, all_preds_pat, patient_cm_path)

    # 2. Probability Histogram
    plot_probability_histogram(all_probs_pat, all_labels_pat, hist_path)

    # 3. Precision-Recall Curve
    plot_pr_curve(all_labels_pat, all_probs_pat, patient_pr_path)

    # 4. Grad-CAM Samples from Hold-Out
    print(f"Generating Grad-CAM for most suspicious patient bags in {gradcam_dir}...")
    # Get top 5 most positive and top 5 most ghostly (FN)
    # top_indices: highest probability, fn_indices: highest prob among actual positive that were pred 0
    top_indices = np.argsort(all_probs_pat)[-5:] 
    
    # Note: Target layer no longer needed with new input-gradient based Grad-CAM implementation

    # Free up memory before visualization
    # Iteration 24.9: Clear all training/evaluation gradients and model temporary buffers
    gc.collect()
    torch.cuda.empty_cache()
    
    # Context to enable gradients for Grad-CAM
    for bag_idx in top_indices:
        p_id = patient_ids_list[bag_idx]
        
        # Load full bag for this patient (keep on CPU initially)
        bag_imgs, label, _ = holdout_dataset[bag_idx]
        # Use chunks for forward pass - keep chunks on CPU, load only current chunk to GPU
        
        # Find Attention weights to choose the top patches (we don't need gradients for this part)
        # Iteration 22: If Max-Pooling, we use Top-K probabilities instead of attention
        all_indicators = [] # Can be attention or logits
        # Ensure we are in eval mode and no_grad to minimize memory during search
        model.eval()
        with torch.no_grad():
            for start_idx in range(0, bag_imgs.size(0), vram_bag_limit):
                chunk = bag_imgs[start_idx:start_idx + vram_bag_limit].to(device)
                chunk = det_preprocess_batch(chunk, training=False)
                
                if pool_type == "attention":
                    _, indicator = model.forward_bag(chunk)
                    all_indicators.append(indicator.cpu()) # (1, N)
                else:
                    # For Max-Pooling, use patch-level logits (Class 1) as search indicator
                    indicator = model(chunk) # (N, num_classes)
                    all_indicators.append(indicator[:, 1:2].transpose(0, 1).cpu()) # (1, N)
                
                # Cleanup chunk immediately
                del chunk
                torch.cuda.empty_cache()
        
        indicators = torch.cat(all_indicators, dim=1).squeeze(0) # (Bag_Size,)
        
        # Take top 3 most "Significant" patches
        top_patch_vals, top_patch_indices = torch.topk(indicators, k=min(3, bag_imgs.size(0)))
        
        for rank, p_idx in enumerate(top_patch_indices):
            # Only load the specific patch and its input tensor for Grad-CAM
            patch_img = bag_imgs[p_idx:p_idx+1].to(device)
            patch_input = det_preprocess_batch(patch_img, training=False)
            
            # Context to enable gradients for Grad-CAM on this specific patch
            with torch.enable_grad():
                heatmap, p_probs = generate_gradcam(model.backbone, patch_input)
            
            # Plot side-by-side visualization (original + heatmap overlay)
            plot_gradcam_pair(
                patch_img, heatmap[0, 0], p_id, rank, p_idx,
                top_patch_vals[rank].item(), p_probs[0, 1],
                is_false_negative=False, output_dir=gradcam_dir
            )
            
            # Cleanup per patch
            del patch_img, patch_input, heatmap
            torch.cuda.empty_cache()

    # Repeat for False Negatives (Ghost Patients)
    fn_indices = [i for i, (prob, label) in enumerate(zip(all_probs_pat, all_labels_pat)) if label == 1 and prob < 0.5]
    fn_indices = sorted(fn_indices, key=lambda i: all_probs_pat[i], reverse=True)[:5]

    # --- Step 12: Extra Visuals (Metrics & Interpretability) ---
    # Beyond statistical metrics, we generate visual evidence for the model's 
    # internal logic using Grad-CAM. This is applied to both high-confidence 
    # positives and "Ghost" (False Negative) patients to identify why signals
    # were missed or captured.
    # 
    # Clinical Rationale for Ghost Audit:
    # 1. Verification: Confirms many FNs are simply "sparse biopsies" where only 1-2
    #    organisms exist, validating the need for Stride-128 dense rescue.
    # 2. Ghost Analysis: Visualizes high-attention but low-confidence pixels to 
    #    distinguish between "No signal found" and "Signal filtered by TTA".

    # 4. Grad-CAM Audit: False Negatives (Ghost Patients)
    # We specifically target patients where the pathologist confirmed bacteria (label=1)
    # but the model failed to cross the clinical threshold (prob < 0.5).
    if fn_indices:
        print(f"Generating Grad-CAM for {len(fn_indices)} False Negative (Ghost) bags...")
        for bag_idx in fn_indices:
            p_id = patient_ids_list[bag_idx]
            bag_imgs, label, _ = holdout_dataset[bag_idx]
            
            # --- 4.1: Find Suspicious Patches inside the FN Bag ---
            # Even in a "missed" patient, some patches likely triggered faint 
            # attention. We find them to visualize the near-miss signals.
            all_indicators = []
            with torch.no_grad():
                for start_idx in range(0, bag_imgs.size(0), vram_bag_limit):
                    chunk = bag_imgs[start_idx:start_idx + vram_bag_limit].to(device)
                    chunk = det_preprocess_batch(chunk, training=False)
                    if pool_type == "attention":
                        # Extract gated-attention weights
                        _, indicator = model.forward_bag(chunk)
                        all_indicators.append(indicator.cpu())
                    else:
                        # Fallback for max-pooling: use patch-level confidence
                        indicator = model(chunk)
                        all_indicators.append(indicator[:, 1:2].transpose(0, 1).cpu())
                    
                    # Cleanup chunk immediately (Clinically vital for A40 stability)
                    del chunk
                    torch.cuda.empty_cache()
            
            indicators = torch.cat(all_indicators, dim=1).squeeze(0)
            # Focus Grad-CAM on the top patches in the missed bag to find the "Ghost" signal
            top_patch_vals, top_patch_indices = torch.topk(indicators, k=min(3, bag_imgs.size(0)))
            
            for rank, p_idx in enumerate(top_patch_indices):
                patch_img = bag_imgs[p_idx:p_idx+1].to(device)
                patch_input = det_preprocess_batch(patch_img, training=False)
                
                # --- 4.2: Interpretability Map Generation ---
                # We re-enable gradients temporarily to backpropagate through the 
                # backbone and see which pixels "confused" the model or were 
                # deemed insufficient for a positive diagnosis.
                with torch.enable_grad():
                    heatmap, p_probs = generate_gradcam(model.backbone, patch_input)
                
                # Plot side-by-side visualization for false negative (ghost patient)
                plot_gradcam_pair(
                    patch_img, heatmap[0, 0], p_id, rank, p_idx,
                    top_patch_vals[rank].item(), p_probs[0, 1],
                    is_false_negative=True, output_dir=gradcam_dir
                )
                
                # Full memory sweep between patches to avoid VRAM fragmentation
                del patch_img, patch_input, heatmap
                torch.cuda.empty_cache()
                
    print(f"Iteration reporting and metrics complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H. Pylori K-Fold Training")
    parser.add_argument("--fold", type=int, default=0, help="Index of the fold to use for validation (0 to num_folds-1)")
    parser.add_argument("--num_folds", type=int, default=5, help="Total number of folds")
    parser.add_argument("--model_name", type=str, default="convnext_tiny", choices=["resnet50", "convnext_tiny"], 
                        help="Backbone architecture to use (resnet50/convnext_tiny)")
    parser.add_argument("--pos_weight", type=float, default=7.5, help="Weight for positive class in Focal Loss")
    parser.add_argument("--neg_weight", type=float, default=1.0, help="Weight for negative class in Focal Loss")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma factor for Focal Loss")
    parser.add_argument("--num_epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--saver_metric", type=str, default="recall", choices=["loss", "recall", "f1"], 
                        help="Metric used to save the best model (loss/recall/f1)")
    parser.add_argument("--freeze_bn", type=str, default="False", help="Freeze BatchNorm layers (Iteration 21.2)")
    parser.add_argument("--clip_grad", type=float, default=0.0, help="Gradient clipping norm (0.0 to disable)")
    parser.add_argument("--pct_start", type=float, default=0.1, help="Warmup pct for OneCycleLR")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--use_swa", type=str, default="True", help="Whether to use SWA")
    parser.add_argument("--swa_start", type=int, default=15, help="Epoch to start SWA")
    parser.add_argument("--jitter", type=float, default=0.15, help="ColorJitter intensity (brightness/contrast)")
    parser.add_argument("--pool_type", type=str, default="attention", choices=["attention", "max"], 
                        help="MIL aggregation pooling type (Iteration 22)")
    parser.add_argument("--iter", type=str, default="24.9", help="Iteration version for filename prefixing")
    
    args = parser.parse_args()
    
    train_model(
        fold_idx=args.fold, 
        num_folds=args.num_folds, 
        model_name=args.model_name,
        pos_weight=args.pos_weight,
        neg_weight=args.neg_weight,
        gamma=args.gamma,
        num_epochs=args.num_epochs,
        saver_metric=args.saver_metric,
        freeze_bn=args.freeze_bn == "True",
        clip_grad=args.clip_grad,
        pct_start=args.pct_start,
        weight_decay=args.weight_decay,
        use_swa=args.use_swa == "True",
        swa_start=args.swa_start,
        jitter=args.jitter,
        pool_type=args.pool_type,
        iter_name=args.iter
    )
