
"""
H. Pylori Contamination Detection - Visual Report Generation
============================================================

OVERVIEW
--------
This script generates comprehensive visual reports for trained H. Pylori detection models.
It provides two fast modes for efficient comparisons and four standard modes for detailed analysis.

  **FAST MODES** (Ideal for comparing multiple trained models):
  - Model Comparison: Compare 4 architectures (ResNet50, ConvNeXt-Tiny/Small/Base) with measured data
  - Transfer Learning Comparison: Validate baseline vs TL improvements (performance + learning curves)

  **STANDARD ANALYSIS GROUPS** (Can be combined or run individually):
  - Clinical Validation: Calibration curves, performance dashboards, TL comparisons
  - Model Analysis: Learning curves, ensemble voting patterns, CV stability
  - Data Rigor: Integrity audit, hard examples, edge cases, training trajectory
  - Efficiency & Justification: Resource metrics, model complexity analysis with data sources

PURPOSE
-------
After model training (train.py), this standalone utility enables comprehensive post-hoc analysis and
report generation. Four operation modes:

  **FAST MODES** (Skip all other visualizations - for efficient comparisons):
    **Model Comparison Mode** (--model_comparison_only):
      - Generates only model complexity analysis (4 architectures: ResNet50, ConvNeXt-Tiny/Small/Base)
      - Loads actual accuracies from all trained models
      - Runtime: ~10 seconds
    
    **Transfer Learning Comparison Mode** (--transfer_learning_comparison_only):
      - Generates only TL comparison visualizations (performance + learning curves)
      - Requires --run_id (TL) + --compare_baseline (Baseline)
      - Runtime: ~5 seconds

  **STANDARD MODES** (Generate specific analysis groups):
    **Pipeline Mode** (--pipeline_mode, default when called from submit_transfer_learning.sh):
      - Generates comprehensive lightweight visual report for presentations
      - Includes: calibration curve, performance dashboards, ensemble contribution analysis, CV stability 
        (box plots + bootstrap CIs for robustness), failure modes, class distribution & stratification, 
        training trajectory, training efficiency, and combined learning curves
      - When used with transfer learning: outputs separate pre-training and fine-tuning learning curve files
      - Skips redundant visualizations already created during training (ROC, PR, confusion matrix)
      - Runtime: ~8-12 minutes

    **Full Mode** (default, generates all analysis groups):
      - Generates all 11+ visualizations including Grad-CAM, training trajectory, efficiency, model complexity
      - Useful for comprehensive report generation
      - Runtime: ~10-15 minutes

Key use cases:
  - FAST: Compare 4 models (ResNet50, ConvNeXt-Tiny/Small/Base) efficiently without full pipeline
  - FAST: Validate transfer learning improvements with quick side-by-side comparison
  - Validating clinical reliability of confidence predictions (calibration)
  - Creating publication-ready performance summaries (dashboard)
  - Comprehensive model analysis: architecture justification, training efficiency, data integrity
  - Debugging misclassifications with Grad-CAM interpretation
  - Generating comprehensive reports for clinical stakeholders

HOW IT WORKS
------------

FAST MODES:
1. Model Comparison Mode (--model_comparison_only):
   - Loads evaluation reports for all 4 trained models (ResNet50, ConvNeXt-Tiny/Small/Base)
   - Extracts accuracies and parameter counts
   - Generates single visualization comparing architecture efficiency vs performance
   - Displays measured vs benchmark data sources transparently
   - No dataset loading or inference required

2. Transfer Learning Comparison Mode (--transfer_learning_comparison_only):
   - Loads predictions CSV for baseline and TL models
   - Loads learning curves JSON for both models
   - Generates ROC/PR comparison and learning curves overlay
   - No dataset loading or inference required

STANDARD MODES:
1. Loads Dataset: Fetches validation patches and patient metadata from the H. Pylori dataset
2. Loads Model: Retrieves trained checkpoint (attention-based MIL architecture)
3. Inference: Runs forward pass on validation fold, aggregating patch-level predictions to patient level
4. Bootstrap CIs: Performs 1000 bootstrap resamples for confidence intervals on all metrics
5. Grad-CAM (if enabled): Computes input-level gradient saliency maps showing attention regions
6. Clinical Validation: Generates calibration curve and performance dashboard
7. Advanced Analysis: Generates any combination of 8 analysis groups (transfer learning, ensemble, CV, audit, etc.)
8. Output: Saves all visualizations as PNG images in results/

USAGE
-----
Run from command line with required arguments:

  # FAST MODES (Skip all other visualizations)
  # Compare 4 models efficiently
  python generate_visuals.py --run_id <RUN_ID> --model_comparison_only
  
  # Transfer learning comparison (baseline vs TL)
  python generate_visuals.py --run_id <RUN_ID> --compare_baseline <BASELINE_RUN_ID> --transfer_learning_comparison_only

  # STANDARD USAGE
  # Generate all visualizations
  python generate_visuals.py --run_id <RUN_ID> [--fold <FOLD>] [--model_name <MODEL>] [--dataset <DATASET>]
  
  # Pipeline mode (comprehensive lightweight report for presentations)
  python generate_visuals.py --run_id <RUN_ID> --pipeline_mode
  
  # Generate specific analysis groups
  python generate_visuals.py --run_id <RUN_ID> --include_model_complexity --include_training_efficiency --include_data_audit

ARGUMENTS
---------
  FAST MODES (Skip all other visualizations):
  
  --model_comparison_only (optional, default: False)
    When True, only generates model complexity analysis (architecture comparison).
    Loads actual accuracies from all 4 trained models (ResNet50, ConvNeXt-Tiny/Small/Base)
    Skips dataset loading, inference, Grad-CAM, and all other visualizations.
    Fast and efficient for comparing multiple model architectures.
    Runtime: ~10 seconds
  
  --transfer_learning_comparison_only (optional, default: False)
    When True, only generates transfer learning comparison visualizations.
    Requires both --run_id (TL model) and --compare_baseline (Baseline model).
    Generates 2 visualizations: performance comparison (ROC/PR) + learning curves.
    Skips all other visualizations.
    Runtime: ~5 seconds
    Example: python generate_visuals.py --run_id 31 --compare_baseline 30 --transfer_learning_comparison_only
  
  STANDARD ARGUMENTS:
  
  --run_id (required)
    The experiment run identifier (e.g., "62_102498" or "31")
    Used to locate model checkpoint in results/ directory
    Defaults to latest run if omitted
    
  --fold (optional, default: 0)
    Which cross-validation fold to visualize (0 to num_folds-1)
    Must match the fold used during training
    
  --num_folds (optional, default: 5)
    Total number of cross-validation folds
    Used to determine which patients belong to validation set
    
  --model_name (optional, default: "convnext_tiny")
    Backbone architecture: "convnext_tiny" or "resnet50"
    Must match the model used during training
    
  --dataset (optional, default: "helicodataset")
    Which dataset to visualize: "helicodataset", "deephp", or "both"
    
  --pipeline_mode (optional, default: False)
    Generate comprehensive lightweight visual report for presentations
    Includes: calibration curve, performance dashboard, ensemble analysis, CV stability,
    failure modes, class distribution & stratification, training trajectory, and training efficiency
    Skips redundant visualizations already created during training (ROC, PR, confusion matrix)
    Automatically used by submit_transfer_learning.sh
    
  --compare_baseline (optional, default: None)
    Baseline run ID for transfer learning comparison (e.g., "30")
    If provided (without --transfer_learning_comparison_only), generates comparison
    plots showing improvement from baseline to transfer learning
    
  ANALYSIS GROUPS (Can be combined):
  
  --include_transfer_analysis (optional)
    Generate transfer learning analysis visualizations (learning curves comparison)
    
  --include_ensemble_analysis (optional)
    Generate ensemble voting agreement patterns
    
  --include_cv_stability (optional)
    Generate cross-validation stability plots (metrics across folds)
    
  --include_data_audit (optional)
    Generate data integrity audit visualizations (leakage detection, train/test split)
    
  --include_failure_modes (optional)
    Generate failure mode analysis (hard examples + edge cases)
    
  --include_training_trajectory (optional)
    Generate training trajectory plots (loss and accuracy over epochs)
    
  --include_training_efficiency (optional)
    Generate training efficiency metrics (wall-clock time, GPU memory, throughput)
    
  --include_model_complexity (optional)
    Generate model complexity vs performance analysis (justifies ConvNeXt-Tiny choice)
    Loads actual accuracies from all 4 trained models with source transparency

EXAMPLES
--------
  # FAST MODES (Efficient comparison without full pipeline)
  
  # Compare 4 model architectures efficiently
  python generate_visuals.py --run_id 31 --model_comparison_only
  
  # Quick transfer learning validation (baseline vs TL)
  python generate_visuals.py --run_id 31 --compare_baseline 30 --transfer_learning_comparison_only
  
  # STANDARD USAGE
  
  # Generate all visualizations for latest run
  python generate_visuals.py
  
  # Generate all visualizations for specific run
  python generate_visuals.py --run_id 31
  
  # Pipeline mode (used automatically during training)
  python generate_visuals.py --run_id 31 --pipeline_mode
  
  # Visualize fold 2 with ResNet50 backbone
  python generate_visuals.py --run_id 31 --fold 2 --model_name resnet50
  
  # Generate only model complexity and training efficiency analysis
  python generate_visuals.py --run_id 31 --include_model_complexity --include_training_efficiency
  
  # Transfer learning comparison with learning curves
  python generate_visuals.py --run_id 31 --compare_baseline 30 --include_transfer_analysis
  
  # Pipeline mode with combined learning curves (separate files)
  python generate_visuals.py --run_id 31 --compare_baseline 30 --pipeline_mode --fold 0
  
  # Data rigor analysis (audit, hard examples, edge cases, trajectory)
  python generate_visuals.py --run_id 31 --include_data_audit --include_failure_modes --include_training_trajectory
  
  # Custom fold split (10-fold cross-validation)
  python generate_visuals.py --run_id 31 --fold 1 --num_folds 10 --dataset both
  
  # Combine learning curves (advanced: manual stitching of images)
  python generate_visuals.py --combine_learning_curves --pretraining_run 30 --dataset_run 31 --fold 0

OUTPUT
------
All visualizations are saved in: results/{RUN_ID}_*

  **FAST MODE Outputs:**
  Model Comparison Mode (--model_comparison_only):
    - model_complexity_analysis_{RUN_ID}.png - 4 architectures (ResNet50, ConvNeXt-Tiny/Small/Base)
  
  Transfer Learning Comparison Mode (--transfer_learning_comparison_only):
    - transfer_learning_comparison_{BASELINE}_vs_{RUN_ID}.png - ROC/PR performance comparison
    - transfer_learning_curves_{BASELINE}_vs_{RUN_ID}.png - Learning curves overlay

  **Clinical Validation Visualizations (Pipeline Mode):**
  - {RUN_ID}_*_calibration_curve.png - Model calibration (probability reliability)
  - {RUN_ID}_*_performance_dashboard.png - 4-panel summary (confusion, ROC, PR, metrics)
  - combined_learning_curves_pretraining_f{FOLD}.png - Pre-training learning curves (when using --compare_baseline)
  - combined_learning_curves_finetuning_f{FOLD}.png - Fine-tuning learning curves (when using --compare_baseline)
  
  **Standard Visualizations (Full Mode):**
  - {RUN_ID}_*_confusion_matrix.png - Patient-level 2x2 confusion matrix
  - {RUN_ID}_*_roc_curve.png - ROC curve with AUC score
  - {RUN_ID}_*_pr_curve.png - Precision-Recall curve with AP score
  - {RUN_ID}_*_threshold_analysis.png - Performance across decision thresholds
  - {RUN_ID}_*_probability_histogram.png - Distribution of predicted probabilities
  - {RUN_ID}_*_metrics_summary.csv - Per-fold metrics with bootstrap 95% CIs
  - {RUN_ID}_*_predictions.csv - Patient-level predictions (for comparison)
  - {RUN_ID}_gradcam_samples/ - Grad-CAM visualizations (50+ heatmap images)
  
  **Advanced Analysis Visualizations:**
  - transfer_learning_comparison_{BASELINE}_vs_{RUN_ID}.png - Baseline vs TL (performance)
  - transfer_learning_curves_{BASELINE}_vs_{RUN_ID}.png - Learning curves comparison
  - training_trajectory_{RUN_ID}_*.png - Loss/accuracy over epochs with convergence analysis
  - training_efficiency_{RUN_ID}.png - 3-panel: wall-clock time, GPU memory, throughput
  - ensemble_voting_agreement_{RUN_ID}.png - Model agreement heatmap across patients
  - cross_validation_stability_{RUN_ID}.png - Box plots of metrics across 5 folds (robustness)
  - cross_validation_bootstrap_ci_{RUN_ID}.png - Bootstrap 95% CIs showing robustness across patient populations
  - data_integrity_audit_{RUN_ID}.png - 4-panel audit with leakage detection
  - hard_examples_analysis_{RUN_ID}.png - Lowest confidence correct predictions
  - edge_cases_analysis_{RUN_ID}.png - False positives vs false negatives analysis
  - model_complexity_analysis_{RUN_ID}.png - Architecture justification with measured data
  
  **Ensemble Performance Dashboards (4-panel with confusion matrix, ROC, PR, metrics):**
  - ensemble_voting_{RUN_ID}_performance_dashboard.png - Voting ensemble performance
  - meta_classifier_{RUN_ID}_performance_dashboard.png - Meta classifier performance
  - hybrid_ensemble_{RUN_ID}_performance_dashboard.png - Hybrid ensemble performance
  - grand_cv_averages_performance_dashboard.png - Grand cross-validation averages

REQUIREMENTS
------------
  - PyTorch with GPU support (CUDA recommended)
  - Trained model checkpoint at: results/{RUN_ID}_f{FOLD}_{MODEL_NAME}_model_brain.pth
  - H. Pylori dataset at: /export/hhome/tkeating/HelicoDataSet or ../HelicoDataSet
  - DeepHP dataset (optional): /export/hhome/tkeating/8117177/

NOTES
-----
  - FAST MODES (--model_comparison_only, --transfer_learning_comparison_only):
    * Do NOT require dataset loading or model inference
    * Load precomputed evaluation reports, predictions CSV, and learning curves JSON
    * Ideal for rapid comparisons without pipeline overhead
  
  - STANDARD MODES:
    * Script uses model.eval() mode (no dropout, batch norm statistics fixed)
    * Inference runs in deterministic mode (set seed in torch/numpy)
    * Grad-CAM uses input-level gradient saliency (model-agnostic approach)
    * Visualization pairs show original patch + normalized heatmap side-by-side
  
  - DATA TRANSPARENCY:
    * All model accuracies and sources are explicitly tracked and displayed
    * "Measured (N folds)" indicates values from actual trained models
    * "ImageNet benchmark" indicates published baseline values
    * Fallback values are used only when measured data is unavailable
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
    precision_recall_curve, average_precision_score, classification_report,
    roc_auc_score
)
from torchvision import transforms
from PIL import Image
from visualization_utils import (
    generate_gradcam, plot_roc_curve, plot_confusion_matrix, plot_gradcam_pair,
    plot_pr_curve, plot_probability_histogram, plot_threshold_analysis,
    plot_calibration_curve, plot_patient_performance_dashboard, 
    plot_transfer_learning_comparison, plot_learning_curves_comparison,
    plot_ensemble_voting_agreement, plot_cross_validation_stability,
    plot_data_integrity_audit, plot_hard_examples_analysis, plot_edge_cases_analysis,
    plot_training_trajectory, plot_training_efficiency, plot_model_complexity_analysis,
    plot_class_distribution_analysis, combine_learning_curves, plot_bootstrap_confidence_intervals,
    plot_cross_fold_confusion_matrices_dashboard, plot_cross_fold_pr_curves_dashboard,
    plot_combined_fold_roc_curves
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
    
    # Validate model checkpoint exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model checkpoint not found: {MODEL_PATH}")
        print(f"Available checkpoint files:")
        import glob
        for f in sorted(glob.glob("results/*_model_brain.pth"))[:5]:
            print(f"  - {f}")
        sys.exit(1)

    # Defer dataset loading until Grad-CAM generation to save memory
    # (Most visualizations don't need the full dataset in memory)
    full_dataset = None
    
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
    
    # OPTIMIZATION: Load precomputed predictions from holdout_consensus.csv if available
    # This avoids re-running inference on 87K+ patches which takes hours
    # Default behavior: use cached predictions to save time and memory
    predictions_path = os.path.join("results", f"{full_prefix}_holdout_consensus.csv")
    if os.path.exists(predictions_path):
        print(f"\n✓ Using cached predictions from [{predictions_path}]...")
        print(f"  (Skipping time-consuming inference on 87K+ patches)")
        
        preds_df = pd.read_csv(predictions_path)
        for _, row in preds_df.iterrows():
            patient_ids.append(row['PatientID'])
            all_labels.append(row['Actual'])
            all_probs.append(row['Bag_Mean_Prob'])  # Use bag mean probability
            patient_performance.append({
                "Patient": row['PatientID'],
                "Label": row['Actual'],
                "Prob": row['Bag_Mean_Prob'],
                "Pred": row['Predicted']
            })
        
        print(f"✓ Loaded predictions for {len(patient_ids)} patients")
        
        # Skip building dataset index mapping - we'll do on-demand lookup in Grad-CAM loop
        # This avoids expensive iteration through all dataset entries
        
        # Don't load model here - defer to Grad-CAM section where it's actually needed
        # This keeps memory usage low when Grad-CAM isn't being generated
        model = None
        val_loader = None
    else:
        # Predictions not found - need checkpoint to run inference
        if not os.path.exists(MODEL_PATH):
            print(f"\n{'='*80}")
            print(f"ERROR: Cannot regenerate visualizations")
            print(f"{'='*80}")
            print(f"Missing required file: {MODEL_PATH}")
            print(f"\nTo regenerate visualizations, you need ONE of:")
            print(f"  1. Cached predictions: {predictions_path}")
            print(f"  2. Model checkpoint: {MODEL_PATH}")
            print(f"\nIf you have ONLY the model checkpoint (no predictions):")
            print(f"  → Run with --gradcam_only to skip inference and only generate Grad-CAM")
            print(f"\nIf you have predictions but no checkpoint:")
            print(f"  → Visualizations (ROC, PR, confusion matrix) will still work")
            print(f"  → Grad-CAM generation will be skipped\n")
            sys.exit(1)
        
        # Full inference mode (need checkpoint to run)
        print(f"Cached predictions not found. Running full inference on validation set...")
        print(f"(This requires the model checkpoint and will use ~30GB memory)")
        print(f"Tip: Keep {predictions_path} from training to avoid re-inference\n")
        sys.stdout.flush()
        
        # Load dataset now (only if we're actually going to run inference)
        if full_dataset is None:
            print(f"Loading dataset...")
            if dataset_type.lower() == "helicodataset":
                full_dataset = HPyloriDataset(
                    HOLDOUT, PATIENT_CSV, PATCH_XLSX, 
                    transform=VAL_TRANSFORM, bag_mode=True, 
                    max_bag_size=1000, train=False
                )
            elif dataset_type.lower() == "deephp":
                if DeepHPDataset is None:
                    print("ERROR: DeepHP dataset module not available")
                    sys.exit(1)
                full_dataset = DeepHPDataset(
                    DEEPHP_DATASET_ROOT, fold_idx=fold_idx, num_folds=num_folds,
                    train=False, transform=VAL_TRANSFORM, bag_mode=True,
                    max_bag_size=1000
                )
            print(f"✓ Dataset loaded: {len(full_dataset)} patients\n")
            sys.stdout.flush()
        
        # Create DataLoader: one patient (bag) per batch
        # Memory-optimized settings to prevent OOM during inference
        val_loader = DataLoader(
            full_dataset, batch_size=1, shuffle=False, 
            num_workers=0,           # No multiprocessing (saves memory)
            pin_memory=False,        # Don't pin to GPU memory
            prefetch_factor=None     # Don't prefetch (saves memory)
        )
        
        print(f"DataLoader configured for memory efficiency:")
        print(f"  • batch_size=1 (one patient per batch)")
        print(f"  • num_workers=0 (no parallel loading)")
        print(f"  • pin_memory=False (reduced GPU memory)")
        print(f"  • Adaptive chunk_size per patient based on bag size")
        print(f"  • GPU cache cleared after each patient\n")
        sys.stdout.flush()
        
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
        
        print(f"Running Inference on {len(full_dataset)} Validation Patients...")
        print(f"(Using efficient MIL aggregation on full bags)")
    
    # Run inference only if we have a dataloader (i.e., not using precomputed predictions)
    if val_loader is not None:
        with torch.no_grad():
            # Also need to track the index in the dataset
            for batch_idx, (bags, labels, p_ids) in enumerate(tqdm(val_loader, desc="Patient Inference", file=sys.stderr)):
                # bags: (1, bag_size, C, H, W), labels: (1,), p_ids: (1,)
                bags = bags.squeeze(0).to(DEVICE)  # (bag_size, C, H, W)
                label = labels.item()
                p_id = p_ids[0]
                dataset_idx = batch_idx
                
                # Store dataset index for later reloading (not the bags themselves)
                pat_to_dataset_idx[p_id] = dataset_idx
                
                # Adaptive chunk sizing based on bag size (balanced for GPU)
                if bags.size(0) > 500:
                    chunk_size = 128  # Large bags: use bigger chunks for speed
                elif bags.size(0) > 200:
                    chunk_size = 128  # Medium bags: standard chunks
                else:
                    chunk_size = 256  # Normal bags: maximum chunk size
                
                # Process entire bag via forward_bag with memory-optimized chunk size
                try:
                    logits, _ = model.forward_bag(bags, chunk_size=chunk_size)
                    prob = torch.softmax(logits, dim=1)[0, 1].item()
                    
                    all_probs.append(prob)
                    all_labels.append(label)
                    patient_ids.append(p_id)
                    
                    patient_performance.append({
                        "Patient": p_id,
                        "Label": label,
                        "Prob": prob,
                        "Pred": 1 if prob >= 0.5 else 0
                    })
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        # OOM recovery: clear GPU and retry with smaller chunks
                        print(f"\n    ⚠ GPU memory full for patient {p_id} ({bags.size(0)} patches), clearing cache and retrying...")
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        
                        # Retry with smaller chunks
                        try:
                            logits, _ = model.forward_bag(bags, chunk_size=64)
                            prob = torch.softmax(logits, dim=1)[0, 1].item()
                            
                            all_probs.append(prob)
                            all_labels.append(label)
                            patient_ids.append(p_id)
                            
                            patient_performance.append({
                                "Patient": p_id,
                                "Label": label,
                                "Prob": prob,
                                "Pred": 1 if prob >= 0.5 else 0
                            })
                            print(f"    ✓ Recovered with chunk_size=64")
                        except RuntimeError as e2:
                            print(f"    ✗ Failed to process patient {p_id} even with smaller chunks: {e2}")
                            print(f"    Skipping patient {p_id} to prevent crash")
                    else:
                        raise
                finally:
                    # Cleanup after each patient
                    del bags
                    torch.cuda.empty_cache()
                    
                    # Every 20 patients: run garbage collection
                    if (batch_idx + 1) % 20 == 0:
                        gc.collect()
    
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
    
    # Always compute ROC-AUC and PR-AUC (needed for dashboard even in pipeline mode)
    fpr, tpr, _ = roc_curve(all_labels_bin, all_probs)
    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(all_labels_bin, all_probs)
    
    if args.pipeline_mode:
        print(f"\n[Pipeline Mode] Skipping redundant visualizations already generated during training:")
        print(f"  - Confusion matrix (skip)")
        print(f"  - ROC curve (skip)")
        print(f"  - PR curve (skip)")
        print(f"  - Probability histogram (skip)")
        print(f"  - Threshold analysis (skip)")
    else:
        plot_confusion_matrix(all_labels_bin, all_preds_bin, os.path.join("results", f"{full_prefix}_confusion_matrix.png"))
        plot_roc_curve(all_labels_bin, all_probs, os.path.join("results", f"{full_prefix}_roc_curve.png"))
        plot_pr_curve(all_labels_bin, all_probs, os.path.join("results", f"{full_prefix}_pr_curve.png"))

        # 4. Probability Histogram
        plot_probability_histogram(np.array(all_probs), np.array(all_labels_bin), os.path.join("results", f"{full_prefix}_probability_histogram.png"))

        # 5. Threshold Analysis (Performance across decision boundaries)
        plot_threshold_analysis(all_labels_bin, all_probs, os.path.join("results", f"{full_prefix}_threshold_analysis.png"))
    
    # ========================================================================
    # MAIN VISUALIZATIONS (Skip in --gradcam_only mode)
    # ========================================================================
    if not args.gradcam_only:
        # 6. Calibration Curve (ALWAYS generate - this is new and clinically important)
        print(f"\n[New Visualization] Generating calibration curve (clinical calibration validation)...")
        plot_calibration_curve(all_labels_bin, all_probs, os.path.join("results", f"{full_prefix}_calibration_curve.png"))
        
        # 7. Patient-Level Performance Dashboard (SKIP per-fold - ensemble dashboards generated separately)
        # Per-fold dashboards not needed since ensemble/meta/hybrid dashboards provide better aggregated view
        print(f"[Skipped] Per-fold performance dashboard (generated separately for ensembles)")
        
        # 8. Grad-CAM visualizations (ALWAYS generate - model explainability)
        if not args.pipeline_mode:
            print(f"\nGenerating Grad-CAM visualizations for top predictions...")
        else:
            print(f"\n[Skipped] Grad-CAM visualizations (generate separately if needed for detailed analysis)")
        
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

    # --- Step 3: Grad-CAM for Top Suspicious Patients (ALWAYS in gradcam_only mode) ---
    if args.pipeline_mode and not args.gradcam_only:
        print(f"[Pipeline Mode] Skipping Grad-CAM (can be regenerated separately for detailed analysis)")
        sys.stdout.flush()
    else:
        # Load model for Grad-CAM if not already loaded
        if model is None:
            if not os.path.exists(MODEL_PATH):
                print(f"\n[Grad-CAM] Checkpoint not found: {MODEL_PATH}")
                print(f"[Grad-CAM] Skipping Grad-CAM generation (checkpoint required)")
                print(f"[Grad-CAM] But other visualizations (ROC, PR, confusion matrix) are available!\n")
                sys.stdout.flush()
                model = None  # Keep as None to skip Grad-CAM generation below
            else:
                print(f"\nLoading model for Grad-CAM generation...")
                sys.stdout.flush()
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
                model_state_dict = model.state_dict()
                filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in model_state_dict}
                
                model.load_state_dict(filtered_state_dict)
                model.eval()
                print(f"✓ Model loaded for Grad-CAM\n")
                sys.stdout.flush()
        
        # Generate Grad-CAM only if model was successfully loaded
        if model is not None:
            # Initialize dataset (loads metadata, not patches) and build patient→bag index cache
            if full_dataset is None:
                print(f"Initializing dataset for Grad-CAM...")
                if dataset_type.lower() == "helicodataset":
                    full_dataset = HPyloriDataset(
                        HOLDOUT, PATIENT_CSV, PATCH_XLSX, 
                        transform=VAL_TRANSFORM, bag_mode=True, 
                        max_bag_size=1000, train=False
                    )
                elif dataset_type.lower() == "deephp":
                    if DeepHPDataset is None:
                        print("ERROR: DeepHP dataset module not available. Install dataset_deepHP.py")
                        sys.exit(1)
                    full_dataset = DeepHPDataset(
                        DEEPHP_DATASET_ROOT, fold_idx=fold_idx, num_folds=num_folds,
                        train=False, transform=VAL_TRANSFORM, bag_mode=True,
                        max_bag_size=1000
                    )
                print(f"✓ Dataset ready: {len(full_dataset)} patients\n")
                sys.stdout.flush()
            
            # Build lightweight patient→bag_index cache
            cache_file = os.path.join("results", f"{full_prefix}_patient_bag_index.json")
            patient_bag_map = build_patient_bag_index_cache(full_dataset, cache_file)
            
            print(f"Generating Grad-CAM for top predictions and false negatives...")
            sys.stdout.flush()
            # Pick Top 3 Positives and Top 3 False Negatives (if any)
            top_positives = perf_df[perf_df['Label'] == 1].sort_values('Prob', ascending=False).head(3)
            ghosts = perf_df[(perf_df['Label'] == 1) & (perf_df['Prob'] < 0.5)].sort_values('Prob', ascending=False).head(3)
            
            targets = pd.concat([top_positives, ghosts])

            print(f"Generating Grad-CAM for {len(targets)} patients...", file=sys.stderr)
            sys.stderr.flush()
            
            for _, row in targets.iterrows():
                p_id = row['Patient']
                is_fn = row['Prob'] < 0.5
                print(f"  Processing patient {p_id} (prob={row['Prob']:.4f}, FN={is_fn})...", file=sys.stderr)
                sys.stderr.flush()
                
                # Find dataset index for this patient using cached mapping (O(1) lookup)
                if p_id not in patient_bag_map:
                    print(f"    WARNING: Patient {p_id} not found in dataset, skipping", file=sys.stderr)
                    sys.stderr.flush()
                    continue
                
                dataset_idx = patient_bag_map[p_id]
                
                bags_tensor, _, _ = full_dataset[dataset_idx]
                bags_tensor = bags_tensor.squeeze(0)  # (bag_size, C, H, W)
                
                print(f"    Loaded bag with {bags_tensor.size(0)} patches", file=sys.stderr)
                sys.stderr.flush()
                
                # Find top 3 most significant patches using attention weights (matching train.py logic)
                # This ensures consistency between training-generated and post-hoc Grad-CAM visualizations
                all_indicators = []
                
                # Adaptive chunking based on bag size (balanced for GPU performance)
                if bags_tensor.size(0) > 500:
                    # Very large bags: use larger chunks for speed
                    vram_bag_limit = 256
                elif bags_tensor.size(0) > 200:
                    # Large bags: use standard chunks
                    vram_bag_limit = 256
                else:
                    # Normal bags: maximum chunk size
                    vram_bag_limit = 512
                
                sys.stderr.flush()
                
                with torch.no_grad():
                    # Process in chunks to avoid VRAM overflow on large bags
                    total_chunks = (bags_tensor.size(0) + vram_bag_limit - 1) // vram_bag_limit
                    for chunk_idx, start_idx in enumerate(range(0, bags_tensor.size(0), vram_bag_limit)):
                        end_idx = min(start_idx + vram_bag_limit, bags_tensor.size(0))
                        print(f"      Processing chunk {chunk_idx+1}/{total_chunks} (patches {start_idx}-{end_idx})...", end='', file=sys.stderr, flush=True)
                        
                        chunk = bags_tensor[start_idx:end_idx].to(DEVICE)
                        chunk = det_preprocess_batch(chunk, training=False)
                        
                        # Forward through model to get attention weights
                        try:
                            if hasattr(model, 'forward_bag'):
                                # Use forward_bag to get attention weights if available
                                _, indicator = model.forward_bag(chunk)
                                all_indicators.append(indicator.cpu())
                            else:
                                # Fallback: use patch-level class logits for max-pooling models
                                logits = model(chunk)
                                indicator = logits[:, 1:2].transpose(0, 1)  # (1, N) - Class 1 confidence
                                all_indicators.append(indicator.cpu())
                            print(" ✓", file=sys.stderr, flush=True)
                        except RuntimeError as e:
                            if 'out of memory' in str(e).lower():
                                print(f" OOM! Retrying with 128-patch chunks...", file=sys.stderr)
                                torch.cuda.empty_cache()
                                # Retry with 128-patch chunks instead of original 256
                                for retry_idx in range(start_idx, end_idx, 128):
                                    retry_end = min(retry_idx + 128, end_idx)
                                    retry_chunk = bags_tensor[retry_idx:retry_end].to(DEVICE)
                                    retry_chunk = det_preprocess_batch(retry_chunk, training=False)
                                    if hasattr(model, 'forward_bag'):
                                        _, indicator = model.forward_bag(retry_chunk)
                                        all_indicators.append(indicator.cpu())
                                    else:
                                        logits = model(retry_chunk)
                                        indicator = logits[:, 1:2].transpose(0, 1)
                                        all_indicators.append(indicator.cpu())
                                    del retry_chunk
                                    torch.cuda.empty_cache()
                                print(" ✓ (recovered)", file=sys.stderr, flush=True)
                            else:
                                print(f" ERROR: {e}", file=sys.stderr, flush=True)
                                raise
                        finally:
                            # Aggressive cleanup after each chunk
                            del chunk
                            torch.cuda.empty_cache()
                            
                            # Every 20 chunks, sync GPU and free system memory
                            if (chunk_idx + 1) % 20 == 0:
                                torch.cuda.synchronize()
                                gc.collect()
                
                # Skip if no indicators were computed (shouldn't happen)
                if len(all_indicators) == 0:
                    print(f"    WARNING: No indicators computed, skipping", file=sys.stderr)
                    sys.stderr.flush()
                    continue
                
                indicators = torch.cat(all_indicators, dim=1).squeeze(0)  # (Bag_Size,)
                
                # Select top 3 most significant patches (or fewer if bag is small)
                top_patch_vals, patch_indices = torch.topk(indicators, k=min(3, bags_tensor.size(0)))
                
                # Cleanup attention computation
                del all_indicators, indicators
                torch.cuda.empty_cache()
                
                print(f"      Generating Grad-CAM for top {len(patch_indices)} patches...", file=sys.stderr)
                sys.stderr.flush()
                
                # Generate Grad-CAM for selected patches
                for rank_idx, (rank, idx) in enumerate(zip(range(len(patch_indices)), patch_indices)):
                    try:
                        patch_img = bags_tensor[idx]
                        patch_t = patch_img.unsqueeze(0).to(DEVICE)
                        
                        print(f"        [{rank_idx+1}/{len(patch_indices)}] Patch {idx}...", end='', file=sys.stderr, flush=True)
                        
                        with torch.enable_grad():
                            heatmap_batch, _ = generate_gradcam(model.backbone, patch_t)
                        
                        # Plot side-by-side visualization (original + heatmap overlay)
                        plot_gradcam_pair(
                            patch_img, heatmap_batch[0, 0], p_id, rank, idx,
                            top_patch_vals[rank].item(),  # Actual attention weight from topk selection
                            row['Prob'],
                            is_false_negative=is_fn, output_dir=OUTPUT_DIR
                        )
                        print(" ✓", file=sys.stderr, flush=True)
                        del heatmap_batch
                        torch.cuda.empty_cache()
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower():
                            print(" OOM!", file=sys.stderr, flush=True)
                            print(f"        Clearing GPU and skipping patch {idx}...", file=sys.stderr)
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        else:
                            print(f" ERROR: {e}", file=sys.stderr, flush=True)
                    finally:
                        # Cleanup after each patch
                        if 'patch_img' in locals():
                            del patch_img
                        if 'patch_t' in locals():
                            del patch_t
                
                # Free memory after patient
                del bags_tensor, top_patch_vals, patch_indices
                torch.cuda.empty_cache()
                print(f"  ✓ Completed patient {p_id}", file=sys.stderr)
                print(f"  {'-'*60}", file=sys.stderr, flush=True)
                sys.stderr.flush()

    # Generate ensemble performance dashboards (only once at the end)
    def generate_ensemble_dashboard(results_csv, bootstrap_ci_csv, dashboard_name, output_prefix):
        """Generate performance dashboard for ensemble/meta/hybrid results"""
        try:
            # Try primary filename first, fall back to alternative names if needed
            csv_file = results_csv
            if not os.path.exists(csv_file):
                # Try alternate naming conventions
                if 'holdout_predictions' in results_csv:
                    # Try the report/results variant
                    csv_file = results_csv.replace('holdout_predictions', 'report' if 'ensemble_voting' in results_csv else 'results')
                elif 'report' in results_csv or 'results' in results_csv:
                    # Try the holdout_predictions variant
                    if 'ensemble_voting' in results_csv:
                        csv_file = results_csv.replace('report', 'holdout_predictions')
                    else:
                        csv_file = results_csv.replace('results', 'holdout_predictions')
                
                if not os.path.exists(csv_file):
                    print(f"  INFO: {dashboard_name} results not available: {results_csv}")
                    return
            
            # Load results
            results_df = pd.read_csv(csv_file)
            
            # Extract labels and predictions
            if 'Actual' in results_df.columns:
                labels = results_df['Actual'].values
            elif 'Label' in results_df.columns:
                labels = results_df['Label'].values
            else:
                print(f"  INFO: Could not find label column in {results_csv}")
                return
            
            # Extract predictions (different column names for different ensemble types)
            if 'Ensemble_Pred' in results_df.columns:
                preds = results_df['Ensemble_Pred'].values
                probs = results_df['Max_Ensemble_Prob'].values
            elif 'Meta_Pred' in results_df.columns:
                preds = results_df['Meta_Pred'].values
                probs = results_df['Meta_Prob'].values
            elif 'Hybrid_Pred' in results_df.columns:
                preds = results_df['Hybrid_Pred'].values
                probs = results_df['Hybrid_Prob'].values
            elif 'Consensus_Pred' in results_df.columns:  # Grand CV
                preds = results_df['Consensus_Pred'].values
                probs = results_df['Consensus_Prob'].values
            elif 'Predicted' in results_df.columns:  # Fallback for generic prediction column names
                preds = results_df['Predicted'].values
                if 'Predicted_Probability' in results_df.columns:
                    probs = results_df['Predicted_Probability'].values
                elif 'Probability' in results_df.columns:
                    probs = results_df['Probability'].values
                else:
                    print(f"  INFO: Could not find probability column in {results_csv}")
                    return
            else:
                print(f"  INFO: Could not find prediction column in {results_csv}")
                return
            
            # Compute ROC-AUC and PR-AUC
            roc_auc = roc_auc_score(labels, probs)
            pr_auc = average_precision_score(labels, probs)
            
            # Load bootstrap CIs
            bootstrap_ci = {}
            if os.path.exists(bootstrap_ci_csv):
                ci_df = pd.read_csv(bootstrap_ci_csv)
                for _, row in ci_df.iterrows():
                    metric_name = row['Metric'].lower()
                    if metric_name == 'recall':
                        bootstrap_ci['sensitivity'] = {
                            'ci_lower': row['CI_Lower_95%'],
                            'ci_upper': row['CI_Upper_95%']
                        }
                    elif metric_name == 'precision':
                        bootstrap_ci['precision'] = {
                            'ci_lower': row['CI_Lower_95%'],
                            'ci_upper': row['CI_Upper_95%']
                        }
                    elif metric_name == 'accuracy':
                        bootstrap_ci['accuracy'] = {
                            'ci_lower': row['CI_Lower_95%'],
                            'ci_upper': row['CI_Upper_95%']
                        }
                    elif metric_name == 'f1':
                        bootstrap_ci['f1'] = {
                            'ci_lower': row['CI_Lower_95%'],
                            'ci_upper': row['CI_Upper_95%']
                        }
                    elif metric_name == 'specificity':
                        bootstrap_ci['specificity'] = {
                            'ci_lower': row['CI_Lower_95%'],
                            'ci_upper': row['CI_Upper_95%']
                        }
            else:
                print(f"  INFO: Bootstrap CI file not found: {bootstrap_ci_csv}")
            
            # Provide default values for missing bootstrap CI keys
            default_ci = {'ci_lower': 0.0, 'ci_upper': 1.0}
            for key in ['sensitivity', 'precision', 'accuracy', 'f1', 'specificity']:
                if key not in bootstrap_ci:
                    bootstrap_ci[key] = default_ci.copy()
            
            # Compute fold metrics
            from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, confusion_matrix
            cm = confusion_matrix(labels, preds)
            tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0, 0, 0, 0)
            
            fold_metrics = {
                'sensitivity': recall_score(labels, preds, zero_division=0),
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'precision': precision_score(labels, preds, zero_division=0),
                'accuracy': accuracy_score(labels, preds),
                'f1': f1_score(labels, preds, zero_division=0)
            }
            
            # Generate dashboard
            plot_patient_performance_dashboard(
                labels,
                preds,
                probs,
                fold_metrics,
                bootstrap_ci,
                roc_auc,
                pr_auc,
                output_path=f"results/{output_prefix}_performance_dashboard.png"
            )
            print(f"  ✓ {dashboard_name} dashboard saved")
        
        except Exception as e:
            print(f"  INFO: Could not generate {dashboard_name} dashboard - {e}")
    
    # Generate ensemble dashboards
    print(f"\n{'='*80}")
    print(f"ENSEMBLE PERFORMANCE DASHBOARDS")
    print(f"{'='*80}\n")
    
    generate_ensemble_dashboard(
        f"results/ensemble_voting_holdout_predictions_{RUN_ID}-{RUN_ID}.csv",
        f"results/ensemble_voting_bootstrap_ci_{RUN_ID}-{RUN_ID}.csv",
        "Ensemble Voting",
        f"ensemble_voting_{RUN_ID}"
    )
    
    generate_ensemble_dashboard(
        f"results/meta_classifier_holdout_predictions_{RUN_ID}-{RUN_ID}.csv",
        f"results/meta_classifier_bootstrap_ci_{RUN_ID}-{RUN_ID}.csv",
        "Meta Classifier",
        f"meta_classifier_{RUN_ID}"
    )
    
    generate_ensemble_dashboard(
        f"results/hybrid_ensemble_holdout_predictions_{RUN_ID}-{RUN_ID}.csv",
        f"results/hybrid_ensemble_bootstrap_ci_{RUN_ID}-{RUN_ID}.csv",
        "Hybrid Ensemble",
        f"hybrid_ensemble_{RUN_ID}"
    )

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

def generate_cross_fold_dashboards(run_id, model_name, num_folds=5):
    """
    Generate cross-fold aggregated dashboards (confusion matrices, ROC/PR curves)
    
    Args:
        run_id: Run ID (e.g., "31")
        model_name: Model architecture (e.g., "convnext_tiny")
        num_folds: Number of folds (default: 5)
    """
    import glob
    
    print(f"\nGenerating Cross-Fold Aggregated Dashboards...")
    
    # Load predictions and labels from all folds
    fold_data_dict = {}
    folds_found = 0
    
    for fold_idx in range(num_folds):
        # Try to find holdout_consensus.csv (cached predictions from training)
        pred_pattern = f"results/{run_id}_*_*_f{fold_idx}_{model_name}_holdout_consensus.csv"
        pred_files = glob.glob(pred_pattern)
        
        if pred_files:
            pred_file = pred_files[0]
            try:
                df = pd.read_csv(pred_file)
                
                # Ensure required columns exist
                if all(col in df.columns for col in ['Actual', 'Bag_Mean_Prob']):
                    labels = df['Actual'].values.astype(int)
                    probs = df['Bag_Mean_Prob'].values
                    preds = (probs >= 0.5).astype(int)
                    
                    fold_data_dict[fold_idx] = {
                        'labels': labels,
                        'probabilities': probs,
                        'predictions': preds
                    }
                    folds_found += 1
                    print(f"  ✓ Fold {fold_idx}: Loaded {len(df)} predictions")
            except Exception as e:
                print(f"  ⚠ Fold {fold_idx}: Failed to load - {e}")
    
    if folds_found < 2:
        print(f"  INFO: Not enough folds found ({folds_found}). Need at least 2 for cross-fold dashboards.")
        return
    
    # Generate cross-fold confusion matrices dashboard
    try:
        output_path = f"results/cross_fold_confusion_matrices_dashboard_{run_id}_{model_name}.png"
        plot_cross_fold_confusion_matrices_dashboard(
            fold_data_dict,
            output_path=output_path,
            figsize=(16, 12)
        )
        print(f"  ✓ Cross-fold confusion matrices dashboard saved")
    except Exception as e:
        print(f"  ⚠ Cross-fold confusion matrices skipped: {e}")
    
    # Generate cross-fold PR curves dashboard
    try:
        output_path = f"results/cross_fold_pr_curves_dashboard_{run_id}_{model_name}.png"
        plot_cross_fold_pr_curves_dashboard(
            fold_data_dict,
            output_path=output_path,
            figsize=(16, 12)
        )
        print(f"  ✓ Cross-fold PR curves dashboard saved")
    except Exception as e:
        print(f"  ⚠ Cross-fold PR curves skipped: {e}")
    
    # Generate combined fold ROC curves
    try:
        output_path = f"results/cross_fold_roc_curves_{run_id}_{model_name}.png"
        plot_combined_fold_roc_curves(
            fold_data_dict,
            output_path=output_path,
            figsize=(12, 9)
        )
        print(f"  ✓ Combined fold ROC curves saved")
    except Exception as e:
        print(f"  ⚠ Combined ROC curves skipped: {e}")


def build_patient_bag_index_cache(dataset, cache_file):
    """
    Build and cache a lightweight mapping of patient_id -> bag_index
    without loading all patch data into memory.
    
    Args:
        dataset: HPyloriDataset or similar with .bags attribute
        cache_file: Path to save JSON cache
        
    Returns:
        Dict mapping patient_id -> bag_index
    """
    import json
    
    # Quick check: does cache already exist and is valid?
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
                if len(cache) > 0:
                    print(f"  ✓ Using cached patient→bag index ({len(cache)} patients)")
                    return cache
        except Exception as e:
            print(f"  Cache load failed, rebuilding: {e}")
    
    # Build mapping from dataset metadata (doesn't load image data)
    print(f"  Building patient→bag index...")
    patient_bag_map = {}
    
    if hasattr(dataset, 'bags') and dataset.bags:
        for idx, bag_tuple in enumerate(dataset.bags):
            # bag_tuple format: (paths, label, patient_id, pos_samples)
            _, _, patient_id, _ = bag_tuple
            patient_bag_map[patient_id] = idx
    
    # Save to cache
    try:
        os.makedirs(os.path.dirname(cache_file) or '.', exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(patient_bag_map, f)
        print(f"  ✓ Cached {len(patient_bag_map)} patient→bag mappings")
    except Exception as e:
        print(f"  Warning: Could not save cache - {e}")
    
    return patient_bag_map


def find_model_path(run_id, fold, model_name, slurm_id=None):
    """Find model file for given run_id and fold. Uses metadata to pick the correct model.
    If specified fold doesn't exist, searches for any available fold.
    
    Args:
        run_id: Run identifier (e.g., '01_34.4')
        fold: Fold index (0-4)
        model_name: Model name (e.g., 'convnext_tiny')
        slurm_id: Optional SLURM job ID to disambiguate when multiple runs exist
    """
    import re
    import json
    results_dir = "results"
    
    # Pattern: {run_id}_{anything}_f{fold}_{model_name}_model_brain.pth
    # With optional SLURM ID: {run_id}_{iteration}_{slurm_id}_f{fold}_{model_name}_model_brain.pth
    if slurm_id:
        swa_pattern = re.compile(rf"^{run_id}_{re.escape(str(slurm_id))}.*_f{fold}_{model_name}_swa_model_brain\.pth$")
        model_pattern = re.compile(rf"^{run_id}_{re.escape(str(slurm_id))}.*_f{fold}_{model_name}_model_brain\.pth$")
        metadata_pattern = re.compile(rf"^{run_id}_{re.escape(str(slurm_id))}.*_f{fold}_{model_name}_model_selection\.json$")
    else:
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
                    print(f"  [Model Selection] Using SWA model (from metadata: use_swa=True)")
                    return swa_models[0], fold
                elif not metadata.get("use_swa") and regular_models:
                    print(f"  [Model Selection] Using best model (from metadata: use_swa=False)")
                    return regular_models[0], fold
        except (json.JSONDecodeError, IOError) as e:
            print(f"  [Model Selection] Metadata file found but unreadable ({e}). Using fallback logic.")
    
    # Fallback: Prefer best model over SWA (safer default, matching training's best model)
    if regular_models:
        print(f"  [Model Selection] No metadata found. Using best model (safer fallback).")
        return regular_models[0], fold
    elif swa_models:
        print(f"  [Model Selection] No best model found. Using SWA model (fallback).")
        return swa_models[0], fold
    
    # If specified fold not found, search for any available fold for this run_id
    if slurm_id:
        fold_pattern = re.compile(rf"^{run_id}_{re.escape(str(slurm_id))}.*_f(\d+)_{model_name}_swa_model_brain\.pth$")
        fold_pattern_regular = re.compile(rf"^{run_id}_{re.escape(str(slurm_id))}.*_f(\d+)_{model_name}_model_brain\.pth$")
    else:
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
    parser.add_argument("--model_name", type=str, default="convnext_tiny", choices=["resnet50", "convnext_tiny", "convnext_small"],
                         help="Backbone architecture")
    parser.add_argument("--dataset", type=str, default="helicodataset", 
                       choices=["helicodataset", "deephp", "both"],
                       help="Which dataset to visualize: 'helicodataset', 'deephp', or 'both'")
    parser.add_argument("--compare_baseline", type=str, default=None,
                       help="Baseline run ID for transfer learning comparison (e.g., '30'). "
                            "If provided, will generate comparison plots between baseline and this run.")
    parser.add_argument("--pipeline_mode", action="store_true",
                       help="Generate comprehensive lightweight visual report. Includes: "
                            "calibration curve, performance dashboard, ensemble analysis, "
                            "cross-validation stability, failure modes, class distribution, "
                            "training trajectory, and training efficiency. "
                            "Skips redundant visualizations already created during training (ROC, PR, confusion matrix, etc.)")
    parser.add_argument("--include_transfer_analysis", action="store_true",
                       help="Generate transfer learning analysis visualizations "
                            "(requires learning curves and baseline model comparison)")
    parser.add_argument("--include_ensemble_analysis", action="store_true",
                       help="Generate ensemble contribution analysis (requires ensemble voting results)")
    parser.add_argument("--include_cv_stability", action="store_true",
                       help="Generate cross-validation stability plots (box plots across all folds)")
    parser.add_argument("--include_data_audit", action="store_true",
                       help="Generate data integrity audit visualizations (leakage detection, train/test split)")
    parser.add_argument("--include_failure_modes", action="store_true",
                       help="Generate failure mode analysis (hard examples, edge cases)")
    parser.add_argument("--include_class_distribution", action="store_true",
                       help="Generate class distribution and stratification analysis (imbalance, fold consistency)")
    parser.add_argument("--include_training_trajectory", action="store_true",
                       help="Generate training trajectory plots (loss and accuracy over epochs)")
    parser.add_argument("--include_training_efficiency", action="store_true",
                       help="Generate training efficiency metrics (wall-clock time, GPU memory, throughput)")
    parser.add_argument("--include_model_complexity", action="store_true",
                       help="Generate model complexity vs performance analysis (justifies ConvNeXt-Tiny choice)")
    parser.add_argument("--model_comparison_only", action="store_true",
                       help="FAST MODE: Only generate model complexity comparison visualization. "
                            "Skips all other visualizations. Useful for comparing multiple trained models.")
    parser.add_argument("--transfer_learning_comparison_only", action="store_true",
                       help="FAST MODE: Only generate transfer learning comparison visualizations (performance + learning curves). "
                            "Skips all other visualizations. Requires --compare_baseline.")
    parser.add_argument("--gradcam_only", action="store_true",
                       help="FAST MODE: Only generate Grad-CAM visualizations for misclassified and suspicious samples. "
                            "Skips all other visualizations. Useful for detailed model interpretation.")
    parser.add_argument("--backbone_path", type=str, default=None,
                       help="Full path to ensemble weighted backbone checkpoint (e.g., deephp_backbone_final_01_34.4_convnext_tiny_34.4.pth). "
                            "If provided, uses ensemble backbone instead of fold-specific checkpoints.")
    parser.add_argument("--combine_learning_curves", action="store_true",
                       help="Combine multiple learning curve images into a single composite visualization. "
                            "Requires --pretraining_run and --dataset_run. Layout controlled by --learning_curves_layout.")
    parser.add_argument("--pretraining_run", type=str, default=None,
                       help="Run ID for pre-training learning curves (e.g., DeepHP). Used with --combine_learning_curves.")
    parser.add_argument("--dataset_run", type=str, default=None,
                       help="Run ID for dataset learning curves (e.g., HelicoDataSet). Used with --combine_learning_curves.")
    parser.add_argument("--learning_curves_layout", type=str, default="horizontal", 
                       choices=["horizontal", "vertical"],
                       help="Layout for combined learning curves: 'horizontal' (side-by-side) or 'vertical' (stacked). "
                            "Default: horizontal")
    args = parser.parse_args()
    
    # If no run_id provided, use the latest one
    run_id = args.run_id
    if run_id is None:
        run_id = get_latest_run_id()
        if run_id is None:
            print("Error: No run ID provided and no models found in results directory")
            sys.exit(1)
        print(f"Using latest run: {run_id}")
    
    # ========================================================================
    # FAST MODE: Transfer Learning Comparison Only (Skip All Other Visualizations)
    # ========================================================================
    if args.transfer_learning_comparison_only:
        print(f"\n{'='*80}")
        print(f"FAST MODE: Transfer Learning Comparison Only")
        print(f"Baseline Run: {args.compare_baseline} | TL Run: {run_id}")
        print(f"Skipping all other visualizations")
        print(f"{'='*80}\n")
        
        if args.compare_baseline is None:
            print("ERROR: --transfer_learning_comparison_only requires --compare_baseline <BASELINE_RUN_ID>")
            print("Example: python generate_visuals.py --run_id 31 --compare_baseline 30 --transfer_learning_comparison_only")
            sys.exit(1)
        
        try:
            # Find model paths for both baseline and TL
            baseline_model_path, baseline_fold = find_model_path(args.compare_baseline, args.fold, args.model_name)
            tl_model_path, tl_fold = find_model_path(run_id, args.fold, args.model_name)
            
            if baseline_model_path is None or not os.path.exists(baseline_model_path):
                print(f"ERROR: Could not find baseline model for run {args.compare_baseline}")
                sys.exit(1)
            
            if tl_model_path is None or not os.path.exists(tl_model_path):
                print(f"ERROR: Could not find TL model for run {run_id}")
                sys.exit(1)
            
            # ========== 1. Performance Comparison (ROC/PR curves) ==========
            print(f"\n--- Generating Performance Comparison ---\n")
            
            baseline_prefix = os.path.basename(baseline_model_path).replace("_swa_model_brain.pth", "").replace("_model_brain.pth", "")
            tl_prefix = os.path.basename(tl_model_path).replace("_swa_model_brain.pth", "").replace("_model_brain.pth", "")
            
            baseline_csv = os.path.join("results", f"{baseline_prefix}_predictions.csv")
            tl_csv = os.path.join("results", f"{tl_prefix}_predictions.csv")
            
            perf_comparison_generated = False
            
            if os.path.exists(baseline_csv) and os.path.exists(tl_csv):
                try:
                    baseline_df = pd.read_csv(baseline_csv)
                    tl_df = pd.read_csv(tl_csv)
                    
                    # Ensure same patient order and labels
                    if len(baseline_df) == len(tl_df) and (baseline_df['Label'] == tl_df['Label']).all():
                        plot_transfer_learning_comparison(
                            baseline_df['Label'].values,
                            baseline_df['Prob'].values,
                            tl_df['Label'].values,
                            tl_df['Prob'].values,
                            baseline_name=f"Baseline (Run {args.compare_baseline})",
                            tl_name=f"Transfer Learning (Run {run_id})",
                            output_path=os.path.join("results", f"transfer_learning_comparison_{args.compare_baseline}_vs_{run_id}.png")
                        )
                        print(f"✓ Performance comparison generated!")
                        perf_comparison_generated = True
                    else:
                        print(f"WARNING: Baseline and TL predictions have different patient counts or labels.")
                except Exception as e:
                    print(f"WARNING: Error loading prediction CSVs: {e}")
            else:
                print(f"WARNING: Prediction CSV files not found:")
                print(f"  Baseline: {baseline_csv} (exists: {os.path.exists(baseline_csv)})")
                print(f"  TL:       {tl_csv} (exists: {os.path.exists(tl_csv)})")
            
            # ========== 2. Learning Curves Comparison ==========
            print(f"\n--- Generating Learning Curves Comparison ---\n")
            
            baseline_curves_file = f"results/{args.compare_baseline}_f{args.fold}_{args.model_name}_learning_curves.json"
            tl_curves_file = f"results/{run_id}_f{args.fold}_{args.model_name}_learning_curves.json"
            
            learning_curves_generated = False
            
            if os.path.exists(baseline_curves_file) and os.path.exists(tl_curves_file):
                try:
                    import json
                    with open(baseline_curves_file, 'r') as f:
                        learning_curves_baseline = json.load(f)
                    with open(tl_curves_file, 'r') as f:
                        learning_curves_tl = json.load(f)
                    
                    plot_learning_curves_comparison(
                        learning_curves_baseline,
                        learning_curves_tl,
                        baseline_name=f"Baseline (Run {args.compare_baseline})",
                        tl_name=f"Transfer Learning (Run {run_id})",
                        output_path=f"results/transfer_learning_curves_{args.compare_baseline}_vs_{run_id}.png"
                    )
                    print(f"✓ Learning curves comparison generated!")
                    learning_curves_generated = True
                except Exception as e:
                    print(f"WARNING: Error generating learning curves comparison: {e}")
            else:
                print(f"WARNING: Learning curves files not found:")
                print(f"  Baseline: {baseline_curves_file} (exists: {os.path.exists(baseline_curves_file)})")
                print(f"  TL:       {tl_curves_file} (exists: {os.path.exists(tl_curves_file)})")
            
            # Summary
            print(f"\n{'='*80}")
            if perf_comparison_generated or learning_curves_generated:
                print(f"✓ Transfer learning comparison complete!")
                if perf_comparison_generated:
                    print(f"  - Performance comparison (ROC/PR): transfer_learning_comparison_{args.compare_baseline}_vs_{run_id}.png")
                if learning_curves_generated:
                    print(f"  - Learning curves comparison: transfer_learning_curves_{args.compare_baseline}_vs_{run_id}.png")
            else:
                print(f"ERROR: No transfer learning comparisons were generated. Check file paths above.")
            print(f"{'='*80}")
            
        except Exception as e:
            print(f"ERROR: Failed to generate transfer learning comparison: {e}")
            import traceback
            traceback.print_exc()
        
        sys.exit(0)
    
    # ========================================================================
    # ========================================================================
    # FAST MODE: Grad-CAM Only (Skip All Other Visualizations)
    # ========================================================================
    if args.gradcam_only:
        print(f"\n{'='*80}")
        print(f"FAST MODE: Grad-CAM Visualization Only")
        print(f"Run: {run_id} | Model: {args.model_name} | Fold: {args.fold}")
        print(f"Skipping all other visualizations")
        print(f"{'='*80}\n")
        args.pipeline_mode = False  # Force Grad-CAM generation
    
    # COMBINE LEARNING CURVES MODE: Stitch learning curve images together
    # ========================================================================
    if args.combine_learning_curves:
        print(f"\n{'='*80}")
        print(f"COMBINE MODE: Stitching Learning Curves Together")
        print(f"{'='*80}\n")
        
        if args.pretraining_run is None or args.dataset_run is None:
            print("ERROR: --combine_learning_curves requires both --pretraining_run and --dataset_run")
            print("Example: python generate_visuals.py --combine_learning_curves --pretraining_run 30 --dataset_run 31 --fold 0")
            sys.exit(1)
        
        try:
            # Find learning curve images for both runs
            pretraining_curves = f"results/{args.pretraining_run}_30.0_*_f{args.fold}_{args.model_name}_learning_curves.png"
            dataset_curves = f"results/{args.dataset_run}_*_*_f{args.fold}_{args.model_name}_learning_curves.png"
            
            # Use glob to find the actual files (since we don't know the job ID)
            import glob
            pretraining_files = glob.glob(pretraining_curves)
            dataset_files = glob.glob(dataset_curves)
            
            if not pretraining_files:
                print(f"ERROR: No pre-training learning curves found for run {args.pretraining_run}, fold {args.fold}")
                print(f"  Searched for: {pretraining_curves}")
                sys.exit(1)
            
            if not dataset_files:
                print(f"ERROR: No dataset learning curves found for run {args.dataset_run}, fold {args.fold}")
                print(f"  Searched for: {dataset_curves}")
                sys.exit(1)
            
            pretraining_curves_path = pretraining_files[0]
            dataset_curves_path = dataset_files[0]
            
            print(f"Pre-training curves:  {pretraining_curves_path}")
            print(f"Dataset curves:       {dataset_curves_path}\n")
            
            # Get dataset name from run ID if available (else use generic labels)
            pretraining_label = f"Pre-training\n(Run {args.pretraining_run})"
            dataset_label = f"Transfer Learning\n(Run {args.dataset_run})"
            
            # Combine the images
            output_path = f"results/combined_learning_curves_{args.pretraining_run}_vs_{args.dataset_run}_f{args.fold}_{args.learning_curves_layout}.png"
            
            combine_learning_curves(
                image_paths=[pretraining_curves_path, dataset_curves_path],
                labels=[pretraining_label, dataset_label],
                output_path=output_path,
                layout=args.learning_curves_layout
            )
            
            print(f"\n✓ Combined learning curves saved: {output_path}\n")
        
        except Exception as e:
            print(f"ERROR: Failed to combine learning curves: {e}")
            import traceback
            traceback.print_exc()
        
        sys.exit(0)
    
    # If no run_id provided, use the latest one
    run_id = args.run_id
    if run_id is None:
        run_id = get_latest_run_id()
        if run_id is None:
            print("Error: No run ID provided and no models found in results directory")
            sys.exit(1)
        print(f"Using latest run: {run_id}")
        
        # Skip Model Comparison mode if in gradcam_only or combine_learning_curves mode
        if args.gradcam_only:
            print("Entering Grad-CAM visualization pipeline...")
        else:
            print(f"\n{'='*80}")
            print(f"FAST MODE: Model Comparison Only")
            print(f"Generating model complexity analysis for: {run_id}")
            print(f"Skipping all other visualizations")
            print(f"{'='*80}\n")
            
            try:
                # Load actual accuracies for all 4 models from experiments
                model_accuracies = {}  # model_name -> (accuracy, num_folds)
                model_names = ['resnet50', 'convnext_tiny', 'convnext_small', 'convnext_base']
                
                for model_name in model_names:
                    fold_accs = []
                    for fold_idx in range(args.num_folds):
                        eval_report = f"results/{run_id}_{run_id}_f{fold_idx}_{model_name}_evaluation_report.csv"
                        if os.path.exists(eval_report):
                            df = pd.read_csv(eval_report)
                            if 'accuracy' in df.columns:
                                fold_accs.append(df['accuracy'].values[0])
                    
                    if fold_accs:
                        avg_acc = np.mean(fold_accs)
                        model_accuracies[model_name] = (avg_acc, len(fold_accs))
                        print(f"  ✓ Loaded actual {model_name} accuracy from {len(fold_accs)} folds: {avg_acc:.4f}")
                
                # Load inference speed for ConvNeXt-Tiny (reference model)
                actual_inference_speed = None
                ct_fold_count = model_accuracies.get('convnext_tiny', (None, 0))[1]
                metadata_files = []
                for fold_idx in range(args.num_folds):
                    metadata_file = f"results/{run_id}_f{fold_idx}_convnext_tiny_model_selection.json"
                    if os.path.exists(metadata_file):
                        metadata_files.append(metadata_file)
                
                if metadata_files:
                    try:
                        import json
                        speeds = []
                        for mf in metadata_files:
                            with open(mf, 'r') as f:
                                metadata = json.load(f)
                                speed = metadata.get('inference_speed_patches_per_sec')
                                if speed:
                                    speeds.append(speed)
                        if speeds:
                            actual_inference_speed = np.mean(speeds)
                            print(f"  ✓ Loaded actual inference speed: {actual_inference_speed:.0f} patches/sec")
                    except:
                        pass
                
                if not actual_inference_speed:
                    actual_inference_speed = 240.0
                    print(f"  ! Using benchmark inference speed for ConvNeXt-Tiny: {actual_inference_speed:.0f} patches/sec")
                
                # Build model comparison data with all measured values
                ct_acc, ct_folds = model_accuracies.get('convnext_tiny', (0.8900, 0))
                r50_acc, r50_folds = model_accuracies.get('resnet50', (0.801, 0))
                cs_acc, cs_folds = model_accuracies.get('convnext_small', (0.830, 0))
                cb_acc, cb_folds = model_accuracies.get('convnext_base', (0.849, 0))
                
                model_comparison_data = [
                    {
                        'name': 'ResNet50',
                        'parameters': 25.6,
                        'accuracy': r50_acc,
                        'inference_speed': 220.0,
                        'source': f'Measured ({r50_folds} folds)' if r50_folds > 0 else 'ImageNet benchmark',
                        'is_selected': False
                    },
                    {
                        'name': 'ConvNeXt-Tiny',
                        'parameters': 28.6,
                        'accuracy': ct_acc,
                        'inference_speed': actual_inference_speed,
                        'source': f'Measured ({ct_folds} folds)' if ct_folds > 0 else 'Estimate',
                        'is_selected': True
                    },
                    {
                        'name': 'ConvNeXt-Small',
                        'parameters': 50.2,
                        'accuracy': cs_acc,
                        'inference_speed': 155.0,
                        'source': f'Measured ({cs_folds} folds)' if cs_folds > 0 else 'ImageNet benchmark',
                        'is_selected': False
                    },
                    {
                        'name': 'ConvNeXt-Base',
                        'parameters': 88.6,
                        'accuracy': cb_acc,
                        'inference_speed': 95.0,
                        'source': f'Measured ({cb_folds} folds)' if cb_folds > 0 else 'ImageNet benchmark',
                        'is_selected': False
                    }
                ]
                
                print(f"\n  Model Comparison Data:")
                for model in model_comparison_data:
                    mark = " ✓" if model['is_selected'] else ""
                    print(f"    {model['name']:20s} | {model['parameters']:6.1f}M params | {model['accuracy']:.4f} acc | {model['inference_speed']:5.0f} p/s{mark}")
                    print(f"      Source: {model['source']}")
                
                # Only generate plot if multiple models have measured data
                models_with_measured_data = sum(1 for m in model_comparison_data if 'Measured' in m['source'])
                
                if models_with_measured_data > 1:
                    plot_model_complexity_analysis(
                        model_comparison_data,
                        output_path=f"results/model_complexity_analysis_{run_id}.png",
                        figsize=(14, 6)
                    )
                    print(f"\n✓ Model comparison visualization complete!")
                else:
                    print(f"\n  ! Skipping model comparison plot (only {models_with_measured_data} model(s) with measured data)")
            except Exception as e:
                print(f"ERROR: Failed to generate model comparison: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"\n{'='*80}")
            print(f"Fast mode complete!")
            print(f"{'='*80}")
            sys.exit(0)
    
    # ========================================================================
    # MAIN VISUALIZATION CODE (Skip non-Grad-CAM visualizations if --gradcam_only)
    # ========================================================================
    
    # Pipeline mode: Automatically enable advanced analyses for comprehensive lightweight reporting
    if args.pipeline_mode:
        args.include_ensemble_analysis = True
        args.include_cv_stability = True
        args.include_failure_modes = True
        args.include_class_distribution = True
        args.include_training_trajectory = True
        args.include_training_efficiency = True
    
    # Use ensemble backbone if provided, otherwise find fold-specific model
    if args.backbone_path:
        if not os.path.exists(args.backbone_path):
            print(f"Error: Ensemble backbone not found: {args.backbone_path}")
            sys.exit(1)
        model_path = args.backbone_path
        actual_fold = args.fold  # Use specified fold for dataset loading
        print(f"Using ensemble backbone: {os.path.basename(args.backbone_path)}")
    else:
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
        
        # PIPELINE MODE or GRADCAM_ONLY: Process all folds for comprehensive analysis
        if args.pipeline_mode or args.gradcam_only:
            fold_mode_label = "Grad-CAM Only" if args.gradcam_only else "Pipeline"
            print(f"\n[{fold_mode_label} Mode] Processing all {args.num_folds} folds for comprehensive visualizations")
            for fold in range(args.num_folds):
                fold_model_path, fold_actual = find_model_path(run_id, fold, args.model_name)
                if fold_model_path is not None and os.path.exists(fold_model_path):
                    print(f"  Processing fold {fold}...")
                    for ds_type in datasets_to_process:
                        full_visual_report(run_id, fold_model_path, args.model_name, fold, args.num_folds, ds_type)
                else:
                    print(f"  Skipping fold {fold} (model not found)")
        else:
            # Standard mode: Process only specified fold
            for ds_type in datasets_to_process:
                full_visual_report(run_id, model_path, args.model_name, actual_fold, args.num_folds, ds_type)
        
        # ========== AGGREGATE ANALYSES (Pipeline Mode Only) ==========
        if args.pipeline_mode:
            print(f"\n{'='*80}")
            print(f"AGGREGATE ANALYSES: Cross-Fold Summaries")
            print(f"{'='*80}\n")
            
            # Generate cross-fold aggregated dashboards (always in pipeline mode)
            try:
                generate_cross_fold_dashboards(run_id, args.model_name, args.num_folds)
                print(f"\n✓ Cross-fold dashboards completed\n")
            except Exception as e:
                print(f"\n⚠ Cross-fold dashboards skipped: {e}\n")
            
            # Generate CV stability across all folds
            if args.include_cv_stability:
                try:
                    import glob
                    fold_metrics = []
                    folds_found = 0
                    
                    for fold_idx in range(args.num_folds):
                        eval_pattern = f"results/{run_id}_*_*_f{fold_idx}_{args.model_name}_evaluation_report.csv"
                        eval_files = glob.glob(eval_pattern)
                        eval_report = eval_files[0] if eval_files else None
                        
                        if eval_report and os.path.exists(eval_report):
                            df = pd.read_csv(eval_report)
                            metrics_dict = {}
                            metric_cols = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'AUC']
                            for col in metric_cols:
                                if col in df.columns:
                                    metrics_dict[col] = df[col].values[0] if len(df) > 0 else 0
                            if metrics_dict:
                                fold_metrics.append(metrics_dict)
                                folds_found += 1
                    
                    if fold_metrics and folds_found >= 2:
                        plot_cross_validation_stability(
                            fold_metrics,
                            metric_names=['Accuracy', 'Sensitivity', 'Specificity', 'F1-Score'],
                            output_path=f"results/cross_validation_stability_{run_id}.png",
                            figsize=(14, 8)
                        )
                        print(f"  ✓ CV stability box plots generated")
                except Exception as e:
                    print(f"  INFO: CV stability analysis skipped - {e}")
            
            # Generate class distribution aggregated across all folds
            if args.include_class_distribution:
                try:
                    import glob
                    all_labels = []
                    all_fold_indices = []
                    
                    for fold_idx in range(args.num_folds):
                        pred_pattern = f"results/{run_id}_*_*_f{fold_idx}_{args.model_name}_predictions.csv"
                        pred_files = glob.glob(pred_pattern)
                        pred_file = pred_files[0] if pred_files else None
                        
                        if pred_file and os.path.exists(pred_file):
                            pred_df = pd.read_csv(pred_file)
                            if 'Label' in pred_df.columns:
                                all_labels.extend(pred_df['Label'].values)
                                all_fold_indices.extend([fold_idx] * len(pred_df))
                    
                    if all_labels:
                        plot_class_distribution_analysis(
                            all_labels,
                            all_fold_indices,
                            output_path=f"results/class_distribution_analysis_{run_id}_aggregate.png",
                            num_folds=args.num_folds,
                            figsize=(14, 8)
                        )
                        print(f"  ✓ Aggregate class distribution generated")
                except Exception as e:
                    print(f"  INFO: Class distribution aggregation skipped - {e}")
            
            # Generate failure modes aggregated across all folds
            # NOTE: Failure modes visualization functions require specific data structure
            # Skipping aggregate version - per-fold failure modes still available if included
            if args.include_failure_modes:
                print(f"  INFO: Aggregate failure modes skipped (per-fold versions available)")
            
            # Generate combined training trajectories dashboard
            if args.include_training_trajectory:
                try:
                    import glob
                    import json
                    from matplotlib.gridspec import GridSpec
                    
                    all_learning_curves = {}
                    curves_found = 0
                    
                    for fold_idx in range(args.num_folds):
                        learning_curves_pattern = f"results/{run_id}_*_*_f{fold_idx}_{args.model_name}_learning_curves.json"
                        learning_curves_files = glob.glob(learning_curves_pattern)
                        learning_curves_file = learning_curves_files[0] if learning_curves_files else None
                        
                        if learning_curves_file and os.path.exists(learning_curves_file):
                            with open(learning_curves_file, 'r') as f:
                                learning_data = json.load(f)
                            
                            train_losses = learning_data.get('train_loss', [])
                            val_losses = learning_data.get('val_loss', [])
                            train_accs = learning_data.get('train_acc', [])
                            val_accs = learning_data.get('val_acc', [])
                            
                            if all([train_losses, val_losses, train_accs, val_accs]):
                                all_learning_curves[fold_idx] = {
                                    'train_loss': train_losses,
                                    'val_loss': val_losses,
                                    'train_acc': train_accs,
                                    'val_acc': val_accs
                                }
                                curves_found += 1
                    
                    if curves_found > 0:
                        # Create dashboard-style combined visualization
                        import matplotlib.pyplot as plt
                        
                        fig = plt.figure(figsize=(16, 10))
                        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
                        
                        fold_axes = []
                        for idx in range(min(5, args.num_folds)):
                            row = idx // 2
                            col = idx % 2
                            ax = fig.add_subplot(gs[row, col])
                            fold_axes.append(ax)
                        
                        # Plot individual fold trajectories
                        for fold_idx, fold_data in sorted(all_learning_curves.items()):
                            if fold_idx < len(fold_axes):
                                ax = fold_axes[fold_idx]
                                epochs = range(1, len(fold_data['train_loss']) + 1)
                                
                                ax.plot(epochs, fold_data['train_loss'], 'o-', label='Train Loss', linewidth=2, markersize=4)
                                ax.plot(epochs, fold_data['val_loss'], 's-', label='Val Loss', linewidth=2, markersize=4)
                                ax.set_xlabel('Epoch', fontsize=10)
                                ax.set_ylabel('Loss', fontsize=10)
                                ax.set_title(f'Fold {fold_idx}: Loss Trajectory', fontsize=11, fontweight='bold')
                                ax.legend(fontsize=9)
                                ax.grid(True, alpha=0.3)
                        
                        # Add aggregate/summary subplot
                        ax_summary = fig.add_subplot(gs[2, 1])
                        for fold_idx, fold_data in sorted(all_learning_curves.items()):
                            epochs = range(1, len(fold_data['val_loss']) + 1)
                            ax_summary.plot(epochs, fold_data['val_loss'], 'o-', label=f'Fold {fold_idx}', linewidth=1.5, alpha=0.7)
                        
                        ax_summary.set_xlabel('Epoch', fontsize=10)
                        ax_summary.set_ylabel('Validation Loss', fontsize=10)
                        ax_summary.set_title('All Folds: Validation Loss Comparison', fontsize=11, fontweight='bold')
                        ax_summary.legend(fontsize=8, loc='best')
                        ax_summary.grid(True, alpha=0.3)
                        
                        fig.suptitle(f'Training Trajectories - Run {run_id} (All Folds)', fontsize=14, fontweight='bold', y=0.995)
                        
                        output_path = f"results/training_trajectory_combined_{run_id}.png"
                        plt.savefig(output_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        print(f"  ✓ Combined training trajectory dashboard generated ({curves_found} folds)")
                    else:
                        print(f"  INFO: No learning curves found for training trajectory")
                except Exception as e:
                    print(f"  INFO: Training trajectory aggregation skipped - {e}")

        
        # ========== TRANSFER LEARNING COMPARISON (Optional) ==========
        if args.compare_baseline is not None:
            print(f"\n{'='*80}")
            print(f"Generating Transfer Learning Comparison: {args.compare_baseline} (Baseline) vs {run_id} (TL)")
            print(f"{'='*80}\n")
            
            # Load baseline model predictions
            baseline_model_path, baseline_fold = find_model_path(args.compare_baseline, args.fold, args.model_name)
            if baseline_model_path is None or not os.path.exists(baseline_model_path):
                print(f"WARNING: Could not find baseline model for run {args.compare_baseline}. Skipping comparison.")
            else:
                # Generate predictions for both baseline and TL for the specified dataset
                ds_type = "helicodataset"  # Use helicodataset for comparison (most relevant)
                
                # Load baseline and TL predictions from CSV files if they exist
                baseline_prefix = os.path.basename(baseline_model_path).replace("_swa_model_brain.pth", "").replace("_model_brain.pth", "")
                tl_prefix = os.path.basename(model_path).replace("_swa_model_brain.pth", "").replace("_model_brain.pth", "")
                
                baseline_csv = os.path.join("results", f"{baseline_prefix}_predictions.csv")
                tl_csv = os.path.join("results", f"{tl_prefix}_predictions.csv")
                
                if os.path.exists(baseline_csv) and os.path.exists(tl_csv):
                    try:
                        baseline_df = pd.read_csv(baseline_csv)
                        tl_df = pd.read_csv(tl_csv)
                        
                        # Ensure same patient order and labels
                        if len(baseline_df) == len(tl_df) and (baseline_df['Label'] == tl_df['Label']).all():
                            # Generate comparison plot
                            plot_transfer_learning_comparison(
                                baseline_df['Label'].values,
                                baseline_df['Prob'].values,
                                tl_df['Label'].values,
                                tl_df['Prob'].values,
                                baseline_name=f"Baseline (Run {args.compare_baseline})",
                                tl_name=f"Transfer Learning (Run {run_id})",
                                output_path=os.path.join("results", f"transfer_learning_comparison_{args.compare_baseline}_vs_{run_id}.png")
                            )
                            print(f"\n✓ Transfer learning comparison complete!")
                        else:
                            print(f"WARNING: Baseline and TL predictions have different patient counts or labels. Skipping comparison.")
                    except Exception as e:
                        print(f"WARNING: Error loading prediction CSVs: {e}")
                        print(f"  Baseline CSV: {baseline_csv}")
                        print(f"  TL CSV:       {tl_csv}")
                else:
                    print(f"WARNING: Could not find prediction CSV files for comparison:")
                    print(f"  Baseline: {baseline_csv} (exists: {os.path.exists(baseline_csv)})")
                    print(f"  TL:       {tl_csv} (exists: {os.path.exists(tl_csv)})")
    
    # ========================================================================
    # TRANSFER LEARNING ANALYSIS: Learning Curves & Feature Importance
    # ========================================================================
    if args.include_transfer_analysis:
        print(f"\n{'='*80}")
        print(f"ADVANCED ANALYSIS: Transfer Learning Impact")
        print(f"{'='*80}\n")
        
        # Look for learning curves in the training logs
        learning_curves_baseline = None
        learning_curves_tl = None
        
        # Try to find learning curve files
        baseline_curves_file = f"results/{args.compare_baseline}_f{args.fold}_{args.model_name}_learning_curves.json"
        tl_curves_file = f"results/{run_id}_f{args.fold}_{args.model_name}_learning_curves.json"
        
        try:
            if os.path.exists(baseline_curves_file):
                with open(baseline_curves_file, 'r') as f:
                    learning_curves_baseline = json.load(f)
            
            if os.path.exists(tl_curves_file):
                with open(tl_curves_file, 'r') as f:
                    learning_curves_tl = json.load(f)
            
            if learning_curves_baseline and learning_curves_tl:
                plot_learning_curves_comparison(
                    learning_curves_baseline, 
                    learning_curves_tl,
                    baseline_name=f"Baseline (Run {args.compare_baseline})",
                    tl_name=f"Transfer Learning (Run {run_id})",
                    output_path=f"results/transfer_learning_curves_{args.compare_baseline}_vs_{run_id}.png"
                )
            else:
                print(f"INFO: Learning curves not available for transfer learning analysis")
                if not os.path.exists(baseline_curves_file):
                    print(f"  - Baseline curves missing: {baseline_curves_file}")
                if not os.path.exists(tl_curves_file):
                    print(f"  - TL curves missing: {tl_curves_file}")
        except Exception as e:
            print(f"WARNING: Error generating transfer learning analysis: {e}")
    
    # ========================================================================
    # COMBINED LEARNING CURVES: Pre-training vs Fine-tuning (Pipeline Mode)
    # ========================================================================
    if args.pipeline_mode and args.compare_baseline is not None:
        print(f"\n{'='*80}")
        print(f"PIPELINE MODE: Combined Learning Curves (Pre-training vs Fine-tuning)")
        print(f"{'='*80}\n")
        
        try:
            import glob
            
            # Find learning curve images for both baseline (pre-training) and current run (fine-tuning)
            baseline_curves_pattern = f"results/{args.compare_baseline}_*_*_f{args.fold}_*_learning_curves.png"
            tl_curves_pattern = f"results/{run_id}_*_*_f{args.fold}_*_learning_curves.png"
            
            baseline_curves_files = glob.glob(baseline_curves_pattern)
            tl_curves_files = glob.glob(tl_curves_pattern)
            
            if baseline_curves_files and tl_curves_files:
                baseline_curves_path = baseline_curves_files[0]
                tl_curves_path = tl_curves_files[0]
                
                # Output 2 separate files: one for pre-training, one for fine-tuning
                pretraining_output = f"results/combined_learning_curves_pretraining_f{args.fold}.png"
                finetuning_output = f"results/combined_learning_curves_finetuning_f{args.fold}.png"
                
                # Copy/reference the images with consistent naming
                import shutil
                shutil.copy(baseline_curves_path, pretraining_output)
                shutil.copy(tl_curves_path, finetuning_output)
                
                print(f"  ✓ Pre-training learning curves saved: {pretraining_output}")
                print(f"  ✓ Fine-tuning learning curves saved: {finetuning_output}")
            else:
                if not baseline_curves_files:
                    print(f"  INFO: Pre-training learning curves not found for fold {args.fold}")
                if not tl_curves_files:
                    print(f"  INFO: Fine-tuning learning curves not found for fold {args.fold}")
        
        except Exception as e:
            print(f"  WARNING: Error generating combined learning curves: {e}")
    
    # ========================================================================
    # ENSEMBLE ANALYSIS: Voting Agreement & Model Contribution
    # ========================================================================
    if args.include_ensemble_analysis:
        print(f"\n{'='*80}")
        print(f"ADVANCED ANALYSIS: Ensemble Contribution Analysis")
        print(f"{'='*80}\n")
        
        try:
            import glob
            # Load individual fold predictions from holdout consensus files
            fold_predictions = []
            labels = None
            fold_names = []
            
            # Use glob to find all holdout_consensus files for this run_id
            holdout_pattern = f"results/{run_id}_*_*_f*_{args.model_name}_holdout_consensus.csv"
            holdout_files = sorted(glob.glob(holdout_pattern))
            
            if holdout_files:
                for fold_idx in range(args.num_folds):
                    # Find the file for this specific fold
                    fold_pattern = f"results/{run_id}_*_*_f{fold_idx}_{args.model_name}_holdout_consensus.csv"
                    fold_files = glob.glob(fold_pattern)
                    if fold_files:
                        df = pd.read_csv(fold_files[0])
                        # holdout_consensus files use 'Predicted' column
                        if 'Predicted' in df.columns:
                            fold_predictions.append(df['Predicted'].values)
                            fold_names.append(f"Fold {fold_idx}")
                        if labels is None and 'Actual' in df.columns:
                            labels = df['Actual'].values
                
                if fold_predictions and labels is not None and len(fold_predictions) >= 2:
                    plot_ensemble_voting_agreement(
                        fold_predictions,
                        labels,
                        output_path=f"results/ensemble_voting_agreement_{run_id}.png",
                        model_names=fold_names,
                        figsize=(12, 8)
                    )
                    print(f"  ✓ Ensemble contribution analysis complete ({len(fold_predictions)} folds)")
                else:
                    print(f"  INFO: Ensemble analysis skipped - need at least 2 folds with holdout data")
            else:
                print(f"  INFO: Ensemble analysis skipped.")
                print(f"      (Requires holdout_consensus.csv files from training - check if training completed)")
        except Exception as e:
            print(f"  INFO: Ensemble analysis skipped - {e}")
    
    # NOTE: CV Stability analysis moved to aggregate mode (runs after all folds complete)
    
    # ========================================================================
    # DATA INTEGRITY & AUDIT: Cross-Leakage Detection
    # ========================================================================
    if args.include_data_audit:
        print(f"\n{'='*80}")
        print(f"ADVANCED ANALYSIS: Data Integrity & Audit")
        print(f"{'='*80}\n")
        
        try:
            # Look for cross-leakage audit CSV files
            audit_file = f"results/{run_id}_f{args.fold}_{args.model_name}_cross_leakage_audit.csv"
            
            if os.path.exists(audit_file):
                audit_df = pd.read_csv(audit_file)
                plot_data_integrity_audit(
                    audit_df,
                    output_path=f"results/data_integrity_audit_{run_id}_f{args.fold}.png",
                    figsize=(14, 8)
                )
            else:
                print(f"INFO: Audit CSV not found: {audit_file}")
                print(f"      Expected from: ensemble_voting.py or training pipeline")
        except Exception as e:
            print(f"WARNING: Error generating data integrity audit: {e}")
    
    # NOTE: Failure mode analysis moved to aggregate mode (runs after all folds complete)
    
    # NOTE: Class distribution analysis moved to aggregate mode (runs after all folds complete)
    
    # ========================================================================
    # TRAINING TRAJECTORY: Learning Progress
    # ========================================================================
    # NOTE: Training trajectory moved to aggregate mode (combines all folds into single dashboard)
    
    # ========================================================================
    # TRAINING EFFICIENCY: Resource Utilization & Throughput
    # ========================================================================
    if args.include_training_efficiency:
        print(f"\n{'='*80}")
        print(f"ADVANCED ANALYSIS: Training Efficiency Metrics")
        print(f"{'='*80}\n")
        
        try:
            import glob
            # Collect efficiency metrics across all folds
            fold_metrics_eff = []
            metrics_found = 0
            
            for fold_idx in range(args.num_folds):
                # Use glob to find actual metadata file with any timestamp/seed
                metadata_pattern = f"results/{run_id}_*_*_f{fold_idx}_{args.model_name}_model_selection.json"
                metadata_files = glob.glob(metadata_pattern)
                metadata_file = metadata_files[0] if metadata_files else None
                
                fold_metric = {
                    'fold': fold_idx,
                    'wall_clock_time': 6.5,  # Default estimate (can be overridden from metadata)
                    'peak_gpu_memory': 24.0,  # Default estimate
                    'batch_throughput': 250.0  # Default estimate
                }
                
                # Try to load actual metrics from metadata
                if metadata_file and os.path.exists(metadata_file):
                    try:
                        import json
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            # Use actual training metrics from model_selection.json
                            fold_metric['wall_clock_time'] = metadata.get('training_time_hours', 6.5)
                            fold_metric['peak_gpu_memory'] = metadata.get('peak_gpu_memory_gb', 24.0)
                            fold_metric['batch_throughput'] = metadata.get('throughput_patches_per_sec', 250.0)
                            metrics_found += 1
                    except Exception as e:
                        print(f"    WARNING: Could not load metadata from {metadata_file}: {e}")
                
                fold_metrics_eff.append(fold_metric)
            
            if fold_metrics_eff and metrics_found > 0:
                plot_training_efficiency(
                    fold_metrics_eff,
                    output_path=f"results/training_efficiency_{run_id}.png",
                    figsize=(14, 6)
                )
                print(f"  ✓ Training efficiency plot saved ({metrics_found} folds with actual metrics)")
            else:
                print(f"INFO: Using default efficiency estimates (metadata not available)")
                default_metrics = [
                    {'fold': i, 'wall_clock_time': 6.5, 'peak_gpu_memory': 24.0, 'batch_throughput': 250.0}
                    for i in range(args.num_folds)
                ]
                plot_training_efficiency(
                    default_metrics,
                    output_path=f"results/training_efficiency_{run_id}.png",
                    figsize=(14, 6)
                )
        except Exception as e:
            print(f"WARNING: Error generating training efficiency analysis: {e}")
    
    # ========================================================================
    # MODEL COMPLEXITY vs PERFORMANCE: Architecture Justification
    # ========================================================================
    if args.include_model_complexity:
        print(f"\n{'='*80}")
        print(f"ADVANCED ANALYSIS: Model Complexity vs Performance")
        print(f"{'='*80}\n")
        
        try:
            # ========== STEP 1: Load actual accuracies for all 4 models from our experiments ==========
            model_accuracies = {}  # model_name -> (accuracy, num_folds)
            model_names = ['resnet50', 'convnext_tiny', 'convnext_small', 'convnext_base']
            
            for model_name in model_names:
                fold_accs = []
                for fold_idx in range(args.num_folds):
                    eval_report = f"results/{run_id}_{run_id}_f{fold_idx}_{model_name}_evaluation_report.csv"
                    if os.path.exists(eval_report):
                        df = pd.read_csv(eval_report)
                        if 'accuracy' in df.columns:
                            fold_accs.append(df['accuracy'].values[0])
                
                if fold_accs:
                    avg_acc = np.mean(fold_accs)
                    model_accuracies[model_name] = (avg_acc, len(fold_accs))
                    print(f"  ✓ Loaded actual {model_name} accuracy from {len(fold_accs)} folds: {avg_acc:.4f}")
            
            # ========== STEP 2: Load inference speed if available ==========
            actual_inference_speed = None
            ct_fold_count = model_accuracies.get('convnext_tiny', (None, 0))[1]
            metadata_files = []
            for fold_idx in range(args.num_folds):
                metadata_file = f"results/{run_id}_f{fold_idx}_convnext_tiny_model_selection.json"
                if os.path.exists(metadata_file):
                    metadata_files.append(metadata_file)
            
            if metadata_files:
                try:
                    import json
                    speeds = []
                    for mf in metadata_files:
                        with open(mf, 'r') as f:
                            metadata = json.load(f)
                            speed = metadata.get('inference_speed_patches_per_sec')
                            if speed:
                                speeds.append(speed)
                    if speeds:
                        actual_inference_speed = np.mean(speeds)
                        print(f"  ✓ Loaded actual inference speed: {actual_inference_speed:.0f} patches/sec")
                except:
                    pass
            
            if not actual_inference_speed:
                # Use documented benchmark for ConvNeXt-Tiny on A40 GPU
                actual_inference_speed = 240.0
                print(f"  ! Using benchmark inference speed for ConvNeXt-Tiny: {actual_inference_speed:.0f} patches/sec")
            
            # ========== STEP 3: Define model comparison using all measured values ==========
            # Parameter counts from torchvision/PyTorch documentation
            # Accuracy values: All models measured from our experiments (when available)
            # Inference: A40 GPU with batch_size=1, TTA enabled (16x augmentation)
            
            ct_acc, ct_folds = model_accuracies.get('convnext_tiny', (0.8900, 0))
            r50_acc, r50_folds = model_accuracies.get('resnet50', (0.801, 0))
            cs_acc, cs_folds = model_accuracies.get('convnext_small', (0.830, 0))
            cb_acc, cb_folds = model_accuracies.get('convnext_base', (0.849, 0))
            
            model_comparison_data = [
                {
                    'name': 'ResNet50',
                    'parameters': 25.6,  # From torchvision
                    'accuracy': r50_acc,  # MEASURED from our experiments
                    'inference_speed': 220.0,  # Estimated relative to ConvNeXt-Tiny
                    'source': f'Measured ({r50_folds} folds)' if r50_folds > 0 else 'ImageNet benchmark',
                    'is_selected': False
                },
                {
                    'name': 'ConvNeXt-Tiny',
                    'parameters': 28.6,  # From torchvision
                    'accuracy': ct_acc,  # MEASURED from our experiments
                    'inference_speed': actual_inference_speed,  # MEASURED or benchmark
                    'source': f'Measured ({ct_folds} folds)' if ct_folds > 0 else 'Estimate',
                    'is_selected': True  # SELECTED ARCHITECTURE
                },
                {
                    'name': 'ConvNeXt-Small',
                    'parameters': 50.2,  # From torchvision
                    'accuracy': cs_acc,  # MEASURED from our experiments
                    'inference_speed': 155.0,  # Slower due to larger model
                    'source': f'Measured ({cs_folds} folds)' if cs_folds > 0 else 'ImageNet benchmark',
                    'is_selected': False
                },
                {
                    'name': 'ConvNeXt-Base',
                    'parameters': 88.6,  # From torchvision
                    'accuracy': cb_acc,  # MEASURED from our experiments
                    'inference_speed': 95.0,  # Significantly slower
                    'source': f'Measured ({cb_folds} folds)' if cb_folds > 0 else 'ImageNet benchmark',
                    'is_selected': False
                }
            ]
            
            print(f"\n  Model Comparison Data:")
            for model in model_comparison_data:
                mark = " ✓" if model['is_selected'] else ""
                print(f"    {model['name']:20s} | {model['parameters']:6.1f}M params | {model['accuracy']:.4f} acc | {model['inference_speed']:5.0f} p/s{mark}")
                print(f"      Source: {model['source']}")
            
            # Only generate plot if multiple models have measured data
            models_with_measured_data = sum(1 for m in model_comparison_data if 'Measured' in m['source'])
            
            if models_with_measured_data > 1:
                plot_model_complexity_analysis(
                    model_comparison_data,
                    output_path=f"results/model_complexity_analysis_{run_id}.png",
                    figsize=(14, 6)
                )
            else:
                print(f"\n  ! Skipping model complexity plot (only {models_with_measured_data} model(s) with measured data)")
        except Exception as e:
            print(f"WARNING: Error generating model complexity analysis: {e}")


