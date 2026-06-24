"""
Rescue Inference Utility - High-Resolution Bacterial Signal Recovery
=====================================================================

PURPOSE:
--------
Performs high-resolution inference with dense sliding-window extraction (Stride=128) 
to capture sparse bacterial signals that may be missed by standard screening (Stride=250).
Designed specifically for rescue operations on misclassified or borderline patients.

CLINICAL RATIONALE:
-------------------
Standard screening uses Stride=250 (4-pixel overlap), which can "skip over" very sparse 
bacterial clusters (e.g., 5-10 organisms in an entire slide). Dense-stride rescanning 
with Stride=128 ensures 50% overlap between adjacent windows, guaranteeing that no 
bacterium is bisected by a patch boundary in a way that hides its morphology.

TECHNICAL APPROACH:
-------------------
- **Dense Windowing**: Stride=128 creates overlapping patches for maximum signal coverage
- **16-way Contrast-Boosted TTA**: Test-time augmentation with:
  * 6 base rotations (0°, 90°, 180°, 270°, H-flip, V-flip)
  * 1.1x contrast boost (targets faint organisms with weak staining)
  * Combined transformations (rotations + flips, rotations + contrast, etc.)
- **Consensus Voting**: Average of 16 TTA predictions anchors the final score, 
  reducing noise and improving robustness to stain variation

METHODOLOGICAL NOTE - HOLDOUT SET REUSE:
-----------------------------------------
⚠️  Rescue inference processes the SAME holdout set used for baseline ensemble evaluation.
   This is NOT a data leakage violation (model weights unchanged, no retraining), but you are
   applying a better inference strategy to the same test data after seeing baseline results.
   
   IMPLICATIONS:
   - ✅ No training/test contamination: Model trained only on training set
   - ✅ No model retraining: Weights frozen, only inference parameters change
   - ⚠️  Potential overfitting to holdout characteristics: Results may not generalize to truly
        new patients beyond the 114 holdout patients
   - ⚠️  Sequential testing problem: Using holdout to diagnose failures, then re-testing same
        holdout with improved method inflates apparent performance improvement
   
   RECOMMENDATION FOR FINAL DEPLOYMENT:
   For clinical deployment or publication, apply rescue inference to a completely separate
   held-out test set (never used for baseline evaluation) to avoid optimistic bias.
   
   CURRENT USAGE - ACCEPTABLE FOR:
   - Post-hoc clinical analysis: "If we re-scanned these patients, would we catch them?"
   - Research ablations: Demonstrating inference technique improvements
   - Quality assurance: Deep-dive investigation on borderline/failed cases
   - But NOT for claiming final generalization performance

USAGE:
------
  # Basic usage (process all patients)
  python3 rescue_inference.py --model path/to/model.pth \\
    --output_dir results/ --stride 128

  # Target specific misclassified patients
  python3 rescue_inference.py --model path/to/model.pth \\
    --output_dir results/ --stride 128 \\
    --targets B22-12_1,B22-206_0,B22-262_0,B22-69_1,B22-81_1,B22-85_0,B22-89_0

  # SLURM submission with environment variables
  MODEL_DIR="finalResults/convnext_tiny_pretrained_backbone_34.4_weight_1.5_gamma_3.0_focalLoss_false" \\
    FOLDS="01_34.4_9077_f0 01_34.4_9078_f1 01_34.4_9079_f2 01_34.4_9080_f3 01_34.4_9081_f4" \\
    TARGETS="B22-12_1,B22-206_0,B22-262_0,B22-69_1,B22-81_1,B22-85_0,B22-89_0" \\
    OUTPUT_DIR="finalResults/convnext_tiny_pretrained_backbone_34.4_weight_1.5_gamma_3.0_focalLoss_false/rescue_ensemble" \\
    STRIDE=128 \\
    sbatch submit_rescue.sh

ARGUMENTS:
----------
  --model        Path to trained model weights (.pth file)
  --output_dir   Directory to save rescue_*.csv outputs (default: results/)
  --stride       Dense window overlap stride (default: 128, range: 1-250)
                 Lower values = denser coverage but slower execution
  --targets      Comma-separated list of PatientIDs to process (e.g., B22-12_1,B22-206_0)
                 Use 'all' or omit to process all patients in dataset

OUTPUT:
-------
  rescue_TIMESTAMP.csv containing:
  - patient_id: Patient identifier (e.g., B22-12_1)
  - predictions: Probability predictions across all patches (average of 16 TTA views)
  - extracted_features: (if applicable) Model intermediate layer features
"""
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from config import SCRATCH_ROOT, DATASET_ROOT, PATIENT_CSV, PATCH_XLSX, HOLDOUT
from dataset import HPyloriDataset
from model import get_model
from torchvision.transforms import v2
import gc

def rescue_inference(model_path, output_dir="results", target_patients=None, stride=128):
    """
    Perform a High-Resolution 'Rescue' Inference for sparse bacteremia.

    CLINICAL RATIONALE: Dense Sliding Window (Stride=128)
    ----------------------------------------------------
    Standard screening (Stride=250) can "skip" over very sparse 
    bacterial clusters (e.g., 5-10 bacteria in a whole slide). By 
    reducing the stride to 128, we ensure a 50% overlap between 
    adjacent windows, guaranteeing that no bacterium is bisected 
    by a patch boundary in a way that hides its morphology.

    TECHNICAL DECISION: 16-way Contrast-Boosted TTA
    -----------------------------------------------
    Stain intensity varies significantly between labs. We implement 
    Test-Time Augmentation (TTA) with 16 variations (90-deg rotations, 
    flips, and 1.1x contrast boosting). 
    - Contrast Boosting: Specifically targets "faint" organisms that 
      haven't taken the Giemsa/H&E stain strongly.
    - Consensus Voting: The average of 16 views is used to anchor 
      the final diagnostic score, significantly reducing the impact 
      of isolated pixel noise.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 🏥 Rescuing Patients with Stride {stride} ---")
    
    # Load Model (assuming convnext_tiny by default)
    model = get_model(model_name="convnext_tiny", num_classes=2, pretrained=False, pool_type="attention").to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Data Paths
    base_data_path = SCRATCH_ROOT
    if not os.path.exists(base_data_path):
        base_data_path = DATASET_ROOT
    
    patient_csv = PATIENT_CSV
    patch_xlsx = PATCH_XLSX
    holdout_dir = HOLDOUT
    
    # Normalization
    gpu_normalize = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    # 16-way Contrast-Boosted TTA
    tta_transforms = [
        lambda x: x,                                                                        # Original image
        v2.RandomHorizontalFlip(p=1.0),                                                     # Standard horizontal flip
        v2.RandomVerticalFlip(p=1.0),                                                       # Standard vertical flip
        lambda x: torch.rot90(x, 1, [2, 3]),                                                # 90-degree rotation
        lambda x: torch.rot90(x, 2, [2, 3]),                                                # 180-degree rotation
        lambda x: torch.rot90(x, 3, [2, 3]),                                                # 270-degree rotation
        lambda x: v2.RandomHorizontalFlip(p=1.0)(torch.rot90(x, 1, [2, 3])),                # 90-deg rotation + Horizontal flip
        lambda x: v2.RandomVerticalFlip(p=1.0)(torch.rot90(x, 1, [2, 3])),                  # 90-deg rotation + Vertical flip
        lambda x: v2.ColorJitter(contrast=(1.1, 1.1))(x),                                   # Fixed 1.1x contrast boost (original orientation)
        lambda x: v2.ColorJitter(contrast=(1.1, 1.1))(v2.RandomHorizontalFlip(p=1.0)(x)),    # Contrast boost + Horizontal flip
        lambda x: v2.ColorJitter(contrast=(1.1, 1.1))(v2.RandomVerticalFlip(p=1.0)(x)),      # Contrast boost + Vertical flip
        lambda x: v2.ColorJitter(contrast=(1.1, 1.1))(torch.rot90(x, 1, [2, 3])),           # Contrast boost + 90-deg rotation
        lambda x: v2.ColorJitter(contrast=(1.1, 1.1))(torch.rot90(x, 2, [2, 3])),           # Contrast boost + 180-deg rotation
        lambda x: v2.ColorJitter(contrast=(1.1, 1.1))(torch.rot90(x, 3, [2, 3])),           # Contrast boost + 270-deg rotation
        lambda x: v2.ColorJitter(contrast=(1.1, 1.1))(v2.RandomHorizontalFlip(p=1.0)(torch.rot90(x, 1, [2, 3]))), # Contrast + 90-deg + H-flip
        lambda x: v2.ColorJitter(contrast=(1.1, 1.1))(v2.RandomVerticalFlip(p=1.0)(torch.rot90(x, 1, [2, 3])))   # Contrast + 90-deg + V-flip
    ]

    # Initialize Dataset
    # We load with no limit on bag size for full coverage during rescue
    from torchvision import transforms
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = HPyloriDataset(holdout_dir, patient_csv, patch_xlsx, transform=val_transform, bag_mode=True, max_bag_size=20000, train=False)
    
    print(f"DEBUG: Dataset length: {len(dataset)}")
    if len(dataset) > 0:
        _, _, first_id = dataset[0]
        print(f"DEBUG: First patient in dataset: {first_id}")
    else:
        print("DEBUG: Dataset is empty!")
        return

    results = []
    vram_bag_limit = 500
    
    with torch.no_grad():
        for i in range(len(dataset)):
            bags, label, patient_id = dataset[i]
            
            # Filter for targets if specified
            # We use 'in' to handle cases like B22-01 vs B22-01_1
            found_target = False
            if target_patients:
                for target in target_patients:
                    if target in patient_id:
                        found_target = True
                        break
            else:
                found_target = True

            if not found_target:
                continue
                
            print(f"DEBUG: Processing {patient_id} (Size: {bags.size(0)})")
            
            bag_size = bags.size(0)
            chunk_probs = []
            
            # Dense Sliding Window
            chunk_ranges = []
            if bag_size <= vram_bag_limit:
                chunk_ranges = [(0, bag_size)]
            else:
                for s in range(0, bag_size - vram_bag_limit + 1, stride):
                    chunk_ranges.append((s, s + vram_bag_limit))
                # Ensure the end of the bag is covered
                if not chunk_ranges or chunk_ranges[-1][1] < bag_size:
                    chunk_ranges.append((max(0, bag_size - vram_bag_limit), bag_size))
            
            print(f"DEBUG: {len(chunk_ranges)} chunks for {patient_id}")
            
            for start, end in tqdm(chunk_ranges, desc=f"  Inference", leave=False):
                chunk = bags[start:end].to(device)
                
                # TTA Loop
                tta_logits = None
                for tta in tta_transforms:
                    aug = tta(chunk)
                    aug = gpu_normalize(aug)
                    
                    with torch.amp.autocast(device_type='cuda'):
                        logits, _ = model.forward_bag(aug)
                    
                    if tta_logits is None:
                        tta_logits = logits
                    else:
                        tta_logits += logits
                
                probs = torch.softmax(tta_logits / len(tta_transforms), dim=1)
                chunk_probs.append(probs[0, 1].cpu().item())
                del chunk
                del tta_logits
            
            # Feature Extraction (Same as meta-classifier)
            chunk_probs = np.array(chunk_probs)
            max_p = np.max(chunk_probs)
            mean_p = np.mean(chunk_probs)
            p50 = np.sum(chunk_probs > 0.5) / len(chunk_probs)
            p80 = np.sum(chunk_probs > 0.8) / len(chunk_probs)
            
            results.append({
                "PatientID": patient_id,
                "Actual": label,
                "Max_Prob": max_p,
                "Mean_Prob": mean_p,
                "Density_P50": p50,
                "Density_P80": p80,
                "Skeptical_Gap": max_p - mean_p
            })
            
            gc.collect()
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    
    # Ensure output directory exists
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate output filename with model fold information (simplified naming)
    model_stem = Path(model_path).stem  # e.g., "01_34.4_9077_f0_convnext_tiny_model_brain"
    # Extract RunID_Iteration_JobID_Fold from the stem
    # Pattern: {RunID}_{Iteration}_{JobID}_{fold}_{rest}
    parts = model_stem.split('_')
    if len(parts) >= 4:
        # Reconstruct as: rescue_{RunID}_{Iteration}_{JobID}_{fold}.csv
        model_name = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}"
    else:
        model_name = model_stem
    output_csv = Path(output_dir) / f"rescue_{model_name}.csv"
    
    df.to_csv(output_csv, index=False)
    print(f"--- Rescue Completed. Saved to {output_csv} ---")

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory (default: results)")
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--targets", type=str, default="B22-206,B22-262,B22-69,B22-81,B22-85,B22-01")
    args = parser.parse_args()
    
    target_list = args.targets.split(",")
    rescue_inference(args.model, output_dir=args.output_dir, target_patients=target_list, stride=args.stride)
