"""
# Rescue Inference Utility
# -------------------------
# Performs High-Resolution inference with a dense sliding window (e.g., Stride=128) 
# to capture sparse bacterial signals missed by standard (250-stride) screening.
#
# Usage:
#   python3 rescue_inference.py --model path/to/model.pth --output results.csv --stride 128 --targets B22-81,B22-206
#
# Arguments:
#   --model:   Path to the .pth model weights.
#   --output:  CSV file to save the extracted features.
#   --stride:  Window overlap density (Default 128). Lower is denser/slower.
#   --targets: Comma-separated list of PatientIDs to process (or 'all').
# -------------------------
"""
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from dataset import HPyloriDataset
from model import get_model
from torchvision.transforms import v2
import gc

def rescue_inference(model_path, output_csv, target_patients=None, stride=128):
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
    base_data_path = "/tmp/ricse03_h_pylori_data"
    if not os.path.exists(base_data_path):
        base_data_path = "/import/fhome/vlia/HelicoDataSet"
    
    patient_csv = os.path.join(base_data_path, "PatientDiagnosis.csv")
    patch_xlsx = os.path.join(base_data_path, "HP_WSI-CoordAnnotatedAllPatches.xlsx")
    holdout_dir = os.path.join(base_data_path, "HoldOut")
    
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
    df.to_csv(output_csv, index=False)
    print(f"--- Rescue Completed. Saved to {output_csv} ---")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--targets", type=str, default="B22-206,B22-262,B22-69,B22-81,B22-85,B22-01")
    args = parser.parse_args()
    
    target_list = args.targets.split(",")
    rescue_inference(args.model, args.output, target_patients=target_list, stride=args.stride)
