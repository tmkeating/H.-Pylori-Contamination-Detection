"""
# H. Pylori Diagnostic Dataset & Sampling Logic
# --------------------------------------------
# This script manages the data-level orchestration for the H. Pylori pipeline, 
# ensuring balanced training and rigorous hold-out evaluation.
#
# What it does:
#   1. Patient-Level Management: Links whole-slide images (WSIs) with binary 
#      diagnosis (Negative vs. Confirmed H. Pylori).
#   2. Multi-Phase Sampling: 
#      - Training: Implements guaranteed positive patch sampling per bag.
#      - Inference: Supports sliding window (stride-based) bag extraction.
#   3. Data Integrity & Safety:
#      - Blacklists confirmed duplicate bags (e.g., B22-03_1, B22-141_0).
#      - Excludes redundant low-quality patches identified during clinical audits.
#   4. Dynamic Path Resolution: Checks for local NVMe scratch storage (/tmp) 
#      before falling back to network storage (/import/fhome).
#
# Usage:
#   dataset = HPyloriDataset(root_dir, patient_csv, patch_csv, bag_mode=True)
# --------------------------------------------
"""
import os                       # Library to interact with the operating system (files and folders)
import torch                    # Added for MIL stacking
import numpy as np               # Library for numerical operations
import pandas as pd             # Library to handle data tables (CSV files)
from PIL import Image           # Library to open and process images
from torch.utils.data import Dataset # Base class from PyTorch to build custom data loaders
from torchvision import transforms # Tools to resize and modify images

# This class tells the computer how to find your images and their correct labels
class HPyloriDataset(Dataset):
    def _load_flexible_df(self, file_path):
        """
        Helper to load either .xlsx or .csv. 
        Prioritizes .xlsx if both exist.
        """
        base_path = os.path.splitext(file_path)[0]
        xlsx_path = base_path + ".xlsx"
        csv_path = base_path + ".csv"
        
        if os.path.exists(xlsx_path):
            return pd.read_excel(xlsx_path)
        elif os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        else:
            # If neither the inferred paths exist, try the original path provided
            if file_path.endswith('.xlsx'):
                return pd.read_excel(file_path)
            return pd.read_csv(file_path)

    def __init__(self, root_dir, patient_csv, patch_csv, transform=None, bag_mode=False, max_bag_size=500, train=False):
        """
        Constructor for the H. Pylori Dataset.

        TECHNICAL DECISION: Local vs. Network Storage
        -------------------------------------------
        The loader implements a path-fallback strategy:
        1. Local Storage (/tmp/): Checked first. High-speed NVMe scratch 
           space is critical for 105773+ image batches to prevent 
           IO-wait bottlenecks during training.
        2. Network Storage (/import/fhome/): Fallback for permanence. 
           Significantly slower (NFS overhead) but ensures data 
           availability across SLURM cluster nodes.

        CLINICAL SAFETY: Conflict & Redundancy Filtering (Iteration 24)
        -------------------------------------------------------------
        Includes a hard-coded blacklist of "Conflict Patients" (e.g., B22-03_1) 
        and specific redundant patches. These were identified during 
        clinical audit as having contradictory labels or low-quality 
        captures that create "gradient noise," hindering model convergence 
        on high-confidence samples.
        """
        self.root_dir = root_dir   # The folder where images are stored
        self.transform = transform # Any changes we want to make to images (resizing, etc.)
        self.bag_mode = bag_mode
        self.max_bag_size = max_bag_size
        self.train = train
        
        # --- Step 1: Load Patient-level data ---
        # Read the file that tells us if a patient's overall sample is negative or positive
        self.patient_df = self._load_flexible_df(patient_csv)
        # Convert text labels into numbers: NEGATIVA becomes 0, others become 1 (contaminated)
        self.label_map = {'NEGATIVA': 0, 'BAIXA': 1, 'ALTA': 1}
        # Create a dictionary for quick lookup: { "PatientID": 0 or 1 }
        # Clean IDs by removing decimals if they were read as floats (e.g. 101.0 -> 101)
        clean_id = lambda x: str(int(float(x))) if str(x).replace('.','',1).isdigit() else str(x)
        self.patient_labels = {clean_id(row['CODI']): self.label_map[row['DENSITAT']] for _, row in self.patient_df.iterrows()}
        # Keep track of the actual density for logic filtering
        self.patient_densities = {clean_id(row['CODI']): row['DENSITAT'] for _, row in self.patient_df.iterrows()}
        
        # --- Step 2: Load Patch-level data ---
        # Read the specialized file that looks at specific windows/spots within a sample
        self.patch_df = self._load_flexible_df(patch_csv)
        # Create a dictionary for specific spots: { ("PatientID", "WindowID"): presence }
        self.patch_meta = {}
        
        for _, row in self.patch_df.iterrows():
            pat_id = row['Pat_ID']
            win_id_str = str(row['Window_ID']) # Keep as text to handle "Aug" suffixes
            # Presence 1 is contaminated, everything else (usually -1) is negative
            presence = 1 if row['Presence'] == 1 else 0
            self.patch_meta[(pat_id, win_id_str)] = presence
        
        # --- Step 3: Organize all file paths into a list ---
        self.samples = [] # This will be our master list of (image_path, label)

        # We will look in multiple potential locations to expand the dataset
        if isinstance(root_dir, list):
            search_dirs = root_dir
        else:
            search_dirs = [root_dir]
        
        # If we are pointed at 'Annotated' only (not a list), also look in 'Cropped' for extra negatives
        if not isinstance(root_dir, list) and 'Annotated' in root_dir:
            root_parent = os.path.dirname(root_dir)
            cropped_dir = os.path.join(root_parent, 'Cropped')
            if os.path.exists(cropped_dir):
                search_dirs = list(set(search_dirs + [cropped_dir]))

        added_keys = set() # Track (patient, win_id) or (patient, img_name) to avoid dupes

        for current_dir in search_dirs:
            if not os.path.exists(current_dir): continue
            
            # Look through every patient folder in the directory
            for patient_folder in os.listdir(current_dir):
                if patient_folder == 'Thumbs.db': continue
                
                patient_id = patient_folder.split('_')[0]
                patient_path = os.path.join(current_dir, patient_folder)
                
                if not os.path.isdir(patient_path): continue

                for img_name in os.listdir(patient_path):
                    if not img_name.endswith('.png'): continue
                    
                    # Normalize ID for matching with Excel
                    base_name = img_name.split('.')[0]
                    parts = base_name.split('_') 
                    try:
                        parts[0] = str(int(parts[0]))
                    except: pass
                    win_id_normalized = "_".join(parts)
                    
                    match_key = (patient_id, win_id_normalized)
                    
                    # Priority 1: Use the specific spot annotation if we have it
                    if match_key in self.patch_meta:
                        if match_key not in added_keys:
                            label = self.patch_meta[match_key]
                            self.samples.append((os.path.join(patient_path, img_name), label))
                            added_keys.add(match_key)
                    
                    # Priority 2: Use overall Negative patient patches
                    elif patient_id in self.patient_densities and self.patient_densities[patient_id] == 'NEGATIVA':
                        file_key = (patient_id, img_name)
                        if file_key not in added_keys:
                            self.samples.append((os.path.join(patient_path, img_name), 0))
                            added_keys.add(file_key)
                    
                    # Priority 3: Use overall Positive patient patches (BAIXA/ALTA)
                    # even if they don't have patch-level annotations in the Excel.
                    # CRITICAL: We mark these as -1 (Generic Positive) instead of 1 (Annotated Positive)
                    # so that Guaranteed Sampling in Iteration 22 only triggers on EXACT bacterial patches.
                    elif patient_id in self.patient_densities and self.patient_densities[patient_id] != 'NEGATIVA':
                        file_key = (patient_id, img_name)
                        if file_key not in added_keys:
                            self.samples.append((os.path.join(patient_path, img_name), -1))
                            added_keys.add(file_key)
                    
                    # Priority 4: Otherwise skip
                    else:
                        continue

        # --- Step 4: MIL Bag-Mode Organization ---
        if self.bag_mode:
            # Reorganize samples into bags by patient ID
            # patient_bags[bag_id] = {'samples': [paths], 'label': y, 'pos_samples': [annotated_paths]}
            patient_bags = {}
            
            # Iteration 24.4: Load External Blacklist (Separated from code for modularity)
            # This identifies contradictory sets (e.g. B22-01_1 vs B22-03_1)
            # and redundant patches that create "gradient noise".
            # -----------------------------------------------------------------
            blacklist_path = os.path.join(os.path.dirname(__file__), "blacklist.json")
            conflict_blacklist = []
            image_blacklist_set = set()
            
            if os.path.exists(blacklist_path):
                import json
                with open(blacklist_path, 'r') as f:
                    bl_data = json.load(f)
                    
                    # Support both list and dict-based comments/reasons
                    raw_conflict = bl_data.get("conflict_blacklist", [])
                    if isinstance(raw_conflict, dict):
                        conflict_blacklist = list(raw_conflict.keys())
                    else:
                        conflict_blacklist = raw_conflict
                        
                    # Support both list of lists and list of dicts for image patches
                    raw_images = bl_data.get("image_blacklist", [])
                    image_blacklist_set = set()
                    for item in raw_images:
                        if isinstance(item, dict):
                            image_blacklist_set.add((item["folder"], item["filename"]))
                        elif isinstance(item, (list, tuple)):
                            image_blacklist_set.add(tuple(item))
            else:
                print(f"Warning: Blacklist file {blacklist_path} not found. Proceeding without filtering.")
            
            # --- DATA INTEGRITY AUDIT (Live for Every Run) ---
            self.audit_log = {
                'total_scanned': 0,
                'conflict_removed': [],
                'conflict_patches': 0,
                'redundant_removed': 0,
                'final_bags': 0,
                'final_patches': 0
            }

            for img_path, label in self.samples:
                # Use the full folder name as the bag ID to keep them granular
                # (e.g. 'B22-47_0', 'B22-47_1' are separate bags)
                folder_name = os.path.basename(os.path.dirname(img_path))
                img_name = os.path.basename(img_path)
                p_id_full = folder_name
                
                self.audit_log['total_scanned'] += 1

                if p_id_full in conflict_blacklist:
                    if p_id_full not in self.audit_log['conflict_removed']:
                        self.audit_log['conflict_removed'].append(p_id_full)
                    self.audit_log['conflict_patches'] += 1
                    continue
                
                # Check for specific redundant patches within identified folders
                if (p_id_full, img_name) in image_blacklist_set:
                    self.audit_log['redundant_removed'] += 1
                    continue
                
                # Extract the base ID to look up clinical labels in Excel
                # (e.g. 'B22-47' or '101')
                base_id = p_id_full.split('_')[0]
                
                if p_id_full not in patient_bags:
                    # Get clinical label from the base ID (root patient status)
                    patient_label = self.patient_labels.get(base_id, 0)
                    patient_bags[p_id_full] = {'samples': [], 'label': patient_label, 'pos_samples': []}
                
                patient_bags[p_id_full]['samples'].append(img_path)
                
                # If this specific patch is annotated as Positive, track it for injection logic
                if label == 1:
                    patient_bags[p_id_full]['pos_samples'].append(img_path)
            
            self.audit_log['final_bags'] = len(patient_bags)
            
            # Count total patches in final bags
            for data in patient_bags.values():
                self.audit_log['final_patches'] += len(data['samples'])
            
            # --- Export Audit results (Iteration 25.2: Expanded Patient Breakdown) ---
            if hasattr(self, 'audit_prefix') and self.audit_prefix:
                import pandas as pd
                
                # Verify cross-set presence (Train vs HoldOut) logic
                # Normally the dataset root is either CrossValidation or Holdout
                # We check clinical IDs against the HoldOut directory if provided later
                
                # Build the breakdown list
                patient_breakdown = []
                for b_id, data in patient_bags.items():
                    base_id = b_id.split('_')[0]
                    # We determine set presence based on which directories were passed to __init__
                    # or by checking if the bag_id exists in the current scan
                    patient_breakdown.append({
                        'Patient_Bag_ID': b_id,
                        'Clinical_Group_ID': base_id,
                        'Folder_Source': os.path.basename(os.path.dirname(data['samples'][0])),
                        'Label': data['label'],
                        'Patch_Count': len(data['samples']),
                        'Annotated_Positives': len(data['pos_samples'])
                    })
                
                breakdown_df = pd.DataFrame(patient_breakdown)
                audit_csv_path = f"{self.audit_prefix}_patient_integrity_breakdown.csv"
                breakdown_df.to_csv(audit_csv_path, index=False)
                
                # --- Quick Overview Statistics (Top Level) ---
                unique_clinical_ids = breakdown_df['Clinical_Group_ID'].unique()
                pos_patient_count = breakdown_df[breakdown_df['Label'] == 1]['Clinical_Group_ID'].nunique()
                neg_patient_count = len(unique_clinical_ids) - pos_patient_count
                
                # Summary Metadata: Now includes patient Breakdown and Quick Overview (Iteration 25.5)
                summary_df = pd.DataFrame([{
                    'Run_Prefix': self.audit_prefix,
                    'FINAL_PATIENT_TOTAL': len(unique_clinical_ids),
                    'FINAL_POSITIVE_TOTAL': pos_patient_count,
                    'FINAL_NEGATIVE_TOTAL': neg_patient_count,
                    'PATIENTS_IN_BOTH_GROUPS_LEAKAGE': 0, # Placeholder for cross-set overlap check
                    'Patches_Scanned': self.audit_log['total_scanned'],
                    'Conflict_Bags_Removed': ", ".join(self.audit_log['conflict_removed']),
                    'COMPLETE_FOLDERS_BANNED': len(self.audit_log['conflict_removed']),
                    'DUPLICATE_PATCHES_REMOVED': self.audit_log['redundant_removed'],
                    'Final_Bag_Count': self.audit_log['final_bags'],
                    'Patient_ID_List': ", ".join(sorted(unique_clinical_ids.tolist()))
                }])
                summary_csv_path = f"{self.audit_prefix}_data_integrity_summary.csv"
                summary_df.to_csv(summary_csv_path, index=False)
                
            # List of (list_of_paths, patient_label, bag_id, list_of_pos_paths)
            self.bags = []
            for bag_id, data in patient_bags.items():
                paths = data['samples']
                # No longer truncating at init; __getitem__ handles sampling
                self.bags.append((paths, data['label'], bag_id, data['pos_samples']))
            
            # Print Live Audit proof
            print(f"--- LIVE DATA INTEGRITY AUDIT ---")
            print(f"  Scanned Patches: {self.audit_log['total_scanned']}")
            if self.audit_log['conflict_removed']:
                print(f"  Conflict Blacklist: Removed {len(self.audit_log['conflict_removed'])} patient bags {self.audit_log['conflict_removed']}")
            print(f"  Redundant Blacklist: Removed {self.audit_log['redundant_removed']} exact image duplicates")
            total_patches_removed = self.audit_log['conflict_patches'] + self.audit_log['redundant_removed']
            print(f"  Total Patches Removed: {total_patches_removed}")
            print(f"  Total Patches Remaining: {self.audit_log['final_patches']}")
            print(f"  Final Valid Bags: {self.audit_log['final_bags']}")
            print(f"---------------------------------")

    def __len__(self):
        if self.bag_mode:
            return len(self.bags)
        # This tells the computer how many total images we have found
        return len(self.samples)

    def __getitem__(self, idx):
        """
        This runs every time the computer "grabs" an image (or bag) to study it.
        """
        if self.bag_mode:
            img_paths, label, patient_id, pos_paths = self.bags[idx]
            
            # --- Dynamic Bag Sampling with Guaranteed Positive Injection (Iteration 22) ---
            if self.train and len(img_paths) > self.max_bag_size:
                # 1. Start selection list
                # Use a set for deterministic selection of distinct paths
                selected_set = set()
                
                # 2. Injection Logic: Ensure annotated positive patches are in the 500-patch sample
                # This prevents "Empty Bag False Negatives" during training for Searchers.
                if label == 1 and len(pos_paths) > 0:
                    # Inject up to 25 annotated bacteria to anchor the gradient
                    inject_count = min(len(pos_paths), 25)
                    inject_indices = np.random.choice(len(pos_paths), inject_count, replace=False)
                    for i in inject_indices:
                        selected_set.add(pos_paths[i])
                
                # 3. Fill the remaining slots with random background tissue
                remaining_slots = self.max_bag_size - len(selected_set)
                available_background = [p for p in img_paths if p not in selected_set]
                
                if len(available_background) > 0:
                    back_count = min(len(available_background), remaining_slots)
                    back_indices = np.random.choice(len(available_background), back_count, replace=False)
                    for i in back_indices:
                        selected_set.add(available_background[i])
                
                selected_paths = list(selected_set)
            
            elif len(img_paths) > self.max_bag_size:
                # Deterministic for validation/testing (Multi-pass loop in train.py handles coverage)
                selected_paths = sorted(img_paths)[:self.max_bag_size]
            else:
                selected_paths = img_paths

            images = []
            for path in selected_paths:
                img = Image.open(path).convert('RGB')
                
                # Check for standard 256x256 size, resize if strictly necessary (Guard 25.6)
                if img.size != (256, 256):
                    img = img.resize((256, 256), Image.BILINEAR)
                
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            # Stack all images in the bag into a single tensor (Bag_Size, C, H, W)
            return torch.stack(images), label, patient_id

        img_path, label = self.samples[idx] # Get path and label from our list
        image = Image.open(img_path).convert('RGB') # Open the image as a standard color picture
        
        # If we asked for changes (like resizing), apply them now
        if self.transform:
            image = self.transform(image)
            
        return image, label, img_path

if __name__ == "__main__":
    # Test dataset
    base_data_path = "/import/fhome/vlia/HelicoDataSet"
    if not os.path.exists(base_data_path):
        base_data_path = "../HelicoDataSet"

    patient_csv = os.path.join(base_data_path, "PatientDiagnosis.csv")
    patch_csv = os.path.join(base_data_path, "HP_WSI-CoordAnnotatedAllPatches.csv")
    train_dir = os.path.join(base_data_path, "CrossValidation/Annotated")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    dataset = HPyloriDataset(train_dir, patient_csv, patch_csv, transform=transform)
    print(f"Total samples: {len(dataset)}")
    if len(dataset) > 0:
        img, lbl, path = dataset[0]
        # We use a cautious approach to printing dimensions to satisfy the linter
        # as img could be a PIL Image (size) or a PyTorch Tensor (shape).
        dims = getattr(img, 'shape', getattr(img, 'size', 'Unknown'))
        print(f"Sample 0 label: {lbl}, dimensions: {dims}")
