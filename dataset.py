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
        Initialization: This runs once when you create the dataset.
        It links the data files with the image folders.
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
        root_parent = os.path.dirname(root_dir)
        search_dirs = [root_dir]
        
        # If we are pointed at 'Annotated', also look in 'Cropped' for extra negatives
        if 'Annotated' in root_dir:
            cropped_dir = os.path.join(root_parent, 'Cropped')
            if os.path.exists(cropped_dir):
                search_dirs = search_dirs + [cropped_dir]

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
                    # even if they don't have patch-level annotations in the Excel
                    elif patient_id in self.patient_densities and self.patient_densities[patient_id] != 'NEGATIVA':
                        file_key = (patient_id, img_name)
                        if file_key not in added_keys:
                            self.samples.append((os.path.join(patient_path, img_name), 1))
                            added_keys.add(file_key)
                    
                    # Priority 4: Otherwise skip
                    else:
                        continue

        # --- Step 4: MIL Bag-Mode Organization ---
        if self.bag_mode:
            # Reorganize samples into bags by patient ID
            patient_bags = {}
            for img_path, label in self.samples:
                # Use the full folder name as the bag ID to keep them granular
                # (e.g. 'B22-47_0', 'B22-47_1' are separate bags)
                folder_name = os.path.basename(os.path.dirname(img_path))
                p_id_full = folder_name
                
                # Extract the base ID to look up clinical labels in Excel
                # (e.g. 'B22-47' or '101')
                base_id = p_id_full.split('_')[0]
                
                if p_id_full not in patient_bags:
                    # Get clinical label from the base ID (root patient status)
                    patient_label = self.patient_labels.get(base_id, 0)
                    patient_bags[p_id_full] = {'samples': [], 'label': patient_label}
                
                patient_bags[p_id_full]['samples'].append(img_path)
            
            # List of (list_of_paths, patient_label, bag_id)
            self.bags = []
            for bag_id, data in patient_bags.items():
                paths = data['samples']
                # No longer truncating at init; __getitem__ handles sampling
                self.bags.append((paths, data['label'], bag_id))

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
            img_paths, label, patient_id = self.bags[idx]
            
            # --- Dynamic Bag Sampling (TOTAL COVERAGE) ---
            if self.train and len(img_paths) > self.max_bag_size:
                # Randomly sample patches to ensure eventually seeing everything
                selected_paths = np.random.choice(img_paths, self.max_bag_size, replace=False)
            elif len(img_paths) > self.max_bag_size:
                # Deterministic for validation/testing
                selected_paths = sorted(img_paths)[:self.max_bag_size]
            else:
                selected_paths = img_paths

            images = []
            for path in selected_paths:
                img = Image.open(path).convert('RGB')
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
