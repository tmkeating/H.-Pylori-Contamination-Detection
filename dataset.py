import os                       # Library to interact with the operating system (files and folders)
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

    def __init__(self, root_dir, patient_csv, patch_csv, transform=None):
        """
        Initialization: This runs once when you create the dataset.
        It links the data files with the image folders.
        """
        self.root_dir = root_dir   # The folder where images are stored
        self.transform = transform # Any changes we want to make to images (resizing, etc.)
        
        # --- Step 1: Load Patient-level data ---
        # Read the file that tells us if a patient's overall sample is negative or positive
        self.patient_df = self._load_flexible_df(patient_csv)
        # Convert text labels into numbers: NEGATIVA becomes 0, others become 1 (contaminated)
        self.label_map = {'NEGATIVA': 0, 'BAIXA': 1, 'ALTA': 1}
        # Create a dictionary for quick lookup: { "PatientID": 0 or 1 }
        self.patient_labels = {row['CODI']: self.label_map[row['DENSITAT']] for _, row in self.patient_df.iterrows()}
        
        # --- Step 2: Load Patch-level data ---
        # Read the specialized file that looks at specific windows/spots within a sample
        self.patch_df = self._load_flexible_df(patch_csv)
        # Create a dictionary for specific spots: { ("PatientID", "WindowID"): 0 or 1 }
        self.patch_labels = {}
        for _, row in self.patch_df.iterrows():
            pat_id = row['Pat_ID']
            win_id_str = str(row['Window_ID']) # Keep as text to handle "Aug" suffixes
            # Presence 1 is contaminated, everything else (usually -1) is negative
            presence = 1 if row['Presence'] == 1 else 0
            self.patch_labels[(pat_id, win_id_str)] = presence
        
        # --- Step 3: Organize all file paths into a list ---
        self.samples = [] # This will be our master list of (image_path, label)
        # Look through every patient folder in the directory
        for patient_folder in os.listdir(root_dir):
            if patient_folder == 'Thumbs.db': # Ignore hidden system files
                continue
            
            # Extract the ID (like 'B22-101') from the folder name
            patient_id = patient_folder.split('_')[0]
            patient_path = os.path.join(root_dir, patient_folder)
            
            # If it's a folder, look inside it
            if os.path.isdir(patient_path):
                for img_name in os.listdir(patient_path):
                    # Only look for PNG images
                    if not img_name.endswith('.png'):
                        continue
                    
                    # Extract the specific "Window ID" from the filename
                    # Files are named like "00180.png" or "00902_Aug1.png"
                    base_name = img_name.split('.')[0] # Removes ".png"
                    parts = base_name.split('_') # Splits "00902" and "Aug1" if present
                    
                    # Remove leading zeros so "00902" becomes "902" to match the CSV
                    try:
                        parts[0] = str(int(parts[0]))
                    except ValueError:
                        pass # If it's not a number, leave it alone
                    
                    # Put the pieces back together (e.g., "902_Aug1")
                    win_id_normalized = "_".join(parts)
                    
                    # --- Step 4: Find the correct label for this specific image ---
                    # Priority 1: Use the specific spot annotation if we have it
                    if (patient_id, win_id_normalized) in self.patch_labels:
                        label = self.patch_labels[(patient_id, win_id_normalized)]
                    # Priority 2: Use the overall patient diagnosis as a fallback
                    elif patient_id in self.patient_labels:
                        label = self.patient_labels[patient_id]
                    # Priority 3: If no data exists, skip this image
                    else:
                        continue
                    
                    # Add the final path and label to our master list
                    self.samples.append((os.path.join(patient_path, img_name), label))

    def __len__(self):
        # This tells the computer how many total images we have found
        return len(self.samples)

    def __getitem__(self, idx):
        """
        This runs every time the computer "grabs" an image to study it.
        It opens the file, transforms it, and hands it to the model.
        """
        img_path, label = self.samples[idx] # Get path and label from our list
        image = Image.open(img_path).convert('RGB') # Open the image as a standard color picture
        
        # If we asked for changes (like resizing), apply them now
        if self.transform:
            image = self.transform(image)
            
        return image, label # Return the ready-to-study image and its answer

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
        img, lbl = dataset[0]
        # We use a cautious approach to printing dimensions to satisfy the linter
        # as img could be a PIL Image (size) or a PyTorch Tensor (shape).
        dims = getattr(img, 'shape', getattr(img, 'size', 'Unknown'))
        print(f"Sample 0 label: {lbl}, dimensions: {dims}")
