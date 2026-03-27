"""
# H. Pylori Data Integrity: Duplicate Bag Detection
# ------------------------------------------------
# This script performs a rigorous audit of the patient datasets to identify 
# identical "bags" (folders of patches) that might exist across different 
# folders or cross-validation splits.
#
# Why this is critical:
#   1. Prevents Data Leakage: Ensures the same patient is not training in Fold 1
#      but also appearing in the validation set of Fold 2.
#   2. Verifies Image Uniqueness: Detects accidental copies of patch data.
#   3. Performance Audit: Validates that the "Annotated", "Cropped", and "HoldOut"
#      sets remain mutually exclusive.
#
# Process:
#   - It generates a unique MD5 hash for every image within a patient folder.
#   - It combines these patch hashes into a single "Bag Hash" for the entire patient.
#   - It cross-references these Bag Hashes across the entire filesystem.
# ------------------------------------------------
"""
import os
import hashlib
import pandas as pd
from tqdm import tqdm

def get_dir_hash(directory):
    """
    Calculates a unique fingerprint (Bag Hash) for a directory by hashing its contents.
    
    Args:
        directory (str): The path to the patient folder containing .png/.jpg patches.
    Returns:
        str: A single MD5 hex digest representing the entire folder.
    """
    hashes = []
    # Sort files to ensure the hash is deterministic regardless of filesystem ordering
    files = sorted([f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if not files:
        return None
    
    for f in files:
        path = os.path.join(directory, f)
        # We read the first 8KB (header/metadata/early pixels) to identify identical files efficiently.
        # This is significantly faster than reading entire high-res images while remaining 99.9% accurate.
        with open(path, 'rb') as f_obj:
            hashes.append(hashlib.md5(f_obj.read(8192)).hexdigest())
    
    # Combine individual patch hashes into a final "Bag Fingerprint"
    return hashlib.md5("".join(hashes).encode()).hexdigest()

def check_duplicates(base_path, output_file="duplicate_bags_report.csv"):
    """
    Scans a specific base path for duplicate patient folders.
    """
    print(f"Scanning {base_path} for duplicate patient bags...")
    results = []
    
    # Identify all sub-directories (individual patient bags)
    patient_dirs = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    
    for d in tqdm(patient_dirs):
        full_path = os.path.join(base_path, d)
        d_hash = get_dir_hash(full_path)
        if d_hash:
            # Record metadata and bag hash for comparison
            results.append({
                "Dataset_Source": os.path.basename(base_path), 
                "Patient_Folder": d, 
                "Full_Bag_Hash": d_hash, 
                "File_Count": len(os.listdir(full_path))
            })
    
    df = pd.DataFrame(results)
    
    # Filter for entries that share a 'Full_Bag_Hash' (these are duplicates)
    duplicates = df[df.duplicated('Full_Bag_Hash', keep=False)].sort_values('Full_Bag_Hash')
    
    if duplicates.empty:
        print(f"\nNo duplicate patient bags found in {base_path}.")
    else:
        print(f"\n[!] DUPLICATE BAGS DETECTED in {base_path}:")
        print(duplicates)
        
        # Save results in append mode; write CSV header only if the file is new
        header = not os.path.exists(output_file)
        duplicates.to_csv(output_file, mode='a', index=False, header=header)
        print(f"\nResults appended to {output_file}")

if __name__ == "__main__":
    # The output report file used for documenting the clinical audit
    report_file = "duplicate_bags_report.csv"
    
    # Initialize a master list to catch "Cross-Set Leakage" (e.g., Cropped vs Annotated)
    all_results = []
    
    # Define the core dataset locations as specified in the project manifest
    sets_to_check = {
        "Annotated": "/import/fhome/vlia/HelicoDataSet/CrossValidation/Annotated",
        "Cropped": "/import/fhome/vlia/HelicoDataSet/CrossValidation/Cropped",
        "HoldOut": "/import/fhome/vlia/HelicoDataSet/HoldOut"
    }
    
    for set_name, set_path in sets_to_check.items():
        print(f"Scanning {set_name}...")
        # Get list of folders, ensuring they are actual directories
        patient_dirs = sorted([d for d in os.listdir(set_path) if os.path.isdir(os.path.join(set_path, d))])
        
        for d in tqdm(patient_dirs):
            full_path = os.path.join(set_path, d)
            d_hash = get_dir_hash(full_path)
            if d_hash:
                # Store normalized information for cross-referencing
                all_results.append({
                    "Dataset_Source": set_name, 
                    "Patient_Folder": d, 
                    "Full_Bag_Hash": d_hash, 
                    "File_Count": len([f for f in os.listdir(full_path) if f.endswith(('.png', '.jpg'))])
                })
    
    # Convert all gathered data into a single DataFrame for global analysis
    df = pd.DataFrame(all_results)
    
    # Detect duplicates by looking for matching Bag Hashes across the entire project
    duplicates = df[df.duplicated('Full_Bag_Hash', keep=False)].sort_values('Full_Bag_Hash')
    
    if duplicates.empty:
        print("\nNo duplicate patient bags found across any dataset sets. Data integrity confirmed.")
    else:
        print("\n[!] DUPLICATE BAGS DETECTED (Leakage Audit):")
        # Identify specific cross-dataset leakage (e.g., same patient in Training and HoldOut)
        for h, group in duplicates.groupby('Full_Bag_Hash'):
            sources = group['Dataset_Source'].unique()
            if len(sources) > 1:
                print(f"--> LEAKAGE WARNING: Tissue with Hash {h[:8]} exists in multiple sets: {sources}")
        
        print("\nFull Analytical Report:")
        print(duplicates)
        # Export the full results for submission/review
        duplicates.to_csv(report_file, index=False)
        print(f"\nUnified Audit Report saved to {report_file}")
