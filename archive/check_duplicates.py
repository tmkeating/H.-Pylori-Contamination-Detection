import os
import hashlib
import pandas as pd
from tqdm import tqdm

def get_dir_hash(directory):
    """Calculates a combined hash of all image files in a directory to detect duplicates."""
    hashes = []
    # Filter for image files only
    files = sorted([f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if not files:
        return None
    
    for f in files:
        path = os.path.join(directory, f)
        # Use a fast hash (md5) of the first 8KB to identify identical images quickly
        with open(path, 'rb') as f_obj:
            hashes.append(hashlib.md5(f_obj.read(8192)).hexdigest())
    
    # Return a single hash of all filtered patch hashes
    return hashlib.md5("".join(hashes).encode()).hexdigest()

def check_duplicates(base_path, output_file="duplicate_bags_report.csv"):
    print(f"Scanning {base_path} for duplicate patient bags...")
    results = []
    
    # Walk through patient folders
    patient_dirs = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    
    for d in tqdm(patient_dirs):
        full_path = os.path.join(base_path, d)
        d_hash = get_dir_hash(full_path)
        if d_hash:
            results.append({"Dataset_Source": os.path.basename(base_path), "Patient_Folder": d, "Full_Bag_Hash": d_hash, "File_Count": len(os.listdir(full_path))})
    
    df = pd.DataFrame(results)
    
    # Find duplicates
    duplicates = df[df.duplicated('Full_Bag_Hash', keep=False)].sort_values('Full_Bag_Hash')
    
    if duplicates.empty:
        print(f"\nNo duplicate patient bags found in {base_path}.")
    else:
        print(f"\n[!] DUPLICATE BAGS DETECTED in {base_path}:")
        print(duplicates)
        
        # Append mode: Write header only if file doesn't exist
        header = not os.path.exists(output_file)
        duplicates.to_csv(output_file, mode='a', index=False, header=header)
        print(f"\nResults appended to {output_file}")

if __name__ == "__main__":
    report_file = "duplicate_bags_report.csv"
    
    # 1. Collect all bags into a single DataFrame to detect cross-set leakage
    all_results = []
    
    sets_to_check = {
        "Annotated": "/import/fhome/vlia/HelicoDataSet/CrossValidation/Annotated",
        "Cropped": "/import/fhome/vlia/HelicoDataSet/CrossValidation/Cropped",
        "HoldOut": "/import/fhome/vlia/HelicoDataSet/HoldOut"
    }
    
    for set_name, set_path in sets_to_check.items():
        print(f"Scanning {set_name}...")
        patient_dirs = sorted([d for d in os.listdir(set_path) if os.path.isdir(os.path.join(set_path, d))])
        
        for d in tqdm(patient_dirs):
            full_path = os.path.join(set_path, d)
            d_hash = get_dir_hash(full_path)
            if d_hash:
                all_results.append({
                    "Dataset_Source": set_name, 
                    "Patient_Folder": d, 
                    "Full_Bag_Hash": d_hash, 
                    "File_Count": len([f for f in os.listdir(full_path) if f.endswith(('.png', '.jpg'))])
                })
    
    df = pd.DataFrame(all_results)
    
    # 2. Find ALL duplicates (internal and cross-set)
    duplicates = df[df.duplicated('Full_Bag_Hash', keep=False)].sort_values('Full_Bag_Hash')
    
    if duplicates.empty:
        print("\nNo duplicate patient bags found across any dataset sets.")
    else:
        print("\n[!] DUPLICATE BAGS DETECTED (Leakage Audit):")
        # Identify specific cross-set leakage
        for h, group in duplicates.groupby('Full_Bag_Hash'):
            sources = group['Dataset_Source'].unique()
            if len(sources) > 1:
                print(f"--> LEAKAGE: Hash {h[:8]} appears in both {sources}")
        
        print("\nFull Report:")
        print(duplicates)
        duplicates.to_csv(report_file, index=False)
        print(f"\nUnified Audit Report saved to {report_file}")
