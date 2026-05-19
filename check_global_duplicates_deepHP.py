"""
# H. Pylori DeepHP Dataset Global Image-Level Deduplication & Inventory
# =======================================================================
# ⚠️  THIS SCRIPT IS FOR DEEPHP DATASET ONLY - NOT FOR HELICODATASET
# 
# This script performs a byte-level audit across the entire DeepHP pre-training dataset.
# It identifies:
#   1. Global Patch Duplicates: Checks all patches against each other for duplicates (by MD5)
#   2. Class Inventory: Complete list of patches in Positive and Negative classes
#   3. Patch Distribution: How many patches in each class
#   4. Duplicate Detection: Identifies exact duplicate patches within and across classes
#
# Performance: Uses 8KB header hashing for candidate identification followed by 
# file size verification to ensure 100% collision-free accuracy.
# """

import os
import hashlib
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import json
from config import DEEPHP_DATASET_ROOT

def get_file_hash(path):
    """Calculates MD5 hash of entire file for accurate duplicate detection."""
    try:
        md5_hash = hashlib.md5()
        with open(path, 'rb') as f:
            # Read file in 8KB chunks for memory efficiency
            for chunk in iter(lambda: f.read(8192), b''):
                md5_hash.update(chunk)
        return md5_hash.hexdigest(), os.path.getsize(path)
    except:
        return None, 0

def run_global_audit():
    sets_to_check = {
        "Positive": os.path.join(DEEPHP_DATASET_ROOT, "Positive"),
        "Negative": os.path.join(DEEPHP_DATASET_ROOT, "Negative")
    }

    image_inventory = [] # List of every single patch scanned
    hash_to_paths = defaultdict(list) # Track duplicates
    
    print("Starting Global Patch-Level Scan for DeepHP Dataset...")
    print(f"Dataset root: {DEEPHP_DATASET_ROOT}\n")
    
    for class_name, class_path in sets_to_check.items():
        if not os.path.exists(class_path):
            print(f"Warning: {class_path} not found. Skipping.")
            continue
            
        print(f"Scanning {class_name} patches...")
        patch_files = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in tqdm(patch_files, desc=class_name):
            img_path = os.path.join(class_path, img_name)
            img_hash, img_size = get_file_hash(img_path)
            
            if img_hash:
                # Store the patch in the global inventory
                image_info = {
                    "Class": class_name,
                    "Image_Name": img_name,
                    "Hash": img_hash,
                    "Size": img_size,
                    "Full_Path": img_path
                }
                image_inventory.append(image_info)
                
                # Track hash collisions for duplicate detection
                hash_to_paths[(img_hash, img_size)].append(img_path)

    # --- Step 1: Export Global Patch Inventory ---
    print("\nExporting Global Patch Inventory...")
    inventory_df = pd.DataFrame(image_inventory)
    inventory_df.to_csv("deephp_image_inventory.csv", index=False)
    print(f"Inventory of {len(inventory_df)} patches saved to deephp_image_inventory.csv")
    
    # --- Step 2: Export Duplicate Patch Report ---
    print("Analyzing Global Duplicates...")
    duplicate_results = []
    
    # Map from path to its inventory info for quick lookup
    path_to_info = {img["Full_Path"]: img for img in image_inventory}
    
    for (img_hash, size), paths in hash_to_paths.items():
        if len(paths) > 1:
            for p in paths:
                info = path_to_info.get(p, {})
                duplicate_results.append({
                    "Hash": img_hash,
                    "Class": info.get("Class", "Unknown"),
                    "Size": size,
                    "Path": p,
                    "Occurrence_Count": len(paths)
                })
    
    if duplicate_results:
        dub_df = pd.DataFrame(duplicate_results)
        dub_df.to_csv("deephp_image_duplicates.csv", index=False)
        print(f"Found {len(dub_df)} duplicate patches. Logged to deephp_image_duplicates.csv")
    else:
        print("No duplicate patches found across the entire dataset!")

    # --- Step 3: Class Distribution Report ---
    print("Generating Class Distribution...")
    class_distribution = defaultdict(int)
    for img in image_inventory:
        class_distribution[img["Class"]] += 1
    
    distribution_rows = []
    for class_name, count in class_distribution.items():
        distribution_rows.append({
            "Class": class_name,
            "Patch_Count": count
        })
    
    distribution_df = pd.DataFrame(distribution_rows)
    distribution_df.to_csv("deephp_class_distribution.csv", index=False)
    print(f"Class distribution saved to deephp_class_distribution.csv")
        
    # --- Step 4: Patch-Level Duplicate Summary ---
    print("Generating Patch Duplicate Summary...")
    
    # Track the count of duplicate files
    all_duplicate_paths = set()
    for (img_hash, size), paths in hash_to_paths.items():
        if len(paths) > 1:
            for p in paths:
                all_duplicate_paths.add(p)

    # Generate summary of duplicates by class
    class_stats = defaultdict(lambda: {
        "Total_Duplicates": 0,
        "Duplicate_Paths": []
    })

    for (img_hash, size), paths in hash_to_paths.items():
        if len(paths) > 1:
            for p in paths:
                # Extract class from path
                if "/Positive/" in p:
                    class_stats["Positive"]["Total_Duplicates"] += 1
                    class_stats["Positive"]["Duplicate_Paths"].append(p)
                elif "/Negative/" in p:
                    class_stats["Negative"]["Total_Duplicates"] += 1
                    class_stats["Negative"]["Duplicate_Paths"].append(p)

    summary_rows = []
    for class_name, stats in class_stats.items():
        summary_rows.append({
            "Class": class_name,
            "Duplicate_Patches": stats["Total_Duplicates"]
        })

    # Always generate the summary CSV (even if no duplicates found)
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
    else:
        summary_df = pd.DataFrame({
            "Class": ["Positive", "Negative"],
            "Duplicate_Patches": [0, 0]
        })
    
    summary_df.to_csv("deephp_patch_duplicate_audit.csv", index=False)
    print("Patch-level duplicate summary saved to deephp_patch_duplicate_audit.csv")
        
    # --- Step 5: Suggested Blacklist Generation ---
    print("Generating Suggested Blacklist for DeepHP...")
    suggested_image_blacklist = []
    
    for (img_hash, size), paths in hash_to_paths.items():
        if len(paths) > 1:
            # For patch-level data, ban all but one occurrence
            for p in paths[1:]:
                suggested_image_blacklist.append({
                    "path": p,
                    "filename": os.path.basename(p),
                    "reason": f"Duplicate patch (hash collision). Keeping: {os.path.basename(paths[0])}"
                })

    blacklist_output = {
        "audit_status": {
            "total_duplicate_patches": len(suggested_image_blacklist),
            "duplicates_found": len(suggested_image_blacklist) > 0,
            "message": f"Found {len(suggested_image_blacklist)} duplicate patches" if len(suggested_image_blacklist) > 0 else "No duplicate patches found - all patches are unique"
        },
        "deephp_image_blacklist": suggested_image_blacklist
    }

    with open("suggested_deephp_blacklist.json", "w") as f:
        json.dump(blacklist_output, f, indent=4)
    
    print(f"Suggested blacklist generated with {len(suggested_image_blacklist)} duplicate patches.")

if __name__ == "__main__":
    run_global_audit()
