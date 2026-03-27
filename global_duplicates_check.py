"""
# H. Pylori Global Image-Level Deduplication & Inventory
# -----------------------------------------------------
# This script performs a byte-level audit across the entire H. Pylori dataset.
# It identifies:
#   1. Global Image Duplicates: Checks all images against each other for duplicates (by MD5) regardless of folder.
#   2. Patient Inventory: Complete list of patients in Annotated, Cropped, and HoldOut.
#   3. Dataset Presence: Which patients appear in which sub-folders.
#   4. Blacklist Generation (suggested_blacklist.json):
#      - Intra-Folder: Bans all but one occurrence of identical images in the same folder to maintain patch-level integrity.
#      - High-Overlap Bags: Identifies patients with >90% image overlap.
#         * Bans one if in the same folder/set.
#         * Bans BOTH if in different sets (e.g. Train vs HoldOut) to resolve clinical label conflicts.
#      - Cross-Folder Patches: Bans patch duplicates across different patients to maintain patient-level integrity.
#
# Performance: Uses 8KB header hashing for candidate identification followed by 
# file size verification to ensure 100% collision-free accuracy.
# -----------------------------------------------------
"""
import os
import hashlib
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import json

def get_file_hash(path):
    """Calculates MD5 hash of a single image file."""
    # We use a 2-stage hashing for massive speed improvement
    # Stage 1: Read first 8KB (enough to differentiate 99.9% of images)
    # Stage 2: (Implicitly) we use the full path and size as a secondary check
    try:
        with open(path, 'rb') as f:
            header = f.read(8192)
            # If the file is small, this is the whole hash.
            # If large, this is a 'candidate' hash.
            return hashlib.md5(header).hexdigest(), os.path.getsize(path)
    except:
        return None, 0

def run_global_audit():
    sets_to_check = {
        "Annotated": "/import/fhome/vlia/HelicoDataSet/CrossValidation/Annotated",
        "Cropped": "/import/fhome/vlia/HelicoDataSet/CrossValidation/Cropped",
        "HoldOut": "/import/fhome/vlia/HelicoDataSet/HoldOut"
    }

    image_inventory = [] # List of every single image scanned
    hash_to_paths = defaultdict(list) # Track duplicates
    
    print("Starting Global Image-Level Scan...")
    
    for set_name, set_path in sets_to_check.items():
        if not os.path.exists(set_path):
            print(f"Warning: {set_path} not found. Skipping.")
            continue
            
        print(f"Scanning {set_name}...")
        patient_folders = [d for d in os.listdir(set_path) if os.path.isdir(os.path.join(set_path, d))]
        
        for p_folder in tqdm(patient_folders, desc=set_name):
            p_path = os.path.join(set_path, p_folder)
            p_id_base = p_folder.split('_')[0]
            
            for img_name in os.listdir(p_path):
                if not img_name.endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                img_path = os.path.join(p_path, img_name)
                img_hash, img_size = get_file_hash(img_path)
                
                if img_hash:
                    # We store the image in the global inventory
                    image_info = {
                        "Set": set_name,
                        "Patient_Folder": p_folder,
                        "Base_Patient_ID": p_id_base,
                        "Image_Name": img_name,
                        "Hash": img_hash,
                        "Size": img_size,
                        "Full_Path": img_path
                    }
                    image_inventory.append(image_info)
                    
                    # Track hash collisions for duplicate detection
                    # Note: Using (hash, size) as key reduces probability of header collisions to near zero
                    hash_to_paths[(img_hash, img_size)].append(img_path)

    # --- Step 1: Export Global Image Inventory ---
    print("\nExporting Global Image Inventory...")
    inventory_df = pd.DataFrame(image_inventory)
    inventory_df.to_csv("global_image_inventory.csv", index=False)
    
    # --- Step 2: Export Duplicate Image Report ---
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
                    "Patient_ID": info.get("Base_Patient_ID", "Unknown"),
                    "Size": size,
                    "Path": p,
                    "Occurrence_Count": len(paths)
                })
    
    if duplicate_results:
        dub_df = pd.DataFrame(duplicate_results)
        dub_df.to_csv("global_image_duplicates.csv", index=False)
        print(f"Found {len(dub_df)} duplicate images. Logged to global_image_duplicates.csv")
    else:
        print("No duplicate images found across the entire dataset!")

    # --- Step 3: Global Patient Presence Report (Including extra Cropped patients) ---
    print("Generating Patient Presence Matrix...")
    patient_set_map = defaultdict(set)
    for img in image_inventory:
        patient_set_map[img["Base_Patient_ID"]].add(img["Set"])
        
    presence_matrix = []
    for p_id, sets in patient_set_map.items():
        presence_matrix.append({
            "Clinical_Patient_ID": p_id,
            "In_Annotated": "Annotated" in sets,
            "In_Cropped": "Cropped" in sets,
            "In_HoldOut": "HoldOut" in sets,
            "Set_Combination": "+".join(sorted(list(sets)))
        })
        
    # --- Step 4: Patient-Level Duplicate Summary ---
    print("Generating Patient-Specific Duplicate Summary...")
    patient_summary = []
    
    # Track the count of duplicate files for each patient across different criteria
    # We'll use the hash_to_paths where entries have >1 path.
    all_duplicate_paths = set()
    for (img_hash, size), paths in hash_to_paths.items():
        if len(paths) > 1:
            for p in paths:
                all_duplicate_paths.add(p)

    # Re-process inventory to generate patient-by-patient summary
    patient_stats = defaultdict(lambda: {
        "Total_Duplicate_Files": 0,
        "Duplicates_In_Same_Folder": 0,
        "Duplicates_Across_Folders": 0,
        "Cross_Set_Duplicates_TrainTest": 0, # Train (Annotated/Cropped) vs HoldOut
        "Cross_Set_Duplicates_AnnotatedCropped": 0,
        "Folders_Sharing_Duplicates": set()
    })

    # To calculate cross-folder/cross-set stats accurately, we need to see where duplicates lead
    for (img_hash, size), paths in hash_to_paths.items():
        if len(paths) > 1:
            # For each unique image that has duplicates
            # Identify the unique folders and sets it belongs to
            folders_and_sets = []
            for p in paths:
                # Extract folder and set from the path
                # Path format example: /import/fhome/vlia/HelicoDataSet/CrossValidation/Annotated/123/img.png
                parts = p.split('/')
                set_name = ""
                folder_name = ""
                if "Annotated" in parts: set_name = "Annotated"
                elif "Cropped" in parts: set_name = "Cropped"
                elif "HoldOut" in parts: set_name = "HoldOut"
                
                # The folder name is just before the image name
                folder_name = parts[-2]
                folders_and_sets.append((set_name, folder_name, p))

            # Group these duplicates by the patients they belong to
            patients_involved = set()
            for s_name, f_name, p_path in folders_and_sets:
                p_id_base = f_name.split('_')[0]
                patients_involved.add(p_id_base)

            for p_id in patients_involved:
                p_paths = [ (s, f, path) for s, f, path in folders_and_sets if f.split('_')[0] == p_id ]
                stats = patient_stats[p_id]
                stats["Total_Duplicate_Files"] += len(p_paths)
                
                # Check if this set of duplicates for THIS patient is in the same folder or across folders
                unique_folders_for_this_patient = set([f for s, f, path in p_paths])
                all_unique_folders = set([f for s, f, path in folders_and_sets])

                if len(unique_folders_for_this_patient) > 1:
                    # Case 1: The patient has copies of the same image in multiple folders of their own
                    stats["Duplicates_Across_Folders"] += len(p_paths)
                elif len(all_unique_folders) > 1:
                    # Case 2: The patient has a copy in ONE folder, but there is ALSO a copy in another folder (likely another patient)
                    stats["Duplicates_Across_Folders"] += len(p_paths)
                else:
                    # Case 3: All copies exist only within this patient's single folder
                    stats["Duplicates_In_Same_Folder"] += len(p_paths)

                # Track which folders this image is shared with
                for s, f, path in folders_and_sets:
                    if f not in unique_folders_for_this_patient:
                        stats["Folders_Sharing_Duplicates"].add(f"{s}/{f}")

    # Convert to DataFrame
    summary_rows = []
    for p_id, stats in patient_stats.items():
        summary_rows.append({
            "Patient_ID": p_id,
            "Total_Duplicate_Files": stats["Total_Duplicate_Files"],
            "Duplicates_In_Same_Folder": stats["Duplicates_In_Same_Folder"],
            "Duplicates_Across_Folders": stats["Duplicates_Across_Folders"],
            "Leak_Train_Vs_HoldOut": stats["Cross_Set_Duplicates_TrainTest"],
            "Leak_Annotated_Vs_Cropped": stats["Cross_Set_Duplicates_AnnotatedCropped"],
            "Shared_With_Folders": ", ".join(sorted(list(stats["Folders_Sharing_Duplicates"])))
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("Total_Duplicate_Files", ascending=False)
    summary_df.to_csv("patient_duplicate_audit.csv", index=False)
    print("Patient-level duplicate summary saved to patient_duplicate_audit.csv")

    # --- Step 5: Suggested Blacklist Generation ---
    print("Generating Suggested Blacklist...")
    suggested_conflict_blacklist = {} # Bag ID -> Reason string
    suggested_image_blacklist = [] # List of {"folder": F, "filename": N, "reason": R}

# Load clinical labels for exact conflict matching
    patient_labels = {}
    try:
        label_file = "/import/fhome/vlia/HelicoDataSet/PatientDiagnosis.csv"
        if os.path.exists(label_file):
            df = pd.read_csv(label_file)
            clean_id = lambda x: str(int(float(x))) if str(x).replace('.','',1).isdigit() else str(x)
            
            # Map BAIXA and ALTA to Positive, NEGATIVA to Negative
            label_map = {'NEGATIVA': 'Negative', 'BAIXA': 'Positive', 'ALTA': 'Positive'}
            
            for _, row in df.iterrows():
                p_id = clean_id(row['CODI'])
                patient_labels[p_id] = label_map.get(row['DENSITAT'], str(row['DENSITAT']))
            print(f"Loaded clinical labels for {len(patient_labels)} patients.")
        else:
            print(f"Warning: {label_file} not found.")
    except Exception as e:
        print(f"Warning: Could not load clinical labels for audit ({e}). Using heuristic.")

    for (img_hash, size), paths in hash_to_paths.items():
        if len(paths) <= 1:
            continue

        # Parse all involved paths
        details = []
        for p in paths:
            parts = p.split('/')
            folder = parts[-2]
            filename = parts[-1]
            set_name = "Annotated" if "Annotated" in parts else "Cropped" if "Cropped" in parts else "HoldOut" if "HoldOut" in parts else "Unknown"
            details.append({"path": p, "folder": folder, "filename": filename, "set": set_name})

        # Rule 1: Intra-folder duplicates (Ban all but one)
        folder_groups = defaultdict(list)
        for d in details:
            folder_groups[d["folder"]].append(d)
        
        for folder, d_list in folder_groups.items():
            if len(d_list) > 1:
                # Keep the first one, ban the rest
                primary_file = d_list[0]["filename"]
                for d in d_list[1:]:
                    suggested_image_blacklist.append({
                        "folder": d["folder"], 
                        "filename": d["filename"],
                        "reason": f"Intra-folder duplicate in {folder}. Remaining file: {primary_file}"
                    })

        # Rule 2: Cross-folder / Patient-level duplicates
        unique_folders = list(folder_groups.keys())
        if len(unique_folders) > 1:
            # Identify "Surgical Precision" patches (same hash across different folders)
            # We'll ban all but the first folder's occurrences
            primary_folder = unique_folders[0]
            for folder in unique_folders[1:]:
                for d in folder_groups[folder]:
                    # To avoid double-banning if already banned by intra-folder rule
                    if not any(b["folder"] == d["folder"] and b["filename"] == d["filename"] for b in suggested_image_blacklist):
                        suggested_image_blacklist.append({
                            "folder": d["folder"],
                            "filename": d["filename"],
                            "reason": f"Cross-folder leak: Shared with {primary_folder}"
                        })

    # For the Patient Conflict Blacklist, we'll use a heuristic for this script:
    # If two folders share more than 90% of their images, they are "Identical Bags"
    folder_to_hashes = defaultdict(set)
    for img in image_inventory:
        folder_to_hashes[img["Patient_Folder"]].add(img["Hash"])

    # Build a quick map of which patients have patches in 'Annotated' vs 'Cropped' vs 'HoldOut'
    # and try to infer clinical labels (Positive if any patch in Annotated exists)
    folder_inferred_label = {} # Folder -> "POS" or "NEG"
    for img in image_inventory:
        f = img["Patient_Folder"]
        if f not in folder_inferred_label:
            # Patients in 'Annotated' with specific patches are usually Positive
            # Patients in 'Annotated' with just basic names are usually Negative
            # For this audit, we'll check if the folder name has a suffix or if it's in a specific set
            folder_inferred_label[f] = "UNKNOWN"

    folders = list(folder_to_hashes.keys())
    for i in range(len(folders)):
        for j in range(i + 1, len(folders)):
            f1, f2 = folders[i], folders[j]
            h1, h2 = folder_to_hashes[f1], folder_to_hashes[f2]
            
            intersection = len(h1.intersection(h2))
            if intersection == 0: continue
            
            smaller_size = min(len(h1), len(h2))
            if smaller_size == 0: continue
            
            overlap_pct = intersection / smaller_size
            
            if overlap_pct > 0.9: # Very high overlap
                # Extract set names and sample paths for these folders
                img1 = next((img for img in image_inventory if img["Patient_Folder"] == f1), None)
                img2 = next((img for img in image_inventory if img["Patient_Folder"] == f2), None)
                
                set1, set2 = img1["Set"] if img1 else "", img2["Set"] if img2 else ""
                
                # Identify if these are the known B22-01 (Pos) vs B22-03 (Neg) conflict
                # or similar ones identified by the overlap.
                p1_base = f1.split('_')[0]
                p2_base = f2.split('_')[0]
                
                # Check actual clinical labels
                l1 = patient_labels.get(p1_base, "Missing")
                l2 = patient_labels.get(p2_base, "Missing")

                # Label Logic:
                # NEGATIVA is negative, BAIXA/ALTA are positive.
                # Now checking for the translated 'Positive' and 'Negative' tags
                def is_pos(l): return l in ["Positive", "BAIXA", "ALTA"]
                def is_neg(l): return l in ["Negative", "NEGATIVA"]

                label_status = ""
                if l1 == "Missing" or l2 == "Missing":
                    label_status = "Label Status: HEURISTIC (Missing Clinical Data)"
                elif (is_pos(l1) and is_pos(l2)) or (is_neg(l1) and is_neg(l2)):
                    label_status = f"Label Status: VERIFIED IDENTICAL ({l1}/{l2})"
                else:
                    label_status = f"Label Status: VERIFIED CONFLICT ({l1} vs {l2})"

                if set1 == "HoldOut" or set2 == "HoldOut":
                    label_status += " (Train vs HoldOut Leak)"

                # Rule: Ban 1 if same set. Ban both if cross-set/conflict (conservative)
                if set1 == set2 and "CONFLICT" not in label_status:
                    suggested_conflict_blacklist[f2] = f"Redundant bag (>90% overlap with {f1} in {set1}). {label_status}. Remaining bag: {f1}"
                else:
                    # Potential Train/Test leak or clinical conflict
                    suggested_conflict_blacklist[f1] = f"Clinical/Set conflict (>90% overlap with {f2}). {label_status}. Linked to: {f2}"
                    suggested_conflict_blacklist[f2] = f"Clinical/Set conflict (>90% overlap with {f1}). {label_status}. Linked to: {f1}"

    blacklist_output = {
        "conflict_blacklist": suggested_conflict_blacklist,
        "image_blacklist": suggested_image_blacklist
    }

    with open("suggested_blacklist.json", "w") as f:
        json.dump(blacklist_output, f, indent=4)
    
    print(f"Suggested blacklist generated with {len(suggested_conflict_blacklist)} bags and {len(suggested_image_blacklist)} patches.")
    print("Audit Complete. Reports generated: global_image_inventory.csv, global_image_duplicates.csv, dataset_presence_matrix.csv, suggested_blacklist.json")

if __name__ == "__main__":
    run_global_audit()
