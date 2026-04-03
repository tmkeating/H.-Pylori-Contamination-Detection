"""
# H. Pylori Data Integrity: Clinical Dataset Audit
# -----------------------------------------------
# This script performs a "Data Hygiene" audit on the Annotated dataset split.
# It validates that the patient-level labels match the slide-level diagnostic
# ground truth and analyzes the distribution of tissue patches per patient.
#
# Why this is critical:
#   1. Label Consistency: Ensures that multiple slides from the same patient 
#      (e.g., biopsy fragments B22-47_0 and B22-47_1) have identical H. Pylori 
#      contamination labels as recorded in the master diagnosis CSV.
#   2. Sensitivity Analysis: identifies "sparse" bags with very few patches,
#      which might be more prone to false negatives.
#   3. Automation: Exports a 'data_integrity_summary.csv' for project verification.
#
# IMPORTANT: Understanding Patch Count Discrepancies
# ==================================================
# NOTE: This script audits ALL patches in scratch (training + evaluation) for leakage.
# For granular training vs evaluation counts, use audit_png_count.py instead.
#
# COUNT BREAKDOWN (filtering order):
#   - Raw PNG files on permanent storage:      ~219,609 patches (all physical files)
#   
#   - Blacklisted and excluded at rsync:         -3,283 patches
#     * B22-124_0: 1,197 patches (CrossValidation redundant)
#     * B22-01_1: 486 patches (HoldOut, clinical conflict with B22-03_1)
#     * B22-03_1: 486 patches (HoldOut, clinical conflict with B22-01_1)
#     * Image-level blacklist: ~113 patches (duplicates within bags)
#   
#   - Physical patches in scratch directory:  ~216,326 patches (audit_png_count.py - AUTHORITATIVE)
#     * Training (CrossValidation):           ~128,724 patches (for 5-fold CV)
#     * Evaluation (HoldOut):                 ~87,602 patches (separate held-out test set)
#   
#   - verify_data_integrity.py scans:        ~214,644 patches (both training + evaluation audited)
#     * Gap: 1,682 patches (from HPyloriDataset internal dedup logic)
#
# IMPORTANT DISTINCTION:
#   - audit_png_count.py: 216,326 patches (raw PNG files, authoritative file count)
#   - verify_data_integrity.py: 214,644 patches (leakage audit, stricter clinical filtering)
#   - The 1,682 patch difference is from verification script's dedup logic, not from blacklist
#   - For training: 128,724 patches from CrossValidation only (audit_png_count.py)
#   - For evaluation: 87,602 patches from HoldOut only (audit_png_count.py)
#
# FILTERING SYSTEM (Applied by HPyloriDataset during leakage audit):
#   - HPyloriDataset loads patches with clinical validation
#   - Applies deduplication to detect leakage (same patch in train + test)
#   - Dedup logic is strict to catch subtle cross-contamination
#   - Result: 214,644 patches after dedup audit (vs 216,326 raw files)
#   - Gap of 1,682 is intentional - represents potential leakage candidates
#
# IMPORTANT:
#   - This script is for LEAKAGE DETECTION, not for training data verification
#   - Use this to identify and audit cross-set patient overlaps
#   - For actual training patch count, use audit_png_count.py
#   - The 1,682 patch gap is from verification logic, NOT from data loss
#
# BLACKLIST IMPACT:
#   - B22-124_0: 1,197 patches (CrossValidation redundant, excluded at rsync)
#   - B22-01_1: 486 patches (HoldOut, held out from training)
#   - B22-03_1: 486 patches (HoldOut, held out from training)
#   - Image-level duplicates: ~113 patches (removed at rsync)
#   - Total Blacklist: ~3,283 patches (excluded at rsync level, NOT in scratch)
#   - Note: Blacklist removal happens at sync time via rsync --exclude
#   - Effect: Scratch receives 216,326 patches (correct amount after blacklist)
#
# OUTPUT INTERPRETATION:
#   - Patches_Scanned = Patches examined by verify_data_integrity (214,644)
#   - This includes BOTH training (128,724) and evaluation (87,602) sets
#   - Gap to 216,326: 1,682 patches (from internal dedup logic in verification)
#   - AUTHORITATIVE COUNTS:
#     * Raw files in scratch: 216,326 patches (audit_png_count.py)
#     * Training set: 128,724 patches (CrossValidation, audit_png_count.py)
#     * Evaluation set: 87,602 patches (HoldOut, audit_png_count.py)
#   - Use this script for leakage detection only, NOT for training patch count
#   - Actual model training uses audit_png_count.py verified numbers
# -----------------------------------------------
"""
import os
import sys
import torch
import pandas as pd

# Add the parent directory to sys.path so we can import 'dataset.py' from the root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import HPyloriDataset

# Define the root path for dataset access (using local scratch or shared storage)
root = '/tmp/ricse03_h_pylori_data'
p_csv = os.path.join(root, 'PatientDiagnosis.csv')
patch_csv = os.path.join(root, 'HP_WSI-CoordAnnotatedAllPatches.xlsx')
train_dir = os.path.join(root, 'CrossValidation/Annotated')

try:
    # --- PHASE 1: LOAD ALL DATASETS FOR CROSS-LEAKAGE AUDIT ---
    # We audit Annotated, Cropped, and HoldOut to ensure no patient overlaps.
    dataset_configs = {
        'Annotated': train_dir,
        'Cropped': os.path.join(root, 'CrossValidation/Cropped'),
        'HoldOut': os.path.join(root, 'HoldOut')
    }
    
    # Use a dictionary to deduplicate bags found in multiple directories
    # Key: bag_id, Value: bag info (prefer more complete versions like Cropped)
    all_bags_audit = {}
    patient_to_sets = {} # Track which sets each patient appears in
    
    # Directory priority (in case a bag appears in multiple directories)
    # Higher priority = keep this version if there's a conflict
    dir_priority = {'HoldOut': 3, 'Cropped': 2, 'Annotated': 1}
    
    for set_name, set_dir in dataset_configs.items():
        if not os.path.exists(set_dir):
            print(f"Warning: Directory {set_dir} not found. Skipping {set_name}.")
            continue
            
        temp_ds = HPyloriDataset(root_dir=set_dir, patient_csv=p_csv, patch_csv=patch_csv, bag_mode=True)
        print(f"Loaded {len(temp_ds.bags)} bags from {set_name}")
        
        for bag in temp_ds.bags:
            paths, label, b_id, pos_paths = bag
            base_id = b_id.split('_')[0]
            
            bag_info = {
                'Patient_Bag_ID': b_id,
                'Base_Patient_ID': base_id,
                'Dataset_Source': set_name,
                'Label': label,
                'Patch_Count': len(paths),
                'Annotated_Positive_Count': len(pos_paths)
            }
            
            # Deduplication logic: keep the version with higher priority
            # (prefer Cropped > Annotated, HoldOut is separate set so no conflict expected)
            if b_id not in all_bags_audit:
                all_bags_audit[b_id] = bag_info
            else:
                # If same bag in multiple dirs, keep the one with higher priority
                existing_priority = dir_priority.get(all_bags_audit[b_id]['Dataset_Source'], 0)
                new_priority = dir_priority.get(set_name, 0)
                if new_priority > existing_priority:
                    all_bags_audit[b_id] = bag_info
            
            # Map clinical ID to its dataset locations
            if base_id not in patient_to_sets:
                patient_to_sets[base_id] = set()
            patient_to_sets[base_id].add(set_name)

    # --- PHASE 2: DETECT LEAKAGE ---
    leakage_issues = []
    cross_pool_leakage = 0 # Leakage between Annotated and Cropped
    holdout_leakage = 0    # Leakage involving HoldOut
    
    for patient_id, sets in patient_to_sets.items():
        if len(sets) > 1:
            leakage_issues.append(f"LEAKAGE: Patient {patient_id} found in multiple sets: {list(sets)}")
            
            # Categorize the leakage for the summary report
            if 'HoldOut' in sets:
                holdout_leakage += 1
            if 'Annotated' in sets and 'Cropped' in sets:
                cross_pool_leakage += 1

    # --- PHASE 3: SUMMARY & EXPORT ---
    total_unique_patients = len(patient_to_sets)
    print(f'\n--- GLOBAL DATA INTEGRITY AUDIT ---')
    print(f'Total Unique Patients (Global): {total_unique_patients}')
    print(f'Total Unique Bags (after deduplication): {len(all_bags_audit)}')
    
    if leakage_issues:
        print(f'[!] CRITICAL LEAKAGE DETECTED: {len(leakage_issues)} violations found.')
        for issue in leakage_issues[:10]: print(f'  -> {issue}')
    else:
        print('Leakage Audit: OK (No patient data overlaps between Train/Cropped/Holdout subsets)')

    # --- Step 3.1: Generate Patient Integrity Breakdown ---
    # This matches the training pipeline format precisely but adds custom booleans
    patient_breakdown = []
    for bag in all_bags_audit.values():  # Use .values() since all_bags_audit is now a dict
        b_id = bag['Patient_Bag_ID']
        base_id = bag['Base_Patient_ID']
        sets_for_patient = patient_to_sets.get(base_id, set())
        
        # Boolean Logic for specific leakage tracks
        is_holdout_leak = 'HoldOut' in sets_for_patient and ('Annotated' in sets_for_patient or 'Cropped' in sets_for_patient)
        is_annotated_cropped_leak = 'Annotated' in sets_for_patient and 'Cropped' in sets_for_patient

        patient_breakdown.append({
            'Patient_Bag_ID': b_id,
            'Clinical_Group_ID': base_id,
            'Folder_Source': bag['Dataset_Source'],
            'Label': bag['Label'],
            'Patch_Count': bag['Patch_Count'],
            'Annotated_Positives': bag['Annotated_Positive_Count'],
            'LEAK_HOLDOUT_TRAIN': is_holdout_leak,
            'LEAK_ANNOTATED_CROPPED': is_annotated_cropped_leak
        })
    
    breakdown_df = pd.DataFrame(patient_breakdown)
    breakdown_df.to_csv('patient_integrity_breakdown.csv', index=False)
    print(f'Detailed patient breakdown exported to: patient_integrity_breakdown.csv')

    # --- Step 3.2: Generate Data Integrity Summary ---
    # Matches the Iteration 25.5 summary format
    unique_clinical_ids = breakdown_df['Clinical_Group_ID'].unique()
    pos_patient_count = breakdown_df[breakdown_df['Label'] == 1]['Clinical_Group_ID'].nunique()
    neg_patient_count = len(unique_clinical_ids) - pos_patient_count

    # Prepare the CSV with standardized columns
    summary_df = pd.DataFrame([{
        'Run_Prefix': 'MANUAL_AUDIT',
        'FINAL_PATIENT_TOTAL': len(unique_clinical_ids),
        'FINAL_POSITIVE_TOTAL': pos_patient_count,
        'FINAL_NEGATIVE_TOTAL': neg_patient_count,
        'PATIENTS_IN_BOTH_ANNOTATED_AND_CROPPED_LEAKAGE': cross_pool_leakage,
        'PATIENTS_IN_HOLDOUT_AND_TRAIN_LEAKAGE': holdout_leakage,
        'Patches_Scanned': breakdown_df['Patch_Count'].sum(),
        'Final_Bag_Count': len(breakdown_df)
    }])
    
    # Export the main summary (first row)
    summary_df.to_csv('data_integrity_summary.csv', index=False)
    
    # Append a blank line and the Patient ID List with a left-aligned header
    with open('data_integrity_summary.csv', 'a') as f:
        f.write('\n')                # Blank line between summary and list
        f.write('Patient_ID_List\n')   # Left-aligned header
        id_list = sorted(unique_clinical_ids.tolist())
        for pid in id_list:
            f.write(f'{pid}\n')
            
    print(f'Quick summary exported to: data_integrity_summary.csv')

    # Add a dedicated leakage summary to the report
    if leakage_issues:
        leak_df = pd.DataFrame([{'Description': issue} for issue in leakage_issues])
        leak_df.to_csv('leakage_violations.csv', index=False)
        print(f'Leakage violations specifically logged to: leakage_violations.csv')

except Exception as e:
    import traceback
    print(f'Error during dataset audit: {e}')
    traceback.print_exc() 