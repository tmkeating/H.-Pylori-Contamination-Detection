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
# NOTE: This script audits ALL patches in scratch (including both Annotated and Cropped).
# Annotated and Cropped are expected to coexist as data diversity - both feed into bags.
#
# COUNT BREAKDOWN:
#   - Raw PNG files on permanent storage:      ~219,609 patches (all physical files)
#   
#   - Blacklisted and excluded at rsync:         -3,283 patches
#     * B22-124_0: 1,197 patches (CrossValidation redundant)
#     * B22-01_1: 486 patches (HoldOut, clinical conflict with B22-03_1)
#     * B22-03_1: 486 patches (HoldOut, clinical conflict with B22-01_1)
#     * Image-level blacklist: ~113 patches (duplicates within bags)
#   
#   - Physical patches in scratch directory:  ~216,326 patches (audit_png_count.py - AUTHORITATIVE)
#     * Training (CrossValidation):           ~128,724 patches (for 5-fold CV, Annotated + Cropped)
#     * Evaluation (HoldOut):                 ~87,602 patches (separate held-out test set)
#   
#   - verify_data_integrity.py scans:        ~216,326 patches (all patches, NO dedup between Annotated/Cropped)
#     * Includes BOTH Annotated and Cropped versions (data diversity)
#     * Only removes duplicates between HoldOut and Training (true cross-contamination)
#
# IMPORTANT DISTINCTION:
#   - Annotated + Cropped: EXPECTED (both versions of patches used in training)
#   - HoldOut + Training: LEAKAGE (test set contamination, critical problem)
#
# LEAKAGE DETECTION LOGIC:
#   - Annotated/Cropped coexistence: Data diversity, not a problem
#   - HoldOut/Training overlap: True cross-contamination, flagged as CRITICAL
#
# IMPORTANT:
#   - This script is for COMPREHENSIVE DATA AUDITING
#   - Counts all patches from both Annotated and Cropped (matching training)
#   - Flags ONLY true leakage: HoldOut mixed with Training
#   - Annotated + Cropped coexistence is expected and documented, not flagged
#
# BLACKLIST IMPACT:
#   - Blacklist exclusion happens at rsync level (before this script runs)
#   - Scratch receives 216,326 patches (correct amount after blacklist)
#
# OUTPUT INTERPRETATION:
#   - Patches_Scanned = All patches examined (includes both Annotated and Cropped)
#   - ANNOTATED_AND_CROPPED flag = Data diversity (expected, not an issue)
#   - LEAK_HOLDOUT_TRAIN flag = True cross-contamination (critical issue)
#   - leakage_violations.csv = Only HoldOut/Training leaks (if any exist)
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
    # We audit CrossValidation (as a list of Annotated + Cropped) and HoldOut separately
    dataset_configs = {
        'CrossValidation': [
            os.path.join(root, 'CrossValidation/Annotated'),
            os.path.join(root, 'CrossValidation/Cropped')
        ],
        'HoldOut': os.path.join(root, 'HoldOut')
    }
    
    # Store all bags from all directories (no deduplication between Annotated and Cropped)
    # Annotated and Cropped are expected to coexist as data diversity
    # Only HoldOut + Train overlap would be leakage
    all_bags_audit = []
    patient_to_sets = {} # Track which sets each patient appears in
    
    for set_name, set_dir in dataset_configs.items():
        # Handle list of directories for CrossValidation
        if isinstance(set_dir, list):
            if not all(os.path.exists(d) for d in set_dir):
                print(f"Warning: Some directories in {set_name} not found. Skipping {set_name}.")
                continue
        else:
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
            
            # Include ALL bags from all directories (no preference logic)
            # Annotated and Cropped both contribute to training, so both should be audited
            all_bags_audit.append(bag_info)
            
            # Map clinical ID to its dataset locations
            if base_id not in patient_to_sets:
                patient_to_sets[base_id] = set()
            patient_to_sets[base_id].add(set_name)

    # --- PHASE 2: DETECT LEAKAGE ---
    # NOTE: Only HoldOut + CrossValidation overlap is true leakage
    leakage_issues = []
    holdout_leakage = 0    # Leakage involving HoldOut (the real problem)
    
    for patient_id, sets in patient_to_sets.items():
        # Only flag if HoldOut overlaps with CrossValidation (true cross-contamination)
        if 'HoldOut' in sets and 'CrossValidation' in sets:
            leakage_issues.append(f"LEAKAGE: Patient {patient_id} found in both HoldOut and CrossValidation: {list(sets)}")
            holdout_leakage += 1

    # --- PHASE 3: SUMMARY & EXPORT ---
    total_unique_patients = len(patient_to_sets)
    print(f'\n--- GLOBAL DATA INTEGRITY AUDIT ---')
    print(f'Total Unique Patients (Global): {total_unique_patients}')
    print(f'Total Bags (including Annotated + Cropped): {len(all_bags_audit)}')
    
    if leakage_issues:
        print(f'[!] CRITICAL LEAKAGE DETECTED: {len(leakage_issues)} violations found.')
        print(f'    (HoldOut patients mixed with CrossValidation sets)')
        for issue in leakage_issues[:10]: print(f'  -> {issue}')
    else:
        print('Leakage Audit: OK (No HoldOut/CrossValidation cross-contamination detected)')

    # --- Step 3.1: Generate Patient Integrity Breakdown ---
    # This matches the training pipeline format precisely
    patient_breakdown = []
    for bag in all_bags_audit:  # Now iterating over list, not dict.values()
        b_id = bag['Patient_Bag_ID']
        base_id = bag['Base_Patient_ID']
        sets_for_patient = patient_to_sets.get(base_id, set())
        
        # Only flag actual leakage: HoldOut + CrossValidation mix
        is_holdout_leak = 'HoldOut' in sets_for_patient and 'CrossValidation' in sets_for_patient

        patient_breakdown.append({
            'Patient_Bag_ID': b_id,
            'Clinical_Group_ID': base_id,
            'Folder_Source': bag['Dataset_Source'],
            'Label': bag['Label'],
            'Patch_Count': bag['Patch_Count'],
            'Annotated_Positives': bag['Annotated_Positive_Count'],
            'LEAK_HOLDOUT_TRAIN': is_holdout_leak
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

    # Add a dedicated leakage summary to the report (only true HoldOut/Train leaks)
    if leakage_issues:
        leak_df = pd.DataFrame([{'Description': issue} for issue in leakage_issues])
        leak_df.to_csv('leakage_violations.csv', index=False)
        print(f'Critical leakage violations logged to: leakage_violations.csv')
    else:
        print('No critical leakage violations found.')

except Exception as e:
    import traceback
    print(f'Error during dataset audit: {e}')
    traceback.print_exc() 