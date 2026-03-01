import os
import torch
import pandas as pd
from dataset import HPyloriDataset

root = '/tmp/ricse03_h_pylori_data'
p_csv = os.path.join(root, 'PatientDiagnosis.csv')
patch_csv = os.path.join(root, 'HP_WSI-CoordAnnotatedAllPatches.xlsx')
train_dir = os.path.join(root, 'CrossValidation/Annotated')

try:
    ds = HPyloriDataset(root_dir=train_dir, patient_csv=p_csv, patch_csv=patch_csv, bag_mode=True)
    
    # Calculate Label distribution
    labels = [bag[1] for bag in ds.bags]
    pos = sum(labels)
    neg = len(labels) - pos
    
    # Calculate patches per bag
    patch_counts = [len(bag[0]) for bag in ds.bags]
    avg_patches = sum(patch_counts) / len(patch_counts)
    min_patches = min(patch_counts)
    max_patches = max(patch_counts)

    print(f'--- DATA HYGIENE REPORT (Annotated) ---')
    print(f'Total Bags: {len(ds.bags)}')
    print(f'Positive (Contaminated): {pos} ({pos/len(ds.bags):.1%})')
    print(f'Negative (Clean): {neg} ({neg/len(ds.bags):.1%})')
    print(f'Avg Patches/Bag: {avg_patches:.1f} (Min: {min_patches}, Max: {max_patches})')
    
    # Check for empty bags
    empty_bags = sum(1 for c in patch_counts if c == 0)
    print(f'Empty Bags: {empty_bags}')
    
    # Verify ID leakage (B22-47_0 vs B22-47_1 should have same label)
    bag_ids = [bag[2] for bag in ds.bags]
    issues = []
    base_id_labels = {}
    for bag in ds.bags:
        paths, label, b_id = bag
        base_id = b_id.split('_')[0]
        if base_id in base_id_labels:
            if base_id_labels[base_id] != label:
                issues.append(f'Mismatch: {base_id} has labels {base_id_labels[base_id]} and {label}')
        else:
            base_id_labels[base_id] = label
    
    if issues:
        print(f'ID Label Inconsistencies: {len(issues)}')
        print(issues[:5])
    else:
        print('ID Label Consistency: OK')

except Exception as e:
    import traceback
    print(f'Error auditing dataset: {e}')
    traceback.print_exc() 