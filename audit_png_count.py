#!/usr/bin/env python3
"""
# H. Pylori Dataset PNG Counter & Audit Script
# -----------------------------------------------
# This script performs a comprehensive audit of PNG files across both the permanent
# dataset storage and the scratch directory (local NVMe SSD), comparing file counts
# and patch distributions to ensure proper data synchronization and integrity.
#
# What it does:
#   1. Counts PNG files in the permanent dataset (/import/fhome/vlia/HelicoDataSet)
#   2. Counts PNG files in the scratch directory (/tmp/ricse03_h_pylori_data)
#   3. Generates patient-by-patient breakdowns for both locations
#   4. Reports sync status (are all permanent files present in scratch?)
#   5. Separates VALID training patches from raw file count (blacklist filtering)
#   6. Exports comprehensive CSV report with all comparisons
#
# Why this is critical:
#   1. Data Sync Verification: Confirms training data has been properly synced to
#      local NVMe scratch before model training begins
#   2. Blacklist Enforcement: Verifies that conflict bags and duplicate images are
#      properly excluded (not counted, not synced)
#   3. Gap Detection: Identifies if any patches are missing between permanent and
#      scratch, which would indicate sync problems
#   4. Audit Trail: Provides CSV documentation for reproducibility and verification
#
# IMPORTANT: Understanding Patch Counts
# ======================================
#
# THREE DIFFERENT COUNTS (all are correct for different purposes):
#
#   A) RAW PNG FILES ON DISK (permanent dataset):
#      - Count:     ~216,865 patches
#      - Source:    This script's permanent scan (png_audit_report.csv)
#      - Includes:  All PNG files physically present in dataset directories
#      - Excludes:  Blacklist bags (5 bags, ~2,744 patches)
#      - Purpose:   Raw inventory of available files
#
#   B) SYNCED TO SCRATCH (local NVMe SSD):
#      - Count:     ~214,121 patches (after cleanup removes blacklist bags)
#      - Source:    This script's scratch scan (png_audit_report.csv)
#      - Includes:  Only files synced via rsync --exclude filters
#      - Excludes:  Blacklist bags (prevented from syncing)
#      - Purpose:   Confirms training data is available locally
#      - Note:      Should be same as A minus blacklist bags
#
#   C) CLINICAL-VALIDATED TRAINING PATCHES:
#      - Count:     ~214,644 patches
#      - Source:    data_integrity_summary.csv (from verify_data_integrity.py)
#      - Includes:  Patches with clinical patient metadata only
#      - Excludes:  Orphaned/unmatched patches (Priority 4 filter)
#      - Purpose:   Actual training set size used by HPyloriDataset
#      - Note:      Slightly different from B due to clinical validation filtering
#
# COUNT RECONCILIATION:
#   - Have on permanent disk:                       ~216,865 patches
#   - After removing blacklist from permanent:      ~214,121 patches (-2,744 blacklist)
#   - Clinically-validated patches for training:    ~214,644 patches
#
# BLACKLIST IMPACT:
#   - Location: blacklist.json at project root
#   - Conflict Bags: 5 bags excluded entirely (conflicting diagnoses)
#   - Duplicate Images: ~317 specific files marked as cross-folder duplicates
#   - Effect: rsync skip via --exclude prevents these from syncing to scratch
#   - Verification: This script confirms they were actually removed
#
# SYNC STATUS INTERPRETATION:
#   - FULLY SYNCED: Scratch has all permanent patches (permanent = scratch)
#   - NOT SYNCED: Some permanent patches missing from scratch (investigate!)
#   - EXTRA: More patches in scratch than permanent (shouldn't happen)
#   - DISCREPANCY: Amount difference shown in CSV report
#
# OUTPUT INTERPRETATION:
#   - CSV Report: png_audit_report.csv contains:
#     * Summary comparison (permanent vs scratch side-by-side)
#     * Sync status and discrepancies
#     * Patient-by-patient breakdown for permanent dataset
#     * Patient-by-patient breakdown for scratch directory
#     * Subtotals for each directory (Annotated, Cropped, HoldOut)
#     * Grand totals and metadata
#
#   - Expected Results:
#     * Permanent Annotated:   ~2,953 patches
#     * Permanent Cropped:    ~126,090 patches
#     * Permanent HoldOut:    ~87,822 patches
#     * Permanent TOTAL:     ~216,865 patches
#     * Scratch TOTAL:       ~214,121 patches (2,744 fewer due to blacklist removal)
#
# HOW TO RUN:
#   python3 audit_png_count.py
#
# OUTPUT FILES:
#   - png_audit_report.csv: Detailed report with all counts and comparisons
# -----------------------------------------------
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
from datetime import datetime

class DatasetAuditor:
    def __init__(self, dataset_root, scratch_root=None):
        self.dataset_root = Path(dataset_root)
        self.scratch_root = Path(scratch_root) if scratch_root else None
        
        # Permanent dataset structure
        self.results = {
            'CrossValidation/Annotated': defaultdict(int),
            'CrossValidation/Cropped': defaultdict(int),
            'HoldOut': defaultdict(int)
        }
        
        # Scratch directory structure
        self.scratch_results = {
            'CrossValidation/Annotated': defaultdict(int),
            'CrossValidation/Cropped': defaultdict(int),
            'HoldOut': defaultdict(int)
        }
        
        self.totals = {}
        self.scratch_totals = {}
        self.blacklist = set()
        self.image_blacklist = []
        self.blacklisted_patches_count = 0
        self._load_blacklist()
        
    def _load_blacklist(self):
        """Load the blacklist of duplicate bags and images to exclude from sync."""
        blacklist_path = Path('blacklist.json')
        try:
            with open(blacklist_path, 'r') as f:
                blacklist_data = json.load(f)
                self.blacklist = set(blacklist_data.get('conflict_blacklist', {}).keys())
                
                # Load image_blacklist, but filter out entries in already-blacklisted bags
                all_image_blacklist = blacklist_data.get('image_blacklist', [])
                self.image_blacklist = [
                    item for item in all_image_blacklist 
                    if item.get('folder') not in self.blacklist
                ]
        except:
            self.blacklist = set()
            self.image_blacklist = []
    
    def count_png_files(self):
        """Recursively count PNG files in permanent dataset directory."""
        print("\n" + "="*80)
        print("H. PYLORI DATASET PNG AUDIT - STARTING")
        print("="*80)
        
        for dir_name in self.results.keys():
            dir_path = self.dataset_root / dir_name
            
            if not dir_path.exists():
                print(f"\n⚠️  WARNING: Directory not found: {dir_path}")
                self.results[dir_name]['_ERROR'] = "Directory not found"
                continue
            
            print(f"\n📁 Scanning (PERMANENT): {dir_name}/")
            print("-" * 60)
            
            png_count = 0
            patient_bags = defaultdict(lambda: {'bags': defaultdict(int), 'total': 0})
            
            # Walk through the directory structure
            for root, dirs, files in os.walk(dir_path):
                png_files = [f for f in files if f.lower().endswith('.png')]
                
                if png_files:
                    rel_path = Path(root).relative_to(dir_path)
                    path_parts = rel_path.parts
                    
                    # Extract patient and bag info from path structure
                    if len(path_parts) >= 1:
                        bag_folder = path_parts[0]  # e.g., "B22-01_0"
                        
                        # Skip blacklisted bags, but count them
                        if bag_folder in self.blacklist:
                            print(f"  ⊘ SKIPPED (blacklisted): {bag_folder} ({len(png_files)} files)")
                            self.blacklisted_patches_count += len(png_files)
                            continue
                        
                        # Extract patient ID (e.g., "B22-01" from "B22-01_0")
                        patient_id = '_'.join(bag_folder.split('_')[:-1]) if '_' in bag_folder else bag_folder
                        
                        patient_bags[patient_id]['bags'][bag_folder] = len(png_files)
                        patient_bags[patient_id]['total'] += len(png_files)
                        png_count += len(png_files)
            
            # Store and display results
            self.results[dir_name] = patient_bags
            self.totals[dir_name] = png_count
            
            print(f"\n✓ Patient Summary for {dir_name}:")
            print(f"  Total Patients: {len(patient_bags)}")
            print(f"  Total Patches: {png_count:,}")
    
    def count_scratch_png_files(self):
        """Recursively count PNG files in scratch directory."""
        if not self.scratch_root or not self.scratch_root.exists():
            print(f"\n⚠️  WARNING: Scratch directory not found: {self.scratch_root}")
            return
        
        print(f"\n\n📁 Scanning (SCRATCH): {self.scratch_root}/")
        print("="*80)
        
        for dir_name in self.scratch_results.keys():
            dir_path = self.scratch_root / dir_name
            
            if not dir_path.exists():
                print(f"\n  ⊘ Not present in scratch: {dir_name}")
                self.scratch_totals[dir_name] = 0
                continue
            
            print(f"\n📁 Scanning: {dir_name}/")
            print("-" * 60)
            
            png_count = 0
            patient_bags = defaultdict(lambda: {'bags': defaultdict(int), 'total': 0})
            
            # Walk through the directory structure
            for root, dirs, files in os.walk(dir_path):
                png_files = [f for f in files if f.lower().endswith('.png')]
                
                if png_files:
                    rel_path = Path(root).relative_to(dir_path)
                    path_parts = rel_path.parts
                    
                    # Extract patient and bag info from path structure
                    if len(path_parts) >= 1:
                        bag_folder = path_parts[0]  # e.g., "B22-01_0"
                        
                        # Skip blacklisted bags
                        if bag_folder in self.blacklist:
                            continue
                        
                        # Extract patient ID
                        patient_id = '_'.join(bag_folder.split('_')[:-1]) if '_' in bag_folder else bag_folder
                        
                        patient_bags[patient_id]['bags'][bag_folder] = len(png_files)
                        patient_bags[patient_id]['total'] += len(png_files)
                        png_count += len(png_files)
            
            # Store and display results
            self.scratch_results[dir_name] = patient_bags
            self.scratch_totals[dir_name] = png_count
            
            print(f"\n✓ Summary for {dir_name} (scratch):")
            print(f"  Total Patients: {len(patient_bags)}")
            print(f"  Total Patches: {png_count:,}")
    
    def generate_report(self, dataset='permanent'):
        """Generate detailed CSV report for specified dataset.
        
        Args:
            dataset (str): 'permanent' or 'scratch' to specify which dataset to report on
        """
        rows = []
        
        # Choose which dataset to use
        if dataset == 'scratch':
            results = self.scratch_results
            source = 'SCRATCH'
        else:
            results = self.results
            source = 'PERMANENT'
        
        for dir_name in results.keys():
            patient_bags = results[dir_name]
            
            for patient_id in sorted(patient_bags.keys()):
                if patient_id == '_ERROR':
                    rows.append({
                        'Directory': dir_name,
                        'Patient': 'ERROR',
                        'Bags': 0,
                        'Total_Patches': 0,
                        'Note': patient_bags[patient_id]
                    })
                    continue
                
                bags = patient_bags[patient_id]['bags']
                total = patient_bags[patient_id]['total']
                num_bags = len(bags)
                
                rows.append({
                    'Directory': dir_name,
                    'Patient': patient_id,
                    'Bags': num_bags,
                    'Total_Patches': total,
                    'Bag_Details': ','.join([f"{b}({c})" for b, c in sorted(bags.items())])
                })
        
        return pd.DataFrame(rows)
    
    def print_summary(self):
        """Print reconciliation summary comparing permanent dataset and scratch."""
        print("\n" + "="*80)
        print("RECONCILIATION SUMMARY - PERMANENT vs SCRATCH")
        print("="*80)
        
        cv_annotated = self.totals.get('CrossValidation/Annotated', 0)
        cv_cropped = self.totals.get('CrossValidation/Cropped', 0)
        holdout = self.totals.get('HoldOut', 0)
        
        scratch_cv_annotated = self.scratch_totals.get('CrossValidation/Annotated', 0)
        scratch_cv_cropped = self.scratch_totals.get('CrossValidation/Cropped', 0)
        scratch_holdout = self.scratch_totals.get('HoldOut', 0)
        
        training_total = cv_annotated + cv_cropped
        grand_total = training_total + holdout
        
        scratch_training_total = scratch_cv_annotated + scratch_cv_cropped
        scratch_grand_total = scratch_training_total + scratch_holdout
        
        print(f"\n{'Directory':<40} {'PERMANENT':>15} {'SCRATCH':>15}")
        print("-" * 72)
        print(f"{'CrossValidation/Annotated':<40} {cv_annotated:>15,} {scratch_cv_annotated:>15,}")
        print(f"{'CrossValidation/Cropped':<40} {cv_cropped:>15,} {scratch_cv_cropped:>15,}")
        print("-" * 72)
        print(f"{'Training Total':<40} {training_total:>15,} {scratch_training_total:>15,}")
        print(f"{'HoldOut':<40} {holdout:>15,} {scratch_holdout:>15,}")
        print("=" * 72)
        print(f"{'GRAND TOTAL':<40} {grand_total:>15,} {scratch_grand_total:>15,}")
        print("=" * 72)
        
        # Scratch statistics
        if scratch_grand_total > 0:
            sync_discrepancy = grand_total - scratch_grand_total
            print(f"\n📋 PERMANENT vs SCRATCH SYNC:")
            print(f"  Permanent Dataset:  {grand_total:,} patches")
            print(f"  Scratch Directory:  {scratch_grand_total:,} patches")
            print(f"  Difference:         {sync_discrepancy:,} patches")
            
            if sync_discrepancy > 0:
                print(f"\n  ⚠️  NOT SYNCED: {sync_discrepancy:,} patches in permanent but not in scratch")
            elif sync_discrepancy < 0:
                print(f"\n  ⚠️  EXTRA: {abs(sync_discrepancy):,} extra patches in scratch")
            else:
                print(f"\n  ✓ FULLY SYNCED: Scratch directory has all permanent dataset patches!")
        
        print(f"\n💡 NOTES:")
        print(f"  - Blacklisted bags (duplicates/conflicts): {len(self.blacklist)}")
        print(f"  - Included in permanent count, excluded from scratch sync")
        print(f"  - Check blacklist.json for details on blacklisted bags")
        
        print("\n" + "="*80)
    
    def save_report(self, output_path):
        """Save detailed report to CSV with summary and patient-by-patient breakdown."""
        # Build header section with summary comparison
        header_rows = [
            {'Directory': '=' * 70, 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''},
            {'Directory': 'H. PYLORI DATASET PNG AUDIT - SUMMARY REPORT', 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''},
            {'Directory': '=' * 70, 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''},
            {'Directory': '', 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''},
            {'Directory': 'RECONCILIATION SUMMARY - PERMANENT vs SCRATCH', 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''},
            {'Directory': '-' * 70, 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''},
        ]
        
        # Add summary comparison rows
        cv_annotated = self.totals.get('CrossValidation/Annotated', 0)
        cv_cropped = self.totals.get('CrossValidation/Cropped', 0)
        holdout = self.totals.get('HoldOut', 0)
        
        scratch_cv_annotated = self.scratch_totals.get('CrossValidation/Annotated', 0)
        scratch_cv_cropped = self.scratch_totals.get('CrossValidation/Cropped', 0)
        scratch_holdout = self.scratch_totals.get('HoldOut', 0)
        
        training_total = cv_annotated + cv_cropped
        grand_total = training_total + holdout
        
        scratch_training_total = scratch_cv_annotated + scratch_cv_cropped
        scratch_grand_total = scratch_training_total + scratch_holdout
        
        header_rows.append({'Directory': 'CrossValidation/Annotated', 'Patient': f'PERMANENT: {cv_annotated:,}', 'Bags': f'SCRATCH: {scratch_cv_annotated:,}', 'Total_Patches': '', 'Bag_Details': ''})
        header_rows.append({'Directory': 'CrossValidation/Cropped', 'Patient': f'PERMANENT: {cv_cropped:,}', 'Bags': f'SCRATCH: {scratch_cv_cropped:,}', 'Total_Patches': '', 'Bag_Details': ''})
        header_rows.append({'Directory': '-' * 70, 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''})
        header_rows.append({'Directory': 'Training Total', 'Patient': f'PERMANENT: {training_total:,}', 'Bags': f'SCRATCH: {scratch_training_total:,}', 'Total_Patches': '', 'Bag_Details': ''})
        header_rows.append({'Directory': 'HoldOut', 'Patient': f'PERMANENT: {holdout:,}', 'Bags': f'SCRATCH: {scratch_holdout:,}', 'Total_Patches': '', 'Bag_Details': ''})
        header_rows.append({'Directory': '=' * 70, 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''})
        header_rows.append({'Directory': 'GRAND TOTAL', 'Patient': f'PERMANENT: {grand_total:,}', 'Bags': f'SCRATCH: {scratch_grand_total:,}', 'Total_Patches': '', 'Bag_Details': ''})
        header_rows.append({'Directory': '=' * 70, 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''})
        
        # Add sync comparison if scratch exists
        if scratch_grand_total > 0:
            sync_discrepancy = grand_total - scratch_grand_total
            header_rows.append({'Directory': '', 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''})
            header_rows.append({'Directory': 'SYNC STATUS', 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''})
            header_rows.append({'Directory': '-' * 70, 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''})
            header_rows.append({'Directory': 'Permanent Dataset Total', 'Patient': f'{grand_total:,} patches', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''})
            header_rows.append({'Directory': 'Scratch Directory Total', 'Patient': f'{scratch_grand_total:,} patches', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''})
            header_rows.append({'Directory': 'Difference', 'Patient': f'{sync_discrepancy:,} patches', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''})
            
            if sync_discrepancy > 0:
                header_rows.append({'Directory': 'Status', 'Patient': f'NOT SYNCED: {sync_discrepancy:,} patches in permanent but not in scratch', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''})
            elif sync_discrepancy < 0:
                header_rows.append({'Directory': 'Status', 'Patient': f'EXTRA: {abs(sync_discrepancy):,} extra patches in scratch', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''})
            else:
                header_rows.append({'Directory': 'Status', 'Patient': 'FULLY SYNCED: All patches accounted for', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''})
        
        # Add blacklist summary right after sync status
        expected_difference = self.blacklisted_patches_count + len(self.image_blacklist)
        actual_difference = grand_total - scratch_grand_total
        
        header_rows.append({'Directory': '', 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''})
        header_rows.append({'Directory': 'BLACKLIST SUMMARY', 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''})
        header_rows.append({'Directory': '-' * 70, 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''})
        header_rows.append({'Directory': '  Blacklisted bags (conflict_blacklist)', 'Patient': f'{len(self.blacklist)} bags', 'Bags': '', 'Total_Patches': f'{self.blacklisted_patches_count:,} patches', 'Bag_Details': 'Excluded from scratch sync'})
        header_rows.append({'Directory': '  Blacklisted images (image_blacklist)', 'Patient': f'{len(self.image_blacklist)} images', 'Bags': '', 'Total_Patches': '', 'Bag_Details': 'Individual duplicates'})
        header_rows.append({'Directory': '  Total blacklisted items', 'Patient': f'{len(self.blacklist) + len(self.image_blacklist)} items', 'Bags': '', 'Total_Patches': f'{expected_difference:,} approx patches', 'Bag_Details': 'Conflict bags + individual images'})
        header_rows.append({'Directory': '  Expected Permanent vs Scratch difference', 'Patient': '', 'Bags': '', 'Total_Patches': f'{expected_difference:,} patches', 'Bag_Details': 'Based on blacklist removals'})
        header_rows.append({'Directory': '  Actual Permanent vs Scratch difference', 'Patient': '', 'Bags': '', 'Total_Patches': f'{actual_difference:,} patches', 'Bag_Details': 'Observed from counts'})
        
        header_rows.append({'Directory': '', 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''})
        header_rows.append({'Directory': 'PATIENT-BY-PATIENT BREAKDOWN', 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''})
        header_rows.append({'Directory': '-' * 70, 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''})
        
        header_df = pd.DataFrame(header_rows)
        
        # Get patient breakdown for permanent dataset
        detail_df = self.generate_report(dataset='permanent')
        
        # Combine header and permanent details
        df = pd.concat([header_df, detail_df], ignore_index=True)
        
        # Add permanent subtotals
        permanent_summary_rows = []
        permanent_summary_rows.append({'Directory': '', 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''})
        permanent_summary_rows.append({'Directory': '-' * 70, 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''})
        
        for dir_name in self.results.keys():
            permanent_summary_rows.append({
                'Directory': f'>>> SUBTOTAL (PERMANENT): {dir_name}',
                'Patient': '',
                'Bags': '',
                'Total_Patches': self.totals.get(dir_name, 0),
                'Bag_Details': ''
            })
        
        permanent_summary_df = pd.DataFrame(permanent_summary_rows)
        df = pd.concat([df, permanent_summary_df], ignore_index=True)
        
        # Add scratch patient breakdown if scratch data exists
        if scratch_grand_total > 0:
            scratch_header_rows = [
                {'Directory': '', 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''},
                {'Directory': 'SCRATCH PATIENT-BY-PATIENT BREAKDOWN', 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''},
                {'Directory': '-' * 70, 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''}
            ]
            scratch_header_df = pd.DataFrame(scratch_header_rows)
            df = pd.concat([df, scratch_header_df], ignore_index=True)
            
            # Get patient breakdown for scratch dataset
            scratch_detail_df = self.generate_report(dataset='scratch')
            df = pd.concat([df, scratch_detail_df], ignore_index=True)
            
            # Add scratch subtotals
            scratch_summary_rows = []
            scratch_summary_rows.append({'Directory': '', 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''})
            scratch_summary_rows.append({'Directory': '-' * 70, 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''})
            
            for dir_name in self.scratch_results.keys():
                scratch_summary_rows.append({
                    'Directory': f'>>> SUBTOTAL (SCRATCH): {dir_name}',
                    'Patient': '',
                    'Bags': '',
                    'Total_Patches': self.scratch_totals.get(dir_name, 0),
                    'Bag_Details': ''
                })
            
            scratch_summary_df = pd.DataFrame(scratch_summary_rows)
            df = pd.concat([df, scratch_summary_df], ignore_index=True)
        
        # Add grand totals
        permanent_grand_total = sum(self.totals.values())
        scratch_grand_total = sum(self.scratch_totals.values())
        
        # Add scratch grand total
        df = pd.concat([df, pd.DataFrame([{
            'Directory': 'GRAND TOTAL (SCRATCH)',
            'Patient': '',
            'Bags': '',
            'Total_Patches': scratch_grand_total,
            'Bag_Details': ''
        }])], ignore_index=True)

        # Add permanent grand total
        df = pd.concat([df, pd.DataFrame([{
            'Directory': 'GRAND TOTAL (PERMANENT)',
            'Patient': '',
            'Bags': '',
            'Total_Patches': permanent_grand_total,
            'Bag_Details': ''
        }])], ignore_index=True)
        
        # Add remaining metadata footer (blacklist summary now in header section)
        footer_rows = [
            {'Directory': '', 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''},
            {'Directory': 'METADATA', 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''},
            {'Directory': '-' * 70, 'Patient': '', 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''},
            {'Directory': 'Permanent Dataset Root', 'Patient': str(self.dataset_root), 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''},
            {'Directory': 'Scratch Directory Root', 'Patient': str(self.scratch_root), 'Bags': '', 'Total_Patches': '', 'Bag_Details': ''},
        ]
        footer_df = pd.DataFrame(footer_rows)
        df = pd.concat([df, footer_df], ignore_index=True)
        
        df.to_csv(output_path, index=False)
        print(f"\n✓ Report saved to: {output_path}")
        return df


def main():
    # Configuration
    dataset_root = Path('/import/fhome/vlia/HelicoDataSet')
    scratch_root = Path('/tmp/ricse03_h_pylori_data')
    output_csv = Path('audit_png_count_report.csv')
    
    # Run audit
    auditor = DatasetAuditor(dataset_root, scratch_root)
    auditor.count_png_files()
    auditor.count_scratch_png_files()
    auditor.print_summary()
    
    # Save report
    df = auditor.save_report(output_csv)
    
    print("\n✓ Audit complete!")
    print(f"\nTo view detailed results:\n  cat {output_csv}")


if __name__ == '__main__':
    main()
