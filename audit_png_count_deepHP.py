#!/usr/bin/env python3
"""
# H. Pylori DeepHP Dataset PNG Counter & Audit Script
# =======================================================
# ⚠️  THIS SCRIPT IS FOR DEEPHP DATASET ONLY - NOT FOR HELICODATASET
# 
# ★ Use this script for DeepHP patch counts: ~394,926 total (111,005 positive + 283,921 negative) ★
# This script performs a comprehensive audit of PNG files in the DeepHP pre-training dataset,
# counting patch distributions and ensuring data integrity.
#
# What it does:
#   1. Counts PNG files in the DeepHP dataset (/export/hhome/ricse03/8117177/Positive and Negative)
#   2. Optionally counts PNG files in scratch directory (/tmp/ricse03_deephp_data)
#   3. Generates class-level breakdowns (Positive vs Negative)
#   4. Reports sync status if scratch exists
#   5. Exports comprehensive CSV report with all counts
#
# Dataset Structure (DeepHP):
#   - Location: /export/hhome/ricse03/8117177/
#   - Positive class: /export/hhome/ricse03/8117177/Positive/ (~111,005 patches)
#   - Negative class: /export/hhome/ricse03/8117177/Negative/ (~283,921 patches)
#   - Total: ~394,926 patches
#   - Used for: Backbone pre-training with 5-fold stratified cross-validation
#
# OUTPUT INTERPRETATION:
#   - Total Patches: Complete inventory of all PNG files
#   - Class Distribution: Positive vs Negative patch counts
#   - Sync Status: If scratch directory exists, compares with permanent storage
#
# HOW TO RUN:
#   python3 audit_png_count_deepHP.py
#
# OUTPUT FILES:
#   - deephp_audit_report.csv: Detailed report with class distributions
"""

import os
import sys
import pandas as pd
from pathlib import Path
from collections import defaultdict
import json

class DeepHPAuditor:
    def __init__(self, dataset_root, scratch_root=None):
        self.dataset_root = Path(dataset_root)
        self.scratch_root = Path(scratch_root) if scratch_root else None
        
        # DeepHP class structure
        self.results = {
            'Positive': 0,
            'Negative': 0
        }
        
        # Scratch directory structure (if present)
        self.scratch_results = {
            'Positive': 0,
            'Negative': 0
        }
        
        self.totals = {}
        self.scratch_totals = {}
    
    def count_png_files(self):
        """Recursively count PNG files in permanent DeepHP dataset directory."""
        print("\n" + "="*80)
        print("H. PYLORI DEEPHP DATASET PNG AUDIT - STARTING")
        print("="*80)
        
        for class_name in self.results.keys():
            class_path = self.dataset_root / class_name
            
            if not class_path.exists():
                print(f"\n⚠️  WARNING: Directory not found: {class_path}")
                self.results[class_name] = 0
                continue
            
            print(f"\n📁 Scanning (PERMANENT): {class_name}/")
            print("-" * 60)
            
            png_count = 0
            
            # Count PNG files directly in class directory
            for file in os.listdir(class_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    png_count += 1
            
            self.results[class_name] = png_count
            print(f"✓ {class_name} patches: {png_count:,}")
        
        self.totals['Total'] = sum(self.results.values())
        print(f"\n✓ Total patches (permanent): {self.totals['Total']:,}")
    
    def count_scratch_png_files(self):
        """Recursively count PNG files in scratch directory."""
        if not self.scratch_root or not self.scratch_root.exists():
            print(f"\n⚠️  WARNING: Scratch directory not found: {self.scratch_root}")
            return
        
        print(f"\n📁 Scanning (SCRATCH): {self.scratch_root}/")
        print("="*80)
        
        for class_name in self.scratch_results.keys():
            class_path = self.scratch_root / class_name
            
            if not class_path.exists():
                print(f"\n  ⊘ Not present in scratch: {class_name}")
                self.scratch_results[class_name] = 0
                continue
            
            print(f"\n📁 Scanning: {class_name}/")
            print("-" * 60)
            
            png_count = 0
            
            # Count PNG files directly in class directory
            for file in os.listdir(class_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    png_count += 1
            
            self.scratch_results[class_name] = png_count
            print(f"✓ {class_name} patches: {png_count:,}")
        
        self.scratch_totals['Total'] = sum(self.scratch_results.values())
        print(f"\n✓ Total patches (scratch): {self.scratch_totals['Total']:,}")
    
    def print_summary(self):
        """Print reconciliation summary comparing permanent dataset and scratch."""
        print("\n" + "="*80)
        print("RECONCILIATION SUMMARY - PERMANENT vs SCRATCH")
        print("="*80)
        
        pos_permanent = self.results.get('Positive', 0)
        neg_permanent = self.results.get('Negative', 0)
        total_permanent = self.totals.get('Total', 0)
        
        pos_scratch = self.scratch_results.get('Positive', 0)
        neg_scratch = self.scratch_results.get('Negative', 0)
        total_scratch = self.scratch_totals.get('Total', 0)
        
        print(f"\n{'Class':<40} {'PERMANENT':>15} {'SCRATCH':>15}")
        print("-" * 72)
        print(f"{'Positive':<40} {pos_permanent:>15,} {pos_scratch:>15,}")
        print(f"{'Negative':<40} {neg_permanent:>15,} {neg_scratch:>15,}")
        print("=" * 72)
        print(f"{'TOTAL':<40} {total_permanent:>15,} {total_scratch:>15,}")
        print("=" * 72)
        
        # Scratch statistics
        if total_scratch > 0:
            sync_discrepancy = total_permanent - total_scratch
            
            print(f"\n📋 PERMANENT vs SCRATCH SYNC:")
            print(f"  Permanent Dataset:  {total_permanent:,} patches")
            print(f"  Scratch Directory:  {total_scratch:,} patches")
            print(f"  Difference:         {sync_discrepancy:,} patches")
            
            if sync_discrepancy == 0:
                print(f"\n  ✓ FULLY SYNCED: Scratch directory has all permanent dataset patches!")
            elif sync_discrepancy > 0:
                print(f"\n  ⚠️  NOT SYNCED: {sync_discrepancy:,} patches in permanent but not in scratch")
            else:
                print(f"\n  ⚠️  EXTRA: {abs(sync_discrepancy):,} extra patches in scratch")
        
        print("\n" + "="*80)
    
    def save_report(self, output_path):
        """Save detailed report to CSV."""
        header_rows = [
            {'Dataset': '=' * 70, 'Positive': '', 'Negative': '', 'Total': ''},
            {'Dataset': 'H. PYLORI DEEPHP DATASET PNG AUDIT - SUMMARY REPORT', 'Positive': '', 'Negative': '', 'Total': ''},
            {'Dataset': '=' * 70, 'Positive': '', 'Negative': '', 'Total': ''},
            {'Dataset': '', 'Positive': '', 'Negative': '', 'Total': ''},
            {'Dataset': 'CLASS DISTRIBUTION', 'Positive': '', 'Negative': '', 'Total': ''},
            {'Dataset': '-' * 70, 'Positive': '', 'Negative': '', 'Total': ''},
        ]
        
        pos_permanent = self.results.get('Positive', 0)
        neg_permanent = self.results.get('Negative', 0)
        total_permanent = self.totals.get('Total', 0)
        
        pos_scratch = self.scratch_results.get('Positive', 0)
        neg_scratch = self.scratch_results.get('Negative', 0)
        total_scratch = self.scratch_totals.get('Total', 0)
        
        header_rows.append({'Dataset': 'Permanent Dataset', 'Positive': pos_permanent, 'Negative': neg_permanent, 'Total': total_permanent})
        
        if total_scratch > 0:
            header_rows.append({'Dataset': 'Scratch Directory', 'Positive': pos_scratch, 'Negative': neg_scratch, 'Total': total_scratch})
        
        header_rows.append({'Dataset': '=' * 70, 'Positive': '', 'Negative': '', 'Total': ''})
        
        # Add sync status if scratch exists
        if total_scratch > 0:
            sync_discrepancy = total_permanent - total_scratch
            header_rows.append({'Dataset': '', 'Positive': '', 'Negative': '', 'Total': ''})
            header_rows.append({'Dataset': 'SYNC STATUS', 'Positive': '', 'Negative': '', 'Total': ''})
            header_rows.append({'Dataset': '-' * 70, 'Positive': '', 'Negative': '', 'Total': ''})
            
            if sync_discrepancy == 0:
                header_rows.append({'Dataset': 'Status', 'Positive': 'FULLY SYNCED', 'Negative': '', 'Total': ''})
            elif sync_discrepancy > 0:
                header_rows.append({'Dataset': 'Status', 'Positive': f'NOT SYNCED: {sync_discrepancy:,} patches missing', 'Negative': '', 'Total': ''})
            else:
                header_rows.append({'Dataset': 'Status', 'Positive': f'EXTRA: {abs(sync_discrepancy):,} patches in scratch', 'Negative': '', 'Total': ''})
        
        # Add metadata footer
        header_rows.append({'Dataset': '', 'Positive': '', 'Negative': '', 'Total': ''})
        header_rows.append({'Dataset': 'METADATA', 'Positive': '', 'Negative': '', 'Total': ''})
        header_rows.append({'Dataset': '-' * 70, 'Positive': '', 'Negative': '', 'Total': ''})
        header_rows.append({'Dataset': 'Permanent Dataset Root', 'Positive': str(self.dataset_root), 'Negative': '', 'Total': ''})
        if self.scratch_root:
            header_rows.append({'Dataset': 'Scratch Directory Root', 'Positive': str(self.scratch_root), 'Negative': '', 'Total': ''})
        
        df = pd.DataFrame(header_rows)
        df.to_csv(output_path, index=False)
        print(f"\n✓ Report saved to: {output_path}")
        return df


def main():
    # Configuration
    from config import DEEPHP_DATASET_ROOT
    
    dataset_root = Path(DEEPHP_DATASET_ROOT)
    scratch_root = Path(f"/tmp/{os.environ.get('USER', 'ricse03')}_deephp_data")
    output_csv = Path('deephp_audit_report.csv')
    
    # Run audit
    auditor = DeepHPAuditor(dataset_root, scratch_root)
    auditor.count_png_files()
    auditor.count_scratch_png_files()
    auditor.print_summary()
    
    # Save report
    df = auditor.save_report(output_csv)
    
    print("\n✓ Audit complete!")
    print(f"\nTo view detailed results:\n  cat {output_csv}")


if __name__ == '__main__':
    main()
