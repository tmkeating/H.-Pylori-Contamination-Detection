"""
Central configuration for H. Pylori dataset paths and constants.
Import this module to access paths consistently across all scripts.

Example:
    from config import DATASET_ROOT, SCRATCH_ROOT
    
    base_data_path = DATASET_ROOT  # /home/tkeating/datasets/HelicoDataSet
    patient_csv = os.path.join(DATASET_ROOT, "PatientDiagnosis.csv")
"""

import os

# Dataset Storage Paths
DATASET_ROOT = "/home/tkeating/datasets/HelicoDataSet"
DEEPHP_DATASET_ROOT = "/home/tkeating/datasets/8117177"  # DeepHP H&E patches (Positive & Negative folders)

# Scratch/Temporary Storage Paths (for syncing datasets to local storage during training)
SCRATCH_ROOT = "/home/tkeating/.scratch/h_pylori_data"  # HelicoDataSet scratch
DEEPHP_SCRATCH_ROOT = "/home/tkeating/.scratch/deephp_data"  # DeepHP scratch

# Virtual Environment Path
VENV_ROOT = "/home/tkeating/venv"  # Python virtual environment for all dependencies

# Allow override via environment variables
DATASET_ROOT = os.environ.get('DATASET_ROOT', DATASET_ROOT)
DEEPHP_DATASET_ROOT = os.environ.get('DEEPHP_DATASET_ROOT', DEEPHP_DATASET_ROOT)
SCRATCH_ROOT = os.environ.get('SCRATCH_ROOT', SCRATCH_ROOT)
DEEPHP_SCRATCH_ROOT = os.environ.get('DEEPHP_SCRATCH_ROOT', DEEPHP_SCRATCH_ROOT)
VENV_ROOT = os.environ.get('VENV_ROOT', VENV_ROOT)

# Metadata Files
PATIENT_CSV = os.path.join(DATASET_ROOT, "PatientDiagnosis.csv")
PATCH_CSV = os.path.join(DATASET_ROOT, "HP_WSI-CoordAnnotatedAllPatches.csv")
PATCH_XLSX = os.path.join(DATASET_ROOT, "HP_WSI-CoordAnnotatedAllPatches.xlsx")

# Dataset Subdirectories
CV_ANNOTATED = os.path.join(DATASET_ROOT, "CrossValidation/Annotated")
CV_CROPPED = os.path.join(DATASET_ROOT, "CrossValidation/Cropped")
HOLDOUT = os.path.join(DATASET_ROOT, "HoldOut")
