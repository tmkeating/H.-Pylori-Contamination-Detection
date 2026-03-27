# H. Pylori Contamination Detection (Iteration 26.0: Clinical-Grade Ensemble)

This project implements a **High-Resolution Multi-Stage MIL Pipeline** for the automated detection of *H. pylori* contamination in histology tissue samples. It features a **Searcher-Rescue** architecture designed to identify sparse bacterium clusters in high-resolution whole-slide imaging.

## Execution Workflow (Step-by-Step)

To reproduce the clinical-grade results (94.7% Accuracy, 98.2% Recall), follow this specific execution order:

### 0. Data Integrity & Deduplication Audit
Before training, run a byte-level MD5 hash audit across the dataset to identify exact duplicated images across the Folders (Annotated, Cropped, HoldOut) to prevent data leakage and skewed metrics.
```bash
sbatch submit_dedupe.sh
```
*Outputs:* `global_image_inventory.csv`, `global_image_duplicates.csv`, `dataset_presence_matrix.csv`, and `patient_duplicate_audit.csv`.

### 1. Training (5-Fold Cross-Validation)
Launch the primary training sweep using the `SEARCHER` profile. This uses ConvNeXt-Tiny with Attention-MIL and SWA.
```bash
sbatch submit_all_folds.sh
```
*Outputs: `results/*_model_brain.pth` and `results/*_patient_consensus.csv`.*

### 2. High-Resolution Rescue (Dense Inference Pass)
Specifically target difficult "Ghost Patients" using the dense Stride-128 rescue scan. This recovers signals from sparse biopsies that were missed by the default Stride-512/Stride-250 sampling. 
```bash
# Update submit_rescue.sh with the correct Searcher Run IDs
sbatch submit_rescue.sh
```
*Outputs: `results/rescue_ensemble/rescue_*.csv`.*

### 3. Final Meta-Ensemble & Hybrid Fusion
Fuse the newer high-precision results (302-306) with the sensitive historic models (299-301) to produce the "Golden Consensus" (94.7% Accuracy).
```bash
# Generate the 94.7% Hybrid Ensemble
python3 ensemble_voting.py --runs 302,303,299,300,301
```
*Outputs: `results/meta_fusion_results_*.csv` (The final Pathology hand-off report).*

### 4. Interpretability Analysis & Reports (Grad-CAM & Metrics)
Generate visual evidence for the model's decisions and patch/patient-level metrics. It bypasses older plotting packages and directly visualizes the confusion matrix and valid ROCs. Ensure you edit `run_visuals.sh` to target your desired `RUN_ID` before submitting.
```bash
sbatch run_visuals.sh
```
*Outputs: `results/*_gradcam_samples/` folder containing heatmaps, plus `*_confusion_matrix.png`, `*_roc_curve.png`, and `*_pr_curve.png` metric reports.*

---

## Core Project Structure

- `dataset.py`: Multi-Pass coverage loader with **16-way Contrast-Boosted TTA**. Handles live data integrity checks.
- `model.py`: **Gated Attention MIL** with **Top-3 Chunk Aggregation** for signal resilience.
- `train.py`: Unified engine featuring **SWA BN Recalibration** and **Grad-CAM Ghost Audits**.
- `generate_visuals.py`: Dedicated analysis script to render interpretable visual clinical layouts using Matplotlib cleanly.
- `global_duplicates_check.py`: High-performance 8KB-header MD5 deduplicator to guarantee strict set isolation.
- `ensemble_voting.py`: Meta-classifier using **Joint-Probability Gating** (Max > 0.39 & Mean > 0.28).
- `profiles.sh`: Centralized hyperparameters (Learning rates, Weights, Data paths).

## 📊 Final Clinical Performance (Hybrid Ensemble: 299-303)

| Metric | Value |
| :--- | :--- |
| **Accuracy** | **94.74%** |
| **Recall (+)** | **98.25%** (Only 1 FN in 114 patients) |
| **Precision (+)**| **91.80%** |
| **The Final Ghost**| `B22-295_0` (Confidence: 0.31) |

## 🛠️ Key Pipeline Features
- **Stride-128 Rescue Pass**: Dense-window overlap to "catch" sparse bacteria that fall in gaps at default strides.
- **Top-3 Mixed MIL**: Balances sensitivity with noise resilience by averaging the top 3 most confident tissue chunks.
- **Contrast-Boosted TTA**: 16-way transforms (8 spatial + 1.1x contrast jitter) to "pop" faint IHC signals.
- **Hybrid Fusion Logic**: Combines precision-weighted modern runs with sensitivity-weighted historical runs.

---

## Hardware & Optimization
- **Compute**: Optimized for **NVIDIA A40/A100 (48GB/80GB)**.
- **Precision**: `torch.set_float32_matmul_precision('high')`.
- **Data Locality**: Automated node-local `/tmp` storage sync via `run_h_pylori.sh`.
