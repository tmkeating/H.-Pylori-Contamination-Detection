# H. Pylori Contamination Detection (Iteration 26.0+: Hybrid Ensemble Fusion)

This project implements a **High-Resolution Multi-Stage MIL Pipeline** for the automated detection of *H. pylori* contamination in histology tissue samples. It features a **Searcher-Rescue** architecture designed to identify sparse bacterium clusters in high-resolution whole-slide imaging, combined with an **intelligent Hybrid Ensemble** that achieves 92.11% accuracy with perfect precision.

## Execution Workflow (Step-by-Step)

To reproduce the results, follow this specific execution order:

### 0. Data Integrity & Deduplication Audit
Before training, run a byte-level MD5 hash audit across the dataset to identify exact duplicated images across the Folders (Annotated, Cropped, HoldOut) to prevent data leakage and skewed metrics.
```bash
sbatch submit_dedupe.sh
```
*Outputs:* `global_image_inventory.csv`, `global_image_duplicates.csv`, `dataset_presence_matrix.csv`, and `patient_duplicate_audit.csv`.

### 1. Training (5-Fold Cross-Validation)
Launch the primary training sweep using the `SEARCHER` profile. This uses ConvNeXt-Tiny with Attention-MIL and SWA.
```bash
PROFILE=SEARCHER MODEL_NAME=convnext_tiny ITER=30.0 ./submit_all_folds.sh
```
*Outputs: `results/*_model_brain.pth` and `results/*_patient_consensus.csv`.*

### 2. Data Integrity and Blacklist Removal Check
After it has been synced, run the .png audit count to ensure the blacklisted files are being properly removed/excluded from the scratch directory.
```bash
python3 audit_png_count.py
```
*Outputs: `audit_png_count_report.csv`.*

### 3. High-Resolution Rescue (Dense Inference Pass)
Specifically target difficult "Ghost Patients" using the dense Stride-128 rescue scan. This recovers signals from sparse biopsies that were missed by the default Stride-512/Stride-250 sampling. 
```bash
# Update submit_rescue.sh with the correct Searcher Run IDs
sbatch submit_rescue.sh
```
*Outputs: `results/rescue_ensemble/rescue_*.csv`.*

### 4. Final Hybrid Ensemble & Fusion (⭐ RECOMMENDED METHOD)
Intelligently fuses multiple fusion approaches (Ensemble Voting, Meta-Classifier, and **Hybrid Ensemble**) to produce the best clinical predictions.
```bash
# Generate the 92.11% Hybrid Ensemble (Production-Ready)
python3 ensemble_voting.py --runs 302,303,299,300,301
```

**Primary Output (Use These Files):**
- `hybrid_ensemble_results_*.csv` - Patient predictions with confidence scores
- `hybrid_ensemble_summary_*.csv` - Clinical metrics (92.11% accuracy, 100% precision, 100% specificity)
- `hybrid_ensemble_roc_pr_*.png` - ROC/PR curves  
- `hybrid_ensemble_threshold_analysis_*.png` - Threshold optimization

**Comparison Outputs (For Analysis):**
- `ensemble_voting_results_*.csv` - Base ensemble voting predictions
- `meta_classifier_results_*.csv` - Random Forest meta-classifier predictions

*Key Result: **92.11% Accuracy | 100% Precision (Zero False Positives) | 100% Specificity***

### 5. (Optional). Interpretability Analysis & Reports (Grad-CAM & Metrics)
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
- `global_duplicates_check.py`: A high-performance byte-level image duplication checker that checks the first 8kb and compares file size for high confidence in results.
- `audit_png_count_report.csv`: Ensures that the blacklist is being adhered to.
- `ensemble_voting.py`: **Hybrid Ensemble Fusion** combining three methods: (1) Majority Vote Ensemble, (2) Random Forest Meta-Classifier (LOO-CV), (3) **Hybrid Ensemble** (RECOMMENDED ⭐)
- `profiles.sh`: Centralized hyperparameters (Learning rates, Weights, Data paths).

## 📊 Final Clinical Performance

| Metric | Ensemble Voting | Meta-Classifier | **Hybrid Ensemble** ⭐ |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 86.84% | 91.23% | **92.11%** |
| **Precision** | 85.00% | 97.96% | **100.00%** |
| **Recall** | 89.47% | 84.21% | 84.21% |
| **Specificity** | 84.21% | 98.25% | **100.00%** |
| **F1 Score** | 87.18% | 90.57% | **91.43%** |
| **False Positives** | 9 | 1 | **0** |
| **False Negatives** | 6 | 9 | 9 |


## 🛠️ Key Pipeline Features
- **Stride-128 Rescue Pass**: Dense-window overlap to "catch" sparse bacteria that fall in gaps at default strides.
- **Top-3 Mixed MIL**: Balances sensitivity with noise resilience by averaging the top 3 most confident tissue chunks.
- **Contrast-Boosted TTA**: 16-way transforms (8 spatial + 1.1x contrast jitter) to "pop" faint IHC signals.
- **Hybrid Ensemble Strategy**: Intelligently combines three fusion methods:
  - **High Confidence Zone (>0.95)**: Uses ensemble voting
  - **Uncertainty Zone (0.35-0.55)**: Uses meta-classifier
  - **Medium Confidence (0.55-0.95)**: Blends both methods (60% ensemble + 40% meta)
  - **Result**: Best-in-class accuracy with zero false positives for clinical safety

---

## Hardware & Optimization
- **Compute**: Optimized for **NVIDIA A40/A100 (48GB/80GB)**.
- **Precision**: `torch.set_float32_matmul_precision('high')`.
- **Data Locality**: Automated node-local `/tmp` storage sync via `run_h_pylori.sh`.
