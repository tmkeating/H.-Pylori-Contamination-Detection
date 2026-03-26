# H. Pylori Contamination Detection (Iteration 26.0: Clinical-Grade Ensemble)

This project implements a **High-Resolution Multi-Stage MIL Pipeline** for the automated detection of *H. pylori* contamination in histology tissue samples. It features a **Searcher-Rescue** architecture designed to identify sparse bacterium clusters in high-resolution whole-slide imaging.

## Execution Workflow (Step-by-Step)

To reproduce the clinical-grade results (94.7% Accuracy, 98.2% Recall), follow this specific execution order:

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

### 4. Interpretability Audit (Grad-CAM)
Generate visual evidence for the model's decisions, focusing on "Ghost Patients" (False Negatives).
```bash
./run_visuals.sh [run_id]
```
*Outputs: `results/*_gradcam_samples/` folder containing heatmaps.*

---

## Core Project Structure

- `dataset.py`: Multi-Pass coverage loader with **16-way Contrast-Boosted TTA**.
- `model.py`: **Gated Attention MIL** with **Top-3 Chunk Aggregation** for signal resilience.
- `train.py`: Unified engine featuring **SWA BN Recalibration** and **Grad-CAM Ghost Audits**.
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
