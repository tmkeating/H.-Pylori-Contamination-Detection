# H. Pylori Contamination Detection (Iteration 22: Precision Searcher)

This project implements a **High-Performance Multi-Backbone MIL Pipeline** for the automated detection of *H. pylori* contamination in histology tissue samples. It is currently in the **"Precision Searcher"** phase, shifting from weighted attention to max-pooling to eliminate signal dilution.

## Project Structure

- `dataset.py`: High-performance loader with **Guaranteed Positive Sampling** to ensure $Y=1$ training bags contain annotated bacteria.
- `model.py`: Modular **HPyNet** supporting **Gated Attention** (Softmax) and **Max-MIL** (Max-Pooling) heads.
- `profiles.sh`: **Single Source of Truth** for experiment hyperparameters (now including `POOL_TYPE`).
- `train.py`: Unified training engine supporting:
    - **Stability Framework**: **Frozen BatchNorm** (`FREEZE_BN`) and **Gradient Clipping** (`CLIP_GRAD`) to handle noisy MIL bag statistics.
    - **Top-K Inference**: Patient-level diagnosis based on the Top-3 patch probabilities to filter sparse artifacts.
    - **Stochastic Weight Averaging (SWA)**: Calibrated at `swa_lr=1e-5` for flatter, more generalizable optima.

## Performance Benchmarks

### Current Model Performance (finalResults/)
The following results represent the current benchmarks for the ConvNeXt-Tiny models across 5 Folds:

| Metric | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 90.52% | 90.52% | 87.93% | 90.52% | 90.52% | **90.00%** |
| **Precision (+)**| 100.00%| 100.00%| 100.00%| 100.00%| 100.00%| **100.00%**|
| **Recall (+)** | 81.03% | 81.03% | 75.86% | 81.03% | 81.03% | **79.99%** |
| **Recall (-)** | 100.00%| 100.00%| 100.00%| 100.00%| 100.00%| **100.00%**|

### Iteration 22: Precision Searcher (Target)
To break the **"95% Recall Barrier"**, the system is transitioning to:
1.  **Max-Pooling MIL**: Routing gradients ONLY to the most suspicious patch, preventing dilution.
2.  **Sampling Injection**: Ensuring $Y=1$ bags always contain labeled bacteria during training.
3.  **Top-K Triage**: Identifying patients if the top-3 patches exceed $P > 0.1$.

## Diagnostic Architecture: The Two-Stage Ensemble
1.  **ConvNeXt-Tiny (The Searcher)**: Optimized via Max-MIL for hyper-sensitivity to sparse bacterial colonies.
2.  **ResNet50 (The Auditor)**: Optimized via Attention-MIL for hyper-precision to veto false detections.

## Hardware & Optimization
- **Compute**: Optimized for **NVIDIA A40/A100 (48GB/80GB)**.
- **Optimization**: Uses **Gradient Checkpointing** and **`torch.set_float32_matmul_precision('high')`**.
- **Data Locality**: Automatic node-local `/tmp` storage synchronization.

## How to Get Started

1. **Environment Setup**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Searcher Pipeline Execution (SLURM)**:
   ```bash
   PROFILE=SEARCHER POOL_TYPE=max ./submit_all_folds.sh
   ```

3. **Inference & Visualization**:
   Visuals are in `results/`, including `_gradcam_samples/` for explainable diagnosis.
