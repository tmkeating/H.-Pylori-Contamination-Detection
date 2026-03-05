# H. Pylori Contamination Detection (Iteration 21: Stability Framework)

This project implements a **High-Performance Multi-Backbone MIL Pipeline** for the automated detection of *H. pylori* contamination in histology tissue samples. It is currently in **Iteration 21**, focusing on stabilizing the ResNet50 and ConvNeXt-Tiny backbones using a centralized profile-driven architecture.

## Project Structure

- `dataset.py`: High-performance data loader with **Dynamic Bag Sampling** (Training) and **Multi-Pass Total Coverage** (Evaluation).
- `model.py`: **Gated Attention MIL** architecture supporting **ConvNeXt-Tiny** (Global Morphology) and **ResNet50** (Local Patterns).
- `profiles.sh`: **Single Source of Truth** for experiment hyperparameters (AUDITOR vs. SEARCHER profiles).
- `train.py`: Unified training engine supporting:
    - **Stability Framework**: **Frozen BatchNorm** (`FREEZE_BN`) and **Gradient Clipping** (`CLIP_GRAD`) to handle noisy MIL bag statistics.
    - **Artifact Suppression**: **Dynamic Color Jitter** (`JITTER`) to neutralize site-specific staining "shortcuts."
    - **Stochastic Weight Averaging (SWA)**: Calibrated at `swa_lr=1e-5` for flatter, more generalizable optima.
- `run_h_pylori.sh`: SLURM-optimized execution script for A40/A100 clusters.
- `ensemble_searcher.py`: Inference-stage merging tool for Multi-Backbone consensus.

## Performance Benchmarks

### Current Model Performance (finalResults/)
The following results represent the stable benchmarks for the ConvNeXt-Tiny ensemble (Iteration 21.0-21.5):

| Metric | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 90.52% | 90.52% | 87.93% | 90.52% | 90.52% | **90.00%** |
| **Precision (+)**| 100.00%| 100.00%| 100.00%| 100.00%| 100.00%| **100.00%**|
| **Recall (+)** | 81.03% | 81.03% | 75.86% | 81.03% | 81.03% | **79.99%** |
| **Recall (-)** | 100.00%| 100.00%| 100.00%| 100.00%| 100.00%| **100.00%**|

## Diagnostic Architecture: Two-Stage Ensemble
1.  **Auditor Profile**: Primary focus on specificity ($PosWeight=1-7.5$). Achieves zero false positives (100% Precision) to establish diagnostic guardrails.
2.  **Searcher Profile**: Primary focus on sensitivity. Designed to flag sparse infections ($P \approx 0.05$) currently diluted by weighted average attention.

## Hardware & Optimization
- **Compute**: Optimized for **NVIDIA A40/A100 (48GB/80GB)**.
- **Optimization**: Uses **Gradient Checkpointing** and **`torch.set_float32_matmul_precision('high')`**.
- **Data Locality**: Automatic node-local `/tmp` storage synchronization.

## Upcoming Development: Iteration 22 (Strategic Pivot)
The next phase (Iteration 22) will transition to **Max-MIL** and **Guaranteed Sampling** to solve the signal dilution and sampling void issues identified during the Iteration 21 stabilization.
