# H. Pylori Contamination Detection (Iteration 14.1)

This project implements a **High-Precision Gated Attention MIL Pipeline** for the automated detection of *H. pylori* contamination in histology tissue samples. It is optimized for clinical safety, prioritizing zero false positives while maintaining high sensitivity.

## Project Structure

- `dataset.py`: High-performance data loader with **Dynamic Bag Sampling** (Training) and **Multi-Pass Total Coverage** (Evaluation).
- `model.py`: **Gated Attention MIL** architecture using **ConvNeXt-Tiny** as a frozen-feature extractor with a sigmoid-gated morphological aggregation head.
- `train.py`: The core engine supporting:
    - **Stochastic Weight Averaging (SWA)**: Calibrated at `swa_lr=1e-5` for flatter, more generalizable optima.
    - **Asymmetric Focal Loss**: Dynamic class weighting (currently `pos_weight=2.2`) to combat sparse-signal dilution.
    - **AdamW Optimization**: Aggressive Weight Decay ($0.1$) to suppress morphology mimics (debris).
- `run_h_pylori.sh`: SLURM-optimized execution script for A40/A100 clusters.
- `generate_visuals.py`: Automated clinical reporting (ROC/PR, Grad-CAM, Probability Histograms).

## Performance (Iteration 14.1 Stability Milestone)

The pipeline has achieved a **Stability Benchmark** through SWA calibration and gated filtering:

| Metric | Value | Status |
| :--- | :--- | :--- |
| **Global Precision** | **100.00%** | ✓ **Zero False Positives** across 116 hold-out cases |
| **Mean Accuracy** | **89.31%** | ↑ Stable cross-validation floor (>88%) |
| **Recall (Sensitivity)** | **78.62%** | ⚖️ Ongoing "Ghost Patient" optimization |
| **Standard Deviation** | **<1.0%** | 💎 Highly consistent inter-fold convergence |

## Diagnostic Architecture: Gated-Attention MIL

To break the heuristic bottleneck of traditional classifiers, this system uses a **Deep Multiple Instance Learning (MIL)** approach:
1.  **Feature Extraction**: ConvNeXt-Tiny extracts high-dimensional vectors ($768D$) for every tissue patch.
2.  **Gated Attention**: A learnable $\tanh(Vx) \odot \sigma(Ux)$ gate acts as a **Morphological Filter**. It assigns near-zero weights to background debris and high weights to bacterial signatures.
3.  **Aggregation**: The bag (patient) prediction is derived from the attention-weighted sum of all instances, allowing the model to focus on solitary bacteria in $2,000+$ background patches.

## Hardware & Hardware Optimization
- **Compute**: Optimized for **NVIDIA A40 (48GB)**.
- **Optimization**: Uses **Gradient Checkpointing** and **`torch.set_float32_matmul_precision('high')`** for 48GB VRAM efficiency and A40-native speed.
- **Data Locality**: Automatic synchronization of dataset to node-local `/tmp` storage to bypass network latency.

## How to Get Started

1. **Environment Setup**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Full Pipeline Execution (SLURM)**:
   ```bash
   MODEL_NAME=convnext_tiny ./submit_all_folds.sh
   ```

3. **Inference & Visualization**:
   Visuals are automatically generated in the `results/` folder upon completion, including `_gradcam_samples/` for explainable diagnosis.

## Key Research Findings
- **The SWA Stability Patch**: SWA with a low learning rate ($1e-5$) is essential for MIL stability; without it, the loss surface is too volatile for sparse bacterial signals.
- **Morphological Mimics**: Weight Decay ($0.1$) is the primary defense against "morphology mimics" (stain artifacts and debris) that otherwise trigger false positives.
- **Ghost Patients**: A subset of patients (~10%) remains "invisible" to single-model feature spaces, requiring multi-stage ensembling.

## Future Research: Iteration 15 (The Searcher Ensemble)
We are currently moving toward a **Two-Stage Multi-Model Ensemble**:
- **Stage 1 (The Searcher)**: Optimized for **100% Recall** via extreme `pos_weight=5.0` and lower Focal Loss $\gamma$.
- **Stage 2 (The Auditor)**: Our existing Iteration 14 model, which "vetos" Searcher detections to maintain the clinical 100% precision standard.
