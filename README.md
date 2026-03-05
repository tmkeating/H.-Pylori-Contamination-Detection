# H. Pylori Contamination Detection (Iteration 21.5: Hyper-Skeptical Auditor)

This project implements a **High-Precision Multi-Backbone MIL Pipeline** for the automated detection of *H. pylori* contamination in histology tissue samples. It is optimized for clinical safety, prioritizing zero false positives through an ensemble-based "Auditor" strategy.

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

## Performance (Iteration 21.5: Hyper-Skeptical ResNet Auditor)

The latest benchmark utilizes a **Hyper-Skeptical ResNet50** configuration (Jobs 105167-105171) to finalize the clinical guardrails:

| Metric | Value | Status |
| :--- | :--- | :--- |
| **Ensemble Precision (+)** | **100.00%** | ✓ **Zero False Positives** in 4/5 holds (Auditor Standard) |
| **Skeptical Recall (+)** | **41.38%** | ⚖️ Intentionally low to guarantee specificity |
| **Negative Recall (-)** | **93.10%** | 💎 Exceptional at identifying clean tissue |
| **Mean Accuracy** | **67.24%** | 🛡️ Conservative diagnostic shift |

### Auditor Configuration:
- **Architecture**: ResNet50 ($3 \times 3$ localist kernels)
- **Skepticism Bias**: `pos_weight=0.25` (4:1 bias toward Negatives)
- **Artifact Shield**: `jitter=0.45` (Extreme color/contrast noise suppression)
- **Stability**: `freeze_bn=True`, `clip_grad=0.3`

## Diagnostic Architecture: Diversity Ensemble

To break the "0.05 Wall" where sparse infections are missed, the system utilizes a **Two-Stage Ensemble**:
1.  **ConvNeXt-Tiny (The Searcher)**: Optimized for sensitivity to detect possible infections.
2.  **ResNet50 (The Auditor)**: Optimized for hyper-precision to veto false detections.

## Hardware & Optimization
- **Compute**: Optimized for **NVIDIA A40 (48GB)**.
- **Optimization**: Uses **Gradient Checkpointing** and **`torch.set_float32_matmul_precision('high')`** for 48GB VRAM efficiency.
- **Data Locality**: Automatic synchronization of dataset to node-local `/tmp` storage.

## How to Get Started

1. **Environment Setup**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Full Pipeline Execution (SLURM)**:
   ```bash
   PROFILE=SEARCHER MODEL_NAME=resnet50 ./submit_all_folds.sh
   ```

3. **Inference & Visualization**:
   Visuals are automatically generated in `results/`, including `_gradcam_samples/` for explainable diagnosis.

## Key Research Findings
- Only tuning weights will not result in a Searcher model that has high precision recall without collapsing accuracy.
- Fundamental changes to the way the Searcher model is trained must be done.