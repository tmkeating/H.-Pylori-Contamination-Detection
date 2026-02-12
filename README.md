# H. Pylori Contamination Detection

This project implements **Fully Pre-trained Transfer Learning** to recognize contaminated samples of cell tissues (H. pylori).

## Project Structure

- `dataset.py`: Contains `HPyloriDataset` class which:
    - Loads patient labels from `PatientDiagnosis.csv`.
    - Maps labels: `NEGATIVA` -> 0, `BAIXA` -> 1, `ALTA` -> 1.
    - Collects images from patient-specific folders.
- `model.py`: Defines the transfer learning model using a pre-trained **ResNet18**.
- `train.py`: Main training script that:
    - **Patient-Level Splitting**: Implements a rigorous validation strategy where data is split by Patient ID. This ensures that patches from the same patient are never shared between training and validation sets, eliminating "Data Leakage."
    - **GPU-Accelerated Macenko Normalization**: Vectorized batch processing on NVIDIA A40 for 7.5x faster training (262 images/sec).
    - **Learning Rate Scheduler**: `ReduceLROnPlateau` to adaptively reduce learning rate when validation loss plateaus.
    - Trains the model and saves the best version as `best_model.pth`.
    - **Advanced Evaluation & Interpretability**: 
      - Automatically runs a full statistical report on the independent validation patients.
      - Generates ROC curves, Confusion Matrices, Precision-Recall curves, and Learning Curves.
      - **Grad-CAM**: Produces heatmaps showing exactly which bacterial structures the model is "looking at" to make its decisions.
      - **Probability Histograms**: Visualizes the confidence distribution of model predictions.
      - **Multi-Tier Consensus**: Implements a dual-gate diagnostic engine (Density + Signal Consistency) to achieve 100% Patient Recall, capturing even "weak stainers" while filtering out artifacts.
- `requirements.txt`: Python packages required.
- `normalization.py`: Hosts the GPU-vectorized Macenko stain normalization algorithm.

## Data Strategy (Expanded Training)

The model utilizes an expanded dataset of **~54,000 images**:
- **High-Fidelity Corpus**: Pathologist-verified patches from the `Annotated` folders.
- **Supplemental Negatives**: High-confidence negative patches from confirmed healthy patients.
- **Scientific Validation**: Validated on a **Patient-Independent Split** (80/20 by Patient ID).
- **Core AI Hardening**: Uses Label Smoothing (0.1) and morphological augmentations (Blur, Grayscale) to improve resilience against staining artifacts.

## Performance Highlights (Final Model: Run 34)

### Key Metrics
- **Patient-Level Recall**: **100%** (4/4 positive patients detected)
- **Patient-Level Accuracy**: **93.5%**
- **Patch-Level Accuracy**: **98%**
- **Throughput**: **262 images/second** (7.5x speedup via GPU-normalization)

### Hardware Performance
- **Vectorized Preprocessing**: Shifted Macenko normalization to GPU using `torch.linalg.pinv`, cutting epoch time from 25 mins to <4 mins on NVIDIA A40.
- **VRAM Utilization**: Efficiently handles 448x448 high-resolution patches at batch-size 64.
- **Cluster Efficiency**: Optimized for SLURM `dcca40` with local NVMe SSD caching.

## Diagnostic Architecture (Multi-Tier Consensus)

The final model moves beyond simple thresholding to a sophisticated consensus logic:
1. **Density Gate**: Flags infection if $\ge 10$ patches show $> 90\%$ confidence (Resilient against focal noise).
2. **Consistency Gate**: Flags infection if mean probability across all patches is $> 50\%$ with high signal stability (Catches "weak stainers" with widespread low-level signal).

## How to Get Started

1. **Environment Setup**:
   Install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

2. **Training and Evaluation**:
   You can run the training script directly:
   ```bash
   python train.py
   ```

3. **Running on SLURM Cluster**:
   For large-scale training on the DCC cluster, use the provided batch script:
   ```bash
   sbatch run_h_pylori.sh
   ```
   This script is pre-configured to:
   - Request 8 CPU cores and 48GB of RAM (optimized for A40).
   - Automatically copies 11GB dataset to local NVMe SSD for fast I/O.
   - Redirect outputs to the `results/` directory automatically.
   - Use the high-performance `dcca40` partition with NVIDIA A40 GPU.

## Key Hyperparameters (Final Model)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Batch Size** | 64 | Tuned for 448x448 resolution on 48GB A40 |
| **Input Resolution** | 448x448 | Required to resolve small bacillary morphology |
| **Loss Weight (Pos)** | 10.0 | Balanced for high sensitivity vs FP reduction |
| **Label Smoothing** | 0.1 | Prevents over-confidence on stain artifacts |
| **Augmentations** | Blur, Grayscale | Forces model to focus on morphology, not color |
| **Consensus N** | 10 | Minimum patches to trigger Density Gate |
| **Consensus P** | 0.90 | Confidence threshold for Density patches |
| **Consistency Gate** | Mean > 0.5 | Catches wide-spread consistent infection |

## Customization

- To adjust diagnostic sensitivity, update the `predict_patient` function in [train.py](train.py).