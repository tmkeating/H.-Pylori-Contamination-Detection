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
      - **Patient Consensus (Max Strategy)**: Flags patient as positive if any patch exceeds 0.90 probability (maximizes sensitivity in medical screening).
- `requirements.txt`: Python packages required.

## Data Strategy (Expanded Training)

The model utilizes an expanded dataset of **~54,000 images**:
- **High-Fidelity Corpus**: Pathologist-verified patches from the `Annotated` folders.
- **Supplemental Negatives**: High-confidence negative patches from confirmed healthy patients.
- **Scientific Validation**: Validated on a **Patient-Independent Split** (80/20 by Patient ID).

## Performance Highlights (Run 26-28: Optimized Pipeline)

### Key Metrics
- **Baseline Recall (Run 10)**: 87% on independent patients
- **Detection Threshold**: 0.2 (favors sensitivity in medical screening)
- **Patient Consensus**: Flags positive if any patch exceeds 0.90 probability
- **PR-AUC**: 0.94+ across diverse staining protocols
- **Loss Weight (Positive)**: 25.0 to compensate class imbalance

### Hardware Performance
- **Training Speed**: 262 images/second (7.5x faster than CPU baseline)
- **Epoch Time**: ~3.5 minutes (down from ~25 minutes)
- **Batch Processing**: Fully vectorized Macenko normalization on GPU
- **Hardware**: NVIDIA A40 (48GB VRAM) with local NVMe SSD caching

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

## Key Hyperparameters (Run 28)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Batch Size** | 128 | Optimized for 48GB A40 VRAM |
| **Learning Rate** | 5e-5 | Base rate for Adam optimizer |
| **Loss Weight (Positive)** | 25.0 | Compensates 54k:1k negative:positive ratio |
| **Detection Threshold** | 0.2 | For patch classification (favors recall) |
| **Patient Consensus Threshold** | 0.90 | For patient-level diagnostic |
| **Epochs** | 15 | With `ReduceLROnPlateau` scheduler |
| **Augmentation** | 90° rotation + color jitter | Robust against image orientation |

## Customization

- To use a different model (e.g., ResNet50), update [model.py](model.py).
- To adjust loss weight for positive class, update the `loss_weights` variable in [train.py](train.py).
- To change detection thresholds, modify `thresh = 0.2` and `max_prob > 0.90` in [train.py](train.py).
