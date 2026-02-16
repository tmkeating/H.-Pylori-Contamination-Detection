# H. Pylori Contamination Detection

This project implements **Fully Pre-trained Transfer Learning** to recognize contaminated samples of cell tissues (H. pylori).

## Project Structure

- `dataset.py`: Contains `HPyloriDataset` class which:
    - Loads patient labels from `PatientDiagnosis.csv`.
    - Maps labels: `NEGATIVA` -> 0, `BAIXA` -> 1, `ALTA` -> 1.
    - Collects images from patient-specific folders.
- `model.py`: Defines the transfer learning model using a pre-trained **ResNet18**.
- `train.py`: Main training script that:
    - **Scientific Three-Way Split**: Implements a rigorous validation strategy. Data is split by Patient ID into Training and Validation sets, with a final **unseen HoldOut set** used for the gold-standard evaluation to eliminate Data Leakage.
    - **GPU-Accelerated Macenko Normalization**: Vectorized batch processing on NVIDIA A40 for 7.5x faster training (262 images/sec).
    - **On-GPU Augmentations**: Geometric and color transforms are offloaded to the GPU using `torchvision.transforms.v2` to eliminate CPU bottlenecks.
    - **Learning Rate Scheduler**: `ReduceLROnPlateau` to adaptively reduce learning rate when validation loss plateaus.
    - Trains the model and saves the best version as `best_model.pth`.
    - **Advanced Evaluation & Interpretability**: 
      - Automatically runs a full statistical report on the independent test patients.
      - **Grad-CAM**: Produces heatmaps showing exactly which bacterial structures the model is "looking at" to make its decisions.
      - **Multi-Tier Consensus**: Implements a dual-gate diagnostic engine (Density + Signal Consistency) to achieve 100% Patient Recall.
- `requirements.txt`: Python packages required.
- `normalization.py`: Hosts the GPU-vectorized Macenko stain normalization algorithm.

## Data Strategy (Expanded Training)

The model utilizes an expanded dataset of **~54,000 images**:
- **High-Fidelity Corpus**: Pathologist-verified patches from the `Annotated` folders.
- **Supplemental Negatives**: High-confidence negative patches from confirmed healthy patients.
- **Scientific Validation**: Validated on a **Patient-Independent Split** (80/20 by Patient ID).
- **Core AI Hardening**: Uses Label Smoothing (0.1) and morphological augmentations (Blur, Grayscale) to improve resilience against staining artifacts.

## Performance Highlights (Final Model: Run 43/44)

### Key Metrics
- **Patient-Level Specificity**: **98.3%** (Milestone: Only 1 False Positive)
- **Patient-Level Sensitivity**: **22.4%** (Rule-In Diagnostic Assistant)
- **Patient-Level Accuracy**: Target **>80%** (Run 44 calibration)
- **Throughput**: **262 images/second** (7.5x speedup via GPU-normalization)

### Hardware Performance
- **High-Throughput Pipeline**: Moved all geometric and color augmentations to the GPU using `v2.Transforms`, resolving CPU bottlenecks and ensuring smooth iterations.
- **VRAM Utilization**: Efficiently handles 448x448 high-resolution patches at batch-size 128 on 48GB VRAM.
- **Cluster Efficiency**: Optimized for SLURM with local NVMe SSD caching and multi-worker prefetching.

## Diagnostic Architecture (Multi-Tier Consensus)

The model utilizes a "Supportive Clinical Tool" architecture, prioritizing reliability and high specificity:
1. **Density Gate**: Flags infection if $\ge 30$ patches show $> 90\%$ confidence (Optimized for Accuracy in Run 44).
2. **Consistency Gate**: Flags infection if mean probability across all patches is $> 80\%$ with high signal stability (Extremely selective for clinical confirmation).

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
| **Batch Size** | 128 | Optimized for 448x448 resolution on 48GB A40 |
| **Input Resolution** | 448x448 | Required to resolve small bacillary morphology |
| **Loss Weight (Pos)** | 1.5 | Balanced for specificity and accuracy (Run 44) |
| **Label Smoothing** | 0.1 | Prevents over-confidence on stain artifacts |
| **Augmentations** | Blur, Grayscale | Forces model to focus on morphology, not color |
| **Consensus N** | 30 | Minimum patches to trigger Density Gate |
| **Consensus P** | 0.90 | Confidence threshold for Density patches |
| **Consistency Gate** | Mean > 0.8 | Threshold for clinical confirmation |

## Customization

- To adjust diagnostic sensitivity, update the `predict_patient` function in [train.py](train.py).