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
    - Trains the model and saves the best version as `best_model.pth`.
    - **Advanced Evaluation & Interpretability**: 
      - Automatically runs a full statistical report on the independent validation patients.
      - Generates ROC curves, Confusion Matrices, Precision-Recall curves, and Learning Curves.
      - **Grad-CAM**: Produces heatmaps showing exactly which bacterial structures the model is "looking at" to make its decisions.
      - **Probability Histograms**: Visualizes the confidence distribution of model predictions.
      - **Patient Consensus**: Groups patch-level results to provide a final diagnostic prediction for the whole patient sample.
- `requirements.txt`: Python packages required.

## Data Strategy (Expanded Training)

The model utilizes an expanded dataset of **~54,000 images**:
- **High-Fidelity Corpus**: Pathologist-verified patches from the `Annotated` folders.
- **Supplemental Negatives**: High-confidence negative patches from confirmed healthy patients.
- **Scientific Validation**: Validated on a **Patient-Independent Split** (80/20 by Patient ID).

## Performance Highlights (Run 10)

- **94.0% Precision**: Extremely low rate of false positives on unseen patient tissues.
- **87.0% Recall**: Strong sensitivity to bacterial presence in independent samples.
- **0.94 PR-AUC**: High clinical confidence across diverse patient staining.
- **93.4% Acc (Hard)**: Performance on difficult, pathologist-reviewed tissue.
- **99.9% Acc (Easy)**: Near-perfect cleaning of supplemental negative tissue.

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
   - Request 8 CPU cores and 32GB of RAM.
   - Redirect outputs to the `results/` directory automatically.
   - Use the high-performance `dcca40` partition.

## Customization

- To use a different model (e.g., ResNet50), update [model.py](model.py).
- To change hyperparameters (learning rate, epochs, batch size), update [train.py](train.py).
