# H. Pylori Contamination Detection

This project implements **Fully Pre-trained Transfer Learning** to recognize contaminated samples of cell tissues (H. pylori).

## Project Structure

- `dataset.py`: Contains `HPyloriDataset` class which:
    - Loads patient labels from `PatientDiagnosis.csv`.
    - Maps labels: `NEGATIVA` -> 0, `BAIXA` -> 1, `ALTA` -> 1.
    - Collects images from patient-specific folders.
- `model.py`: Defines the transfer learning model using a pre-trained **ResNet18**.
- `train.py`: Main training script that:
    - Splits the annotated data into training and validation sets.
    - Trains the model and saves the best version as `best_model.pth`.
    - **Advanced Evaluation & Interpretability**: 
      - Automatically runs a full statistical report on the `HoldOut` test set.
      - Generates ROC curves, Confusion Matrices, Precision-Recall curves, and Learning Curves.
      - **Grad-CAM**: Produces heatmaps showing exactly which bacterial structures the model is "looking at" to make its decisions.
      - **Probability Histograms**: Visualizes the confidence distribution of model predictions.
- `requirements.txt`: Python packages required.

## Data Strategy (Expanded Training)

The model now utilizes an expanded dataset of **~54,000 images**:
- **High-Fidelity Corpus**: Pathologist-verified patches from the `Annotated` folders.
- **Supplemental Negatives**: High-confidence negative patches from the `Cropped` folders of patients with a confirmed 100% negative diagnosis. This increases the model's exposure to healthy tissue variations by over 20x.

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
