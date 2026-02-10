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
    - **Advanced Evaluation**: Automatically runs a full statistical report on the `HoldOut` test set after training, generating ROC curves, confusion matrices, and machine-readable CSVs.
- `requirements.txt`: Python packages required.

## How to Get Started

1. **Environment Setup**:
   Install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

2. **Training and Evaluation**:
   Run the training script (this will automatically evaluate the model once finished):
   ```bash
   python train.py
   ```

## Customization

- To use a different model (e.g., ResNet50), update [model.py](model.py).
- To change hyperparameters (learning rate, epochs, batch size), update [train.py](train.py).
