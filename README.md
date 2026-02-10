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
    - Evaluates the final results on the `HoldOut` test set.
- `requirements.txt`: Python packages required.

## How to Get Started

1. **Environment Setup**:
   Install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

2. **Training**:
   Run the training script:
   ```bash
   python train.py
   ```

3. **Inference**:
   The script will automatically evaluate on the `HoldOut` set after training completes.

## Customization

- To use a different model (e.g., ResNet50), update [model.py](model.py).
- To change hyperparameters (learning rate, epochs, batch size), update [train.py](train.py).
