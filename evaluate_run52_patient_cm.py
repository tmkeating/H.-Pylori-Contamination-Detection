
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- Config ---
RUN_ID = "52_102093"
CSV_PATH = f"finalResults/{RUN_ID}_patient_consensus.csv"

def generate_patient_cm():
    print(f"--- Generating Patient-Level Confusion Matrix for Run {RUN_ID} ---")
    
    # 1. Load Data
    df = pd.read_csv(CSV_PATH)
    
    # Clean up any potential whitespace
    df['Actual'] = df['Actual'].str.strip()
    df['Predicted'] = df['Predicted'].str.strip()
    
    y_true = df['Actual']
    y_pred = df['Predicted']
    
    # labels order to ensure "Positive" and "Negative" are correctly placed
    labels = ["Negative", "Positive"]
    
    # 2. Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # 3. Plot
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='d')
    
    plt.title(f'Patient-Level Confusion Matrix (Run {RUN_ID})')
    
    output_path = f"{RUN_ID}_patient_confusion_matrix.png"
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved patient-level confusion matrix to {output_path}")
    
    # Print metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    print(f"\nPatient-Level Summary Metrics:")
    print(f"Accuracy:    {accuracy:.2%}")
    print(f"Sensitivity: {sensitivity:.2%}")
    print(f"Specificity: {specificity:.2%}")
    print(f"Total Patients: {len(df)}")

if __name__ == "__main__":
    generate_patient_cm()
